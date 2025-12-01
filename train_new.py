import os
import sys
import shutil
import logging
import json
import gc
import random
import glob
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import numpy as np
import transformers
from transformers import Trainer, BitsAndBytesConfig
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import HfApi

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "ClickNoow/Terese-v12-Training-Dataset"
REPO_ID = "ClickNoow/Terese-v12-adapter"
OUTPUT_DIR = "Terese_v12/qwen25-rehearsal-sft"
IGNORE_INDEX = -100

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

class SavePeftModelCallback(transformers.TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}", "adapter_model")
        
        kwargs["model"].save_pretrained(checkpoint_folder)
        kwargs.get("tokenizer", self.tokenizer).save_pretrained(checkpoint_folder)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)

def _tokenize_fn(strings, tokenizer):
    tokenized_list = [
        tokenizer(
            text,
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]
    return dict(input_ids=input_ids, labels=labels, input_ids_lens=input_ids_lens, labels_lens=labels_lens)

def preprocess(sources, targets, tokenizer):
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = [x.copy() for x in input_ids]
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in labels], batch_first=True, padding_value=IGNORE_INDEX)
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

def train_tokenize_function(examples, tokenizer):
    sources = [PROMPT.format_map(dict(instruction=i)) for i in examples["instruction"]]
    targets = [f"{o}\n{tokenizer.eos_token}" for o in examples["output"]]
    return preprocess(sources, targets, tokenizer)

def main():
    logging.basicConfig(level=logging.INFO)
    try:
        import wandb
        wandb.init(project="qwen-simple-sft", name="qwen25-7b-lora-4bit-auto-upload")
    except ImportError:
        pass

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=1024, padding_side="right", use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
    )
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="sdpa", device_map="auto"
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.0, init_lora_weights=True
    )
    model = get_peft_model(model, peft_config)
    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name: module = module.to(torch.float32)

    raw_train_dataset = load_dataset(DATA_PATH, split="train")
    if "question" in raw_train_dataset.column_names: raw_train_dataset = raw_train_dataset.rename_column("question", "instruction")
    if "response" in raw_train_dataset.column_names: raw_train_dataset = raw_train_dataset.rename_column("response", "output")
    
    train_dataset = raw_train_dataset.filter(lambda x: x.get('instruction') and x.get('output')).map(
        train_tokenize_function, batched=True, batch_size=500, remove_columns=raw_train_dataset.column_names, fn_kwargs={"tokenizer": tokenizer}
    )

    training_args = transformers.TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=3, per_device_train_batch_size=16, gradient_accumulation_steps=2,
        save_strategy="steps", save_steps=100, save_total_limit=1, learning_rate=2e-4, weight_decay=0.0,
        warmup_ratio=0.03, logging_steps=5, lr_scheduler_type="cosine", fp16=True, gradient_checkpointing=True,
        report_to="wandb" if "wandb" in sys.modules else "none"
    )

    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, train_dataset=train_dataset,
        data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    )
    trainer.add_callback(SavePeftModelCallback(tokenizer))
    
    logger.info("🏋️ Starting training...")
    trainer.train()
    
    # --- Auto Upload Logic ---
    logger.info("Training complete. Searching for latest checkpoint to upload...")
    
    # Find the checkpoint directory with the highest number
    checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    if not checkpoints:
        logger.error(f"No checkpoints found in {OUTPUT_DIR}. Cannot upload.")
        return

    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    adapter_path = os.path.join(latest_checkpoint, "adapter_model")
    
    if not os.path.exists(adapter_path):
        logger.error(f"Adapter folder not found at {adapter_path}")
        return

    logger.info(f"Found adapter at: {adapter_path}")
    logger.info(f"Uploading to Hugging Face: {REPO_ID}")

    try:
        api = HfApi()
        api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=adapter_path, repo_id=REPO_ID, repo_type="model")
        logger.info(f"\n✅ Success! Adapter uploaded to https://huggingface.co/{REPO_ID}")
    except Exception as e:
        logger.error(f"Upload failed: {e}")

if __name__ == "__main__":
    main()