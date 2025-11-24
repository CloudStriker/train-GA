import sys
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import logging
import json
import gc

import torch
import transformers
from transformers import Trainer
from datasets import Dataset, load_dataset
import datasets
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel, LoraRuntimeConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("⚠️  bitsandbytes not available. Quantization will be disabled.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available. Experiment tracking will be disabled.")

IGNORE_INDEX = -100
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
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        tokenizer = kwargs.get("tokenizer", self.tokenizer)
        tokenizer.save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: 
            return None
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0: 
            return None
        latest_ckpt_dir = os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples["instruction"]]
    targets = [f"{output}\n{tokenizer.eos_token}" for output in examples["output"]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def load_sft_dataset(file_path: str, max_examples: int = None) -> Dataset:
    logger.info(f"Loading SFT dataset from {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_examples and len(data) >= max_examples:
                break
                
            try:
                item = json.loads(line.strip())
                if 'instruction' in item and 'output' in item:
                    if len(item['instruction']) + len(item['output']) < 8000:
                        data.append(item)
                else:
                    logger.warning(f"Line {line_num}: Missing required fields, skipping")
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error - {e}")
                continue
    
    logger.info(f"Loaded {len(data)} examples from {file_path}")
    dataset = Dataset.from_list(data)
    return dataset

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def build_model(model_name, use_quantization=True, lora_rank=32):
    clear_memory()
    compute_dtype = torch.float16
    
    quantization_config = None
    if use_quantization and BITSANDBYTES_AVAILABLE:
        logger.info("Loading model with 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif use_quantization and not BITSANDBYTES_AVAILABLE:
        logger.warning("bitsandbytes not available. Falling back to FP16.")
    else:
        logger.info("Loading model in FP16 precision with LoRA")
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )
    
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    
    if use_quantization and BITSANDBYTES_AVAILABLE:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    logger.info(f'Init LoRA modules with rank {lora_rank}...')
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        inference_mode=False,
        r=lora_rank, 
        lora_alpha=lora_rank,
        lora_dropout=0.0,
        init_lora_weights=True,
    )
    model = get_peft_model(model, peft_config)

    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)
    
    clear_memory()
    return model

def main():    
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    
    DATA_PATH = "ClickNoow/Terese-v11-Training-Dataset"
    
    OUTPUT_DIR = "Terese_v11/qwen25-rehearsal-sft"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO)
    logger.info("🚀 Starting simple Qwen 2.5 training...")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Data: {DATA_PATH}")
    logger.info(f"Output: {OUTPUT_DIR}")
    
    if WANDB_AVAILABLE:
        wandb.init(
            project="qwen-simple-sft",
            name="qwen25-7b-lora-4bit-simple",
            config={
                "model": MODEL_NAME,
                "lora_rank": 32,
                "bits": 4,
                "learning_rate": 2e-4,
                "batch_size": 1,
                "gradient_accumulation_steps": 32,
                "max_length": 1024,
            }
        )
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_NAME,
        model_max_length=1024,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("✅ Tokenizer loaded")
    
    model = build_model(MODEL_NAME, use_quantization=True, lora_rank=32)
    logger.info("✅ Model loaded")
    print(model)

    logger.info(f"📥 Downloading dataset from Hugging Face: {DATA_PATH}")
    try:
        raw_train_dataset = load_dataset(DATA_PATH, split="train")
    except Exception as e:
        logger.error(f"Failed load dataset: {e}")
        return

    column_names = raw_train_dataset.column_names
    logger.info(f"dataset: {column_names}")

    mappings = {
        "question": "instruction",
        "input": "instruction", 
        "response": "output",
        "answer": "output"
    }
    
    for old_name, new_name in mappings.items():
        if old_name in column_names and new_name not in column_names:
            logger.info(f"Renaming column '{old_name}' to '{new_name}'")
            raw_train_dataset = raw_train_dataset.rename_column(old_name, new_name)

    raw_train_dataset = raw_train_dataset.filter(
        lambda x: x.get('instruction') is not None and x.get('output') is not None
    )
    
    raw_train_dataset = raw_train_dataset.filter(
        lambda x: len(str(x['instruction']) + str(x['output'])) < 8000
    )

    raw_train_dataset = raw_train_dataset.shuffle(seed=42)
        
    train_dataset = raw_train_dataset.map(
        train_tokenize_function,
        batched=True,
        batch_size=500,
        num_proc=2,
        remove_columns=raw_train_dataset.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer}
    )
    
    logger.info(f"✅ Dataset loaded: {len(train_dataset)} examples")
    
    for index in random.sample(range(len(train_dataset)), min(3, len(train_dataset))):
        logger.info(f"Sample {index}: {train_dataset[index]['input_ids'][:20]}...")

    training_args = transformers.TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        learning_rate=2e-4,
        weight_decay=0.0,
        warmup_ratio=0.03,
        logging_steps=5,
        lr_scheduler_type="cosine",
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="wandb" if WANDB_AVAILABLE else "none",
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model, 
        processing_class=tokenizer, 
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    trainer.add_callback(SavePeftModelCallback(tokenizer))
    
    clear_memory()
    
    logger.info("🏋️  Starting training...")
    trainer.train()
    trainer.save_state()
    
    logger.info("💾 saving final model...")
    trainer.save_model(OUTPUT_DIR)
    
    logger.info("✅ Training completed! Adapter saved.")
    logger.info(f"📁 Adapter saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
