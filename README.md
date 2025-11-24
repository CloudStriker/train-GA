# How To
1. Create python environment using uv or python -m
   ```bash
   uv venv
   # or
   python -m venv venv
   ```
2. Use that environment
3. Install all the requirements
   ```bash
   uv pip install -r requirements.txt
   # or
   pip install -r requirements.txt
   ```
4. (Optional) Insert hf auth token to download QWEN Model
   ```bash
   hf auth login
   ```
5. Prepare the dataset, the dataset is 5000 rows (4500 + 500 rehearsal)
6. Change the name of the dataset file in `train.py` line 244
7. Run the script
   ```bash
   uv run train.py
   or
   python train.py
   ```