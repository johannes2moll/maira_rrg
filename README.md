# MAIRA-RRG

This repository contains scripts and configurations for fine-tuning MAIRA-2 using LoRA.
All parameters for training can be set in config.yaml

Train on findings by changing the create_dataset and train 'impression' parts.
## Setup

### Installation Steps

Follow these steps to set up the environment and get the project running:

```bash
# Step 1: Clone the Repository
git clone https://github.com/alothomas/maira_rrg.git
cd maira_rrg

# Step 2: Create Conda Environment
conda create -n maira python=3.10
conda activate maira

# Step 3: Install Requirements
pip install -r requirements.txt

# Step 4: Prepare the Data
# Place required JSON files in the 'data/' directory
# Ensure paths in 'config.yaml' (e.g., mimic_cxr_dir) are correctly set (set mimic images path).

# Step 5: Generate Dataset Splits
python create_dataset.py

# Step 6: Train the Model
./train.sh

# Step 7: Generate Prediction on Test Set
python run_model.py

# Step 8: Evaluate
python eval_model.py
