# MAIRA-RRG

This repository contains scripts and configurations for fine-tuning **MAIRA-2** using **LoRA**.

## Introduction

**MAIRA-RRG** is designed to streamline the fine-tuning process of the MAIRA-2 model leveraging Low-Rank Adaptation (**LoRA**). This repository includes all necessary scripts and configurations to get you started quickly.

## Setup and Usage

### Prerequisites

Before you begin, ensure you have the following installed:

- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.10
- Git

### Installation Steps

Follow these steps to set up the environment and get the project running:

```bash
# Step 1: Clone the Repository
git clone https://github.com/yourusername/MAIRA-RRG.git
cd MAIRA-RRG

# Step 2: Create Conda Environment
conda create -n maira python=3.10
conda activate maira

# Step 3: Install Requirements
pip install -r requirements.txt

# Step 4: Prepare the Data
# Place required JSON files in the 'data/' directory
# Ensure paths in 'config.yaml' (e.g., mimic_cxr_dir) are correctly set.

# Step 5: Generate Dataset Splits
python create_dataset.py

# Step 6: Preprocess Data
python preprocess.py

# Step 7: Train the Model
python train.py
