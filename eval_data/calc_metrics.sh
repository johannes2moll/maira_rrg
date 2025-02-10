#!/bin/bash

#MODEL="llama-1b"  #llama-1b, llama-3b, llama-8b, llama-70b, vicuna, medalpaca-7b, medalpaca-13b, mistral-nemo, mistral-7b, phi3
MODEL=''
CASE_ID=0
PRED_FILE_MIMIC='generated_reports_radialog_mimic_impression.json'
PRED_FILE_CHEXPERT=''
REF_DATA_PATH="StanfordAIMI/srrg_findings_impression"
OUTPUT_FILE="metrics_radialog_mimic_impression.json"

python src/calc_metrics.py \
    --pred_file_mimic "$PRED_FILE_MIMIC" \
    --pred_file_chexpert "$PRED_FILE_CHEXPERT" \
    --ref_data_path "$REF_DATA_PATH" \
    --output_file "$OUTPUT_FILE" 
