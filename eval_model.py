import pandas as pd 
import numpy as np
import evaluate
from radgraph import F1RadGraph
from f1chexbert import F1CheXbert
import json

file_name = 'generated_reports.csv'

def main():
    # load data from csv file
    data = pd.read_csv(file_name)
    #{"input": list_inp, "generated": list_gen, "reference": list_refs}
    predictions = data["generated"]
    references = data["reference"]

    print(f"Loaded {len(predictions)} predictions")
    print(f"Loaded {len(references)} references")
    print("prediction: ",predictions[0])
    print("reference: ",references[0])
    # Initialize the metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    # BLEU
    bleu_results = bleu.compute(predictions=predictions, references=[[ref] for ref in references])

    # ROUGE
    rouge_results = rouge.compute(predictions=predictions, references=references)

    # BERTScore (using default RoBERTa model)
    bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en")

    # F1-RadGraph (custom implementation)
    predictions = predictions.tolist()
    references = references.tolist()
    #f1_radgraph = F1RadGraph(reward_level="partial")
    #f1_radgraph_results, _, _, _ = f1_radgraph(hyps=predictions, refs=references)
    f1_radgraph_results = 0.0
    
    f1chexbert = F1CheXbert(device="cuda")
    per_sample_chexbert_scores = []

    for pred, target in zip(predictions, references):
        # Compute F1 score for each sample individually
        _, _, _, class_report = f1chexbert(hyps=[pred], refs=[target])  # Wrapping pred and target in lists
        per_sample_f1 = class_report["micro avg"]["f1-score"]
        per_sample_chexbert_scores.append(per_sample_f1)

    metrics = {
            "BLEU": bleu_results["bleu"],
            "ROUGE-1": rouge_results["rouge1"],
            "ROUGE-2": rouge_results["rouge2"],
            "ROUGE-L": rouge_results["rougeL"],
            "BERT": np.mean(bertscore_results["f1"]),
            "F1-Radgraph": f1_radgraph_results,
            "CheXbert": np.mean(per_sample_chexbert_scores)
        }
    
    # Save metrics to a file
    output_metrics_file = f"evaluation_metrics.json"
    with open(output_metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {output_metrics_file}")

if __name__ == "__main__":
    main()