import pandas as pd 
import numpy as np
import evaluate
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

    metrics = {
            "BLEU": bleu_results["bleu"],
            "ROUGE-1": rouge_results["rouge1"],
            "ROUGE-2": rouge_results["rouge2"],
            "ROUGE-L": rouge_results["rougeL"],
            "BERT": np.mean(bertscore_results["f1"]),
            

        }
    
    # Save metrics to a file
    output_metrics_file = f"evaluation_metrics.json"
    with open(output_metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {output_metrics_file}")

if __name__ == "__main__":
    main()