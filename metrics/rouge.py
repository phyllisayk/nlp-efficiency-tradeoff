# 

from .metrics_utils import normalize_text, metric_max_over_ground_truths
from datasets import load_metric


def compute_rouge(predictions, references, rouge_types=None, use_stemmer=False):
    rouge_list = ['rouge1', 'rouge2', 'rougeL']
    scores = {}

    predictions = [normalize_text(prediction).split() for prediction in predictions]
    references = [normalize_text(reference).split() for reference in references]

    metric = load_metric("rouge")
    rouge = metric.compute(predictions=predictions, references=references, use_agregator=False)

    for rouge_type in rouge_list:
        score_list = [score.fmeasure for score in rouge[rouge_type]]
        scores[rouge_type] = (sum(score_list) * 100.0) / len(score_list)
    
    return scores
       
