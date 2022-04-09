#!/usr/bin/env python
# coding=utf-8 
"""
Common the f1 score
"""
# Code copied from SQUAD and SCROLLS: 
# https://huggingface.co/datasets/tau/scrolls/tree/main/metrics
# https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py

from collections import Counter

from .metrics_utils import normalize_text, metric_max_over_ground_truths

def f1_score(prediction, ground_truth):
    """Compute the f1 score between prediction and ground truth"""
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_f1(preds, refs):
    """Get the f1 score for all the predictions"""
    f1 = 0
    for prediction, ground_truths in zip(preds, refs):
        if not isinstance(ground_truths, list):
            ground_truths = [ground_truths]
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    return 100.0 * f1 / len(preds)