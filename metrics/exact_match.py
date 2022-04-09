#!/usr/bin/env python
# coding=utf-8 
"""
Compute the exact match score
"""
# Code copied from SQUAD and SCROLLS:
# https://huggingface.co/datasets/tau/scrolls/tree/main/metrics 
# https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py

from .metrics_utils import normalize_text, metric_max_over_ground_truths

def EM_score(prediction, truth):
    """Compute the EM score between prediction and ground truth"""
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_exact_match(preds, refs):
    """Get the EM score for all the predictions"""
    exact_match = 0
    for prediction, ground_truths in zip(preds, refs):
        if not isinstance(ground_truths, list):
            ground_truths = [ground_truths]
        em_score = metric_max_over_ground_truths(EM_score, prediction, ground_truths)
        exact_match += em_score
        
    return 100.0 * exact_match / len(preds)
