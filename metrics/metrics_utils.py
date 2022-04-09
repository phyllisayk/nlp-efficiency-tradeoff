#!/usr/bin/env python
# coding=utf-8 
"""
Common functions used across the different metrics
"""
# Code copied from SQUAD and SCROLLS:
# https://huggingface.co/datasets/tau/scrolls/tree/main/metrics 
# https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py

import re
import string

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compare each prediction with all its possible ground truths and return the max score"""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)