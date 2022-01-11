# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""Official evaluation script for the MRQA Workshop Shared Task.
Adapted fromt the SQuAD v1.1 official evaluation script.
Usage:
    python official_eval.py dataset_file.jsonl.gz prediction_file.json
"""
# Copied and adapted from https://github.com/mrqa/MRQA-Shared-Task-2019/blob/b86bbcc63352786f0ca246b76191f062dce4b52b/mrqa_official_eval.py

import string
import re
import json
import gzip
from collections import Counter


def normalize_answer(s):
    r""" Lower text and remove punctuation, articles and extra whitespace. """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    r""" Compute F1 score between float predictions and labels. """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def read_predictions(prediction_file):
    with open(prediction_file) as f:
        predictions = json.load(f)
    return predictions


def read_answers(gold_file):
    answers = {}
    with gzip.open(gold_file, 'rb') as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            if i == 0 and 'header' in example:
                continue
            for qa in example['qas']:
                answers[qa['qid']] = qa['answers']
    return answers
