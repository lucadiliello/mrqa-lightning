import collections
import logging
import re
import string
from collections import Counter
from typing import Dict, List

from datasets.arrow_dataset import Dataset

from utilities.utilities import list_to_dict


logger = logging.getLogger(__name__)


def make_predictions(
    all_examples: Dataset,
    all_features: Dataset,
    all_results: List[Dict],
    n_best_size: int,
    max_answer_length: int,
):
    index_to_features = collections.defaultdict(list)
    for feature in all_features:
        index_to_features[feature['index']].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result['unique_id']] = result

    all_predictions = collections.OrderedDict()

    for example in all_examples:
        features = index_to_features[example['index']]
        preliminary_predictions = []

        for feature_index, feature in enumerate(features):
            result = unique_id_to_result[feature['unique_id']]
            start_indexes = get_best_indexes(result['start_logits'], n_best_size)
            end_indexes = get_best_indexes(result['end_logits'], n_best_size)
            covered_tokens = feature['covered_tokens']

            # whether a token's context is maximised in the given span
            feature['token_is_max_context'] = list_to_dict(feature['token_is_max_context'])

            for start_index in start_indexes:
                if start_index >= len(feature['tokens']):
                    continue
                if start_index not in covered_tokens:
                    continue
                if not feature['token_is_max_context'].get(start_index, False):
                    continue

                for end_index in end_indexes:
                    if end_index >= len(feature['tokens']):
                        continue
                    if end_index not in covered_tokens:
                        continue
                    if end_index < start_index:
                        continue
                    if (end_index - start_index) >= max_answer_length:
                        continue

                    prelim_pred = dict(
                        feature_index=feature_index,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result['start_logits'][start_index],
                        end_logit=result['end_logits'][end_index]
                    )
                    preliminary_predictions.append(prelim_pred)

        preliminary_predictions = sorted(
            preliminary_predictions, key=lambda x: (x['start_logit'] + x['end_logit']), reverse=True
        )

        # get exact text span of best prediction
        final_text = ""
        if preliminary_predictions:
            pred = preliminary_predictions[0]
            if pred['start_index'] > 0:
                offset_mapping = features[pred['feature_index']]['offset_mapping']
                # retrieve start and last character in original context
                start_char = offset_mapping[pred['start_index']][0]
                end_char = offset_mapping[pred['end_index']][1]
                final_text = example['context'][start_char:end_char]
            else:
                final_text = None

        all_predictions[example['question_id']] = final_text

    return all_predictions


def get_best_indexes(logits, n_best_size):
    r""" Get the n-best logits from a list. """
    index_and_scores = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    return [x[0] for x in index_and_scores[:n_best_size]]


def get_raw_scores(dataset, predictions):
    r""" Get list of EM and F1 for each question. """
    answers = {example['question_id']: example['all_answers'] for example in dataset}

    exact_scores = {}
    f1_scores = {}
    for qid, ground_truths in answers.items():
        prediction = predictions[qid]
        exact_scores[qid] = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1_scores[qid] = metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    return exact_scores, f1_scores


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    r""" Compare prediction with every ground truth and return max pair score. """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths, default=prediction is None)


def normalize_answer(text: str):
    r""" Lower text and remove punctuation, articles and extra whitespace. """
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text.lower() if ch not in exclude)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    return text


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
    r""" Exact match requires exact prediction of text span. """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))
