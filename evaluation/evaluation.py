import collections
import math
import logging
from mrqa_official_eval import exact_match_score, f1_score, metric_max_over_ground_truths
from transformers import BasicTokenizer

logger = logging.getLogger(__name__)


def make_predictions(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
):
    example_index_to_features = collections.defaultdict(list)
    unique_id_to_result = {}

    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_pred = _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result.start_logits[start_index],
                        end_logit=result.end_logits[end_index]
                    )
                    prelim_predictions.append(prelim_pred)

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True
        )

        _NbestPrediction = collections.namedtuple(
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []

        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break

            feature = features[pred.feature_index]

            if pred.start_index > 0:
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)
                final_text = get_final_text(tok_text, orig_text)

                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True

            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest_pred = _NbestPrediction(
                text=final_text,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit
            )
            nbest.append(nbest_pred)

        if not nbest:
            nbest_pred = _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0)
            nbest.append(nbest_pred)

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None

        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)
        nbest_json = []

        for i, entry in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json


def get_final_text(pred_text, orig_text):
    r""" Project the tokenized prediction back to the original text. """
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()

        for (i, c) in enumerate(text):
            if c == " ":
                continue

            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)

        ns_text = "".join(ns_chars)

        return (ns_text, ns_to_s_map)

    tokenizer = BasicTokenizer()
    tok_text = " ".join(tokenizer.tokenize(orig_text))
    start_position = tok_text.find(pred_text)

    if start_position == -1:
        logger.debug(
            f"Unable to find text: '{pred_text}' in '{orig_text}'"
        )
        return orig_text

    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        logger.debug(
            f"Length not equal after stripping spaces: '{orig_ns_text}' vs '{tok_ns_text}'"
        )
        return orig_text

    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        logger.debug("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        logger.debug("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    r""" Get the n-best logits from a list. """
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []

    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])

    return best_indexes


def _compute_softmax(scores):
    r""" Compute softmax probability over raw logits. """
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def get_raw_scores(dataset, predictions):
    r""" Get list of EM and F1 for each question. """
    answers = {}

    for example in dataset:
        for qa in example['qas']:
            answers[qa['qid']] = qa['answers']

    exact_scores = {}
    f1_scores = {}

    for qid, ground_truths in answers.items():
        if qid not in predictions:
            print('Missing prediction for %s' % qid)
            continue

        prediction = predictions[qid]
        exact_scores[qid] = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1_scores[qid] = metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    return exact_scores, f1_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    r""" Group together predictions over the questions and average. """
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            exact=sum(exact_scores.values()) / total,
            f1=sum(f1_scores.values()) / total,
            total=total,
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            exact=sum(exact_scores[k] for k in qid_list) / total,
            f1=sum(f1_scores[k] for k in qid_list) / total,
            total=total,
        )