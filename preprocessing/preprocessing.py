import collections
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import gzip
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from multiprocessing import Pool
from functools import partial


@dataclass
class MRQAExample:
    r""" A single training/test example for the MRQA dataset.
    For examples without an answer, the start and end position are -1.
    """
    qas_id: int
    question_text: str
    doc_tokens: List[str]
    orig_answer_text: str = None
    start_position: int = None
    end_position: int = None


@dataclass
class InputFeatures:
    r""" A single set of features of data. """

    unique_id: int
    example_index: int
    doc_span_index: int
    tokens: List
    token_to_orig_map: List
    token_is_max_context: bool
    input_ids: List
    input_mask: List
    segment_ids: List
    start_position: int = None
    end_position: int = None


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def process_entry(entry: Dict, is_training: bool = True) -> MRQAExample:
    r"""
    Process a single dataset entry.
    """

    examples = []
    paragraph_text = entry["context"]
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True

    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False

        char_to_word_offset.append(len(doc_tokens) - 1)

    for qa in entry["qas"]:
        qas_id = qa["qid"]
        question_text = qa["question"]
        start_position = None
        end_position = None
        orig_answer_text = None

        if is_training:
            answers = qa["detected_answers"]
            spans = sorted([span for spans in answers for span in spans['char_spans']])
            # take first span
            char_start, char_end = spans[0][0], spans[0][1]
            orig_answer_text = paragraph_text[char_start:char_end+1]
            start_position, end_position = char_to_word_offset[char_start], char_to_word_offset[char_end]

        example = MRQAExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position
        )

        examples.append(example)

    return examples


def read_mrqa_examples(input_file, is_training) -> Tuple[List, List]:
    r"""
    Read a MRQA json file into a list of MRQAExample.
    """

    if isinstance(input_file, str):
        input_file = [input_file]

    input_data = []
    for f in input_file:
        with gzip.GzipFile(f, 'r') as reader:
            # skip header
            content = reader.read().decode('utf-8').strip().split('\n')[1:]
            input_data += [json.loads(line) for line in content]

    examples = [x for example in tqdm(input_data, desc="Processing...") for x in process_entry(example, is_training=is_training)]
    return input_data, examples


def convert_examples_to_features(
    examples: List,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    doc_stride: int,
    max_query_length: int,
    is_training: bool,
    preprocessing_workers: int,
) -> List[InputFeatures]:

    mp_function = partial(
        convert_example_to_features,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=is_training
    )

    with Pool(processes=preprocessing_workers) as pool:
        features = [
            x for processed_batch in pool.map(
                func=mp_function,
                iterable=tqdm(enumerate(examples), total=len(examples), desc="Converting samples to features"),
                chunksize=1000,
            ) for x in processed_batch
        ]
        unique_id_start = 10000000
        for i, feat in enumerate(features):
            feat.unique_id = unique_id_start + i

    return features


def convert_example_to_features(
    example: MRQAExample,
    tokenizer: PreTrainedTokenizerBase = None,
    max_sequence_length: int = None,
    doc_stride: int = None,
    max_query_length: int = None,
    is_training: bool = None,
) -> List[InputFeatures]:
    r""" Loads a data file into a list of `InputBatch`s. """

    example_index, example = example
    features = []
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0: max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []

    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None

    if is_training:
        tok_start_position = -1
        tok_end_position = -1

    if is_training:
        tok_start_position = orig_to_tok_index[example.start_position]

        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.orig_answer_text
        )

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_sequence_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])

    doc_spans = []
    start_offset = 0

    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset

        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))

        if start_offset + length == len(all_doc_tokens):
            break

        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)

        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)

        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_sequence_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_sequence_length
        assert len(input_mask) == max_sequence_length
        assert len(segment_ids) == max_sequence_length

        start_position = None
        end_position = None

        if is_training:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        feature = InputFeatures(
            unique_id=None,
            example_index=example_index,
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position
        )
        features.append(feature)

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    r""" Returns tokenized answer spans that better match the annotated answer. """
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:new_end + 1])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    r"""Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None

    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1

        if position < doc_span.start:
            continue
        if position > end:
            continue

        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length

        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index