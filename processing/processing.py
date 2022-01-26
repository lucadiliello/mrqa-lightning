import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from datasets import Dataset, load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from utilities.utilities import (
    SPLITS_TO_SUBSETS,
    SUBSETS_TO_IDS,
    UNIQUE_ID_START,
    check_is_max_context,
    clean_text,
    dict_to_list,
)


logger = logging.getLogger(__name__)


def load_from_datasets(
    split: str,
    subsets: List[str],
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    doc_stride: int,
    max_query_length: int,
    preprocessing_workers: int = 16,
    load_from_cache_file: bool = False,
) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    r""" Read datasets from datasets library and preprocess examples. """

    # at the moment we only accept right padding
    assert tokenizer.padding_side == 'right', "left padding not supported yet"

    # checks
    if split not in SPLITS_TO_SUBSETS:
        raise ValueError(f'split {split} not in available splits {SPLITS_TO_SUBSETS.keys()}')

    for subset in subsets:
        if subset not in SPLITS_TO_SUBSETS[split]:
            raise ValueError(f'domain {subset} not in available subsets {SPLITS_TO_SUBSETS[split]} for split {split}')

    logger.info("Loading original data from disk")

    ###########
    # LOADING #
    ###########
    original = load_dataset('mrqa', keep_in_memory=False, split=split).filter(
        lambda a: a['subset'] in subsets,
        num_proc=preprocessing_workers,
        load_from_cache_file=load_from_cache_file,
    )

    #################
    # PREPROCESSING #
    #################
    examples = original.map(
        process_entry,
        with_indices=True,
        num_proc=preprocessing_workers,
        remove_columns=original.column_names,
        desc="Preprocessing examples",
        load_from_cache_file=load_from_cache_file,
        fn_kwargs=dict(tokenizer=tokenizer)
    )

    #######################
    # CONVERT TO FEATURES #
    #######################
    features = examples.map(
        convert_examples_to_features,
        num_proc=preprocessing_workers,
        batched=True,
        batch_size=100,
        remove_columns=examples.column_names,
        desc="Tokenizing examples",
        load_from_cache_file=load_from_cache_file,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
        )
    )

    ####################
    # CREATING DATASET #
    ####################
    features = features.add_column('unique_id', list(range(UNIQUE_ID_START, len(features) + UNIQUE_ID_START)))

    columns_to_keep = ('index', 'subset', 'unique_id', 'start_position', 'end_position')
    columns_to_keep = columns_to_keep + tuple(tokenizer.model_input_names)

    # Removing columns
    cols_to_remove = set(features.column_names) - set(columns_to_keep)
    dataset = features.remove_columns(cols_to_remove)

    return examples, features, dataset


def process_entry(entry: Dict, index: int, tokenizer: PreTrainedTokenizerBase = None) -> Dict:
    r""" Process and clean a single dataset entry. """
    # get all spans in order and clean data
    char_spans = sorted(set([
        (start, end)
        for span in entry['detected_answers']['char_spans']
        for start, end in zip(span['start'], span['end'])
    ]))

    # map old tokens to new, skipping unwanted junk. remap also gold spans
    context, char_spans = entry['context'], char_spans # clean_text(tokenizer, entry['context'], spans=char_spans)

    # all answers
    all_extracted_answers = [context[char_start:char_end + 1] for char_start, char_end in char_spans]

    # take first span for training
    if char_spans:
        selected_span = char_spans[0]
        answer = all_extracted_answers[0]
        char_start_position, char_end_position = selected_span
    else:
        answer = char_start_position = char_end_position = None

    # remove eventual multiple spaces
    question = clean_text(tokenizer, entry['question'])

    example = dict(
        context=context,
        subset=SUBSETS_TO_IDS[entry['subset']],
        index=index,
        answer=answer,
        gold_answers=set(entry['answers']),
        char_start_position=char_start_position,
        char_end_position=char_end_position,
        question_id=entry['qid'],
        question_text=question,
    )
    return example


def convert_examples_to_features(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase = None,
    max_sequence_length: int = None,
    doc_stride: int = None,
    max_query_length: int = None,
) -> Dict[str, List]:
    r""" Convert samples with annotations in one or more tokenized features. """
    results = [
        _convert_example_to_features(
            *features,
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
        )
        for features in zip(
            examples['index'],
            examples['question_id'],
            examples['context'],
            examples['question_text'],
            examples['char_start_position'],
            examples['char_end_position'],
            examples['subset'],
        )
    ]
    res = {k: [x for dic in results for x in dic[k]] for k in results[0].keys()}
    return res


def _convert_example_to_features(
    index: int,
    question_id: str,
    context: str,
    question_text: str,
    char_start_position: int,
    char_end_position: int,
    subset: int,
    tokenizer: PreTrainedTokenizerBase = None,
    max_sequence_length: int = None,
    doc_stride: int = None,
    max_query_length: int = None,
) -> Dict[str, List]:
    r""" Convert a sample with annotations in one or more tokenized features. """

    # clip question to max length in tokens with this trick
    question_text = tokenizer.convert_tokens_to_string(
        tokenizer.tokenize(question_text, add_special_tokens=False)[:max_query_length]
    )

    # for all tokenizers compatibility
    encoded = tokenizer(
        question_text,
        context,
        truncation='only_second',
        padding="max_length",
        max_length=max_sequence_length,
        stride=doc_stride,
        add_special_tokens=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    num_features = len(encoded['input_ids'])
    all_sequence_ids = [encoded.sequence_ids(i) for i in range(num_features)]

    # map: token position in the span -> bool (is max context)
    token_is_max_context = list(check_is_max_context(all_sequence_ids, doc_stride=doc_stride))

    res = defaultdict(list)
    for feature_index, input_ids in enumerate(encoded['input_ids']):

        tok_start_position = 0
        tok_end_position = 0
    
        # convert positions from words relativity to tokens
        if char_start_position is not None and char_end_position is not None:
            start_position = encoded.char_to_token(feature_index, char_start_position, 1)
            end_position = encoded.char_to_token(feature_index, char_end_position, 1)

            # if gold answer is contained in actual span
            if start_position is not None and end_position is not None:
                tok_start_position = start_position
                tok_end_position = end_position

        # offset mapping only on context
        covered_tokens = [i for i, seq_id in enumerate(all_sequence_ids[feature_index]) if seq_id == 1]

        res['index'].append(index)
        res['question_id'].append(question_id)
        res['subset'].append(subset)
        res['tokens'].append(encoded.tokens(feature_index))
        res['covered_tokens'].append(covered_tokens)
        res['offset_mapping'].append(encoded['offset_mapping'][feature_index])
        res['token_is_max_context'].append(dict_to_list(token_is_max_context[feature_index]))

        res['input_ids'].append(input_ids)
        if 'attention_mask' in encoded:
            res['attention_mask'].append(encoded['attention_mask'][feature_index])
        if 'token_type_ids' in encoded:
            res['token_type_ids'].append(encoded['token_type_ids'][feature_index])

        res['start_position'].append(tok_start_position)  # position of start token in span
        res['end_position'].append(tok_end_position)  # position of end token in span

    return res
