from collections import defaultdict
from typing import List, Dict, Tuple
import logging
from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from functools import partial
from datasets import load_dataset


logger = logging.getLogger(__name__)


AVAILABLE_SPLITS = ('train', 'validation', 'test')
AVAILABLE_SUBSETS = (
    'SearchQA',
    'TriviaQA-web',
    'HotpotQA',
    'SQuAD',
    'NaturalQuestionsShort',
    'NewsQA',
    'TextbookQA',
    'DuoRC.ParaphraseRC',
    'RelationExtraction',
    'DROP',
    'BioASQ',
    'RACE'
)
SUBSETS_TO_IDS = dict([(s, i) for i, s in enumerate(AVAILABLE_SUBSETS)])
UNIQUE_ID_START = 10000000


def load_from_datasets(
    split: str,
    subsets: List[str],
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    doc_stride: int,
    max_query_length: int,
    is_training: bool = False,
    preprocessing_workers: int = 16,
) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    r""" Read datasets from datasets library and preprocess examples. """

    # at the moment we only accept right padding
    assert tokenizer.padding_side == 'right', "left padding not supported yet"

    # checks
    if split not in AVAILABLE_SPLITS:
        raise ValueError(f'split {split} not in available splits {AVAILABLE_SPLITS}')

    for subset in subsets:
        if not subset in AVAILABLE_SUBSETS:
            raise ValueError(f'domain {subset} not in available domains {AVAILABLE_SUBSETS}')

    original = load_dataset('mrqa', keep_in_memory=False, split=split).filter(lambda a: a['subset'] in subsets, num_proc=preprocessing_workers)

    logger.info(f"Preprocessing datasets of split {split} using subsets {subsets}")
    process_entry_partial = partial(process_entry, is_training=is_training, subsets_to_ids=SUBSETS_TO_IDS)

    # examples preprocessing
    examples = original.map(
        process_entry_partial,
        with_indices=True,
        num_proc=preprocessing_workers,
        remove_columns=original.column_names,
        desc="Preprocessing examples",
    )

    mp_function = partial(
        convert_examples_to_features,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=is_training
    )

    # creating features and tokenizing
    features = examples.map(
        mp_function,
        num_proc=preprocessing_workers,
        batched=True,
        batch_size=2,
        remove_columns=examples.column_names,
        desc="Tokenizing examples",
    )
    # adding unique indexes
    features = features.add_column('unique_id', list(range(UNIQUE_ID_START, len(features) + UNIQUE_ID_START)))

    columns_to_keep = ('example_index', 'input_ids', 'attention_mask', 'token_type_ids', 'subset', 'unique_id')
    if is_training is True:
        columns_to_keep = columns_to_keep + ('start_position', 'end_position')

    cols_to_remove = set(features.column_names) - set(columns_to_keep)
    dataset = features.remove_columns(cols_to_remove)

    return original, examples, features, dataset


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def process_entry(entry: Dict, index: int, is_training: bool = True, subsets_to_ids: Dict = None) -> Dict:
    r"""
    Process a single dataset entry.
    """

    paragraph_text = entry['context']
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

    question_id = entry['qid']
    question_text = entry['question']
    start_position = None
    end_position = None
    orig_answer_text = None

    if is_training:
        answers = entry["detected_answers"]
        spans = sorted([(start, end) for span in answers['char_spans'] for start, end in zip(span['start'], span['end'])])
        # take first span
        char_start, char_end = spans[0][0], spans[0][1]
        orig_answer_text = paragraph_text[char_start:char_end + 1]
        start_position, end_position = char_to_word_offset[char_start], char_to_word_offset[char_end]

    example = dict(
        subset=subsets_to_ids[entry['subset']],
        example_index=index,
        question_id=question_id,
        question_text=question_text,
        doc_tokens=doc_tokens,
        orig_answer_text=orig_answer_text,
        start_position=start_position,
        end_position=end_position
    )
    return example


def convert_examples_to_features(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase = None,
    max_sequence_length: int = None,
    doc_stride: int = None,
    max_query_length: int = None,
    is_training: bool = None,
) -> Dict[str, List]:
    r""" Loads a data file into a list of `InputBatch`s. """

    results = [
        _convert_example_to_features(
            index,
            question_text,
            doc_tokens,
            orig_answer_text,
            start_position,
            end_position,
            subset,
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
        )
        for (
            index, question_text, doc_tokens, orig_answer_text, start_position, end_position, subset
        ) in zip(
            examples['example_index'],
            examples['question_text'],
            examples['doc_tokens'],
            examples['orig_answer_text'],
            examples['start_position'],
            examples['end_position'],
            examples['subset'],
        )
    ]
    res = {k: [x for dic in results for x in dic[k]] for k in results[0].keys()}
    return res


def _convert_example_to_features(
    index,
    question_text,
    doc_tokens,
    orig_answer_text,
    start_position,
    end_position,
    subset,
    tokenizer: PreTrainedTokenizerBase = None,
    max_sequence_length: int = None,
    doc_stride: int = None,
    max_query_length: int = None,
    is_training: bool = None,
) -> Dict[str, List]:

    res = defaultdict([])

    # doc -> "hi I am a powerfull sequence of tokens"
    # doc_tokens -> ["hi", "I", "am", "a", "powerfull", "sequence", "of", "tokens"]
    # tokenized -> ['hi', 'ĠI', 'Ġam', 'Ġa', 'Ġpower', 'full', 'Ġsequence', 'Ġof', 'Ġtokens']
    orig_to_tok_index = []  # [0, 1, 2, 3, 4, 6, 7, 8]
    tok_to_orig_index = []  # [0, 1, 2, 3, 4, 4, 5, 6, 7]
    all_doc_tokens = []  # ['hi', 'ĠI', 'Ġam', 'Ġa', 'Ġpower', 'full', 'Ġsequence', 'Ġof', 'Ġtokens']

    for i, token in enumerate(doc_tokens):
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

        tok_start_position = orig_to_tok_index[start_position]

        if end_position < len(doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[end_position + 1] - 1   # to catch the last subtoken on the target word
        else:
            tok_end_position = len(all_doc_tokens) - 1

        tok_start_position, tok_end_position = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, orig_answer_text
        )

    query_tokens = tokenizer.tokenize(question_text)[:max_query_length]

    # The number_of_special_tokens accounts for example to [CLS], [SEP] and [SEP]
    number_of_special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    max_tokens_for_doc = max_sequence_length - len(query_tokens) - number_of_special_tokens

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.

    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset

        if length > max_tokens_for_doc:
            length = max_tokens_for_doc

        doc_spans.append((start_offset, length))

        if start_offset + length == len(all_doc_tokens):
            break

        start_offset += min(length, doc_stride)

    # for each span of the sliding window
    # doc_span_index -> index
    # doc_span -> (start_index, length of span)
    for doc_span_index, (doc_span_start, doc_span_length) in enumerate(doc_spans):

        encoded = tokenizer(
            tokenizer.convert_tokens_to_string(query_tokens),
            tokenizer.convert_tokens_to_string(all_doc_tokens[doc_span_start:doc_span_start + doc_span_length]),
            truncation='only_second',
            padding="max_length",
            max_length=max_sequence_length,
        )
        start_position_second_sequence = encoded.sequence_ids().index(1)  # get first position index of second sequence

        token_to_orig_map = {}
        token_is_max_context = {}
        for i in range(doc_span_length):
            split_token_index = doc_span_start + i
            token_to_orig_map[start_position_second_sequence + i] = tok_to_orig_index[split_token_index]
            is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
            token_is_max_context[start_position_second_sequence + i] = int(is_max_context)

        start_position = None
        end_position = None

        if is_training:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span_start
            doc_end = doc_span_start + doc_span_length - 1
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

        tokens = [t for t, mask in zip(encoded.tokens(), encoded['attention_mask']) if mask]

        res['example_index'].append(index)
        res['subset'].append(subset)
        res['doc_span_index'].append(doc_span_index)
        res['tokens'].append(tokens)
        res['token_to_orig_map'].append(list(token_to_orig_map.items()))
        res['token_is_max_context'].append(list(token_is_max_context.items()))
        res['input_ids'].append(encoded['input_ids'])
        if 'attention_mask' in encoded:
            res['attention_mask'].append(encoded['attention_mask'])
        if 'token_type_ids' in encoded:
            res['token_type_ids'].append(encoded['token_type_ids'])
        res['start_position'].append(start_position)
        res['end_position'].append(end_position)

    return res


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
    r""" Check if this is the 'max context' doc span for the token. """
    best_score = None
    best_span_index = None

    for span_index, (doc_span_start, doc_span_length) in enumerate(doc_spans):
        end = doc_span_start + doc_span_length - 1

        if position < doc_span_start:
            continue
        if position > end:
            continue

        num_left_context = position - doc_span_start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span_length

        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
