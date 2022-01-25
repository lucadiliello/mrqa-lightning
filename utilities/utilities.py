import logging
import re
from argparse import Namespace
from typing import Dict, List, Tuple, Union

import numpy
from datasets import Dataset
from pytorch_lightning.utilities.distributed import rank_zero_only
from transformers import PreTrainedTokenizerBase


logger = logging.getLogger(__name__)
SPLITS_TO_SUBSETS = {
    'train': ('NewsQA', 'NaturalQuestionsShort', 'TriviaQA-web', 'SearchQA', 'HotpotQA', 'SQuAD'),
    'validation': ('NewsQA', 'NaturalQuestionsShort', 'TriviaQA-web', 'SearchQA', 'HotpotQA', 'SQuAD'),
    'test': ('RACE', 'DuoRC.ParaphraseRC', 'BioASQ', 'TextbookQA', 'RelationExtraction', 'DROP'),
}
SUBSETS_TO_IDS = dict(
    [(s, i) for i, s in enumerate(sorted(set([sub for arr in SPLITS_TO_SUBSETS.values() for sub in arr])))]
)
UNIQUE_ID_START = 10000000


class ExtendedNamespace(Namespace):
    r""" Simple object for storing attributes.

    Implements equality by attribute names and values, and provides a simple
    string representation.

    This version is enhanced with dictionary capabilities.
    """

    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    @classmethod
    def from_namespace(cls, other_namespace: Namespace):
        new = cls()
        new.__dict__ = other_namespace.__dict__
        return new

    def __eq__(self, other):
        if not isinstance(other, Namespace):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        yield from self.__dict__.items()

    def __len__(self):
        return len(self.__dict__)


def _split_datasets_on_column_value(
    dataset: Dataset,
    column: str = 'subset',
    values: List = None,
    preprocessing_workers: int = 1
) -> Tuple[Dataset]:
    r""" Split a dataset in many smaller datasets based on the 'subset' column. """
    if values is None:
        values = list(sorted(set(dataset[column])))
    else:
        values = [SUBSETS_TO_IDS[v] for v in values]

    return [
        dataset.filter(lambda a: a[column] == value, num_proc=preprocessing_workers)
        for value in values
    ]


def split_datasets_on_column_value(
    *dataset: Dataset,
    column: str = 'subset',
    values: List = None,
    preprocessing_workers: int = 1
) -> Union[Tuple[Dataset], List[Tuple[Dataset]]]:
    r""" Split one or many dataset in many smaller datasets based on the 'subset' column. """
    res = [
        _split_datasets_on_column_value(d, column=column, values=values, preprocessing_workers=preprocessing_workers)
        for d in dataset
    ]
    return res[0] if len(dataset) == 1 else res


def is_whitespace(c):
    return (c == " ") or (c == "\t") or (c == "\r") or (c == "\n") or (ord(c) == 0x202F)


def do_overlap(a: Tuple, b: Tuple) -> bool:
    return min(a[1], b[1]) - max(a[0], b[0]) > 0


def dict_to_list(dictionary: Dict) -> List:
    r""" Flatten a dictionary to a list of key-value pairs. Inverse of `list_to_dict`. """
    return list(dictionary.items())


def list_to_dict(_list: List) -> Dict:
    r""" Create a dictionary from a list of key-value pairs. Inverse of `list_to_dict`. """
    return dict(_list)


def improve_answer_span(doc_tokens, tok_start_position, tok_end_position, tokenizer, answer):
    r""" Returns tokenized answer spans that better match the annotated answer. """
    answer = tokenizer.convert_tokens_to_string(tokenizer.tokenize(answer))

    for new_start in range(tok_start_position, tok_end_position + 1):
        for new_end in range(tok_end_position, new_start - 1, -1):
            text_span = tokenizer.convert_tokens_to_string(doc_tokens[new_start:new_end + 1])
            if text_span == answer:
                return new_start, new_end

    return tok_start_position, tok_end_position


def get_position_score(doc_span_start, doc_span_length, position):
    r""" Get score of a token in a span. The score represents the level of context available for that token. """
    return min(
        position - doc_span_start,  # left context
        (doc_span_start + doc_span_length - 1) - position  # right context
    ) + 0.001 * doc_span_length


def check_is_max_context(all_sequence_ids, doc_stride: int):
    r""" Check if this is the 'max context' doc span for the token. """

    # extract spans sequence
    doc_token_spans = []
    last_span_length = doc_stride  # fake previous span just to avoid if-else below
    relative_start_position = 0

    for sequence_ids in all_sequence_ids:
        segment_length = sum(int(s == 1) for s in sequence_ids)

        # update start of relative (from start of second span) and absolute (from beginning of input ids) position
        relative_start_position += (last_span_length - doc_stride)
        absolute_start_position = sequence_ids.index(1) + relative_start_position

        last_span_length = segment_length
        doc_token_spans.append((relative_start_position, absolute_start_position, segment_length))

    # get largest interval to create numpy array
    start_position = min(doc_span_start for doc_span_start, _, _ in doc_token_spans)
    end_position = max(doc_span_start + doc_span_length for doc_span_start, _, doc_span_length in doc_token_spans)

    # each element should be the max_context vector of each span
    res = numpy.zeros(shape=(len(doc_token_spans), end_position - start_position))

    # for every position in every span compute the context score
    for span_index, (doc_span_start, _, doc_span_length) in enumerate(doc_token_spans):
        for position in range(doc_span_start, doc_span_start + doc_span_length):
            res[span_index, position - start_position] = get_position_score(doc_span_start, doc_span_length, position)

    # find best position along each span
    output = numpy.equal(res, res.max(axis=0, keepdims=True)).astype(int).tolist()

    # return
    for o, (doc_span_start, abs_start_position, doc_span_length) in zip(output, doc_token_spans):
        interval = o[doc_span_start - start_position:doc_span_start + doc_span_length - start_position]
        yield {i + abs_start_position: value for i, value in enumerate(interval)}


def clean_text(
    tokenizer: PreTrainedTokenizerBase, text: str, spans: List[Tuple] = None
) -> Union[str, Tuple[str, List]]:
    r""" Clean a string and return mapping to old string.
    Also fixes eventual spans referring te the old string.
    """
    special_tokens_to_remove = set(['[DOC]', '[PAR]', '[TLE]', '[SEP]'] + tokenizer.all_special_tokens)
    regex = re.compile(
        r"(\s+)|(" + "|".join(re.escape(x) for x in special_tokens_to_remove) + ")"
    )

    # for each match, remove it from context and adjust spans position
    old_char_to_new_char = {}
    new_text = ""
    old_index = 0

    while old_index < len(text):
        match = regex.search(text, old_index)
        if match is None:
            # add last mapping to new_char_to_old_char
            old_char_to_new_char.update({old_index + i: len(new_text) + i for i in range(len(text) - old_index)})
            new_text += text[old_index:]
            break

        # advance copying old in new
        span = match.span()
        old_char_to_new_char.update({old_index + i: len(new_text) + i for i in range(span[0] - old_index)})
        new_text += text[old_index:span[0]]
        old_index = span[0]

        # now work on match
        if match.group()[0] is not None:
            # stripping whitespaces
            if not new_text.endswith(" ") and len(new_text):
                new_text += " "
        else:
            # removing special reserved tokens
            pass

        old_index = span[1]

    new_text = new_text.strip()

    if spans is not None:
        # fix position of spans after cleaning
        spans = [
            (old_char_to_new_char[span_start], old_char_to_new_char[span_end])
            for span_start, span_end in spans
            if span_start in old_char_to_new_char and span_end in old_char_to_new_char
        ]
        return new_text, spans
    else:
        return new_text


@rank_zero_only
def print_results(subsets: List[str], results: List[Dict]):
    for name, res in zip(subsets, results):
        logger.info(f"Results for test set {name}: {res}")
