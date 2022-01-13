from argparse import ArgumentParser
import os
from typing import List
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.warnings import rank_zero_warn
from requests.models import MissingSchema
import torch
from transformers_lightning.datamodules import SuperDataModule
import logging
from dataset.dataset import MapDataset
from preprocessing.preprocessing import convert_examples_to_features, read_mrqa_examples
import requests


logger = logging.getLogger(__name__)


DATA_PATHS = dict(
    squad_train='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz',
    newsqa_train='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NewsQA.jsonl.gz',
    triviaqa_train='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/TriviaQA-web.jsonl.gz',
    searchqa_train='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SearchQA.jsonl.gz',
    hotpotqa_train='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/HotpotQA.jsonl.gz',
    naturalquestions_train='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz',

    squad_dev='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz',
    newsqa_dev='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz',
    triviaqa_dev='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz',
    searchqa_dev='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SearchQA.jsonl.gz',
    hotpotqa_dev='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz',
    naturalquestions_dev='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz',

    bioasq_dev='http://participants-area.bioasq.org/MRQA2019/',
    drop_dev='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DROP.jsonl.gz',
    duorc_dev='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DuoRC.ParaphraseRC.jsonl.gz',
    race_dev='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RACE.jsonl.gz',
    relationextraction_dev='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RelationExtraction.jsonl.gz',
    textbookqa_dev='https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TextbookQA.jsonl.gz',
)


class QuestionAnsweringDataModule(SuperDataModule):
    r""" Train, validation and prediction. """

    def __init__(self, hyperparameters, trainer, tokenizer):
        super().__init__(hyperparameters, trainer)
        self.tokenizer = tokenizer
        self.train_cache = os.path.join(self.hyperparameters.cache_dir, "train_features_" + self.hyperparameters.name)
        self.valid_cache = os.path.join(self.hyperparameters.cache_dir, "valid_features_" + self.hyperparameters.name)
        self.test_cache = os.path.join(self.hyperparameters.cache_dir, "test_features_" + self.hyperparameters.name)
        self.predict_cache = os.path.join(self.hyperparameters.cache_dir, "predict_features_" + self.hyperparameters.name)

    def do_train(self) -> bool:
        r""" Whether to do training. """
        return len(self.hyperparameters.train_datasets) > 0

    def do_validation(self) -> bool:
        r""" Whether to do validation. """
        return len(self.hyperparameters.dev_datasets) > 0

    def do_test(self):
        r""" Whether to do testing. """
        return len(self.hyperparameters.test_datasets) > 0

    def do_predict(self):
        r""" Whether to do prediction. """
        return len(self.hyperparameters.predict_datasets) > 0

    def download_dataset(self, name) -> str:
        r""" Download dataset and return path on disk. """
        url = DATA_PATHS[name]
        filepath = os.path.join(self.hyperparameters.cache_dir, name + ".jsonl.gz")
        if not os.path.isfile(filepath):
            logger.info(f"Downloading dataset {name} from {url} to {filepath}")
            r = requests.get(url, allow_redirects=True)
            with open(filepath, "wb") as fo:
                fo.write(r.content)
        else:
            logger.info(f"Dataset {name} from {url} already downloaded, reusing it.")
        return filepath

    def get_datasets(self, datasets: List[str], concat: bool = True, is_training: bool = True):
        r""" Download or just load files. """
        filenames = []
        for name in datasets:
            if os.path.isfile(name):
                filenames.append(name)
            else:
                try:
                    name = self.download_dataset(name)
                except KeyError:
                    logger.error(
                        f"Invalid dataset name {name}, choose one of "
                        "{list(DATA_PATHS.keys())} or provide path to a jsonl.gz file."
                    )
                except (ConnectionError, MissingSchema):
                    logger.error(
                        f"Check your internet connection because this url is not available {DATA_PATHS[name]}."
                    )
                filenames.append(name)

        if concat is True:
            rank_zero_warn("Going to concatenate and shuffle all training data together. ")
            data, examples = read_mrqa_examples(input_file=filenames, is_training=True)
        else:
            data, examples = zip(*[read_mrqa_examples(input_file=f, is_training=is_training) for f in filenames])

        return data, examples

    def prepare_data(self):
        r""" Download and preprocess data on master node. """
        # train data
        os.makedirs(self.hyperparameters.cache_dir, exist_ok=True)

        if self.do_train():
            logger.info("Preparing train data")
            train_original, train_examples = self.get_datasets(
                self.hyperparameters.train_datasets, concat=True, is_training=True
            )
            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=self.tokenizer,
                max_sequence_length=self.hyperparameters.max_sequence_length,
                doc_stride=self.hyperparameters.doc_stride,
                max_query_length=self.hyperparameters.max_query_length,
                is_training=True,
                preprocessing_workers=self.hyperparameters.preprocessing_workers,
            )
            torch.save((train_original, train_features), self.train_cache)

        # dev data
        if self.do_validation():
            logger.info("Preparing validation data")
            valid_original, valid_examples = self.get_datasets(
                self.hyperparameters.dev_datasets, concat=True, is_training=False
            )
            valid_features = convert_examples_to_features(
                examples=valid_examples,
                tokenizer=self.tokenizer,
                max_sequence_length=self.hyperparameters.max_sequence_length,
                doc_stride=self.hyperparameters.doc_stride,
                max_query_length=self.hyperparameters.max_query_length,
                is_training=False,
                preprocessing_workers=self.hyperparameters.preprocessing_workers,
            )
            torch.save((valid_original, valid_examples, valid_features), self.valid_cache)

        # test data
        if self.do_test():
            logger.info("Preparing test data")
            # list of datasets for test
            test_original, test_examples = self.get_datasets(
                self.hyperparameters.test_datasets,  concat=False, is_training=False
            )
            test_features = [
                convert_examples_to_features(
                    examples=examples,
                    tokenizer=self.tokenizer,
                    max_sequence_length=self.hyperparameters.max_sequence_length,
                    doc_stride=self.hyperparameters.doc_stride,
                    max_query_length=self.hyperparameters.max_query_length,
                    is_training=False,
                    preprocessing_workers=self.hyperparameters.preprocessing_workers,
                )
                for examples in test_examples
            ]
            torch.save((test_original, test_examples, test_features), self.test_cache)

        # predict data
        if self.do_predict():
            logger.info("Preparing predict data")
            # list of datasets for test
            predict_original, predict_examples = self.get_datasets(
                self.hyperparameters.predict_datasets, concat=False, is_training=False
            )
            predict_features = [
                convert_examples_to_features(
                    examples=examples,
                    tokenizer=self.tokenizer,
                    max_sequence_length=self.hyperparameters.max_sequence_length,
                    doc_stride=self.hyperparameters.doc_stride,
                    max_query_length=self.hyperparameters.max_query_length,
                    is_training=False,
                    preprocessing_workers=self.hyperparameters.preprocessing_workers,
                )
                for examples in predict_examples
            ]
            torch.save((predict_original, predict_examples, predict_features), self.predict_cache)

    def setup(self, stage) -> None:
        r""" Called on every process. """ 
        if stage == TrainerFn.FITTING.value or stage == TrainerFn.VALIDATING.value:
            if self.do_train():
                self.train_original, self.train_features = torch.load(self.train_cache)
                self.train_dataset = MapDataset([
                    dict(
                        id=i,
                        input_ids=a.input_ids,
                        input_mask=a.input_mask,
                        segment_ids=a.segment_ids,
                        start_positions=a.start_position,
                        end_positions=a.end_position,
                    ) for i, a in enumerate(self.train_features)
                ])
                logger.info(f"Training dataset has length {len(self.train_dataset)}")

            if self.do_validation():
                self.valid_original, self.valid_examples, self.valid_features = torch.load(self.valid_cache)
                self.valid_dataset = MapDataset([
                    dict(
                        id=i,
                        input_ids=a.input_ids,
                        input_mask=a.input_mask,
                        segment_ids=a.segment_ids,
                    ) for i, a in enumerate(self.valid_features)
                ])
                logger.info(f"Validation dataset has length {len(self.valid_dataset)}")

        elif stage == TrainerFn.TESTING.value:
            if self.do_test():
                self.test_original, self.test_examples, self.test_features = torch.load(self.test_cache)
                self.test_dataset = []
                for test_features in self.test_features:
                    dataset = MapDataset([
                        dict(
                            id=i,
                            input_ids=a.input_ids,
                            input_mask=a.input_mask,
                            segment_ids=a.segment_ids,
                        ) for i, a in enumerate(test_features)
                    ])
                    self.test_dataset.append(dataset)
                logger.info(f"Test datasets have length {[len(d) for d in self.test_dataset]}")

        elif stage == TrainerFn.PREDICTING.value:
            if self.do_predict():
                self.predict_original, self.predict_examples, self.predict_features = torch.load(self.predict_cache)
                self.predict_dataset = []
                for predict_features in self.predict_features:
                    dataset = MapDataset([
                        dict(
                            id=i,
                            input_ids=a.input_ids,
                            input_mask=a.input_mask,
                            segment_ids=a.segment_ids,
                        ) for i, a in enumerate(predict_features)
                    ])
                    self.predict_dataset.append(dataset)
                logger.info(f"Predict datasets have length {[len(d) for d in self.predict_dataset]}")

    def train_dataloader(self):
        r""" Return the training dataloader. """
        if self.do_train():
            params = dict(shuffle=True) if not self.hyperparameters.iterable else dict()
            return self.default_dataloader(self.train_dataset, self.hyperparameters.batch_size, **params)
        return None

    def val_dataloader(self):
        r""" Return the validation dataloader. """
        if self.do_validation():
            return self.default_dataloader(self.valid_dataset, self.hyperparameters.val_batch_size, shuffle=False)
        return None

    def test_dataloader(self):
        r""" Return the test dataloader. """
        if self.do_test():
            return [
                self.default_dataloader(dataset, self.hyperparameters.test_batch_size, shuffle=False)
                for dataset in self.test_dataset
            ]
        return None

    def predict_dataloader(self):
        r""" Return the predict dataloader(s). """
        if self.do_predict():
            return [
                self.default_dataloader(dataset, self.hyperparameters.predict_batch_size, shuffle=False)
                for dataset in self.predict_datasetù
            ]
        return None

    @staticmethod
    def add_datamodule_specific_args(parser: ArgumentParser):
        super(QuestionAnsweringDataModule, QuestionAnsweringDataModule).add_datamodule_specific_args(parser)
        parser.add_argument(
            '--cache_dir',
            type=str,
            default=os.path.join(os.path.expanduser('~'), ".cache", "mrqa-lightning"),
            required=False,
            help="Cache folder for temporary files"
        )
        parser.add_argument("--train_datasets", default=[], type=str, nargs='+')
        parser.add_argument("--dev_datasets", default=[], type=str, nargs='+')
        parser.add_argument("--test_datasets", default=[], type=str, nargs='+')
        parser.add_argument("--predict_datasets", default=[], type=str, nargs='+')

        parser.add_argument("--max_sequence_length", default=512, type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                                    "longer than this will be truncated, and sequences shorter than this will be padded.")
        parser.add_argument("--doc_stride", default=128, type=int,
                            help="When splitting up a long document into chunks, "
                                    "how much stride to take between chunks.")
        parser.add_argument("--max_query_length", default=64, type=int,
                            help="The maximum number of tokens for the question. Questions longer than this will "
                                    "be truncated to this length.")
        parser.add_argument("--max_answer_length", default=30, type=int,
                            help="The maximum length of an answer that can be generated. "
                                    "This is needed because the start "
                                    "and end predictions are not conditioned on one another.")
        parser.add_argument('--preprocessing_workers', default=16, type=int, help="Preprocessing processes to use to prepare the dataset")
