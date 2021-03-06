import logging
import os
from argparse import ArgumentParser
from multiprocessing import cpu_count

from datasets.load import load_from_disk
from pytorch_lightning.trainer.states import TrainerFn
from transformers_lightning.datamodules import SuperDataModule

from processing.processing import load_from_datasets
from utilities.utilities import split_datasets_on_column_value


logger = logging.getLogger(__name__)


class QuestionAnsweringDataModule(SuperDataModule):
    r""" Train, validation and test. """

    def __init__(self, hyperparameters, trainer, tokenizer):
        super().__init__(hyperparameters, trainer)
        self.tokenizer = tokenizer

        assert self.hyperparameters.train_split is None or self.hyperparameters.train_subsets is not None
        assert self.hyperparameters.val_split is None or self.hyperparameters.val_subsets is not None
        assert self.hyperparameters.test_split is None or self.hyperparameters.test_subsets is not None

        # generate cache file names
        for name in ('examples', 'features', 'dataset'):
            for split in ('train', 'valid', 'test', 'predict'):
                setattr(
                    self,
                    f'{split}_{name}_cache',
                    os.path.join(self.hyperparameters.cache_dir, f"{split}_{name}_{self.hyperparameters.name}")
                )

    def do_train(self) -> bool:
        r""" Whether to do training. """
        return self.hyperparameters.train_split is not None

    def do_validation(self) -> bool:
        r""" Whether to do validation. """
        return self.hyperparameters.val_split is not None

    def do_test(self):
        r""" Whether to do testing. """
        return self.hyperparameters.test_split is not None

    def do_predict(self):
        r""" Whether to do predictions. """
        return False

    def prepare_data(self):
        r""" Download and preprocess data on master node. """
        # train data
        os.makedirs(self.hyperparameters.cache_dir, exist_ok=True)

        if self.do_train():
            logger.info("Preparing train data")
            _, _, train_dataset = load_from_datasets(
                split=self.hyperparameters.train_split,
                tokenizer=self.tokenizer,
                max_sequence_length=self.hyperparameters.max_sequence_length,
                doc_stride=self.hyperparameters.doc_stride,
                max_query_length=self.hyperparameters.max_query_length,
                dataset=self.hyperparameters.dataset,
                subsets=self.hyperparameters.train_subsets,
                preprocessing_workers=self.hyperparameters.preprocessing_workers,
                load_from_cache_file=not self.hyperparameters.disable_cache,
            )
            train_dataset.save_to_disk(self.train_dataset_cache)

        # dev data
        if self.do_validation():
            logger.info("Preparing validation data")
            valid_examples, valid_features, valid_dataset = load_from_datasets(
                split=self.hyperparameters.val_split,
                tokenizer=self.tokenizer,
                max_sequence_length=self.hyperparameters.max_sequence_length,
                doc_stride=self.hyperparameters.doc_stride,
                max_query_length=self.hyperparameters.max_query_length,
                dataset=self.hyperparameters.dataset,
                subsets=self.hyperparameters.val_subsets,
                preprocessing_workers=self.hyperparameters.preprocessing_workers,
                load_from_cache_file=not self.hyperparameters.disable_cache,
            )
            valid_examples.save_to_disk(self.valid_examples_cache)
            valid_features.save_to_disk(self.valid_features_cache)
            valid_dataset.save_to_disk(self.valid_dataset_cache)

        # test data
        if self.do_test():
            logger.info("Preparing test data")
            # list of datasets for test
            test_examples, test_features, test_dataset = load_from_datasets(
                split=self.hyperparameters.test_split,
                tokenizer=self.tokenizer,
                max_sequence_length=self.hyperparameters.max_sequence_length,
                doc_stride=self.hyperparameters.doc_stride,
                max_query_length=self.hyperparameters.max_query_length,
                dataset=self.hyperparameters.dataset,
                subsets=self.hyperparameters.test_subsets,
                preprocessing_workers=self.hyperparameters.preprocessing_workers,
                load_from_cache_file=not self.hyperparameters.disable_cache,
            )
            test_examples.save_to_disk(self.test_examples_cache)
            test_features.save_to_disk(self.test_features_cache)
            test_dataset.save_to_disk(self.test_dataset_cache)

    def setup(self, stage) -> None:
        r""" Called on every process. """
        if stage == TrainerFn.FITTING.value or stage == TrainerFn.VALIDATING.value:
            if self.do_train():
                self.train_dataset = load_from_disk(self.train_dataset_cache)
                logger.info(f"Training dataset has length {len(self.train_dataset)}")

            if self.do_validation():
                self.valid_examples = load_from_disk(self.valid_examples_cache)
                self.valid_features = load_from_disk(self.valid_features_cache)
                self.valid_dataset = load_from_disk(self.valid_dataset_cache)
                logger.info(f"Validation dataset has length {len(self.valid_dataset)}")

        elif stage == TrainerFn.TESTING.value:
            if self.do_test():
                test_examples = load_from_disk(self.test_examples_cache)
                test_features = load_from_disk(self.test_features_cache)
                test_dataset = load_from_disk(self.test_dataset_cache)

                # split in multiple datasets based on column value
                all_data = split_datasets_on_column_value(
                    test_examples,
                    test_features,
                    test_dataset,
                    column='subset',
                    preprocessing_workers=self.hyperparameters.preprocessing_workers,
                )

                self.test_examples, self.test_features, self.test_dataset = all_data
                logger.info(f"Test datasets have length {[len(d) for d in self.test_dataset]}")

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

    @staticmethod
    def add_datamodule_specific_args(parser: ArgumentParser):
        super(QuestionAnsweringDataModule, QuestionAnsweringDataModule).add_datamodule_specific_args(parser)
        parser.add_argument(
            '--cache_dir',
            type=str,
            default=os.path.join(os.path.expanduser('~'), ".cache", "mrqa-lightning"),
            required=False,
            help="Cache folder for temporary datasets."
        )
        parser.add_argument(
            "--dataset",
            default='mrqa',
            type=str,
            help="Dataset name or path on disk."
        )
        parser.add_argument('--train_split', default=None, type=str, required=False, help="Training split name.")
        parser.add_argument('--val_split', default=None, type=str, required=False, help="Validation split name.")
        parser.add_argument('--test_split', default=None, type=str, required=False, help="Test split name.")
        parser.add_argument("--train_subsets", default=None, type=str, nargs='+', help="Train subsets to use.")
        parser.add_argument("--val_subsets", default=None, type=str, nargs='+', help="Validation subsets to use.")
        parser.add_argument("--test_subsets", default=None, type=str, nargs='+', help="Test subsets to use.")
        parser.add_argument(
            "--max_sequence_length",
            default=512,
            type=int,
            help=(
                "The maximum total input sequence length after WordPiece tokenization. "
                "Sequences longer than this will be truncated, and sequences shorter than this will be padded."
            )
        )
        parser.add_argument(
            "--doc_stride",
            default=128,
            type=int,
            help="When splitting up a long document into chunks, how much stride to take between chunks."
        )
        parser.add_argument(
            "--max_query_length",
            default=64,
            type=int,
            help=(
                "The maximum number of tokens for the question. "
                "Questions longer than this will be truncated to this length."
            )
        )
        parser.add_argument(
            "--max_answer_length",
            default=30,
            type=int,
            help=(
                "The maximum length of an answer that can be generated. This is needed because the start and "
                "end predictions are not conditioned on one another."
            )
        )
        parser.add_argument(
            '--preprocessing_workers',
            default=cpu_count(),
            type=int,
            help="Preprocessing processes to use to prepare the dataset."
        )
