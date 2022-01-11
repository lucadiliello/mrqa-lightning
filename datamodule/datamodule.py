from argparse import ArgumentParser
import os
from pytorch_lightning.trainer.states import TrainerFn
import torch
from transformers_lightning.datamodules import SuperDataModule
import logging
from dataset.dataset import MapDataset
from preprocessing.preprocessing import convert_examples_to_features, read_mrqa_examples


logger = logging.getLogger(__name__)


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
        return self.hyperparameters.train_file is not None

    def do_validation(self) -> bool:
        r""" Whether to do validation. """
        return self.hyperparameters.dev_file is not None

    def do_test(self):
        r""" Whether to do testing. """
        return self.hyperparameters.test_file is not None

    def do_predict(self):
        r""" Whether to do prediction. """
        return self.hyperparameters.predict_file is not None

    def prepare_data(self) -> None:
        # train data
        if self.do_train():
            logger.info("Preparing train data")
            train_original, train_examples = read_mrqa_examples(input_file=self.hyperparameters.train_file, is_training=True)
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
            valid_original, valid_examples = read_mrqa_examples(input_file=self.hyperparameters.dev_file, is_training=False)
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
            test_original, test_examples = read_mrqa_examples(input_file=self.hyperparameters.test_file, is_training=False)
            test_features = convert_examples_to_features(
                examples=test_examples,
                tokenizer=self.tokenizer,
                max_sequence_length=self.hyperparameters.max_sequence_length,
                doc_stride=self.hyperparameters.doc_stride,
                max_query_length=self.hyperparameters.max_query_length,
                is_training=False,
                preprocessing_workers=self.hyperparameters.preprocessing_workers,
            )
            torch.save((test_original, test_examples, test_features), self.test_cache)
        
        # predict data
        if self.do_predict():
            logger.info("Preparing predict data")
            predict_original, predict_examples = read_mrqa_examples(input_file=self.hyperparameters.predict_file, is_training=False)
            predict_features = convert_examples_to_features(
                examples=predict_examples,
                tokenizer=self.tokenizer,
                max_sequence_length=self.hyperparameters.max_sequence_length,
                doc_stride=self.hyperparameters.doc_stride,
                max_query_length=self.hyperparameters.max_query_length,
                is_training=False,
                preprocessing_workers=self.hyperparameters.preprocessing_workers,
            )
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
                self.test_dataset = MapDataset([
                    dict(
                        id=i,
                        input_ids=a.input_ids,
                        input_mask=a.input_mask,
                        segment_ids=a.segment_ids,
                    ) for i, a in enumerate(self.test_features)
                ])
                logger.info(f"Test dataset has length {len(self.test_dataset)}")

        elif stage == TrainerFn.PREDICTING.value:
            if self.do_predict():
                self.predict_original, self.predict_examples, self.predict_features = torch.load(self.predict_cache)
                self.predict_dataset = MapDataset([
                    dict(
                        id=i,
                        input_ids=a.input_ids,
                        input_mask=a.input_mask,
                        segment_ids=a.segment_ids,
                    ) for i, a in enumerate(self.predict_features)
                ])
                logger.info(f"Predict dataset has length {len(self.predict_dataset)}")
