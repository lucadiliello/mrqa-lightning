from argparse import ArgumentParser
from typing import List

import torch
from torchmetrics import Accuracy
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from transformers_lightning.models import TransformersModel

from evaluation.evaluation import get_raw_scores, make_predictions


class QuestionAnsweringModel(TransformersModel):
    r""" QA functionalities with EM and F1 computation. """

    def __init__(self, hyperparameters):
        r""" Instantiate config, model, tokenizer and metrics. """
        super().__init__(hyperparameters)

        self.tokenizer = AutoTokenizer.from_pretrained(hyperparameters.pre_trained_model)
        self.config = AutoConfig.from_pretrained(self.hyperparameters.pre_trained_model)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.hyperparameters.pre_trained_model, config=self.config
        )

        self.train_start_acc = Accuracy()
        self.train_end_acc = Accuracy()

    def training_step(self, batch, *args):
        r""" Simple training step without complete evaluation.
        Computing only accuracy of first and last position.
        """
        input_ids, start_position, end_position = batch['input_ids'], batch['start_position'], batch['end_position']

        attention_mask = batch.get('attention_mask', None)
        token_type_ids = batch.get('token_type_ids', None)

        results = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_position,
            end_positions=end_position
        )
        self.train_start_acc(results.start_logits.argmax(dim=-1), start_position)
        self.train_end_acc(results.end_logits.argmax(dim=-1), end_position)

        self.log('training/loss', results.loss, on_step=True)
        self.log('training/start_acc', self.train_start_acc, on_step=True)
        self.log('training/end_acc', self.train_end_acc, on_step=True)

        return results.loss

    def eval_step(self, batch, *args):
        r""" Generic eval step. Used for both validation and test. """
        input_ids, unique_indexes, start_position, end_position = (
            batch['input_ids'], batch['unique_id'], batch['start_position'], batch['end_position']
        )

        attention_mask = batch.get('attention_mask', None)
        token_type_ids = batch.get('token_type_ids', None)

        results = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_position,
            end_positions=end_position
        )
        return results.loss, unique_indexes, results.start_logits, results.end_logits

    # all methods are similar apart from logging names and input data
    def validation_step(self, *args, **kwargs):
        r""" Exactly as `eval_step`. """
        return self.eval_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        r""" Exactly as `eval_step`. """
        return self.eval_step(*args, **kwargs)

    def validation_epoch_end(self, outputs: List) -> None:
        r"""
        Collects results from the various steps, syncronizes from the different processes and
        finally computes the metrics for QA.
        """
        loss = torch.stack([o[0] for o in outputs]).mean()
        unique_indexes = torch.cat([o[1] for o in outputs], dim=0)
        batch_start_logits = torch.cat([o[2] for o in outputs], dim=0)
        batch_end_logits = torch.cat([o[3] for o in outputs], dim=0)

        unique_indexes = self.all_gather(unique_indexes).view(-1).detach().cpu().tolist()

        batch_start_logits = self.all_gather(batch_start_logits)
        batch_start_logits = batch_start_logits.view(-1, batch_start_logits.shape[-1]).detach().cpu().tolist()
        batch_end_logits = self.all_gather(batch_end_logits)
        batch_end_logits = batch_end_logits.view(-1, batch_end_logits.shape[-1]).detach().cpu().tolist()

        all_results = [
            dict(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits)
            for start_logits, end_logits, unique_id in zip(batch_start_logits, batch_end_logits, unique_indexes)
        ]

        preds = make_predictions(
            self.trainer.datamodule.valid_examples,
            self.trainer.datamodule.valid_features,
            all_results,
            self.hyperparameters.n_best_size,
            self.hyperparameters.max_answer_length,
        )

        original = self.trainer.datamodule.valid_examples
        exact_raw, f1_raw = get_raw_scores(original, preds)

        exact_match = torch.tensor(list(exact_raw.values()), dtype=torch.float).mean()
        f1 = torch.tensor(list(f1_raw.values()), dtype=torch.float).mean()

        self.log('validation/loss', loss)
        self.log('validation/em', exact_match)
        self.log('validation/f1', f1)

    def test_epoch_end(self, outputs):
        r"""
        Collects results from the various steps, syncronizes from the different processes and
        finally computes the metrics for QA.
        """
        if isinstance(outputs[0], tuple):
            outputs = [outputs]

        for dataloader_idx, dataloader_outputs in enumerate(outputs):
            loss = torch.stack([o[0] for o in dataloader_outputs]).mean()
            unique_indexes = torch.cat([o[1] for o in dataloader_outputs], dim=0)
            batch_start_logits = torch.cat([o[2] for o in dataloader_outputs], dim=0)
            batch_end_logits = torch.cat([o[3] for o in dataloader_outputs], dim=0)

            unique_indexes = self.all_gather(unique_indexes).view(-1).detach().cpu().tolist()

            batch_start_logits = self.all_gather(batch_start_logits)
            batch_start_logits = batch_start_logits.view(-1, batch_start_logits.shape[-1]).detach().cpu().tolist()
            batch_end_logits = self.all_gather(batch_end_logits)
            batch_end_logits = batch_end_logits.view(-1, batch_end_logits.shape[-1]).detach().cpu().tolist()

            all_results = [
                dict(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits)
                for start_logits, end_logits, unique_id in zip(batch_start_logits, batch_end_logits, unique_indexes)
            ]

            preds = make_predictions(
                self.trainer.datamodule.test_examples[dataloader_idx],
                self.trainer.datamodule.test_features[dataloader_idx],
                all_results,
                self.hyperparameters.n_best_size,
                self.hyperparameters.max_answer_length,
            )

            original = self.trainer.datamodule.test_examples[dataloader_idx]
            exact_raw, f1_raw = get_raw_scores(original, preds)

            exact_match = torch.tensor(list(exact_raw.values()), dtype=torch.float).mean()
            f1 = torch.tensor(list(f1_raw.values()), dtype=torch.float).mean()

            name = self.hyperparameters.test_subsets[dataloader_idx]
            self.log(f'test/{name}/loss', loss)
            self.log(f'test/{name}/em', exact_match)
            self.log(f'test/{name}/f1', f1)

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser):
        super(QuestionAnsweringModel, QuestionAnsweringModel).add_model_specific_args(parser)
        parser.add_argument(
            "--pre_trained_model", default=None, type=str, required=True, help="Path or name of pre-trained model."
        )
        parser.add_argument(
            "--n_best_size",
            default=20,
            type=int,
            help="The total number of n-best predictions to generate in the nbest_predictions.json output file."
        )
