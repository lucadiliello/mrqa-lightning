import torch
from typing import List
from transformers_lightning.models import TransformersModel
from transformers import AutoConfig, AutoModelForQuestionAnswering
from dataclasses import dataclass
from evaluation.evaluation import make_predictions, get_raw_scores, make_eval_dict


@dataclass
class RawResult:
    unique_id: int
    start_logits: List[int]
    end_logits: List[int]


class QuestionAnsweringModel(TransformersModel):
    r""" QA functionalities with EM and F1 computation. """

    def __init__(self, hyperparameters, tokenizer) -> None:
        super().__init__(hyperparameters)
        self.config = AutoConfig.from_pretrained(self.hyperparameters.pre_trained_model)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.hyperparameters.pre_trained_model, config=self.config)
        self.tokenizer = tokenizer

    def training_step(self, batch, *args):
        input_ids, input_mask, segment_ids, start_positions, end_positions = (
            batch['input_ids'], batch['input_mask'], batch['segment_ids'], batch['start_positions'],  batch['end_positions']
        )
        results = self.model(
            input_ids,
            input_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        self.log('training/loss', results.loss, prog_bar=True, on_epoch=True)
        return results.loss

    def validation_step(self, batch, *args):
        input_ids, input_mask, segment_ids, ids = (
            batch['input_ids'], batch['input_mask'], batch['segment_ids'], batch['id']
        )

        results = self.model(input_ids, input_mask)
        batch_start_logits, batch_end_logits = results.start_logits, results.end_logits
        return {'ids': ids, 'batch_start_logits': batch_start_logits, 'batch_end_logits': batch_end_logits}

    def validation_epoch_end(self, outputs: List) -> None:
        ids = torch.cat([o['ids'] for o in outputs], dim=0)
        batch_start_logits = torch.cat([o['batch_start_logits'] for o in outputs], dim=0)
        batch_end_logits = torch.cat([o['batch_end_logits'] for o in outputs], dim=0)

        ids = self.all_gather(ids, sync_grads=False).view(-1).detach().cpu().tolist()
        batch_start_logits = self.all_gather(batch_start_logits, sync_grads=False)
        batch_end_logits = self.all_gather(batch_end_logits, sync_grads=False)
        batch_start_logits = batch_start_logits.view(-1, batch_start_logits.shape[-1]).detach().cpu().tolist()
        batch_end_logits = batch_end_logits.view(-1, batch_end_logits.shape[-1]).detach().cpu().tolist()

        all_results = []
        for i, example_index in enumerate(ids):
            start_logits = batch_start_logits[i]
            end_logits = batch_end_logits[i]
            eval_feature = self.trainer.datamodule.valid_features[example_index]
            unique_id = eval_feature.unique_id
            raw_result = RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits
            )
            all_results.append(raw_result)

        preds, _ = make_predictions(
            self.trainer.datamodule.valid_examples,
            self.trainer.datamodule.valid_features,
            all_results,
            self.hyperparameters.n_best_size,
            self.hyperparameters.max_answer_length,
        )

        exact_raw, f1_raw = get_raw_scores(self.trainer.datamodule.valid_original, preds)
        result = make_eval_dict(exact_raw, f1_raw)

        self.log('validation/em', result['exact'], prog_bar=True, on_epoch=True)
        self.log('validation/f1', result['f1'], prog_bar=True, on_epoch=True)

    def test_step(self, batch, *args):
        input_ids, input_mask, segment_ids, ids = (
            batch['input_ids'], batch['input_mask'], batch['segment_ids'], batch['id']
        )

        results = self.model(input_ids, input_mask)
        batch_start_logits, batch_end_logits = results.start_logits, results.end_logits
        return {'ids': ids, 'batch_start_logits': batch_start_logits, 'batch_end_logits': batch_end_logits}

    def test_epoch_end(self, outputs: List) -> None:
        ids = torch.cat([o['ids'] for o in outputs], dim=0)
        batch_start_logits = torch.cat([o['batch_start_logits'] for o in outputs], dim=0)
        batch_end_logits = torch.cat([o['batch_end_logits'] for o in outputs], dim=0)

        ids = self.all_gather(ids, sync_grads=False).view(-1).detach().cpu().tolist()
        batch_start_logits = self.all_gather(batch_start_logits, sync_grads=False)
        batch_end_logits = self.all_gather(batch_end_logits, sync_grads=False)
        batch_start_logits = batch_start_logits.view(-1, batch_start_logits.shape[-1]).detach().cpu().tolist()
        batch_end_logits = batch_end_logits.view(-1, batch_end_logits.shape[-1]).detach().cpu().tolist()

        all_results = []
        for i, example_index in enumerate(ids):
            start_logits = batch_start_logits[i]
            end_logits = batch_end_logits[i]
            eval_feature = self.trainer.datamodule.valid_features[example_index]
            unique_id = eval_feature.unique_id
            raw_result = RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits
            )
            all_results.append(raw_result)

        preds, _ = make_predictions(
            self.trainer.datamodule.valid_examples,
            self.trainer.datamodule.valid_features,
            all_results,
            self.hyperparameters.n_best_size,
            self.hyperparameters.max_answer_length,
        )

        exact_raw, f1_raw = get_raw_scores(self.trainer.datamodule.valid_original, preds)
        result = make_eval_dict(exact_raw, f1_raw)

        self.log('test/em', result['exact'], prog_bar=True, on_epoch=True)
        self.log('test/f1', result['f1'], prog_bar=True, on_epoch=True)

    def predict_step(self, batch, *args):
        input_ids, input_mask, segment_ids, ids = (
            batch['input_ids'], batch['input_mask'], batch['segment_ids'], batch['id']
        )

        results = self.model(input_ids, input_mask)
        batch_start_logits, batch_end_logits = results.start_logits, results.end_logits
        return {'ids': ids, 'batch_start_logits': batch_start_logits, 'batch_end_logits': batch_end_logits}

    def predict_epoch_end(self, outputs: List) -> None:
        ids = torch.cat([o['ids'] for o in outputs], dim=0)
        batch_start_logits = torch.cat([o['batch_start_logits'] for o in outputs], dim=0)
        batch_end_logits = torch.cat([o['batch_end_logits'] for o in outputs], dim=0)

        ids = self.all_gather(ids, sync_grads=False).view(-1).detach().cpu().tolist()
        batch_start_logits = self.all_gather(batch_start_logits, sync_grads=False)
        batch_end_logits = self.all_gather(batch_end_logits, sync_grads=False)
        batch_start_logits = batch_start_logits.view(-1, batch_start_logits.shape[-1]).detach().cpu().tolist()
        batch_end_logits = batch_end_logits.view(-1, batch_end_logits.shape[-1]).detach().cpu().tolist()

        all_results = []
        for i, example_index in enumerate(ids):
            start_logits = batch_start_logits[i]
            end_logits = batch_end_logits[i]
            eval_feature = self.trainer.datamodule.valid_features[example_index]
            unique_id = eval_feature.unique_id
            raw_result = RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits
            )
            all_results.append(raw_result)

        preds, nbest_preds = make_predictions(
            self.trainer.datamodule.eval_examples,
            self.trainer.datamodule.eval_features,
            all_results,
            self.hyperparameters.n_best_size,
            self.hyperparameters.max_answer_length,
        )

        exact_raw, f1_raw = get_raw_scores(self.trainer.datamodule.eval_original, preds)
        result = make_eval_dict(exact_raw, f1_raw)

        self.log('predict/em', result['exact'], prog_bar=True, on_epoch=True)
        self.log('predict/f1', result['f1'], prog_bar=True, on_epoch=True)

        return result, preds, nbest_preds