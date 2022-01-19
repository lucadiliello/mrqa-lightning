import torch
from typing import List
from transformers_lightning.models import TransformersModel
from transformers import AutoConfig, AutoModelForQuestionAnswering
from evaluation.evaluation import make_predictions, get_raw_scores, make_eval_dict


class QuestionAnsweringModel(TransformersModel):
    r""" QA functionalities with EM and F1 computation. """

    def __init__(self, hyperparameters, tokenizer) -> None:
        super().__init__(hyperparameters)
        self.config = AutoConfig.from_pretrained(self.hyperparameters.pre_trained_model)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.hyperparameters.pre_trained_model, config=self.config)
        self.tokenizer = tokenizer

    def training_step(self, batch, *args):
        
        input_ids, start_position, end_position = batch['input_ids'], batch['start_position'],  batch['end_position']

        attention_mask = batch.get('attention_mask', None)
        token_type_ids = batch.get('token_type_ids', None)

        results = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_position,
            end_positions=end_position
        )

        self.log('training/loss', results.loss, prog_bar=True, on_epoch=True)
        return results.loss

    def validation_step(self, batch, *args):
        input_ids, ids = batch['input_ids'], batch['example_index']

        attention_mask = batch.get('attention_mask', None)
        token_type_ids = batch.get('token_type_ids', None)

        results = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

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

        all_results = [
            dict(
                unique_id=self.trainer.datamodule.valid_dataset[example_index]['unique_id'],
                start_logits=batch_start_logits[i],
                end_logits=batch_end_logits[i]
            )
            for i, example_index in enumerate(ids)
        ]

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
        input_ids, ids = batch['input_ids'], batch['example_index']

        attention_mask = batch.get('attention_mask', None)
        token_type_ids = batch.get('token_type_ids', None)

        results = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        batch_start_logits, batch_end_logits = results.start_logits, results.end_logits
        return {'ids': ids, 'batch_start_logits': batch_start_logits, 'batch_end_logits': batch_end_logits}

    def test_epoch_end(self, all_outputs: List) -> None:
        r""" Evaluate a list of predictions for each test dataloader. """
        if isinstance(all_outputs[0], dict):
            all_outputs = [all_outputs]

        for dataloader_idx, outputs in enumerate(all_outputs):
            ids = torch.cat([o['ids'] for o in outputs], dim=0)
            batch_start_logits = torch.cat([o['batch_start_logits'] for o in outputs], dim=0)
            batch_end_logits = torch.cat([o['batch_end_logits'] for o in outputs], dim=0)

            ids = self.all_gather(ids, sync_grads=False).view(-1).detach().cpu().tolist()
            batch_start_logits = self.all_gather(batch_start_logits, sync_grads=False)
            batch_end_logits = self.all_gather(batch_end_logits, sync_grads=False)
            batch_start_logits = batch_start_logits.view(-1, batch_start_logits.shape[-1]).detach().cpu().tolist()
            batch_end_logits = batch_end_logits.view(-1, batch_end_logits.shape[-1]).detach().cpu().tolist()

            all_results = [
                dict(
                    unique_id=self.trainer.datamodule.test_dataset[dataloader_idx][example_index]['unique_id'],
                    start_logits=batch_start_logits[i],
                    end_logits=batch_end_logits[i]
                )
                for i, example_index in enumerate(ids)
            ]

            preds, _ = make_predictions(
                self.trainer.datamodule.test_examples[dataloader_idx],
                self.trainer.datamodule.test_features[dataloader_idx],
                all_results,
                self.hyperparameters.n_best_size,
                self.hyperparameters.max_answer_length,
            )

            exact_raw, f1_raw = get_raw_scores(self.trainer.datamodule.test_original[dataloader_idx], preds)
            result = make_eval_dict(exact_raw, f1_raw)

            self.log(f'test/{dataloader_idx}/em', result['exact'], prog_bar=True, on_epoch=True)
            self.log(f'test/{dataloader_idx}/f1', result['f1'], prog_bar=True, on_epoch=True)

    def predict_step(self, batch, *args):
        input_ids, ids = batch['input_ids'], batch['example_index']

        attention_mask = batch.get('attention_mask', None)
        token_type_ids = batch.get('token_type_ids', None)

        results = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        batch_start_logits, batch_end_logits = results.start_logits, results.end_logits
        return {'ids': ids, 'batch_start_logits': batch_start_logits, 'batch_end_logits': batch_end_logits}

    def predict_epoch_end(self, all_outputs: List) -> None:
        r""" Evaluate a list of predictions for each predict dataloader. """
        if isinstance(all_outputs[0], dict):
            all_outputs = [all_outputs]

        res = []
        for dataloader_idx, outputs in enumerate(all_outputs):
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
                unique_id = self.trainer.datamodule.predict_features[dataloader_idx][example_index].unique_id
                raw_result = dict(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits
                )
                all_results.append(raw_result)

            preds, nbest_preds = make_predictions(
                self.trainer.datamodule.predict_examples[dataloader_idx],
                self.trainer.datamodule.predict_features[dataloader_idx],
                all_results,
                self.hyperparameters.n_best_size,
                self.hyperparameters.max_answer_length,
            )

            exact_raw, f1_raw = get_raw_scores(self.trainer.datamodule.predict_original[dataloader_idx], preds)
            result = make_eval_dict(exact_raw, f1_raw)

            self.log(f'predict/{i}/em', result['exact'], prog_bar=True, on_epoch=True)
            self.log(f'predict/{i}/f1', result['f1'], prog_bar=True, on_epoch=True)

            res.append(result, preds, nbest_preds)

        return res
