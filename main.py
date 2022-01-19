# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
# Adapterd by Luca Di Liello from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_mrqa.py

import argparse
import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.seed import seed_everything
from transformers import AutoTokenizer
from transformers_lightning.callbacks import TransformersModelCheckpointCallback
from transformers_lightning.defaults import DefaultConfig
from datamodule.datamodule import QuestionAnsweringDataModule
from model.model import QuestionAnsweringModel

from utilities.utilities import ExtendedNamespace


PRED_FILE = "predictions.json"
TEST_FILE = "test_results.txt"


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main(hyperparameters):

    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    tokenizer = AutoTokenizer.from_pretrained(hyperparameters.pre_trained_model)

    # set the random seed
    seed_everything(seed=hyperparameters.seed, workers=True)

    # instantiate PL model
    model = QuestionAnsweringModel(hyperparameters, tokenizer)

    # default tensorboard logger
    test_tube_logger = TestTubeLogger(
        save_dir=os.path.join(hyperparameters.output_dir, 'tensorboard'),
        name=hyperparameters.name,
    )
    loggers = [test_tube_logger]

    # save pre-trained models to
    save_transformers_callback = TransformersModelCheckpointCallback(hyperparameters)

    # and normal checkpoints with
    checkpoints_dir = os.path.join(hyperparameters.output_dir, 'checkpoints', hyperparameters.name)
    checkpoint_callback_hyperparameters = {'verbose': True, 'dirpath': checkpoints_dir}

    if hyperparameters.monitor is not None:
        checkpoint_callback_hyperparameters = {
            **checkpoint_callback_hyperparameters,
            'monitor': hyperparameters.monitor,
            'save_last': True,
            'mode': 'max',
            'save_top_k': 1,
        }

    checkpoint_callback = ModelCheckpoint(**checkpoint_callback_hyperparameters)

    # all callbacks
    callbacks = [
        save_transformers_callback,
        checkpoint_callback,
    ]

    # early stopping if defined
    if hyperparameters.early_stopping:
        if hyperparameters.monitor is None:
            raise ValueError("cannot use early_stopping without a monitored variable")

        early_stopping_callback = EarlyStopping(
            monitor=hyperparameters.monitor,
            patience=hyperparameters.patience,
            verbose=True,
            mode=hyperparameters.monitor_direction,
        )
        callbacks.append(early_stopping_callback)

    # disable find unused parameters to improve performance
    kwargs = dict()
    if hyperparameters.strategy == "ddp":
        kwargs['strategy'] = DDPPlugin(find_unused_parameters=False)

    # instantiate PL trainer
    trainer = pl.Trainer.from_argparse_args(
        hyperparameters,
        default_root_dir=hyperparameters.output_dir,
        logger=loggers,
        callbacks=callbacks,
        weights_summary='full',
        profiler='simple',
        **kwargs,
    )

    # DataModules
    datamodule = QuestionAnsweringDataModule(hyperparameters, trainer, tokenizer)

    # Train!
    if datamodule.do_train():
        trainer.fit(model, datamodule=datamodule)

    # Test!
    if datamodule.do_test():
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    DefaultConfig.add_defaults_args(parser)

    # add model / callback / logger specific parameters
    QuestionAnsweringModel.add_model_specific_args(parser)
    TransformersModelCheckpointCallback.add_callback_specific_args(parser)
    QuestionAnsweringDataModule.add_datamodule_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --devices --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--monitor', type=str, help='Value to monitor for best checkpoint', default=None)
    parser.add_argument(
        '--monitor_direction', type=str, help='Monitor value direction for best', default='max', choices=['min', 'max']
    )
    parser.add_argument('--early_stopping', type=bool, default=False, help="Use early stopping")
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        required=False,
        help="Number of non-improving validations to wait before early stopping"
    )
    parser.add_argument("--name", type=str, required=True, help="Run name")

    # I/O folders
    parser.add_argument('--predictions_dir', type=str, default="predictions", required=False, help="Predictions folder")
    parser.add_argument("--pre_trained_model", default=None, type=str, required=True)
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                                "output file.")
    parser.add_argument('--seed', type=int, default=1337,
                        help="random seed for initialization")

    # get NameSpace of paramters
    hyperparameters = parser.parse_args()
    hyperparameters = ExtendedNamespace.from_namespace(hyperparameters)
    hyperparameters.num_sanity_val_steps = 0

    main(hyperparameters)
