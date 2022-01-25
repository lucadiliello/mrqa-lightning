# mrqa-lightning

MRQA test suite on PyTorch Lightning.

Easily run the MRQA test suite with any `*ForQuestionAnswering` model from `transformers`.
`pytorch-lightning` will manage all the hardware resources, allowing you to run on CPU, GPU,
Multi GPU and Multi Node - Multi GPU without changing a single line of code.


## Advantages

- Updated to use latest versions of `pytorch-lightning` and `transformers`
- Works with every `*ForQuestionAnswering` in the `transformers` library
- Much faster datasets preparation using multiprocessing
- Much easier training on different machines thanks to `pytorch-lightning`
- Checkpointing and better logging
- `datasets` library really reduces memory usage
- tokenization works for every model, and not just for `BERT` as in the original project


## Prepare environment

Start by installing the requirements:
```bash
pip install -r requirements.txt
```

## Get the data

The training/dev/test datasets are automatically downloaded. You can choose training and validation sets among:
- `NewsQA`
- `NaturalQuestionsShort`
- `TriviaQA-web`
- `SearchQA`
- `HotpotQA`
- `SQuAD`

and test sets among:
- `RACE`
- `DuoRC.ParaphraseRC`
- `BioASQ`
- `TextbookQA`
- `RelationExtraction`
- `DROP`


## Run the fine-tuning

Now run your fine-tuning on HotPotQA:
```bash
python main.py \
    --accelerator gpu \
    --pre_trained_model roberta-base \
    --name roberta-base-finetuned-hotpotqa \
    --train_subsets HotpotQA SearchQA \
    --val_subsets HotpotQA \
    --test_subsets DROP RACE \
    --batch_size 16 \
    --val_batch_size 32 \
    --test_batch_size 32 \
    --accumulate_grad_batches 2 \
    --learning_rate 1e-5 \
    --max_epochs 4 \
    --max_sequence_length 512 \
    --doc_stride 128 \
    --val_check_interval 0.2 \
    --output_dir outputs/mrqa-training/hotpot_search \
    --num_warmup_steps 100 \
    --monitor validation/f1 \
    --patience 5 \
    --early_stopping True \
    --num_workers 8 \
```

If you wans to use many (say 8) GPUs, set `--accelerator gpu --strategy ddp --devices 8`.
Please refer to [pytorch-lighting doc](https://pytorch-lightning.readthedocs.io/en/stable/) for the training hyperparameters.

Another example with multiple GPUs and many train files:

```bash
python main.py \
    --accelerator gpu --devices 8 --strategy ddp \
    --pre_trained_model roberta-base \
    --name roberta-base-finetuned-hotpotqa \
    --train_subsets HotpotQA \
    --val_subsets HotpotQA \
    --test_subsets DROP \
    --batch_size 32 \
    --val_batch_size 32 \
    --test_batch_size 32 \
    --accumulate_grad_batches 2 \
    --learning_rate 1e-5 \
    --max_epochs 4 \
    --max_sequence_length 512 \
    --doc_stride 128 \
    --val_check_interval 0.2 \
    --output_dir outputs/mrqa-training/hotpotqa \
    --num_warmup_steps 100 \
    --monitor validation/f1 \
    --patience 5 \
    --early_stopping True \
    --num_workers 8 \
```

Get all the available hyperparameters with:

```bash
python main.py --help
```

## FAQ
- `batch_size` is per single device (GPU, TPU, ...)
- `accumulate_grad_batches` enables you to use larger batch sizes when memory is a contraint
- `max_steps` set the max number of steps instead of `max_epochs`
- use only `test_subsets` if you want to test only a model


## Troubleshooting
- Delete cache folder and let the script create datasets again. The default cache folder is `~/.cache/mrqa-lightning`.


## Copyright
Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved.
Project adapted by Luca Di Liello from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_mrqa.py
