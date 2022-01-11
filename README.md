# mrqa-lightning
MRQA test suite on PyTorch Lightning

Easily run the MRQA test suite with any `*ForQuestionAnswering` model from `transformers`.
`pytorch-lightning` will manage all the hardware resources, allowing you to run on CPU, GPU,
Multi GPU and Multi Node - Multi GPU without changing a single line of code.

## Avdantages

- Updated to use latest versions of `pytorch-lightning` and `transformers`
- Works with every `*ForQuestionAnswering` in the `transformers` library
- Much faster datasets preparation using multiprocessing
- Much easier training on different machines thanks to `pytorch-lightning`
- Checkpointing and better logging

## Prepare environment

Start by installing the requirements:
```bash
pip install -r requirements.txt
```

## Get the data

Download the training/dev/eval files from the
(original repository)[https://github.com/mrqa/MRQA-Shared-Task-2019#datasets] into some
data folder `data_dir`. You can use many files for training: examples of different datasets
will be shuffled together.

## Run the fine-tuning

Now run your fine-tuning on HotPotQA:
```bash
python main.py \
    --accelerator gpu \
    --pre_trained_model roberta-base \
    --name roberta-base-finetuned-mrqa \
    --train_file /science/lucadiliello/datasets/mrqa/HotpotQA-train.jsonl.gz \
    --dev_file /science/lucadiliello/datasets/mrqa/HotpotQA-dev.jsonl.gz \
    --batch_size 32 \
    --val_batch_size 32 \
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

or maybe on many GPUs my simply setting `--accelerator gpu --strategy ddp --devices 8` if you have a machine with 8 GPUs.
Please refer to (pytorch-lighting doc)[https://pytorch-lightning.readthedocs.io/en/stable/] for the training hyperparameters.

Use
```bash
python main.py --help
```

to get all the available hyperparameters.

## FAQ
- `batch_size` is per single device (GPU, TPU, ...)
- ``