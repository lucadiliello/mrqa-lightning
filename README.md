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

## Prepare environment

Start by installing the requirements:
```bash
pip install -r requirements.txt
```

## Get the data

The training/dev/test datasets can be automatically downloaded or provided by the user. Available datasets for automatic download include:

- `squad_train`
- `newsqa_train`
- `triviaqa_train`
- `searchqa_train`
- `hotpotqa_train`
- `naturalquestions_train`
- `squad_dev`
- `newsqa_dev`
- `triviaqa_dev`
- `searchqa_dev`
- `hotpotqa_dev`
- `naturalquestions_dev`
- `bioasq_dev`
- `drop_dev`
- `duorc_dev`
- `race_dev`
- `relationextraction_dev`
- `textbookqa_dev`

otherwise just provide the path to your `*.jsonl.gz` file.

## Run the fine-tuning

Now run your fine-tuning on HotPotQA:
```bash
python main.py \
    --accelerator gpu \
    --pre_trained_model roberta-base \
    --name roberta-base-finetuned-hotpotqa \
    --train_datasets hotpotqa_train \
    --dev_datasets hotpotqa_train \
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

If you wans to use many (say 8) GPUs, set `--accelerator gpu --strategy ddp --devices 8`.
Please refer to [pytorch-lighting doc](https://pytorch-lightning.readthedocs.io/en/stable/) for the training hyperparameters.

Another example with multiple GPUs and many train files:

```bash
python main.py \
    --accelerator gpu --devices 8 --strategy ddp \
    --pre_trained_model roberta-base \
    --name roberta-base-finetuned-hotpotqa \
    --train_datasets data_dir/HotpotQA-train.jsonl.gz data_dir/TriviaQA-train.jsonl.gz \
    --dev_datasets data_dir/HotpotQA-dev.jsonl.gz \
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

Pass custom train files like:
```bash
    --train_datasets data_dir/HotpotQA-train.jsonl.gz \
    --dev_datasets data_dir/HotpotQA-dev.jsonl.gz \
```

Get all the available hyperparameters with:

```bash
python main.py --help
```


## FAQ
- `batch_size` is per single device (GPU, TPU, ...)
- `accumulate_grad_batches` enables you to use larger batch sizes when memory is a contraint
- `max_steps` set the max number of steps instead of `max_epochs`
- use `test_datasets` or `predict_datasets` for testing only or for creating predictions
- predictions are placed into the `--prediction_dir` folder
