# qlora.py
import os
import time
import glob
from argparse import ArgumentParser

import torch
from torchinfo import summary
from datasets import concatenate_datasets, DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from trl import SFTTrainer
try:
    from trl import DataCollatorForCompletionOnly
    _HAS_TRL_COLLATOR = True
except Exception:
    DataCollatorForCompletionOnly = None
    _HAS_TRL_COLLATOR = False

from llm_unlearning.version1.utils import get_data_path, compute_metrics, preprocess_logits_for_metrics, CustomCallback, CompletionOnlyCollator


def get_args():
    parser = ArgumentParser(description="Fine-tune distilgpt2 with LoRA on CPU")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (sst2 or yelp)")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="HF model ID (default: distilgpt2)")
    parser.add_argument("--output_path", type=str, default="checkpoints/distilgpt2_qlora", help="Save path")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Train batch size (CPU-safe)")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Eval batch size (CPU-safe)")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs (keep small on CPU)")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_bias", type=str, default="none", choices={"lora_only", "none", "all"}, help="LoRA bias mode")
    return parser.parse_args()


def get_lora_model(model_name, rank=4, alpha=16, lora_dropout=0.05, bias="none"):
    # CPU-only model, no quantization
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=lora_dropout,
        bias=bias,
    )
    return model, tokenizer, peft_config


def _load_parquet_split_files(dirpath, pattern):
    files = sorted(glob.glob(os.path.join(dirpath, pattern)))
    if not files:
        return None
    ds = None
    for f in files:
        part = load_dataset("parquet", data_files=f)["train"]
        ds = part if ds is None else concatenate_datasets([ds, part])
    return ds


def get_unlearn_dataset_and_collator(data_path, tokenizer, max_length=128):
    prompt_template = lambda text, label: f"""### Text: {text}\n\n### Question: What is the sentiment?\n\n### Sentiment: {label}"""

    def _preprocess(examples):
        return {"text": prompt_template(examples["text"], examples["label_text"])}

    response_template = "\n### Sentiment:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    if _HAS_TRL_COLLATOR and DataCollatorForCompletionOnly is not None:
        collator = DataCollatorForCompletionOnly(response_template_ids, tokenizer=tokenizer)
    else:
        collator = CompletionOnlyCollator(response_template_ids, tokenizer=tokenizer, max_length=max_length)

    raw = load_dataset(data_path)

    # Normalize to a DatasetDict with expected keys
    if all(k in raw for k in ("train_retain", "train_forget", "test_retain", "test_forget")):
        dataset = DatasetDict({
            "train_retain": raw["train_retain"],
            "train_forget": raw["train_forget"],
            "test_retain": raw["test_retain"],
            "test_forget": raw["test_forget"],
        })
    else:
        if "train" in raw and "test" in raw:
            train = raw["train"]
            test = raw["test"]
            if "is_forget" in train.column_names:
                train_retain = train.filter(lambda x: x["is_forget"] in (0, False))
                train_forget = train.filter(lambda x: x["is_forget"] in (1, True))
                test_retain = test.filter(lambda x: x["is_forget"] in (0, False))
                test_forget = test.filter(lambda x: x["is_forget"] in (1, True))
            elif "subset" in train.column_names:
                train_retain = train.filter(lambda x: x.get("subset", "") == "retain")
                train_forget = train.filter(lambda x: x.get("subset", "") == "forget")
                test_retain = test.filter(lambda x: x.get("subset", "") == "retain")
                test_forget = test.filter(lambda x: x.get("subset", "") == "forget")
            else:
                if os.path.isdir(data_path):
                    train_retain = _load_parquet_split_files(data_path, "train_retain*.parquet")
                    train_forget = _load_parquet_split_files(data_path, "train_forget*.parquet")
                    test_retain = _load_parquet_split_files(data_path, "test_retain*.parquet")
                    test_forget = _load_parquet_split_files(data_path, "test_forget*.parquet")
                else:
                    train_retain = train
                    train_forget = train.select([])
                    test_retain = test
                    test_forget = test.select([])

                train_retain = train_retain or train
                train_forget = train_forget or train.select([])
                test_retain = test_retain or test
                test_forget = test_forget or test.select([])

            dataset = DatasetDict({
                "train_retain": train_retain,
                "train_forget": train_forget,
                "test_retain": test_retain,
                "test_forget": test_forget,
            })
        else:
            if os.path.isdir(data_path):
                train_retain = _load_parquet_split_files(data_path, "train_retain*.parquet") or load_dataset(data_path)["train"]
                train_forget = _load_parquet_split_files(data_path, "train_forget*.parquet") or load_dataset(data_path)["train"].select([])
                test_retain = _load_parquet_split_files(data_path, "test_retain*.parquet") or load_dataset(data_path)["train"].select([])
                test_forget = _load_parquet_split_files(data_path, "test_forget*.parquet") or load_dataset(data_path)["train"].select([])
                dataset = DatasetDict({
                    "train_retain": train_retain,
                    "train_forget": train_forget,
                    "test_retain": test_retain,
                    "test_forget": test_forget,
                })
            else:
                raise RuntimeError("Unable to interpret dataset layout. Expected train/test or parquet files under the dataset dir.")

    # Format and prompt-map each split
    for split in ("train_retain", "train_forget", "test_retain", "test_forget"):
        ds = dataset[split]
        if "text" in ds.column_names and "label_text" in ds.column_names:
            dataset[split] = ds.map(_preprocess, batched=False)
        elif "text" in ds.column_names and "label" in ds.column_names:
            def _map_label(ex):
                return {"label_text": str(ex["label"])}
            dataset[split] = ds.map(_map_label, batched=False)
            dataset[split] = dataset[split].map(_preprocess, batched=False)
        else:
            raise RuntimeError(f"Dataset split {split} missing required columns 'text' and 'label_text'/'label'.")

    # Remove original label columns and set format
    for split in dataset:
        cols_to_remove = [c for c in ("label", "label_text") if c in dataset[split].column_names]
        if cols_to_remove:
            try:
                dataset[split] = dataset[split].remove_columns(cols_to_remove)
            except Exception:
                pass
        dataset[split].set_format("torch")

    return dataset, collator


def main(args):
    model, tokenizer, lora_config = get_lora_model(
        args.model_name,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
    )

    data_path = get_data_path(args.dataset)
    dataset, collator = get_unlearn_dataset_and_collator(data_path, tokenizer, max_length=args.max_length)

    os.makedirs(args.output_path, exist_ok=True)

    # Wrap model with LoRA explicitly and freeze base params
    peft_model = get_peft_model(model, lora_config)

    # Ensure base model weights frozen (defensive)
    for n, p in peft_model.named_parameters():
        if "lora" not in n and "adapter" not in n:
            p.requires_grad = False

    # Print parameter counts to confirm LoRA attach
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params (should be small with LoRA): {trainable_params:,}")

    # Show model summary after LoRA applied
    summary(peft_model)

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=5,
        report_to="none",
    )

    # Build concatenated train/eval datasets for the trainer
    train_concat = concatenate_datasets([dataset["train_retain"], dataset["train_forget"]])
    eval_concat = concatenate_datasets([dataset["test_retain"], dataset["test_forget"]])

    # Pass the already-wrapped PEFT model and do not pass peft_config to avoid double-wrap
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        peft_config=None,
        train_dataset=train_concat,
        eval_dataset=eval_concat,
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(CustomCallback(trainer))

    start = time.perf_counter()
    trainer.train()
    print("Training runtime (s):", time.perf_counter() - start)


if __name__ == "__main__":
    args = get_args()
    main(args)
