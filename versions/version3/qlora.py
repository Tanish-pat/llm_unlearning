# python3 qlora.py --dataset sst2 --model_name meta-llama/Llama-2-7b-hf --output_path checkpoints/qlora_llama2_7b_sst2 --train_batch_size 8 --eval_batch_size 8 --num_epochs 1 --max_length 32 --gradient_accumulation_steps 8 --save_every_epoch
# qlora.py
import os
import time
import glob
from argparse import ArgumentParser
from tqdm import tqdm

import torch
torch.backends.cudnn.benchmark = True

from torch.utils.data import DataLoader
from torchinfo import summary
from datasets import concatenate_datasets, DatasetDict, load_dataset
from peft import get_peft_model, PeftConfig, PeftModel, LoraConfig, TaskType

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from trl import DataCollatorForCompletionOnly as _TRL_DataCollator
    _HAS_TRL_COLLATOR = True
except Exception:
    _TRL_DataCollator = None
    _HAS_TRL_COLLATOR = False

from utils import get_data_path, compute_metrics, preprocess_logits_for_metrics, CustomCallback, CompletionOnlyCollator

# small memory helper
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")  # reduce fragmentation (may help)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # suppress warnings
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # disable telemetry

def get_args():
    parser = ArgumentParser(description="Fine-tune a large LLM with QLoRA (4-bit) + PEFT (manual loop)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (sst2 or yelp)")
    parser.add_argument("--model_name", type=str, required=True, help="HF model ID (e.g. meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--output_path", type=str, default="checkpoints/qlora_model", help="Save path")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Train batch size (GPU-safe)")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Eval batch size (GPU-safe)")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--lora_bias", type=str, default="none", choices={"lora_only", "none", "all"}, help="LoRA bias mode")
    parser.add_argument("--save_every_epoch", action="store_true", help="Save checkpoint after each epoch")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps (use to emulate larger batch sizes)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers")
    return parser.parse_args()


def get_lora_model(model_name, rank=8, alpha=32, lora_dropout=0.1, bias="none"):
    """
    Load model and tokenizer. If CUDA available, attempt 4-bit loading via bitsandbytes.
    Returns: (model, tokenizer, peft_config)
    """
    use_cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            # load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
            use_safetensors=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True)

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


def get_unlearn_dataset_and_collator(data_path, tokenizer, max_length=1024):
    prompt_template = lambda text, label: f"""### Text: {text}\n\n### Question: What is the sentiment?\n\n### Sentiment: {label}"""

    def _preprocess(examples):
        return {"text": prompt_template(examples["text"], examples.get("label_text", str(examples.get("label", ""))))}

    response_template = "\n### Sentiment:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    if _HAS_TRL_COLLATOR and _TRL_DataCollator is not None:
        collator = _TRL_DataCollator(response_template_ids, tokenizer=tokenizer)
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
                    train_forget = train.select([]) if hasattr(train, "select") else train
                    test_retain = test
                    test_forget = test.select([]) if hasattr(test, "select") else test

                train_retain = train_retain or train
                train_forget = train_forget or (train.select([]) if hasattr(train, "select") else train)
                test_retain = test_retain or test
                test_forget = test_forget or (test.select([]) if hasattr(test, "select") else test)

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


def move_batch_to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


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

    # Attach LoRA
    peft_model = get_peft_model(model, lora_config)

    # Freeze everything and enable only LoRA params
    for _, p in peft_model.named_parameters():
        p.requires_grad = False

    enabled = 0
    try:
        for p in peft_model.lora_parameters():
            p.requires_grad = True
            enabled += 1
    except Exception:
        for n, p in peft_model.named_parameters():
            if "lora" in n or "adapter" in n:
                p.requires_grad = True
                enabled += 1

    if enabled == 0:
        raise RuntimeError("No LoRA parameters enabled. Inspect PEFT attach step.")

    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params (LoRA): {trainable_params:,}")

    if trainable_params == 0:
        raise RuntimeError("No trainable parameters detected. Aborting.")

    # try to enable gradient checkpointing on the base model to save activation memory
    try:
        # prefer base_model if peft wrapper hides method
        base = getattr(peft_model, "base_model", peft_model)
        if hasattr(base, "gradient_checkpointing_disable"):
            base.gradient_checkpointing_disable()
    except Exception:
        pass

    summary(peft_model)

    use_cuda = torch.cuda.is_available()
    # determine primary device (if model uses device_map it may already be sharded)
    try:
        primary_device = next(peft_model.parameters()).device
    except StopIteration:
        primary_device = torch.device("cuda" if use_cuda else "cpu")
    print("Primary model device:", primary_device)

    # move the PEFT wrapper explicitly to the device if it's not already (safe no-op if already sharded)
    try:
        peft_model.to(primary_device)
    except Exception:
        pass

    # Prepare data loaders (do tokenization + collator on the CPU and move each batch to GPU)
    train_concat = concatenate_datasets([dataset["train_retain"], dataset["train_forget"]])
    eval_concat = concatenate_datasets([dataset["test_retain"], dataset["test_forget"]])

    train_loader = DataLoader(train_concat, batch_size=args.train_batch_size, shuffle=True,
                              collate_fn=collator, num_workers=args.num_workers)
    eval_loader  = DataLoader(eval_concat, batch_size=args.eval_batch_size, shuffle=False,
                              collate_fn=collator, num_workers=args.num_workers)

    params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    accum_steps = max(1, args.gradient_accumulation_steps)
    print("Using gradient_accumulation_steps =", accum_steps)

    # Training loop (manual)
    for epoch in range(1, args.num_epochs + 1):
        peft_model.train()
        epoch_loss = 0.0
        n_steps = 0
        start = time.perf_counter()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        optimizer.zero_grad()
        for step, batch in pbar:
            try:
                batch = move_batch_to_device(batch, primary_device)
                outputs = peft_model(**batch, return_dict=True, use_cache=False)
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    # fallback: try to compute cross-entropy from logits + labels
                    logits = getattr(outputs, "logits", None)
                    labels = batch.get("labels") if isinstance(batch, dict) else None
                    if logits is None or labels is None:
                        raise RuntimeError("Model did not return loss and logits/labels not available to compute it.")
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                # scale loss for accumulation
                loss = loss / accum_steps
                loss.backward()
            except torch.cuda.OutOfMemoryError as oom:
                # try to recover gracefully and give the user hints
                print("CUDA OOM during forward/backward. Emptying cache and trying to continue. Consider reducing batch size or max_length.")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                # bubble up serious errors
                raise

            if (step + 1) % accum_steps == 0:
                # gradient clipping (only trainable params)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                # track loss (unscale)
                epoch_loss += loss.item() * accum_steps
                n_steps += 1
                pbar.set_postfix({"loss": f"{(epoch_loss / max(1, n_steps)):.6f}"})

        epoch_time = time.perf_counter() - start
        avg_loss = epoch_loss / max(1, n_steps)
        print(f"Epoch {epoch} finished. avg_loss={avg_loss:.6f} steps={n_steps} time_s={epoch_time:.2f}")

        # evaluation (compute loss on eval set)
        peft_model.eval()
        eval_loss = 0.0
        eval_steps = 0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Eval"):
                batch = move_batch_to_device(batch, primary_device)
                outputs = peft_model(**batch, return_dict=True, use_cache=False)
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    logits = getattr(outputs, "logits", None)
                    labels = batch.get("labels") if isinstance(batch, dict) else None
                    if logits is None or labels is None:
                        continue
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                eval_loss += loss.item()
                eval_steps += 1
        if eval_steps:
            print(f"Epoch {epoch} eval_loss={eval_loss / eval_steps:.6f}")

        # save checkpoint after epoch if requested
        if args.save_every_epoch:
            epoch_path = os.path.join(args.output_path, f"epoch_{epoch}")
            os.makedirs(epoch_path, exist_ok=True)
            try:
                peft_model.save_pretrained(epoch_path)
                tokenizer.save_pretrained(epoch_path)
                print("Saved epoch checkpoint to:", epoch_path)
            except Exception as e:
                print("Warning: failed to save epoch checkpoint:", e)

    # final save (safe merge)
    try:
        if hasattr(peft_model, "merge_and_unload"):
            try:
                base_model = peft_model.merge_and_unload()
                base_model.save_pretrained(args.output_path)
            except KeyError as ke:
                # fallback: PEFT merge failed, save PEFT wrapper instead
                print(f"Warning: merge_and_unload failed ({ke}), saving PEFT model directly.")
                peft_model.save_pretrained(args.output_path)
        else:
            peft_model.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        print("Saved final PEFT model and tokenizer to:", args.output_path)
    except Exception as e:
        print("Warning: failed to save final model/tokenizer:", e)

if __name__ == "__main__":
    args = get_args()
    main(args)
