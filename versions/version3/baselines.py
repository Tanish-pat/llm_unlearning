# python3 baselines.py --dataset sst2 --model_checkpoints checkpoints/qlora_llama2_7b_sst2 --unlearn_method gradient_ascent --output_path unlearned_checkpoints/qlora_llama2_7b_sst2_ga --train_batch_size 8 --eval_batch_size 8 --num_epochs 1 --max_length 32
# baselines.py
import os
from copy import deepcopy
from argparse import ArgumentParser
import numpy as np
import datetime
import time
import pickle
import random

import torch
from torchinfo import summary

from datasets import load_dataset, concatenate_datasets
import evaluate
from peft import get_peft_model, PeftConfig, PeftModel, LoraConfig, TaskType
from transformers import AutoTokenizer, TrainerState, TrainerControl, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback, DataCollatorForLanguageModeling


from utils import get_data_path, compute_metrics, preprocess_logits_for_metrics, get_logits_from_base_model, CustomCallback

POS_WEIGHT, NEG_WEIGHT = (1.0, 1.0)
os.environ["WANDB_DISABLED"] = "true"

def get_args():
    parser = ArgumentParser(description="Fine-tune an LLM model with PEFT")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        help="name of dataset",
    )
    parser.add_argument(
        "--model_checkpoints",
        type=str,
        default=None,
        required=True,
        help="Path to checkpoints for base model to be unlearned",
    )
    parser.add_argument(
        "--unlearn_method",
        type=str,
        default=None,
        required=True,
        choices={"gradient_ascent", "random_label", "gradient_ascent_kl", "gradient_ascent_descent"},
        help="Name of baseline unlearn method"
    )
    parser.add_argument(
        "--logits_path",
        type=str,
        default=None,
        required=False,
        help="Path to save original logits to use for KL loss, used by GA+KL",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=False,
        help="Path to store the unlearned model",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        required=False,
        help="Maximum length of the input sequences",
    )
    parser.add_argument(
        "--set_pad_id",
        action="store_true",
        help="Set the id for the padding token, needed by models such as Mistral-7B",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Train batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="Eval batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of epochs"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay"
    )

    arguments = parser.parse_args()
    return arguments


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset['train_retain'],
                                   metric_key_prefix="eval_train_retrain")
            self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset['train_forget'],
                                   metric_key_prefix="eval_train_forget")
            return control_copy

def get_base_model(model_checkpoints, max_length=1024):
    """
    Load a 4-bit QLoRA checkpoint (LoRA adapters + quantized base model) for unlearning.
    Returns: model (PEFT), tokenizer, dummy lora_config
    """
    from types import SimpleNamespace
    from peft import PeftModel, LoraConfig, get_peft_model
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    import os

    # ----------------------
    # Load tokenizer
    # ----------------------
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Llama-2-7b-hf",
        use_fast=True,
        truncation=True,
        padding=True,
        max_length=max_length,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ----------------------
    # QLoRA: 4-bit config
    # ----------------------
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # ----------------------
    # Load base model
    # ----------------------
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Llama-2-7b-hf",
        device_map="auto",
        quantization_config=quant_config,
        trust_remote_code=True
    )

    # ----------------------
    # Attach LoRA adapter if checkpoint exists
    # ----------------------
    lora_checkpoint_path = os.path.join(model_checkpoints, "epoch_1")
    if os.path.exists(os.path.join(lora_checkpoint_path, "adapter_model.safetensors")):
        model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
        print(f"Loaded QLoRA checkpoint from {lora_checkpoint_path}")
    else:
        model = base_model
        print(f"No LoRA checkpoint found, using base model only.")

    # ----------------------
    # Make only floating-point params trainable
    # ----------------------
    for n, p in model.named_parameters():
        if p.is_floating_point():
            p.requires_grad = True
        else:
            p.requires_grad = False

    # ----------------------
    # Dummy lora_config to satisfy baselines.py
    # ----------------------
    lora_config = SimpleNamespace()
    lora_config.base_model_name_or_path = model_checkpoints
    lora_config.inference_mode = False

    return model, tokenizer, lora_config

def make_completion_only_collator(tokenizer, response_template_ids):
    base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def collator(features):
        batch = base_collator(features)
        labels = batch["labels"]

        # Mask everything except tokens after the response template
        for i, feature in enumerate(features):
            input_ids = feature["input_ids"]

            # Ensure tensor is on CPU for comparison
            input_ids_cpu = input_ids.cpu() if isinstance(input_ids, torch.Tensor) and input_ids.is_cuda else input_ids
            response_token = torch.tensor(response_template_ids[0], dtype=input_ids_cpu.dtype)

            matches = (input_ids_cpu == response_token).nonzero(as_tuple=True)[0]
            if matches.numel() > 0:
                start = matches[0].item()
            else:
                start = input_ids_cpu.size(0)  # fallback: ignore all tokens

            labels[i, :start] = -100  # mask tokens before response

        batch["labels"] = labels
        return batch

    return collator

def get_unlearn_dataset_and_collator(
        data_path,
        tokenizer,
        unlearn_method,
        col_to_delete,
        max_length,
        truncation,
        add_prefix_space=True,
):
    prompt_template = lambda text, label: f"""### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment: {label}"""

    def _preprocessing_sentiment(examples):
        return tokenizer(prompt_template(examples['text'], examples['label_text']), truncation=truncation, max_length=max_length )

    response_template = "\n### Sentiment:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    data_collator = make_completion_only_collator(tokenizer, response_template_ids)


    # data = load_dataset(data_path)
    data = load_dataset(
        "parquet",
        data_files={
            "train_forget": os.path.join(data_path, "train_forget_0.parquet"),
            "train_retain": os.path.join(data_path, "train_retain_0.parquet"),
            "test_forget": os.path.join(data_path, "test_forget_0.parquet"),
            "test_retain": os.path.join(data_path, "test_retain_0.parquet"),
        },
    )
    # Add flags to distinguish forget sets
    if unlearn_method.lower() in ['gradient_ascent_kl', 'gradient_ascent_descent']:
        data['train_forget'] = data['train_forget'].map(lambda item: {"is_forget": 1})
        data['train_retain'] = data['train_retain'].map(lambda item: {"is_forget": 0})
        data['train'] = concatenate_datasets([data['train_retain'], data['train_forget']])

        data['train_forget'] = data['train_forget'].remove_columns('is_forget')
        data['train_retain'] = data['train_retain'].remove_columns('is_forget')

        data['train'] = data['train'].map(lambda item, idx: {"index": idx}, with_indices=True)

    # Assign random labels to forget samples
    if unlearn_method.lower() in ['random_label']:
        random_labels = ['neutral', 'unknown']
        train_forget_flip = deepcopy(data['train_forget'])
        train_forget_flip = train_forget_flip.map(lambda item: {"label_text": random_labels[random.randint(0, len(random_labels)-1)]})
        data['train'] = train_forget_flip

        del train_forget_flip

    data = data.map(_preprocessing_sentiment, batched=False)
    data = data.remove_columns(col_to_delete)
    data.set_format("torch")

    print(data)

    return data, data_collator

def get_gradient_ascent_trainer():
    class GradientAscent(Trainer):
        def __init__(self, **kwargs):
            # remove deprecated tokenizer arg to avoid warnings
            kwargs.pop("tokenizer", None)
            super().__init__(**kwargs)
            self.name = "GA"

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Use model.training to detect whether we are in train() mode.
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # Only apply gradient ascent during training.
            if model.training:
                loss = -loss
            return (loss, outputs) if return_outputs else loss

    return GradientAscent


def get_random_label_trainer():
    class RandomLabel(Trainer):
        def __init__(self, **kwargs):
            kwargs.pop("tokenizer", None)
            super().__init__(**kwargs)
            self.name = "RL"

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # No special modification for random-label baseline; behave normally
            return (loss, outputs) if return_outputs else loss

    return RandomLabel


def get_gradient_ascent_plus_descent_trainer():
    class GradientAscentPlusDescent(Trainer):
        def __init__(self, **kwargs):
            kwargs.pop("tokenizer", None)
            super().__init__(**kwargs)
            self.name = "GA+GD"

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            # If not training or missing forget flags, fall back to normal loss
            if (not model.training) or ("is_forget" not in inputs) or ("index" not in inputs):
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                return (loss, outputs) if return_outputs else loss

            # Training path: separate forget vs retain samples and do GA on forget set
            is_forget_indicators = inputs.pop("is_forget")
            is_retain_indicator = 1 - is_forget_indicators
            sample_indices = inputs.pop("index")

            outputs = model(**inputs)
            logits = outputs.get("logits")
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Select splits (safe guards if empty)
            fgt_shift_logits = shift_logits[is_forget_indicators > 0]
            fgt_shift_labels = shift_labels[is_forget_indicators > 0]
            rtn_shift_logits = shift_logits[is_forget_indicators < 1]
            rtn_shift_labels = shift_labels[is_forget_indicators < 1]

            ce_loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            fgt_ce_loss = ce_loss_fct(
                fgt_shift_logits.view(-1, self.model.config.vocab_size),
                fgt_shift_labels.view(-1)
            ) if fgt_shift_logits.numel() > 0 else torch.tensor(0.0, device=logits.device)

            rtn_ce_loss = ce_loss_fct(
                rtn_shift_logits.view(-1, self.model.config.vocab_size),
                rtn_shift_labels.view(-1)
            ) if rtn_shift_logits.numel() > 0 else torch.tensor(0.0, device=logits.device)

            # gradient ascent on forget set (negate)
            fgt_ce_loss = -fgt_ce_loss
            loss = fgt_ce_loss + rtn_ce_loss

            return (loss, outputs) if return_outputs else loss

    return GradientAscentPlusDescent


def get_gradient_ascent_plus_kl_trainer():
    class GradientAscentPlusKL(Trainer):
        def __init__(self, original_logits, **kwargs):
            kwargs.pop("tokenizer", None)
            super().__init__(**kwargs)
            self.name = "GA+KL"
            self.original_logits = original_logits

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")

            # If not training or missing forget flags, fall back to normal loss
            if (not model.training) or ("is_forget" not in inputs) or ("index" not in inputs):
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                return (loss, outputs) if return_outputs else loss

            is_forget_indicators = inputs.pop("is_forget")
            is_retain_indicator = 1 - is_forget_indicators
            sample_indices = inputs.pop("index")

            outputs = model(**inputs)
            logits = outputs.get("logits")

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # CE for forget set
            fgt_shift_logits = shift_logits[is_forget_indicators > 0]
            fgt_shift_labels = shift_labels[is_forget_indicators > 0]
            ce_loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            fgt_ce_loss = ce_loss_fct(
                fgt_shift_logits.view(-1, self.model.config.vocab_size),
                fgt_shift_labels.view(-1)
            ) if fgt_shift_logits.numel() > 0 else torch.tensor(0.0, device=logits.device)

            # Build prev logits for the batch using same device as logits
            device = logits.device
            prev_logits_for_output = []
            for idx in sample_indices:
                # original_logits keyed by integer index -> convert with idx.item()
                prev = torch.tensor(self.original_logits[idx.item()], device=device)
                prev_logits_for_output.append(prev)

            if len(prev_logits_for_output) == 0:
                # no prev logits -> fallback to CE-only behavior
                fgt_ce_loss = -fgt_ce_loss
                return (fgt_ce_loss, outputs) if return_outputs else -fgt_ce_loss

            prev_logits = torch.stack(prev_logits_for_output, dim=0)

            # Keep only retain rows
            rtn_prev_logits = prev_logits[is_retain_indicator > 0]
            # Masked logits corresponding to output tokens
            label_mask = labels != -100
            rtn_logits = shift_logits[label_mask]  # flattened by mask
            # If there are retain rows, pick corresponding positions for them
            if rtn_prev_logits.numel() == 0 or rtn_logits.numel() == 0:
                rtn_kl_loss = torch.tensor(0.0, device=device)
            else:
                # Align shapes: rtn_logits should match rtn_prev_logits in rows
                # We'll compute KL per-token and average using batchmean
                # Ensure shapes: (num_tokens, vocab) for both
                # If rtn_prev_logits is shaped (N, V) and rtn_logits (M, V) but selection aligns,
                # we assume the mapping in self.original_logits produced the same token positions.
                kl_loss_fct = torch.nn.KLDivLoss(reduction='batchmean')
                try:
                    rtn_kl_loss = kl_loss_fct(
                        torch.log_softmax(rtn_logits, dim=-1),
                        torch.softmax(rtn_prev_logits, dim=-1)
                    )
                except Exception:
                    # fallback safe zero
                    rtn_kl_loss = torch.tensor(0.0, device=device)

            # gradient ascent on forget CE
            fgt_ce_loss = -fgt_ce_loss
            loss = fgt_ce_loss + rtn_kl_loss

            return (loss, outputs) if return_outputs else loss

    return GradientAscentPlusKL

def main(args):
    # Sync wandb
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models

    model_name = None
    if 'llama2_7b' in args.model_checkpoints.lower():
        model_name = 'llama-2-7b-hf'
    elif 'llama2_13b' in args.model_checkpoints.lower():
        model_name = 'llama-2-13b-hf'
    elif 'opt-1.3b' in args.model_checkpoints.lower():
        model_name = 'opt-1.3b'
    else:
        raise ValueError(f"Unsupported model checkpoint: {args.model_checkpoints}")

    os.environ["WANDB_PROJECT"] = f'baseline_{model_name}_{args.dataset.lower()}'

    data_path = get_data_path(args.dataset)

    model, tokenizer, lora_config = get_base_model(
        args.model_checkpoints,
        max_length=args.max_length
    )

    dataset, collator = get_unlearn_dataset_and_collator(
        data_path=data_path,
        tokenizer=tokenizer,
        unlearn_method=args.unlearn_method.lower(),
        col_to_delete = ['text', 'label', 'label_text'],
        max_length=args.max_length,
        truncation=True,
    )

    if args.set_pad_id:
        model.config.pad_token_id = model.config.eos_token_id

    # move model to GPU device
    # if model.device.type != 'cuda':
    #     model = model.to('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load logits if needed by trainer
    if args.unlearn_method in ['gradient_ascent_kl']:
        if args.logits_path is None:
            args.logits_path = f'saved_logits/{model_name}_{args.dataset.lower()}-{args.forget_size}.pkl'

        if not os.path.exists(args.logits_path):
            print('Saving original logits from base model')
            # original_logits = get_logits_from_base_model(original_model, collator, dataset)
            original_logits = get_logits_from_base_model(model, collator, dataset)
            torch.save(original_logits, "logits_from_"+args.model_checkpoints.split("/")[-2]+".pt")
            original_logits = torch.load("logits_from_"+args.model_checkpoints.split("/")[-2]+".pt")
            new_original_logits = {}
            for k in original_logits.keys():
                new_original_logits[k.item()] = original_logits[k].numpy()

            with open(args.logits_path, 'wb') as f:
                pickle.dump(new_original_logits, f, protocol=pickle.HIGHEST_PROTOCOL)

            print('Completed saving logits from base model')

        with open(args.logits_path, 'rb') as f:
            print('Loading original logits from base model')
            original_logits = pickle.load(f)

    if args.output_path is None:
        args.output_path = f'unlearn_checkpoints/{args.unlearn_method}_{model_name.lower()}_{args.dataset.lower()}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    os.makedirs(args.output_path, exist_ok=True)

    with open(os.path.join(args.output_path, 'arguments.txt'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write(f'{k}: {v}\n')
        print(f"Output path: {args.output_path}")


    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        eval_strategy="no",
        save_strategy="no",
        group_by_length=True,
        gradient_checkpointing=True,
        fp16=False,
        report_to=None,
        run_name=f'{args.unlearn_method.lower()}_lr={args.lr}',
        max_grad_norm=0.3,
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )

    if args.unlearn_method.lower() == "gradient_ascent":
        custom_loss = get_gradient_ascent_trainer()
        trainer = custom_loss(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train_forget'],
            eval_dataset={"train_retain": dataset['train_retain'], "train_forget": dataset['train_forget']},
            data_collator=collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics
        )

    elif args.unlearn_method.lower() == "random_label":
        custom_loss = get_random_label_trainer()
        trainer = custom_loss(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset={"train_retain": dataset['train_retain'],
                      "train_forget": dataset['train_forget']},
            data_collator=collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics
        )

    elif args.unlearn_method.lower() == "gradient_ascent_kl":
        custom_loss = get_gradient_ascent_plus_kl_trainer()
        trainer = custom_loss(
            model=model,
            original_logits=original_logits,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset={"train_retain": dataset['train_retain'],
                      "train_forget": dataset['train_forget']},
            data_collator=collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics
        )

    elif args.unlearn_method.lower() == "gradient_ascent_descent":
        custom_loss = get_gradient_ascent_plus_descent_trainer()
        trainer = custom_loss(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset={"train_retain": dataset['train_retain'],
                      "train_forget": dataset['train_forget']},
            data_collator=collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics
        )

    trainer.add_callback(CustomCallback(trainer))
    start = time.perf_counter()
    trainer.train()
    runtime = (time.perf_counter()-start)
    print(f"Total training time (s): {runtime:.1f}")
    print(f"Here are the final results:")
    metrics = trainer.evaluate(eval_dataset={"train_retain": dataset['train_retain'], "train_forget": dataset['train_forget']})
    print(metrics)
    with open(os.path.join(args.output_path, 'final_results.txt'), 'w') as f:
        f.write(f"Total training time (s): {runtime:.1f}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    args = get_args()
    main(args)
