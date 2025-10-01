# utils.py
import os
from copy import deepcopy
import evaluate
import torch
from transformers import TrainingArguments, TrainerState, TrainerControl, TrainerCallback
from transformers import PreTrainedTokenizerBase

# Try to use TRL collator if present
try:
    from trl import DataCollatorForCompletionOnly as _TRL_DATA_COLLATOR
    _HAS_TRL_COLLATOR = True
except Exception:
    _TRL_DATA_COLLATOR = None
    _HAS_TRL_COLLATOR = False


class CompletionOnlyCollator:
    """
    Collator that handles either raw-text examples or already-tokenized inputs.
    Masks prefix tokens (before response template) with -100 so only completion contributes to loss.
    """
    def __init__(self, response_template_ids, tokenizer: PreTrainedTokenizerBase, max_length=128):
        self.response_template_ids = list(response_template_ids) if response_template_ids is not None else []
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def _find_subsequence(haystack, needle):
        if not needle:
            return -1
        n, m = len(haystack), len(needle)
        for i in range(n - m + 1):
            if haystack[i:i + m] == needle:
                return i
        return -1

    def __call__(self, features):
        if len(features) == 0:
            return {}

        first = features[0]

        # Case A: already tokenized inputs (contain 'input_ids')
        if isinstance(first, dict) and ("input_ids" in first):
            seqs = []
            for ex in features:
                ids = ex["input_ids"]
                if torch.is_tensor(ids):
                    ids = ids.tolist()
                seqs.append(ids)

            encodings = [{"input_ids": s} for s in seqs]
            toks = self.tokenizer.pad(
                encodings,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
            )
            input_ids = toks["input_ids"]
            attention_mask = toks.get("attention_mask", None)
        else:
            # Case B: raw text examples
            texts = [f["text"] if isinstance(f, dict) and "text" in f else f for f in features]
            toks = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = toks["input_ids"]
            attention_mask = toks.get("attention_mask", None)

        labels = input_ids.clone().fill_(-100)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        for i, ids in enumerate(input_ids.tolist()):
            pos = self._find_subsequence(ids, self.response_template_ids)
            if pos == -1:
                try:
                    last_nonpad = max(j for j, tok in enumerate(ids) if tok != pad_id)
                    labels[i, last_nonpad] = input_ids[i, last_nonpad]
                except Exception:
                    pass
            else:
                start = pos + len(self.response_template_ids)
                if start < input_ids.shape[1]:
                    labels[i, start:] = input_ids[i, start:]

        out = {"input_ids": input_ids, "labels": labels}
        if attention_mask is not None:
            out["attention_mask"] = attention_mask
        return out


# compatibility alias (use TRL collator if available)
DataCollatorForCompletionOnlyLM = _TRL_DATA_COLLATOR or CompletionOnlyCollator


def get_data_path(dataset):
    dataset = dataset.lower()
    if dataset == "sst2":
        return "datasets/Unlearning_SST2"
    elif dataset == "yelp":
        return "datasets/Unlearning_Yelp_Polarity"
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")
    prec_metric = evaluate.load("precision")
    rec_metric = evaluate.load("recall")

    logits, labels = eval_pred
    predictions = logits[:, :-1]
    labels = labels[:, 1:]
    mask = labels != -100

    preds_flat = []
    refs_flat = []
    for i in range(predictions.shape[0]):
        idxs = mask[i].nonzero(as_tuple=False).squeeze(-1)
        if idxs.numel() == 0:
            continue
        preds_flat.extend(predictions[i][idxs].tolist())
        refs_flat.extend(labels[i][idxs].tolist())

    preds_flat = [int(x) for x in preds_flat]
    refs_flat = [int(x) for x in refs_flat]

    if len(preds_flat) == 0:
        return {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

    return {
        "f1": f1_metric.compute(predictions=preds_flat, references=refs_flat, average="weighted")["f1"],
        "accuracy": acc_metric.compute(predictions=preds_flat, references=refs_flat)["accuracy"],
        "precision": prec_metric.compute(predictions=preds_flat, references=refs_flat, average="micro")["precision"],
        "recall": rec_metric.compute(predictions=pred_flat, references=refs_flat, average="micro")["recall"] if len(refs_flat) > 0 else 0.0,
    }


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        try:
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
        except Exception:
            pass
        return deepcopy(control)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        try:
            if isinstance(self._trainer.eval_dataset, dict):
                for key, ds in self._trainer.eval_dataset.items():
                    self._trainer.evaluate(eval_dataset=ds, metric_key_prefix=f"eval_{key}")
            else:
                self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset, metric_key_prefix="eval")
        except Exception:
            pass
        return deepcopy(control)
