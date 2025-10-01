# python3 analyze_max_length.py --dataset sst2 --model_name meta-llama/Llama-2-7b-hf --split train --text_column sentence
# analyze_max_length.py
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import numpy as np

def main(args):
    # Load dataset
    dataset = load_dataset(args.dataset)
    split = args.split
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")

    texts = dataset[split][args.text_column]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Compute token lengths
    token_lengths = [len(tokenizer(text)['input_ids']) for text in texts]
    token_lengths = np.array(token_lengths)

    # Statistics
    max_len = int(token_lengths.max())
    mean_len = float(token_lengths.mean())
    p90 = int(np.percentile(token_lengths, 90))
    p95 = int(np.percentile(token_lengths, 95))
    p99 = int(np.percentile(token_lengths, 99))

    print(f"Dataset: {args.dataset} | Split: {split}")
    print(f"Text column: {args.text_column}")
    print(f"Max token length: {max_len}")
    print(f"Mean token length: {mean_len:.1f}")
    print(f"90th percentile: {p90}")
    print(f"95th percentile: {p95}")
    print(f"99th percentile: {p99}")
    print("\nRecommended max_length (95th percentile):", p95)
    print("Optional faster max_length (90th percentile):", p90)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze dataset token lengths to determine optimal max_length for LLM training")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name or local path")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model tokenizer")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to analyze (default: train)")
    parser.add_argument("--text_column", type=str, default="sentence", help="Name of the text column in dataset")
    args = parser.parse_args()
    main(args)
