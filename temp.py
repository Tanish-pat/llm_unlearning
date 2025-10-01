# load_model_info.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def main(checkpoint_dir):
    """
    Loads a trained QLoRA model and tokenizer from `checkpoint_dir`
    and prints relevant info for sharing or inspection.
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    print("Tokenizer loaded successfully!")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Pad token: {tokenizer.pad_token}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True
    )
    print("Model loaded successfully!")
    print(f"Total parameters: {model.num_parameters():,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Non-trainable parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

    # Example: share config
    config_dict = model.config.to_dict()
    print("\nModel config keys:", list(config_dict.keys()))

    return model, tokenizer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load a QLoRA model + tokenizer and show info")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the trained QLoRA model checkpoint directory"
    )
    args = parser.parse_args()
    main(args.checkpoint_dir)
