# sys_info.py
import platform
import torch
import os

def system_info():
    print("=== SYSTEM INFO ===")
    print("OS:", platform.system(), platform.release(), platform.version())
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA-compatible GPU detected.")

    try:
        import bitsandbytes as bnb
        print("bitsandbytes version:", bnb.__version__)
    except Exception as e:
        print("bitsandbytes not available:", e)

    print("Environment Variables:")
    for k in ["CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "TRANSFORMERS_VERBOSITY"]:
        print(f"{k} =", os.environ.get(k, "not set"))

if __name__ == "__main__":
    system_info()
