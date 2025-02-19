# utils.monitoring.py
    # model_memory_need
    # local_memory_usage

from transformers import AutoModelForCausalLM
import torch
import psutil

import os
from pathlib import Path

def model_cache_state():
    """Lists the models stored in the Hugging Face cache directory with their respective sizes."""
    
    # Hugging Face cache directory
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    if not hf_cache_dir.exists():
        print("❌ Hugging Face cache directory not found.")
        return
    
    print(f"📁 Hugging Face Model Cache: {hf_cache_dir}\n")

    total_size = 0
    for model_dir in hf_cache_dir.glob("models--*"):
        size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / 1e9  # Convert to GB
        total_size += size
        print(f"📌 {size:.2f} GB\t| {model_dir.name}")

    print(f"\n🛑 Total Cache Size: {total_size:.2f} GB")


def model_memory_need(model=None, model_name=None):
    """Prints model total parameters, precision, and estimated memory need"""
    # Load model
    if model is None and model_name is None:
        return None
    if model is None :
        llm = AutoModelForCausalLM.from_pretrained(model_name)
    else :
        llm = model.llm.pipeline.model
        model_name = model.llm.model_id

    model_type = type(llm)

    
    # Compute total parameters
    total_params = sum(p.numel() for p in llm.parameters())

    # Check precision
    precision = next(llm.parameters()).dtype

    # Estimate memory requirement
    bits_per_param = {
        torch.float32: 4,  # FP32 = 4 bytes per parameter
        torch.float16: 2,  # FP16 = 2 bytes per parameter
        torch.bfloat16: 2, # BF16 = 2 bytes per parameter
        torch.int8: 1,     # INT8 = 1 byte per parameter
        torch.int4: 0.5    # 4-bit quantization
    }
    
    # Get memory per parameter in bytes
    memory_per_param = bits_per_param.get(precision, 4)  # Default FP32 if unknown
    estimated_memory_gb = (total_params * memory_per_param) / 1e9  # Convert bytes to GB

    # print(f"Model: {model_name}")
    # print(f"Type: {model_type}")
    # print(f"Total Parameters: {total_params / 1e9:.2f} Billion")
    # print(f"Precision: {precision}")
    # print(f"Estimated memory needed: {estimated_memory_gb:.2f} GB")

    return {
        'model_name' : model_name,
        'model_type' : model_type,
        'total_params_Billion' : round(total_params / 1e9, 2),
        'precision' : precision,
        'estimated_memory_gb' : round(estimated_memory_gb, 2),
    }


def pytorch_print_cache():
    current = torch.mps.current_allocated_memory() / 1e9
    driver = torch.mps.driver_allocated_memory() / 1e9
    available_memory = psutil.virtual_memory().available / 1e9 

    print("Allocated by MPS:", round(current, 2), "GB")
    print("Allocated by driver:", round(driver, 2), "GB")
    print("Available system memory:", round(available_memory, 2), "GB")



def pytorch_empty_cache():
    pytorch_print_cache()

    torch.mps.empty_cache()  # Clears PyTorch's unused memory
    torch.mps.synchronize()  # Ensures all pending ops are done

    print("Cache cleared.")
    pytorch_print_cache()



if __name__ == "__main__":
    model_memory_need(model_name="microsoft/Phi-3-mini-4k-instruct")
    pytorch_print_cache()