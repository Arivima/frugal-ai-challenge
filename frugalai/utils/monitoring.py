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


def get_model_nb_params(model):
    return sum(p.numel() for p in model.parameters())

def get_model_precision(model):
    return next(model.parameters()).dtype


def langchain_memory_need(model=None, model_name=None):
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

    return {
        'model_name' : model_name,
        'model_type' : model_type,
        'total_params_Billion' : round(total_params / 1e9, 2),
        'precision' : precision,
        'estimated_memory_gb' : round(estimated_memory_gb, 2),
    }

def model_memory_need(model=None):

    model_type = type(model)

    # Compute total parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Check precision
    precision = next(model.parameters()).dtype

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
    estimated_memory_gb = (total_params * memory_per_param) / (1024 ** 3)

    return {
        'model_type' : model_type,
        'total_params_Billion' : round(total_params / (1024 ** 3), 2),
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

# def print_memory_status():
#     if hasattr(torch, 'mps') and torch.mps.is_available():
#         # allocation : Returns total GPU memory allocated by Metal driver for the process in bytes.
#         driver = torch.mps.driver_allocated_memory() / (1024 ** 3)
#         # actual use : Returns the current GPU memory occupied by tensors in bytes.
#         current = torch.mps.current_allocated_memory() / (1024 ** 3)
#         # Returns recommended max Working set size for GPU memory in bytes.
#         reco = torch.mps.recommended_max_memory() / (1024 ** 3)
#         # memory requested by pytorch to handle interbnal buffer, cache, memory fragmentation, allocation for future use, 
#         # kernel workspaces and temp buffers needed during computation
#         overhead = driver - current
#         available_memory = psutil.virtual_memory().available / (1024 ** 3)

#         print(f'torch.mps.current_allocated_memory:\t{round(current, 2)} GB')
#         print(f'torch.mps.driver_allocated_memory:\t{round(driver, 2)} GB')
#         print(f'torch.mps.recommended_max_memory:\t{round(reco, 2)} GB')
#         print(f'overhead | used by pytorch operations:\t{round(overhead, 2)} GB')
#         print(f'recommended available memory:\t\t{round(reco - driver, 2)} GB')
#         print(f'actual available memory:\t\t{round(available_memory, 2)} GB')
#     else:
#         print("MPS (Metal Performance Shaders) is not available on this system.")


def pytorch_empty_cache():
    pytorch_print_cache()

    torch.mps.empty_cache()  # Clears PyTorch's unused memory
    torch.mps.synchronize()  # Ensures all pending ops are done

    print("Cache cleared.")
    pytorch_print_cache()




import torch
import psutil
import os
import gc
from typing import Dict, Any, Optional, Tuple


def get_memory_status() -> Dict[str, Any]:
    """
    Gathers comprehensive memory information for both CPU and MPS devices.
    
    Returns:
        Dict containing memory metrics for available devices
    """
    memory_info = {}
    
    # CPU memory information
    memory_info["cpu"] = {
        "total": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "available": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        "used": round(psutil.virtual_memory().used / (1024 ** 3), 2),
        "percent": psutil.virtual_memory().percent,
        "process_used": round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3), 2)
    }
    
    # MPS memory information (if available)
    if hasattr(torch, 'mps') and torch.mps.is_available():
        # Returns the current GPU memory occupied by tensors in bytes
        current = torch.mps.current_allocated_memory() / (1024 ** 3)
        # Returns total GPU memory allocated by Metal driver for the process in bytes
        driver = torch.mps.driver_allocated_memory() / (1024 ** 3)
        # Returns recommended max Working set size for GPU memory in bytes
        reco = torch.mps.recommended_max_memory() / (1024 ** 3)
        
        memory_info["mps"] = {
            "tensor_allocated": round(current, 2),
            "driver_allocated": round(driver, 2),
            "recommended_max": round(reco, 2),
            "overhead": round(driver - current, 2),
            "available_in_pool": round(reco - driver, 2)
        }
    
    # Count tensors by device
    memory_info["tensor_counts"] = count_tensors_by_device()
    
    return memory_info


def count_tensors_by_device() -> Dict[str, int]:
    """
    Counts the number of tensors allocated on each device.
    
    Returns:
        Dict with device names as keys and tensor counts as values
    """
    tensor_counts = {"cpu": 0, "mps": 0, "other": 0}
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                device_str = str(obj.device)
                if "cpu" in device_str:
                    tensor_counts["cpu"] += 1
                elif "mps" in device_str:
                    tensor_counts["mps"] += 1
                else:
                    tensor_counts["other"] += 1
        except:
            pass
    
    return tensor_counts


def print_memory_status_across_devices():
    """
    Prints a formatted report of memory usage across CPU and MPS devices.
    """
    memory_info = get_memory_status()
    
    print("\n" + "="*60)
    print(f"{'MEMORY USAGE REPORT':^60}")
    print("="*60)
    
    # CPU memory section
    print("\n" + "-"*20 + " CPU MEMORY " + "-"*20)
    cpu_info = memory_info["cpu"]
    print(f"Total System Memory:         {cpu_info['total']:.2f} GB")
    print(f"Available System Memory:     {cpu_info['available']:.2f} GB")
    print(f"Used System Memory:          {cpu_info['used']:.2f} GB ({cpu_info['percent']}%)")
    print(f"Current Process Memory:      {cpu_info['process_used']:.2f} GB")
    
    # MPS memory section (if available)
    if "mps" in memory_info:
        print("\n" + "-"*20 + " MPS MEMORY " + "-"*20)
        mps_info = memory_info["mps"]
        print(f"Tensor Allocated Memory:     {mps_info['tensor_allocated']:.2f} GB")
        print(f"Overhead (PyTorch Internal): {mps_info['overhead']:.2f} GB")
        print(f"Driver Allocated Memory:     {mps_info['driver_allocated']:.2f} GB")
        print(f"Recommended Maximum Memory:  {mps_info['recommended_max']:.2f} GB")
        print(f"Available in Memory Pool:    {mps_info['available_in_pool']:.2f} GB")
    
    # Tensor count by device
    print("\n" + "-"*20 + " TENSOR COUNTS " + "-"*20)
    tensor_counts = memory_info["tensor_counts"]
    print(f"CPU Tensors:                 {tensor_counts['cpu']}")
    print(f"MPS Tensors:                 {tensor_counts['mps']}")
    if tensor_counts["other"] > 0:
        print(f"Other Device Tensors:        {tensor_counts['other']}")
    
    print("\n" + "="*60 + "\n")






import torch
import math
from typing import Any, Dict, Optional, Union
import sys

def estimate_ft_memory_requirements(
    model, 
    tokenizer=None, 
    training_args=None, 
    dataset_sample=None,
    verbose=True
):
    """
    Estimates memory requirements for fine-tuning a model by extracting information
    directly from the provided objects.
    
    Args:
        model: The model to be fine-tuned (PyTorch or Hugging Face model)
        tokenizer: Optional tokenizer (to determine sequence length if not specified)
        training_args: Optional training arguments (for batch size, precision, etc.)
        dataset_sample: Optional sample from dataset (to determine sequence length)
        verbose: Whether to print detailed information
    
    Returns:
        Dict containing estimated memory requirements in GB
    """
    memory_estimates = {}
    
    # Get model architecture details
    model_info = _get_model_info(model)
    memory_estimates['model_info'] = model_info
    
    # Extract training parameters
    training_params = _extract_training_params(model, tokenizer, training_args, dataset_sample)
    memory_estimates['training_params'] = training_params
    
    # Calculate memory for model parameters
    param_memory = _calculate_parameter_memory(model_info, training_params)
    memory_estimates['parameters'] = param_memory
    
    # Calculate optimizer states memory
    optimizer_memory = _calculate_optimizer_memory(model_info, training_params)
    memory_estimates['optimizer'] = optimizer_memory
    
    # Calculate gradient memory
    gradient_memory = _calculate_gradient_memory(model_info, training_params)
    memory_estimates['gradients'] = gradient_memory
    
    # Calculate activation memory
    activation_memory = _calculate_activation_memory(model_info, training_params)
    memory_estimates['activations'] = activation_memory
    
    # Calculate total memory with safety margin
    total_memory = (
        param_memory['total_gb'] + 
        optimizer_memory['total_gb'] + 
        gradient_memory['total_gb'] + 
        activation_memory['total_gb']
    )
    
    # Add safety margin (10%)
    memory_estimates['total'] = {
        'raw_gb': total_memory,
        'with_margin_gb': total_memory * 1.1
    }
    
    if verbose:
        _print_memory_report(memory_estimates)
    
    return memory_estimates


def _get_model_info(model):
    """Extract model architecture information"""
    model_info = {
        'param_count': sum(p.numel() for p in model.parameters()),
        'trainable_param_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'dtype': next(model.parameters()).dtype,
        'device': next(model.parameters()).device,
        'model_class': model.__class__.__name__
    }
    
    # Try to detect if it's a transformer model and get relevant attributes
    if hasattr(model, 'config'):
        config = model.config
        model_info['has_config'] = True
        
        # Extract common transformer attributes if available
        for attr in ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 
                    'intermediate_size', 'max_position_embeddings']:
            if hasattr(config, attr):
                model_info[attr] = getattr(config, attr)
    
    # Bytes per parameter based on dtype
    dtype_size_map = {
        torch.float32: 4,
        torch.float: 4,
        torch.float16: 2,
        torch.half: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.int32: 4,
        torch.long: 8,
        torch.int64: 8
    }
    
    model_info['bytes_per_param'] = dtype_size_map.get(model_info['dtype'], 4)
    
    return model_info


def _extract_training_params(model, tokenizer, training_args, dataset_sample):
    """Extract training parameters from the provided objects"""
    training_params = {
        'precision': 'fp32',  # Default
        'batch_size': 8,      # Default
        'seq_length': 512,    # Default
        'gradient_accumulation_steps': 1,  # Default
        'optimizer_type': 'adamw',  # Default
    }
    
    # Update precision based on model dtype
    model_dtype = next(model.parameters()).dtype
    if model_dtype == torch.float16 or model_dtype == torch.half:
        training_params['precision'] = 'fp16'
    elif hasattr(torch, 'bfloat16') and model_dtype == torch.bfloat16:
        training_params['precision'] = 'bf16'
        
    # Extract sequence length from tokenizer if available
    if tokenizer is not None:
        if hasattr(tokenizer, 'model_max_length'):
            training_params['seq_length'] = tokenizer.model_max_length
    
    # Extract batch size and other training params from training_args if available
    if training_args is not None:
        # Handle both Transformers TrainingArguments and custom args objects
        param_mapping = {
            'batch_size': ['per_device_train_batch_size', 'batch_size', 'train_batch_size'],
            'gradient_accumulation_steps': ['gradient_accumulation_steps'],
            'fp16': ['fp16', 'use_fp16', 'mixed_precision'],
            'bf16': ['bf16', 'use_bf16'],
            'optimizer_type': ['optimizer_type', 'optim', 'optimizer']
        }
        
        for param, possible_names in param_mapping.items():
            for name in possible_names:
                if hasattr(training_args, name):
                    value = getattr(training_args, name)
                    if param == 'fp16' and value:
                        training_params['precision'] = 'fp16'
                    elif param == 'bf16' and value:
                        training_params['precision'] = 'bf16'
                    elif param in ['batch_size', 'gradient_accumulation_steps']:
                        training_params[param] = value
                    elif param == 'optimizer_type':
                        training_params[param] = str(value).lower()
    
    # Try to estimate sequence length from dataset sample if available
    if dataset_sample is not None and tokenizer is not None:
        try:
            if hasattr(dataset_sample, 'items') or isinstance(dataset_sample, (list, tuple)):
                # Take first item if it's a collection
                sample = dataset_sample[0] if isinstance(dataset_sample, (list, tuple)) else next(iter(dataset_sample.items()))
                
                # Check if the sample has text-like fields
                for key in ['text', 'input_text', 'sentence', 'content', 'input']:
                    if key in sample:
                        tokenized = tokenizer(sample[key], truncation=False, padding=False)
                        if 'input_ids' in tokenized and len(tokenized['input_ids']) > 0:
                            training_params['seq_length'] = len(tokenized['input_ids'])
                            break
        except:
            # If we can't extract from dataset, keep the default value
            pass
    
    # Default optimizer bytes per parameter based on optimizer type
    optimizer_memory_map = {
        'adamw': 12,  # 8 bytes for momentum, 4 for variance
        'adam': 12,   # Same as AdamW
        'sgd': 4      # With momentum
    }
    
    # Add optimizer bytes per parameter info
    training_params['optimizer_bytes_per_param'] = optimizer_memory_map.get(
        training_params['optimizer_type'].lower(), 8)  # Default to 8 if unknown
    
    # Get effective batch size
    training_params['effective_batch_size'] = (
        training_params['batch_size'] * training_params['gradient_accumulation_steps'])
    
    return training_params


def _calculate_parameter_memory(model_info, training_params):
    """Calculate memory required for model parameters"""
    # Base parameter memory in bytes
    param_bytes = model_info['param_count'] * model_info['bytes_per_param']
    
    # Convert to different units
    param_kb = param_bytes / 1024
    param_mb = param_kb / 1024
    param_gb = param_mb / 1024
    
    return {
        'bytes': param_bytes,
        'kb': param_kb,
        'mb': param_mb,
        'gb': param_gb,
        'total_gb': param_gb
    }


def _calculate_optimizer_memory(model_info, training_params):
    """Calculate memory required for optimizer states"""
    trainable_param_count = model_info['trainable_param_count']
    optimizer_bytes_per_param = training_params['optimizer_bytes_per_param']
    
    # Calculate total bytes for optimizer states
    optimizer_bytes = trainable_param_count * optimizer_bytes_per_param
    
    # Convert to different units
    optimizer_kb = optimizer_bytes / 1024
    optimizer_mb = optimizer_kb / 1024
    optimizer_gb = optimizer_mb / 1024
    
    return {
        'bytes': optimizer_bytes,
        'kb': optimizer_kb,
        'mb': optimizer_mb,
        'gb': optimizer_gb,
        'total_gb': optimizer_gb
    }


def _calculate_gradient_memory(model_info, training_params):
    """Calculate memory required for gradients"""
    trainable_param_count = model_info['trainable_param_count']
    
    # Gradients are usually stored in fp32 regardless of model precision
    gradient_bytes = trainable_param_count * 4
    
    # Convert to different units
    gradient_kb = gradient_bytes / 1024
    gradient_mb = gradient_kb / 1024
    gradient_gb = gradient_mb / 1024
    
    return {
        'bytes': gradient_bytes,
        'kb': gradient_kb,
        'mb': gradient_mb,
        'gb': gradient_gb,
        'total_gb': gradient_gb
    }


def _calculate_activation_memory(model_info, training_params):
    """
    Estimate memory required for activations during training
    This is a rough estimate and can vary based on model architecture
    """
    # Default activation memory for transformer-like models
    activation_memory_gb = 0
    
    # Check if we have transformer config attributes
    if all(key in model_info for key in ['hidden_size', 'num_hidden_layers']):
        batch_size = training_params['batch_size']
        seq_length = training_params['seq_length']
        hidden_size = model_info['hidden_size']
        num_layers = model_info['num_hidden_layers']
        
        # Bytes per activation based on training precision
        bytes_per_activation = 2 if training_params['precision'] in ['fp16', 'bf16'] else 4
        
        # Rough formula for transformer activations
        # Each position in each layer needs storage proportional to hidden_size
        # Multiply by a constant factor to account for various model-specific factors
        constant_factor = 4  # Empirical constant for transformers
        
        activation_bytes = (
            batch_size * seq_length * hidden_size * num_layers * 
            bytes_per_activation * constant_factor
        )
        
        # Convert to GB
        activation_gb = activation_bytes / (1024**3)
        
        # Also estimate peak activation memory which can be higher due to attention patterns
        peak_activation_gb = activation_gb * 1.5  # 50% more for peak usage
        
        return {
            'bytes': activation_bytes,
            'gb': activation_gb,
            'peak_gb': peak_activation_gb,
            'total_gb': peak_activation_gb  # Use peak for total estimate
        }
    else:
        # Fallback for non-transformer models - rough empirical estimation
        # Based on parameter count and batch size
        param_gb = model_info['param_count'] * 4 / (1024**3)
        batch_size = training_params['batch_size']
        
        # Rough approximation based on empirical observations
        activation_gb = param_gb * 0.1 * batch_size
        peak_activation_gb = activation_gb * 1.5
        
        return {
            'bytes': activation_gb * (1024**3),
            'gb': activation_gb,
            'peak_gb': peak_activation_gb,
            'total_gb': peak_activation_gb
        }


def _print_memory_report(memory_estimates):
    """Print a detailed memory report"""
    model_info = memory_estimates['model_info']
    training_params = memory_estimates['training_params']
    
    print("\n" + "="*70)
    print(f"{'FINE-TUNING MEMORY ESTIMATION REPORT':^70}")
    print("="*70)
    
    # Model information
    print("\n" + "-"*25 + " MODEL INFORMATION " + "-"*25)
    print(f"Model class:                 {model_info['model_class']}")
    print(f"Total parameters:            {model_info['param_count']:,}")
    print(f"Trainable parameters:        {model_info['trainable_param_count']:,}")
    print(f"Model precision:             {model_info['dtype']}")
    
    if 'hidden_size' in model_info:
        print(f"Hidden size:                 {model_info['hidden_size']}")
        print(f"Number of layers:            {model_info['num_hidden_layers']}")
    
    # Training parameters
    print("\n" + "-"*25 + " TRAINING PARAMETERS " + "-"*25)
    print(f"Training precision:          {training_params['precision']}")
    print(f"Batch size:                  {training_params['batch_size']}")
    print(f"Sequence length:             {training_params['seq_length']}")
    print(f"Gradient accumulation steps: {training_params['gradient_accumulation_steps']}")
    print(f"Effective batch size:        {training_params['effective_batch_size']}")
    print(f"Optimizer:                   {training_params['optimizer_type']}")
    
    # Memory breakdown
    print("\n" + "-"*25 + " MEMORY BREAKDOWN (GB) " + "-"*25)
    print(f"Model parameters:            {memory_estimates['parameters']['gb']:.2f} GB")
    print(f"Optimizer states:            {memory_estimates['optimizer']['gb']:.2f} GB")
    print(f"Gradients:                   {memory_estimates['gradients']['gb']:.2f} GB")
    print(f"Peak activations:            {memory_estimates['activations']['peak_gb']:.2f} GB")
    
    # Total estimate
    print("\n" + "-"*25 + " TOTAL MEMORY ESTIMATE " + "-"*25)
    print(f"Raw estimate:                {memory_estimates['total']['raw_gb']:.2f} GB")
    print(f"With 10% safety margin:      {memory_estimates['total']['with_margin_gb']:.2f} GB")
    
    # Memory optimization suggestions
    print("\n" + "-"*25 + " OPTIMIZATION SUGGESTIONS " + "-"*25)
    
    # Calculate how much memory would be saved with different optimizations
    current_total = memory_estimates['total']['raw_gb']
    
    # 1. Mixed precision if not already using it
    if training_params['precision'] == 'fp32':
        mixed_precision_savings = (
            memory_estimates['parameters']['gb'] * 0.5 + 
            memory_estimates['activations']['peak_gb'] * 0.5
        )
        print(f"Using mixed precision:       Save ~{mixed_precision_savings:.2f} GB")
    
    # 2. Gradient accumulation
    current_batch = training_params['batch_size']
    if current_batch > 1:
        halved_batch = max(1, current_batch // 2)
        activation_ratio = halved_batch / current_batch
        activation_savings = memory_estimates['activations']['peak_gb'] * (1 - activation_ratio)
        print(f"Reducing batch size to {halved_batch}: Save ~{activation_savings:.2f} GB")
    
    # 3. Using 8-bit optimizer
    if training_params['optimizer_type'].lower() in ['adam', 'adamw']:
        optimizer_savings = memory_estimates['optimizer']['gb'] * 0.5
        print(f"Using 8-bit optimizer:       Save ~{optimizer_savings:.2f} GB")
    
    # 4. Gradient checkpointing
    if 'hidden_size' in model_info:
        checkpoint_savings = memory_estimates['activations']['peak_gb'] * 0.7
        print(f"Gradient checkpointing:      Save ~{checkpoint_savings:.2f} GB")
    
    print("\n" + "="*70)






if __name__ == "__main__":
    model_memory_need(model_name="microsoft/Phi-3-mini-4k-instruct")
    pytorch_print_cache()