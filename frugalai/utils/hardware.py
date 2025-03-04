# utils.hardware.py

import torch

def get_device():
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    # Fall back to CPU
    else:
        device = torch.device("cpu")

    print('device :', device)
    return device

import platform
import subprocess
import psutil
import os

def get_mac_specs():
    """Get detailed specifications of a Mac machine."""
    specs = {}
    
    # Basic system info
    specs["system"] = platform.system()
    specs["version"] = platform.version()
    specs["machine"] = platform.machine()
    specs["processor"] = platform.processor()
    
    # More detailed CPU info
    try:
        cpu_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        specs["cpu_detailed"] = cpu_info
    except:
        specs["cpu_detailed"] = "Could not retrieve detailed CPU info"
    
    # Number of cores
    specs["physical_cores"] = psutil.cpu_count(logical=False)
    specs["logical_cores"] = psutil.cpu_count(logical=True)
    
    # Memory information
    mem = psutil.virtual_memory()
    specs["total_memory_gb"] = round(mem.total / (1024**3), 2)
    specs["available_memory_gb"] = round(mem.available / (1024**3), 2)
    
    # Disk information
    disk = psutil.disk_usage('/')
    specs["total_disk_gb"] = round(disk.total / (1024**3), 2)
    specs["free_disk_gb"] = round(disk.free / (1024**3), 2)
    
    # Check if MPS (Metal Performance Shaders) is available through PyTorch
    specs["has_pytorch"] = False
    specs["has_mps"] = False
    try:
        import torch
        specs["has_pytorch"] = True
        specs["pytorch_version"] = torch.__version__
        specs["has_mps"] = hasattr(torch, 'mps') and torch.backends.mps.is_available()
        if specs["has_mps"]:
            # Try to get MPS device memory if available
            try:
                mps_mem = subprocess.check_output(["system_profiler", "SPDisplaysDataType"]).decode()
                # This is a rough extraction and might need adjustment
                if "VRAM" in mps_mem:
                    vram_lines = [line for line in mps_mem.split('\n') if "VRAM" in line]
                    if vram_lines:
                        specs["gpu_memory"] = vram_lines[0].strip()
            except:
                specs["gpu_memory"] = "Could not retrieve GPU memory info"
    except ImportError:
        pass
    
    # Check for Apple Silicon
    specs["is_apple_silicon"] = platform.machine() == 'arm64'
    
    # Get model name
    try:
        model_name = subprocess.check_output(["sysctl", "-n", "hw.model"]).decode().strip()
        specs["model_name"] = model_name
    except:
        specs["model_name"] = "Could not retrieve model name"
    
    # Get OS version
    try:
        os_version = subprocess.check_output(["sw_vers", "-productVersion"]).decode().strip()
        specs["os_version"] = os_version
    except:
        specs["os_version"] = "Could not retrieve OS version"
    
    return specs

def print_mac_specs():
    """Print formatted Mac specifications."""
    specs = get_mac_specs()
    
    print("\n" + "="*60)
    print(f"{'MAC SPECIFICATIONS':^60}")
    print("="*60)
    
    print(f"\n{'System Information':^60}")
    print("-"*60)
    print(f"Model: {specs.get('model_name', 'Unknown')}")
    print(f"OS: macOS {specs.get('os_version', 'Unknown')}")
    print(f"Architecture: {specs.get('machine', 'Unknown')} ({'Apple Silicon' if specs.get('is_apple_silicon') else 'Intel'})")
    
    print(f"\n{'CPU Information':^60}")
    print("-"*60)
    print(f"Processor: {specs.get('cpu_detailed', 'Unknown')}")
    print(f"Physical cores: {specs.get('physical_cores', 'Unknown')}")
    print(f"Logical cores: {specs.get('logical_cores', 'Unknown')}")
    
    print(f"\n{'Memory Information':^60}")
    print("-"*60)
    print(f"Total RAM: {specs.get('total_memory_gb', 'Unknown')} GB")
    print(f"Available RAM: {specs.get('available_memory_gb', 'Unknown')} GB")
    
    print(f"\n{'Storage Information':^60}")
    print("-"*60)
    print(f"Total Disk: {specs.get('total_disk_gb', 'Unknown')} GB")
    print(f"Free Disk: {specs.get('free_disk_gb', 'Unknown')} GB")
    
    print(f"\n{'GPU/ML Information':^60}")
    print("-"*60)
    if specs.get("has_pytorch"):
        print(f"PyTorch Version: {specs.get('pytorch_version', 'Unknown')}")
        print(f"MPS Available: {'Yes' if specs.get('has_mps') else 'No'}")
        if specs.get('has_mps') and 'gpu_memory' in specs:
            print(f"GPU Memory: {specs.get('gpu_memory', 'Unknown')}")
    else:
        print("PyTorch: Not installed")
    
    print("\n" + "="*60)
    print(f"{'FINE-TUNING RECOMMENDATION':^60}")
    print("="*60)
    
    # Provide recommendations based on specs
    ram = specs.get('total_memory_gb', 0)
    is_apple_silicon = specs.get('is_apple_silicon', False)
    has_mps = specs.get('has_mps', False)
    
    if ram >= 32 and is_apple_silicon and has_mps:
        print("\nYour Mac appears suitable for fine-tuning Phi-3-mini with optimizations.")
        print("Recommendations:")
        print("- Use parameter-efficient fine-tuning methods like LoRA/QLoRA")
        print("- Enable gradient checkpointing")
        print("- Use small batch sizes (1-2) with gradient accumulation")
        print("- Consider using 4-bit quantization (QLoRA)")
    elif ram >= 16 and is_apple_silicon:
        print("\nYour Mac may be able to fine-tune Phi-3-mini with significant optimizations.")
        print("Recommendations:")
        print("- Use QLoRA with 4-bit quantization")
        print("- Set a very small batch size (1) with gradient accumulation")
        print("- Enable gradient checkpointing")
        print("- Consider reducing context length (e.g., 512 instead of 4096)")
    else:
        print("\nFine-tuning Phi-3-mini directly on this Mac may be challenging.")
        print("Recommendations:")
        print("- Consider using smaller models (e.g., Phi-2 or DistilBERT)")
        print("- Look into cloud-based fine-tuning options")
        print("- Try adapter-based methods with CPU training")
    
    print("\n" + "="*60)



if __name__ == "__main__":
    device = get_device()
    print('device:', device)
