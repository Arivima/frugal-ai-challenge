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


import platform
import subprocess
import psutil
import os
import sys
import cpuinfo
import shutil
import re
import socket
from datetime import datetime

def get_device_ml_info():
    """Get ML-related device information (GPU/TPU)."""
    ml_info = {
        "has_pytorch": False,
        "pytorch_version": None,
        "has_cuda": False,
        "cuda_version": None,
        "has_mps": False,
        "gpus": [],
        "has_tensorflow": False,
        "tensorflow_version": None,
    }
    
    # Check for PyTorch
    try:
        import torch
        ml_info["has_pytorch"] = True
        ml_info["pytorch_version"] = torch.__version__
        
        # Check for CUDA
        ml_info["has_cuda"] = torch.cuda.is_available()
        if ml_info["has_cuda"]:
            ml_info["cuda_version"] = torch.version.cuda
            
            # Get NVIDIA GPU details
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_info = {
                    "name": torch.cuda.get_device_name(i),
                    "index": i,
                    "memory_total": torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
                    "memory_allocated": torch.cuda.memory_allocated(i) / (1024**3),  # GB
                }
                ml_info["gpus"].append(gpu_info)
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            ml_info["has_mps"] = True
    except ImportError:
        pass
    
    # Check for TensorFlow
    try:
        import tensorflow as tf
        ml_info["has_tensorflow"] = True
        ml_info["tensorflow_version"] = tf.__version__
        
        # Add TF-specific GPU info if not already collected via PyTorch
        if not ml_info["gpus"] and hasattr(tf, "config") and hasattr(tf.config, "list_physical_devices"):
            gpus = tf.config.list_physical_devices('GPU')
            for i, gpu in enumerate(gpus):
                gpu_info = {
                    "name": gpu.name,
                    "index": i,
                }
                ml_info["gpus"].append(gpu_info)
    except ImportError:
        pass
    
    return ml_info

def get_linux_gpu_info():
    """Get GPU information on Linux systems."""
    gpu_info = []
    
    # Try nvidia-smi for NVIDIA GPUs
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu", 
                                          "--format=csv,noheader,nounits"], 
                                          universal_newlines=True)
        for i, line in enumerate(result.strip().split('\n')):
            values = line.split(', ')
            if len(values) >= 4:
                gpu = {
                    "name": values[0],
                    "index": i,
                    "memory_total": float(values[1]) / 1024,  # GB
                    "memory_used": float(values[2]) / 1024,  # GB
                    "temperature": float(values[3]),
                }
                gpu_info.append(gpu)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Try lspci for other GPUs if no NVIDIA GPUs found
    if not gpu_info:
        try:
            result = subprocess.check_output(["lspci", "-v"], universal_newlines=True)
            gpu_lines = [line for line in result.split('\n') if re.search('VGA|3D|Display|Graphics', line)]
            
            for i, line in enumerate(gpu_lines):
                gpu = {
                    "name": line.split(':')[-1].strip(),
                    "index": i,
                }
                gpu_info.append(gpu)
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    return gpu_info

def get_windows_gpu_info():
    """Get GPU information on Windows systems."""
    gpu_info = []
    
    try:
        # Use Windows Management Instrumentation (WMI) to query GPU info
        import wmi
        w = wmi.WMI()
        
        for i, gpu in enumerate(w.Win32_VideoController()):
            gpu_data = {
                "name": gpu.Name,
                "index": i,
                "driver_version": gpu.DriverVersion,
                "memory_total": getattr(gpu, "AdapterRAM", 0) / (1024**3) if hasattr(gpu, "AdapterRAM") else None,
            }
            gpu_info.append(gpu_data)
    except ImportError:
        # Fallback to using subprocess and Windows command-line tools
        try:
            result = subprocess.check_output(["wmic", "path", "win32_VideoController", "get", 
                                             "Name,AdapterRAM,DriverVersion", "/format:csv"], 
                                             universal_newlines=True)
            
            lines = result.strip().split('\n')
            if len(lines) > 1:  # First line is header
                headers = lines[0].split(',')
                name_idx = headers.index("Name") if "Name" in headers else -1
                ram_idx = headers.index("AdapterRAM") if "AdapterRAM" in headers else -1
                
                for i, line in enumerate(lines[1:]):
                    values = line.split(',')
                    if len(values) > max(name_idx, ram_idx) and name_idx >= 0:
                        gpu = {
                            "name": values[name_idx].strip(),
                            "index": i,
                        }
                        
                        if ram_idx >= 0 and values[ram_idx].strip():
                            try:
                                gpu["memory_total"] = float(values[ram_idx]) / (1024**3)  # GB
                            except (ValueError, TypeError):
                                pass
                        
                        gpu_info.append(gpu)
        except (subprocess.SubprocessError, FileNotFoundError, ValueError):
            pass
    
    return gpu_info

def get_mac_gpu_info():
    """Get GPU information on macOS systems."""
    gpu_info = []
    
    try:
        # Use system_profiler to get GPU information
        result = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], 
                                         universal_newlines=True)
        
        # Parse system_profiler output (this is complex since it's not structured)
        sections = result.split("\n\n")
        current_gpu = None
        
        for i, section in enumerate(sections):
            if "Chipset Model:" in section:
                lines = section.strip().split('\n')
                gpu_data = {"index": i}
                
                for line in lines:
                    line = line.strip()
                    
                    if "Chipset Model:" in line:
                        gpu_data["name"] = line.split("Chipset Model:")[-1].strip()
                    
                    if "VRAM" in line:
                        vram_match = re.search(r"(\d+)\s*(?:MB|GB)", line)
                        if vram_match:
                            vram_value = float(vram_match.group(1))
                            # Convert to GB if in MB
                            if "MB" in line:
                                vram_value /= 1024
                            gpu_data["memory_total"] = vram_value
                
                if "name" in gpu_data:
                    gpu_info.append(gpu_data)
        
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return gpu_info

def get_system_specs():
    """Get detailed specifications of the system in a platform-agnostic way."""
    specs = {}
    
    # Basic system info
    specs["hostname"] = socket.gethostname()
    specs["system"] = platform.system()
    specs["system_version"] = platform.version()
    specs["system_release"] = platform.release()
    specs["architecture"] = platform.machine()
    specs["platform"] = platform.platform()
    specs["python_version"] = sys.version
    specs["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # CPU Information
    specs["cpu"] = {}
    
    # Try py-cpuinfo first (most reliable cross-platform)
    try:
        cpu_details = cpuinfo.get_cpu_info()
        specs["cpu"]["brand"] = cpu_details.get("brand_raw", "Unknown")
        specs["cpu"]["vendor"] = cpu_details.get("vendor_id_raw", "Unknown")
        specs["cpu"]["arch"] = cpu_details.get("arch", "Unknown")
        specs["cpu"]["bits"] = cpu_details.get("bits", "Unknown")
        specs["cpu"]["frequency"] = cpu_details.get("hz_actual_friendly", "Unknown")
    except Exception:
        # Fallback methods
        if specs["system"] == "Darwin":  # macOS
            try:
                specs["cpu"]["brand"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], 
                                                               universal_newlines=True).strip()
            except (subprocess.SubprocessError, FileNotFoundError):
                specs["cpu"]["brand"] = platform.processor() or "Unknown"
                
        elif specs["system"] == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            specs["cpu"]["brand"] = line.split(":")[-1].strip()
                            break
            except FileNotFoundError:
                specs["cpu"]["brand"] = platform.processor() or "Unknown"
                
        elif specs["system"] == "Windows":
            try:
                specs["cpu"]["brand"] = platform.processor() or "Unknown"
                
                # Try more detailed info via WMI or registry
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                   r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                    specs["cpu"]["brand"] = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            except Exception:
                pass
        
        else:
            specs["cpu"]["brand"] = platform.processor() or "Unknown"
    
    # CPU Cores
    specs["cpu"]["physical_cores"] = psutil.cpu_count(logical=False) or "Unknown"
    specs["cpu"]["logical_cores"] = psutil.cpu_count(logical=True) or "Unknown"
    
    # Memory Information
    mem = psutil.virtual_memory()
    specs["memory"] = {
        "total_gb": round(mem.total / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2),
        "used_gb": round(mem.used / (1024**3), 2),
        "percent_used": mem.percent
    }
    
    # Disk Information
    specs["disk"] = {}
    
    # Get all partitions
    partitions = []
    for part in psutil.disk_partitions(all=False):
        if os.path.exists(part.mountpoint):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                partitions.append({
                    "device": part.device,
                    "mountpoint": part.mountpoint,
                    "filesystem": part.fstype,
                    "total_gb": round(usage.total / (1024**3), 2),
                    "used_gb": round(usage.used / (1024**3), 2),
                    "free_gb": round(usage.free / (1024**3), 2),
                    "percent_used": usage.percent
                })
            except PermissionError:
                continue
    
    specs["disk"]["partitions"] = partitions
    
    # Calculate total and free space across all partitions
    total_space = sum(part["total_gb"] for part in partitions)
    free_space = sum(part["free_gb"] for part in partitions)
    
    specs["disk"]["total_gb"] = round(total_space, 2)
    specs["disk"]["free_gb"] = round(free_space, 2)
    
    # Network Information
    specs["network"] = {}
    
    # Get all network interfaces
    interfaces = []
    for name, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:  # IPv4
                interfaces.append({
                    "name": name,
                    "address": addr.address,
                    "netmask": addr.netmask,
                    "broadcast": getattr(addr, "broadcast", None)
                })
    
    specs["network"]["interfaces"] = interfaces
    
    # Machine-specific detection
    specs["is_virtual_machine"] = False
    
    # Check for common hypervisor signatures
    if specs["system"] == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo_content = f.read().lower()
                if any(x in cpuinfo_content for x in ["vmware", "virtualbox", "xen", "kvm", "qemu"]):
                    specs["is_virtual_machine"] = True
        except FileNotFoundError:
            pass
        
        # Check for Docker
        specs["is_docker"] = os.path.exists("/.dockerenv")
        
        # Check for various cloud providers
        cloud_detection_files = {
            "aws": ["/sys/hypervisor/uuid"],
            "gcp": ["/sys/class/dmi/id/product_name"],
            "azure": ["/sys/class/dmi/id/product_name", "/sys/class/dmi/id/product_uuid"]
        }
        
        specs["cloud_provider"] = None
        
        for provider, files in cloud_detection_files.items():
            matches = False
            for file_path in files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r") as f:
                            content = f.read().lower()
                            if provider == "aws" and "ec2" in content:
                                matches = True
                            elif provider == "gcp" and "google" in content:
                                matches = True
                            elif provider == "azure" and "microsoft" in content:
                                matches = True
                    except:
                        continue
            
            if matches:
                specs["cloud_provider"] = provider
                break
    
    elif specs["system"] == "Windows":
        # Check for VM-related services
        vm_services = ["vmicheartbeat", "vmicvss", "vmicshutdown", "vmicexchange"]
        for service in vm_services:
            try:
                result = subprocess.check_output(["sc", "query", service], universal_newlines=True)
                if "RUNNING" in result:
                    specs["is_virtual_machine"] = True
                    break
            except:
                continue
    
    elif specs["system"] == "Darwin":  # macOS
        try:
            result = subprocess.check_output(["system_profiler", "SPHardwareDataType"], universal_newlines=True)
            if any(x in result.lower() for x in ["vmware", "virtualbox", "parallels"]):
                specs["is_virtual_machine"] = True
        except:
            pass
    
    # GPU/ML Information
    specs["ml_info"] = get_device_ml_info()
    
    # Platform-specific GPU info collection
    if not specs["ml_info"]["gpus"]:
        if specs["system"] == "Linux":
            specs["ml_info"]["gpus"] = get_linux_gpu_info()
        elif specs["system"] == "Windows":
            specs["ml_info"]["gpus"] = get_windows_gpu_info()
        elif specs["system"] == "Darwin":
            specs["ml_info"]["gpus"] = get_mac_gpu_info()
    
    # Detect Apple Silicon
    if specs["system"] == "Darwin" and specs["architecture"] == "arm64":
        specs["is_apple_silicon"] = True
    else:
        specs["is_apple_silicon"] = False
    
    # Get model name for Mac
    if specs["system"] == "Darwin":
        try:
            model_name = subprocess.check_output(["sysctl", "-n", "hw.model"], universal_newlines=True).strip()
            specs["model_name"] = model_name
        except:
            specs["model_name"] = "Unknown Mac model"
    
    return specs

def format_size(size_gb):
    """Format size for better readability."""
    if size_gb >= 1024:
        return f"{size_gb/1024:.2f} TB"
    else:
        return f"{size_gb:.2f} GB"

def print_system_specs():
    """Print formatted system specifications."""
    specs = get_system_specs()
    
    print("\n" + "="*60)
    print(f"{'SYSTEM SPECIFICATIONS':^60}")
    print("="*60)
    
    # System Info
    print(f"\n{'System Information':^60}")
    print("-"*60)
    print(f"Hostname: {specs['hostname']}")
    print(f"OS: {specs['system']} {specs.get('system_release', '')} {specs.get('system_version', '')}")
    
    if specs['system'] == 'Darwin' and 'model_name' in specs:
        print(f"Model: {specs.get('model_name', 'Unknown')}")
        print(f"Architecture: {specs.get('architecture', 'Unknown')} ({'Apple Silicon' if specs.get('is_apple_silicon') else 'Intel'})")
    else:
        print(f"Architecture: {specs.get('architecture', 'Unknown')}")
    
    if specs.get('is_virtual_machine'):
        print("Environment: Virtual Machine")
        if specs.get('cloud_provider'):
            print(f"Cloud Provider: {specs['cloud_provider'].upper()}")
    elif specs.get('is_docker'):
        print("Environment: Docker Container")
    else:
        print("Environment: Physical Machine")
    
    # CPU Info
    print(f"\n{'CPU Information':^60}")
    print("-"*60)
    print(f"Processor: {specs['cpu'].get('brand', 'Unknown')}")
    print(f"Physical cores: {specs['cpu'].get('physical_cores', 'Unknown')}")
    print(f"Logical cores: {specs['cpu'].get('logical_cores', 'Unknown')}")
    if 'frequency' in specs['cpu']:
        print(f"Frequency: {specs['cpu']['frequency']}")
    
    # Memory Info
    print(f"\n{'Memory Information':^60}")
    print("-"*60)
    print(f"Total RAM: {format_size(specs['memory']['total_gb'])}")
    print(f"Available RAM: {format_size(specs['memory']['available_gb'])}")
    print(f"Used RAM: {format_size(specs['memory']['used_gb'])} ({specs['memory']['percent_used']}%)")
    
    # Disk Info
    print(f"\n{'Storage Information':^60}")
    print("-"*60)
    print(f"Total Disk: {format_size(specs['disk']['total_gb'])}")
    print(f"Free Disk: {format_size(specs['disk']['free_gb'])}")
    
    if len(specs['disk']['partitions']) > 1:
        print("\nMounted Partitions:")
        for part in specs['disk']['partitions']:
            print(f"  {part['mountpoint']} ({part['filesystem']}): {format_size(part['total_gb'])} total, {format_size(part['free_gb'])} free")
    
    # GPU/ML Info
    print(f"\n{'GPU/ML Information':^60}")
    print("-"*60)
    
    if specs["ml_info"]["has_pytorch"]:
        print(f"PyTorch Version: {specs['ml_info']['pytorch_version']}")
        if specs["ml_info"]["has_cuda"]:
            print(f"CUDA Available: Yes (v{specs['ml_info']['cuda_version']})")
        else:
            print("CUDA Available: No")
        
        if specs["ml_info"]["has_mps"]:
            print("MPS Available: Yes (Apple Silicon acceleration)")
    else:
        print("PyTorch: Not installed")
    
    if specs["ml_info"]["has_tensorflow"]:
        print(f"TensorFlow Version: {specs['ml_info']['tensorflow_version']}")
    
    # GPU Information
    if specs["ml_info"]["gpus"]:
        print("\nGPU Information:")
        for i, gpu in enumerate(specs["ml_info"]["gpus"]):
            print(f"  GPU {i+1}: {gpu.get('name', 'Unknown')}")
            if 'memory_total' in gpu:
                print(f"    Memory: {format_size(gpu['memory_total'])}")
            if 'memory_used' in gpu:
                print(f"    Memory Used: {format_size(gpu['memory_used'])}")
            if 'memory_allocated' in gpu:
                print(f"    Memory Allocated: {format_size(gpu['memory_allocated'])}")
            if 'temperature' in gpu:
                print(f"    Temperature: {gpu['temperature']}°C")
    else:
        print("\nNo dedicated GPU detected.")
    
    # ML Training Recommendations
    print("\n" + "="*60)
    print(f"{'ML TRAINING RECOMMENDATION':^60}")
    print("="*60)
    
    # Helper variables for recommendations
    ram_gb = specs['memory']['total_gb']
    has_gpu = bool(specs["ml_info"]["gpus"])
    has_cuda = specs["ml_info"]["has_cuda"]
    has_mps = specs["ml_info"]["has_mps"]
    is_apple_silicon = specs.get('is_apple_silicon', False)
    
    if has_cuda:
        total_gpu_memory = sum(gpu.get('memory_total', 0) for gpu in specs["ml_info"]["gpus"])
        if total_gpu_memory >= 24:
            print("\nYour system has substantial GPU resources suitable for training medium to large models.")
            print("Recommendations:")
            print("- Can handle full fine-tuning of medium-sized models (up to ~7B parameters)")
            print("- Suitable for LoRA/QLoRA on larger models (up to ~70B parameters)")
            print("- Can use larger batch sizes for faster training")
        elif total_gpu_memory >= 8:
            print("\nYour system has good GPU resources suitable for training smaller models.")
            print("Recommendations:")
            print("- Use parameter-efficient fine-tuning methods like LoRA/QLoRA")
            print("- Enable gradient checkpointing")
            print("- Consider 8-bit or 4-bit quantization for larger models")
            print("- Use appropriate batch sizes based on available memory")
        else:
            print("\nYour system has limited GPU resources.")
            print("Recommendations:")
            print("- Use QLoRA with 4-bit quantization")
            print("- Use very small batch sizes with gradient accumulation")
            print("- Consider reducing context length")
            print("- Focus on smaller models (< 1B parameters)")
    elif has_mps and is_apple_silicon:
        if ram_gb >= 32:
            print("\nYour Apple Silicon Mac is suitable for training with MPS acceleration.")
            print("Recommendations:")
            print("- Use parameter-efficient fine-tuning methods like LoRA/QLoRA")
            print("- Enable gradient checkpointing")
            print("- Use small batch sizes (1-2) with gradient accumulation")
            print("- Consider 4-bit quantization (QLoRA)")
        elif ram_gb >= 16:
            print("\nYour Apple Silicon Mac may be able to train with significant optimizations.")
            print("Recommendations:")
            print("- Use QLoRA with 4-bit quantization")
            print("- Set a very small batch size (1) with gradient accumulation")
            print("- Enable gradient checkpointing")
            print("- Consider reducing context length")
        else:
            print("\nTraining directly on this Apple Silicon Mac may be challenging.")
            print("Recommendations:")
            print("- Consider using smaller models")
            print("- Look into cloud-based training options")
            print("- Try adapter-based methods with minimal parameters")
    else:
        print("\nYour system does not have GPU acceleration for ML training.")
        if ram_gb >= 64:
            print("With high RAM available, you can still train on CPU for smaller models.")
            print("Recommendations:")
            print("- Use CPU-optimized models")
            print("- Consider distilled or quantized versions of models")
            print("- Be prepared for significantly longer training times")
        else:
            print("Recommendations:")
            print("- Consider cloud-based training options")
            print("- Focus on inference rather than training")
            print("- Use pre-trained models with minimal adaptation")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    device = get_device()
    print('device:', device)
