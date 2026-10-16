# Get GPU available memory
from rag.config import DEVICE
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from pprint import pprint


def get_mac_mem():
    # 2. Calculate Total Unified Memory (RAM)
    # On a Mac, the GPU shares the total system memory.
    total_memory_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    # 0.7 since we can realistically only use upto 70% of mem
    total_memory_gb = round(0.7*(total_memory_bytes / (2**30)))

    print(f"Available Unified Memory: {total_memory_gb} GB")
    return total_memory_gb


def get_nvidia_mem():
    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = round(gpu_memory_bytes / (2**30))
    print(f"Available GPU memory: {gpu_memory_gb} GB")
    return gpu_memory_gb


def check_processing_power(device: str = DEVICE):
    available_mem = 0

    if device == 'cuda':
        available_mem = get_nvidia_mem()
        print(f"Nvidia GPU is available - {available_mem}")

    # 1. Check if Apple's MPS (Metal Performance Shaders) is available
    elif device == 'mps':
        available_mem = get_mac_mem()
        print(f"Apple Silicon GPU (MPS) is available - {available_mem}")

    else:
        available_mem = get_mac_mem()
        print(f"MPS is not available. PyTorch will use CPU - {available_mem}")

    return available_mem


# llm_mem_eligibility_tiers = {
#     "google/gemma-2b-it": 5.1
#     "google/gemma-2b-it": 8.1
#     "google/gemma-2b-it": 19.0
# }

def check_model_compatibility(
    available_mem, 
    llm_mem_eligibility_tiers=(5.1, 8.1, 19.0)
):
    # Note: the following is Gemma focused, however, there are more and more LLMs of the 2B and 7B size appearing for local use.
    if available_mem < llm_mem_eligibility_tiers[0]:
        print(f"Your available GPU memory is {available_mem}GB, you may not have enough memory to run a Gemma LLM locally without quantization.")
        use_quantization_config = False 
        model_id = None
    elif available_mem < llm_mem_eligibility_tiers[1]:
        print(f"GPU memory: {available_mem} | Recommended model: Gemma 2B in 4-bit precision.")
        use_quantization_config = True 
        model_id = "google/gemma-2b-it"
    elif available_mem < llm_mem_eligibility_tiers[2]:
        print(f"GPU memory: {available_mem} | Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision.")
        use_quantization_config = False 
        model_id = "google/gemma-2b-it"
    elif available_mem >= llm_mem_eligibility_tiers[2]:
        print(f"GPU memory: {available_mem} | Recommend model: Gemma 7B in 4-bit or float16 precision.")
        use_quantization_config = False 
        model_id = "google/gemma-7b-it"

    print(f"use_quantization_config set to: {use_quantization_config}")
    print(f"model_id set to: {model_id}")

    return { 'model_id': model_id, 'use_quantization_config': use_quantization_config }


def load_llm(
    model_id: str,
    use_quantization_config: bool,
    device: str = DEVICE,
    low_cpu_mem_usage: bool = False
):

    model_config = { 'torch_dtype': torch.float16, 'attn_implementation': 'sdpa' }

    if (device == 'cuda'):
        from transformers import BitsAndBytesConfig
        model_config['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_compute_dtype = torch.float16
        )

        # Bonus: flash attention 2 = faster attention mechanism
        # Flash Attention 2 requires a GPU with a compute capability score of 8.0+ (Ampere, Ada Lovelace, Hopper and above): https://developer.nvidia.com/cuda-gpus 

        if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
            model_config['attn_implementation'] = "flash_attention_2"
    else:
        model_config['attn_implementation'] = "sdpa" # scaled dot product attention

    print(f"Using attention implementation: {model_config['attn_implementation']}")

    # 3. Instantiate tokenizer (tokenizer turns text into tokens)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

    # 4. Instantiate the model 
    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = model_id,
        torch_dtype = model_config['torch_dtype'],
        quantization_config = model_config.get('quantization_config') if use_quantization_config else None,
        low_cpu_mem_usage = low_cpu_mem_usage, # use as much memory as we can
        attn_implementation = model_config['attn_implementation']
    )

    if not use_quantization_config:
        llm_model.to(device)

    return { 'llm_model': llm_model, 'tokenizer': tokenizer }

def get_model_num_params(model: torch.nn.Module):
    return sum([param.numel() for param in model.parameters()])

def get_model_mem_size(model: torch.nn.Module):
    # Get model parameters and buffer sizes
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    # Calculate model sizes
    model_mem_bytes = mem_params + mem_buffers
    model_mem_mb = model_mem_bytes / (1024**2)
    model_mem_gb = model_mem_bytes / (1024**3) 

    pprint({"model_mem_bytes": model_mem_bytes,
            "model_mem_mb": round(model_mem_mb, 2), 
            "model_mem_gb": round(model_mem_gb, 2)})

    return {
        "model_mem_bytes": model_mem_bytes,
        "model_mem_mb": round(model_mem_mb, 2), 
        "model_mem_gb": round(model_mem_gb, 2)
    }

def init_model(
    model_id: str | None = None,
    use_quantization_config: bool | None = None,
    device: str = DEVICE
):
    available_mem = check_processing_power(device = device)

    model_compatibility = check_model_compatibility(
        available_mem = available_mem,
    )
    
    if model_id is None:
        model_id = model_compatibility['model_id']

    if model_id is None:
        raise RuntimeError(f"Not enough memory ({available_mem} GB) to run a local LLM.")

    if use_quantization_config is None:
        use_quantization_config = model_compatibility['use_quantization_config']

    print(f'Model Overrirded - {model_id}');

    llm = load_llm(
        model_id = model_id,
        use_quantization_config = use_quantization_config,
        device = device,
        low_cpu_mem_usage = False
    )

    model_num_params = get_model_num_params(model = llm['llm_model'])
    model_mem = get_model_mem_size(model = llm['llm_model'])

    return {
        'model_id': model_id,
        'llm_model': llm['llm_model'],
        'tokenizer': llm['tokenizer'],
        'model_num_params': model_num_params,
        'model_mem': model_mem,
        'available_mem': available_mem,
        'use_quantization_config': use_quantization_config
    }

