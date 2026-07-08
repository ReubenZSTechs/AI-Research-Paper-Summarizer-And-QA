from dotenv import load_dotenv

load_dotenv()

import argparse
import os
from pathlib import Path
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier



def quantize_model(model_name: str, quantized_name: str):
    quant_path = Path(os.getenv("HUGGINGFACE_QUANTIZE_FOLDERPATH")).expanduser() / quantized_name
    quant_path.mkdir(exist_ok=True, parents=True)

    offload_path = Path(os.getenv("HUGGINGFACE_QUANTIZE_FOLDERPATH")).expanduser() / "offload_tmp"
    offload_path.mkdir(exist_ok=True, parents=True)

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map={"":0}, 
        dtype=torch.bfloat16, 
        # max_memory={0: "13GiB", "disk": "20GiB"},
        low_cpu_mem_usage=True,
        # offload_folder=str(offload_path),
        # offload_state_dict=True
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading calibration dataset")
    quant_config = GPTQModifier(
    scheme="W4A16",
    targets="Linear",
    offload_hessians=True,
    ignore=[
        'lm_head',
        're:.*vision_tower.*',
        're:.*audio_tower.*',
        "re:.*visual.*",
        're:.*per_layer_model_projection.*',
        're:.*per_layer_.*',       # catches related per-layer embedding projections if present
        're:.*altup.*',            # Gemma 3n/4 AltUp modules, if present, are also often unsuitable for GPTQ
    ]
)

    print(f"Calibrating model with quant settings")
    oneshot(
        model=model,
        dataset="ultrachat_200k",
        splits="train_sft[:64]",
        recipe=quant_config,
        output_dir=str(quant_path),
        max_seq_length=256,
        num_calibration_samples=16,
        pipeline="basic"
    )

    print(f"Saving tokenizer to {quant_path}")
    tokenizer.save_pretrained(str(quant_path))

    print(f"Model quantized to {quant_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--quantized_name", required=True)
    
    args = parser.parse_args()

    quantize_model(model_name=args.model_name, quantized_name=args.quantized_name)