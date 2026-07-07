from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from pathlib import Path

def merge_lora(base_model: str, adapter_path: str, output_path: str):
    print(f"Merging LoRA adapter from {adapter_path} into base model {base_model} and saving to {output_path}...")

    model, tokenizer = FastLanguageModel.from_pretrained(model_name=base_model, max_seq_length=2048, load_in_4bit=True)

    model.load_adapter(adapter_path)

    Path(output_path).mkdir(parents=True, exist_ok=True)

    model.save_pretrained_merged(output_path, tokenizer, save_method="merged_16bit")

    print(f"Model saved to {output_path}")