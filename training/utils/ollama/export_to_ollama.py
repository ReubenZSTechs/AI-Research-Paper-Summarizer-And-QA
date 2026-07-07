from pathlib import Path
import yaml

from utils.ollama.convert_to_gguf import convert_to_gguf
from utils.ollama.generate_modelfile import generate_modelfile
from utils.ollama.register_ollama_model import register_ollama_model
from utils.ollama.merge_lora import merge_lora
from utils.ollama.verify_ollama_model import verify_ollama_model


def deploy_model(config_path: str):
    # Load YAML file
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    name = config['name']
    base_model = config['base_model']
    system_prompt = config['system_prompt']

    generation = config['generation']

    paths = config['paths']

    adapter = paths['adapter_dir']
    merged = paths['merged_dir']
    gguf = paths['gguf_dir']
    modelfile = paths['modelfile']

    merge_lora(base_model=base_model, adapter_path=adapter, output_path=merged)

    convert_to_gguf(merged_path=merged, gguf_output=gguf)

    generate_modelfile(model_name=name, output_path=modelfile, gguf_path=gguf, system_prompt=system_prompt, generation=generation)

    register_ollama_model(model_name=name, modelfile_path=modelfile)

    verify_ollama_model(model_name=name)