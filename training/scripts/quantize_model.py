import argparse
from pathlib import Path


from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def quantize_model(model_name: str, quantized_name: str):
    quant_path = Path("C:/Users/reube/.cache/huggingface/quantized") / quantized_name
    quant_path.mkdir(exist_ok=True, parents=True)

    print(f"Loading {model_name}...")
    model = AutoAWQForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quant_config = {
        'zero_point': True,
        'q_group_size': 128,
        'w_bit': 4,
        'version': 'GEMM'
    }

    print(f"Quantizing model...")
    model.quantize(tokenizer=tokenizer, quant_config=quant_config)

    print(f"Saving model to {quant_path}")
    model.save_quantized(str(quant_path))
    tokenizer.save_pretrained(str(quant_path))

    print(f"Model quantized to {quant_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--quantized_name", required=True)
    
    args = parser.parse_args()

    quantize_model(model_name=args.model_name, quantized_name=args.quantized_name)