from pathlib import Path

def generate_modelfile(model_name: str, output_path: str, gguf_path: str, system_prompt: str, generation):
    print(f"Generating modelfile for model {model_name} and saving to {output_path}...")

    content = f"""
        FROM {gguf_path}

        PARAMETER temperature {generation['temperature']}
        PARAMETER top_p {generation['top_p']}
        PARAMETER repeat_penalty {generation['repeat_penalty']}
        PARAMETER max_tokens {generation['max_tokens']}

        SYSTEM \"\"\"
        {system_prompt}
        \"\"\"
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding='utf-8') as f:
        f.write(content)

    print(f"Model file generated and saved to {output_path}")