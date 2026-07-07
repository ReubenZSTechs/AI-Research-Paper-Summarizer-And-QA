import subprocess

def register_ollama_model(model_name: str, modelfile_path: str):
    print(f"Registering model {model_name} with modelfile {modelfile_path}...")

    command = [
        'ollama',
        'create',
        model_name,
        '--f',
        modelfile_path
    ]

    subprocess.run(command, check=True)

    print(f"Model {model_name} registered successfully.")