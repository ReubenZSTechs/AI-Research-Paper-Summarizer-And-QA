import requests

def verify_ollama_model(model_name: str) -> bool:
    print(f"Verifying model {model_name}...")


    url = f"http://localhost:11434/api/generate"
    
    response = requests.post(url, json={
        "model": model_name,
        "prompt": "Hello, world!",
        "stream": False
    })

    if response.status_code == 200:
        print(f"Model {model_name} is working correctly.")
        return True
    else:
        print(f"Model {model_name} failed to generate a response. Status code: {response.status_code}")
        return False