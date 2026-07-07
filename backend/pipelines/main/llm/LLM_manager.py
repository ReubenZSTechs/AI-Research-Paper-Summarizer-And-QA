from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel

import torch
import yaml
from threading import Thread


class LLMManager:
    def __init__(self):
        pass


    def init_config(self, config_path: str):
        try:
            with open(config_path, "r", encoding='utf-8') as f:
                config = yaml.safe_load(f)

            return config
        
        except FileNotFoundError as e:
            print(f"Configuration file not found. Got {config_path}\n\nError:\n{e}")
            return
        
    
    def load_model(self, model_name: str, load_in_4bit: bool, adapter_path: str = None):
        if load_in_4bit and torch.cuda.is_available():
            print(f"Model {model_name} loading in Q4")
            quantize_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

        elif load_in_4bit and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA GPU needed to load in 4bit")
        
        else:
            print(f"Model {model_name} loaded in default")
            quantize_config = None

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantize_config,
            device_map={"":0}, # Force in GPU only
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

        if adapter_path:
            model = PeftModel.from_pretrained(
                model,
                adapter_path
            )

        model.eval()

        return model, tokenizer
    

class LLMAgent:
    def __init__(self, agent_config, main_model, main_tokenizer):
        self.llm_manager = LLMManager()
        self.agent_config = self.llm_manager.init_config(config_path=agent_config)

        self.model, self.tokenizer = main_model, main_tokenizer
        
        try:
            self.agent_system_prompt = self.agent_config['system_prompt']
            self.agent_generation_config = self.agent_config['generation']
        except KeyError as e:
            raise KeyError(f"Missing configuration keys: {e.args[0]}")
        

    def _build_user_prompt(self, user_prompt):
        messages = [
            {
                'role': 'system',
                'content': self.agent_system_prompt
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors='pt')

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        return inputs
    

    def generate_response(self, user_prompt):
        inputs = self._build_user_prompt(user_prompt=user_prompt)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.agent_generation_config)

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

        result = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return result
    

    def generate_stream_response(self, user_prompt):
        inputs = self._build_user_prompt(user_prompt=user_prompt)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_args = {
            **inputs,
            **self.agent_generation_config,
            "streamer": streamer
        }

        thread = Thread(target=self.model.generate, kwargs=generation_args)

        thread.start()

        for text in streamer:
            yield text

        thread.join()


    def query(self, user_prompt):
        for chunk in self.generate_stream_response(user_prompt=user_prompt):
            print(chunk, end="", flush=True)

