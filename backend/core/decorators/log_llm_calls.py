from functools import wraps

import json
from time import perf_counter


def log_llm_calls(func):
    @wraps(func)
    def wrapper(self, user_prompt, *args, **kwargs):
        status = "success"
        result = None
        usage = None

        start_time = perf_counter()

        try:
            result = func(self, user_prompt, *args, **kwargs)
            return result
        
        except Exception as e:
            status = "error"
            raise

        finally:
            end_time = perf_counter()
            latency = start_time - end_time

            data_payload = {
                "input_prompt": user_prompt,
                "model_name": self.model_config['agent']['base_model'],
                "status": status
            }

            if status == "success":
                usage = getattr(self, "last_usage", None)
                if usage:
                    prompt_tokens = usage.prompt_tokens
                    generated_tokens = usage.completion_tokens
                    token_generation_metric = round(generated_tokens / latency, 5)
                else:
                    prompt_tokens = None
                    generated_tokens = None

                data_payload['prompt_tokens'] = prompt_tokens
                data_payload['generated_tokens'] = generated_tokens
                data_payload['token_generation'] = token_generation_metric
                data_payload['total_tokens'] = prompt_tokens + generated_tokens
                data_payload['model_latency'] = latency
                data_payload['output'] = result

                with open("backend/core/logging/llm_call_logs.jsonl", "a", encoding='utf-8') as f:
                    f.write(json.dumps(data_payload) + "\n")

    return wrapper


def log_output_call(func):
    @wraps(func)
    def wrapper(self, user_prompt, *args, **kwargs):
        status = "success"
        usage = None
        output = ""

        start_time = perf_counter()

        try:
            for chunk in func(self, user_prompt, *args, **kwargs):
                output += chunk
                yield chunk

            usage = getattr(self, "last_usage", None)
        
        except Exception as e:
            status = "error"
            raise

        finally:
            end_time = perf_counter()
            latency = start_time - end_time

            data_payload = {
                "input_prompt": user_prompt,
                "model_name": self.model_config['agent']['base_model'],
                "status": status
            }

            if status == "success":
                usage = getattr(self, "last_usage", None)
                if usage:
                    prompt_tokens = usage.prompt_tokens
                    generated_tokens = usage.completion_tokens
                    token_generation_metric = round(generated_tokens / latency, 5)
                else:
                    prompt_tokens = None
                    generated_tokens = None

                data_payload['prompt_tokens'] = prompt_tokens
                data_payload['generated_tokens'] = generated_tokens
                data_payload['token_generation'] = token_generation_metric
                data_payload['total_tokens'] = prompt_tokens + generated_tokens
                data_payload['model_latency'] = latency
                data_payload['output'] = output

                with open("backend/core/logging/llm_call_logs.jsonl", "a", encoding='utf-8') as f:
                    f.write(json.dumps(data_payload) + "\n")

    return wrapper

                