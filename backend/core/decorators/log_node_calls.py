from functools import wraps

import json
from time import perf_counter


def log_node_call(node_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            status = "success"
            result = None
            usage = None

            start_time = perf_counter()

            try:
                result = func(*args, **kwargs)
                return result
            
            except Exception as e:
                status = "error"
                raise

            finally:
                end_time = perf_counter()
                node_latency = end_time - start_time

                data_payload = {
                    'node_name': node_name,
                    'status': status,
                    'latency': round(node_latency * 1000, 5) # In miliseconds
                }

                if status == "success":
                    with open("backend/core/logging/node_call_logs.jsonl", "a", encoding='utf-8') as f:
                        f.write(json.dumps(data_payload) + "\n")
        
        return wrapper
    return decorator
