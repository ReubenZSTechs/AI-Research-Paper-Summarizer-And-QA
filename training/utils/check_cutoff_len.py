import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

lengths = []
with open("training/data/formatted/main/answer/answer_dpo.jsonl") as f:
    for line in f:
        record = json.loads(line)
        prompt = record["system"] + record["instruction"] + record["input"]
        chosen_len = len(tokenizer.encode(prompt + record["chosen"]))
        rejected_len = len(tokenizer.encode(prompt + record["rejected"]))
        lengths.append(max(chosen_len, rejected_len))

lengths.sort()
n = len(lengths)
print(f"max:  {lengths[-1]}")
print(f"p99:  {lengths[int(n * 0.99)]}")
print(f"p95:  {lengths[int(n * 0.95)]}")
print(f"mean: {sum(lengths) / n:.0f}")