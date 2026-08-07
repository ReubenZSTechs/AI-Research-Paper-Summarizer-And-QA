from collections import defaultdict
from pathlib import Path

import json
import random


CONFIG = {
    'MAIN_DATA_FOLDERPATH': Path("./training/data/formatted/main"),
    'AGENT_ROLES_ARR': ['answer', 'prompt_rewriter', 'summarizer', 'synthesizer'],
    'TRAIN_RATIO': 0.9,
    'VAL_RATIO': 0.05,
    'TEST_RATIO': 0.05,
    'SFT_FOLDERPATH': Path("./training/data/formatted/sft"),
    'DPO_FOLDERPATH': Path("./training/data/formatted/dpo"),
    'SEED': 42,
}

SPLIT_NAMES = ('train', 'val', 'test')


def write_data_corpus(path, payload_arr):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding='utf-8') as f:
        for payload in payload_arr:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_sft_payload(raw_data):
    system_prompt = raw_data['system']
    user_prompt = f"{raw_data['instruction']}\n{raw_data['input']}"
    label = raw_data['output']

    data_payload = {
        'messages': [
            {
                'from': 'system',
                'value': system_prompt
            },
            {
                'from': 'human',
                'value': user_prompt
            },
            {
                'from': 'gpt',
                'value': label
            }
        ]
    }

    return data_payload


def build_dpo_payload(raw_data):
    system_prompt = raw_data['system']
    user_prompt = f"{raw_data['instruction']}\n{raw_data['input']}"
    chosen_label = raw_data['chosen']
    rejected_label = raw_data['rejected']

    data_payload = {
        'messages': [
            {
                'from': 'system',
                'value': system_prompt
            },
            {
                'from': 'human',
                'value': user_prompt
            }
        ],
        "chosen": {
            'from': 'gpt',
            'value': chosen_label
        },
        'rejected': {
            'from': 'gpt',
            'value': rejected_label
        }
    }

    return data_payload


def build_dataset_info(agent, training_stage, training_type, filename):
    dataset_key = f"{agent}_{training_stage}_{training_type}"
    dataset_value = {
        "file_name": f"{agent}/{filename}",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages"
        }
    }

    if training_type == "dpo":
        dataset_value["ranking"] = True
        dataset_value["columns"]["chosen"] = "chosen"
        dataset_value["columns"]["rejected"] = "rejected"

    return dataset_key, dataset_value


def group_by_source(data_corpus):
    grouped = defaultdict(list)

    for record in data_corpus:
        source = record.get("meta", {}).get("source", "__unsourced__")
        grouped[source].append(record)

    return grouped


def split_sources(grouped, rng):
    sources = sorted(grouped)
    rng.shuffle(sources)

    total = sum(len(grouped[source]) for source in sources)

    # Not enough distinct sources to grant a group-level split -> fall back
    # to a flat per-record split so val/test aren't starved.
    if len(sources) < 3:
        return None  # signal to caller: use flat split instead

    train_target = total * CONFIG['TRAIN_RATIO']
    val_target = total * CONFIG['VAL_RATIO']

    assigned = {name: [] for name in SPLIT_NAMES}
    filled = 0

    for source in sources:
        if filled < train_target:
            assigned['train'].append(source)
        elif filled < train_target + val_target:
            assigned['val'].append(source)
        else:
            assigned['test'].append(source)

        filled += len(grouped[source])

    return assigned


def flat_split_records(data_corpus, builder, rng):
    records = [builder(r) for r in data_corpus]
    rng.shuffle(records)

    n = len(records)
    train_end = int(n * CONFIG['TRAIN_RATIO'])
    val_end = train_end + int(n * CONFIG['VAL_RATIO'])

    return {
        'train': records[:train_end],
        'val': records[train_end:val_end],
        'test': records[val_end:],
    }


def build_split_records(grouped, assigned, builder, rng):
    splits = {}

    for name in SPLIT_NAMES:
        records = [
            builder(record)
            for source in assigned[name]
            for record in grouped[source]
        ]
        rng.shuffle(records)
        splits[name] = records

    return splits


def report(agent, training_type, splits, source_counts):
    total = sum(len(records) for records in splits.values())

    if total == 0:
        print(f"  {agent:<16} {training_type:<4} no records")
        return

    cells = []
    for name in SPLIT_NAMES:
        count = len(splits[name])
        share = count / total * 100
        cells.append(f"{name}={count:,} ({share:.0f}%)")

    sources = " ".join(f"{name}:{source_counts[name]}" for name in SPLIT_NAMES)
    print(f"  {agent:<16} {training_type:<4} {total:>7,} total  " + "  ".join(cells))
    print(f"  {'':<16} {'':<4} sources {sources}")


if __name__ == "__main__":
    rng = random.Random(CONFIG['SEED'])

    dataset_info = {
        'sft': {},
        'dpo': {},
    }

    target_folders = {
        'sft': CONFIG['SFT_FOLDERPATH'],
        'dpo': CONFIG['DPO_FOLDERPATH'],
    }

    builders = {
        'sft': build_sft_payload,
        'dpo': build_dpo_payload,
    }

    for agent in CONFIG['AGENT_ROLES_ARR']:
        agent_foldername = CONFIG['MAIN_DATA_FOLDERPATH'] / agent

        if not agent_foldername.exists():
            print(f"Missing folder for {agent}, skipping")
            continue

        files = sorted(f.name for f in agent_foldername.rglob("*.jsonl") if f.is_file())

        for file in files:
            training_type = file.split("_")[-1].split(".")[0]

            if training_type not in target_folders:
                print(f"Unknown training type in {file}, skipping")
                continue

            filepath = agent_foldername / file

            try:
                with open(filepath, "r", encoding='utf-8') as f:
                    data_corpus = [json.loads(line) for line in f if line.strip()]

            except FileNotFoundError as e:
                print(f"File not found\n\n{e}")
                continue

            if not data_corpus:
                print(f"Empty corpus for {agent} {training_type}, skipping")
                continue

            grouped = group_by_source(data_corpus)
            assigned = split_sources(grouped, rng)

            if assigned is None:
                splits = flat_split_records(data_corpus, builders[training_type], rng)
                source_counts = {name: len(splits[name]) for name in SPLIT_NAMES}  # not source-based here
            else:
                splits = build_split_records(grouped, assigned, builders[training_type], rng)
                source_counts = {name: len(assigned[name]) for name in SPLIT_NAMES}

            for stage in SPLIT_NAMES:
                filename = f"{agent}_{stage}_{training_type}.jsonl"
                output_path = target_folders[training_type] / agent / filename

                write_data_corpus(output_path, splits[stage])

                key, value = build_dataset_info(agent, stage, training_type, filename)
                dataset_info[training_type][key] = value

            report(agent, training_type, splits, source_counts)

    for training_type, entries in dataset_info.items():
        if not entries:
            continue

        info_path = target_folders[training_type] / "dataset_info.json"
        info_path.parent.mkdir(parents=True, exist_ok=True)

        with open(info_path, "w", encoding='utf-8') as f:
            json.dump(entries, f, indent=2)

        print(f"\nWrote {len(entries)} entries to {info_path}")