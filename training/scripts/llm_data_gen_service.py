import hashlib
import json
import os
import random
import re
import signal
import threading
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import yaml
from tqdm import tqdm

from backend.services.call_llm_service import Agent


CONFIG = {
    "CHUNK_FILE": Path("training/data/processed/chunks_retrieval.jsonl"),
    "TEACHER_AGENT_CONFIG": Path("training/configs/agents/data_generation_agent/teacher.yaml"),
    "ANSWER_AGENT_CONFIG": Path("backend/pipelines/agents/inference_only/answer_agent.yaml"),
    "SUMMARIZER_AGENT_CONFIG": Path("backend/pipelines/agents/inference_only/summarizer_agent.yaml"),
    "SYNTHESIZER_AGENT_CONFIG": Path("backend/pipelines/agents/inference_only/synthesizer_agent.yaml"),
    "PROMPT_REWRITER_AGENT_CONFIG": Path("backend/pipelines/agents/inference_only/prompt_rewriter_agent.yaml"),
    "OUTPUT_DIR": Path("training/data/formatted/main"),
    "SKIP_LOG": Path("training/logs/chunk_skips.jsonl"),
    "CHECKPOINT_FILE": Path("training/checkpoint/dataset_generation_state.json"),
    "CONCURRENCY": 32,
    "CHECKPOINT_EVERY": 1,
    "SEED": 42,
    "CHUNK_BUDGET": {"answer": 5000, "summarizer": 5000},
    "PAIRS_PER_CHUNK": 2,
    "WINDOW_SIZE": 3,
    "WINDOWS_PER_SOURCE": 6,
    "SYNTHESIS_GROUP": 4,
    "REWRITER_MAX_RECORDS": 8000,
    "DISTRACTOR_RATE": 0.40,
    "UNANSWERABLE_RATE": 0.07,
    "DPO_RATE": 0.55,
    "MIN_WORDS": 60,
    "MIN_ALPHA_RATIO": 0.70,
    "MIN_GROUNDING": 0.45,
    "MAX_LENGTH_RATIO": 2.5,
    "REQUEST_RETRIES": 2,
}

WRITER_NAMES = (
    "answer_sft", "answer_dpo", "summarizer_sft", "summarizer_dpo",
    "synthesizer_sft", "synthesizer_dpo", "prompt_rewriter_sft",
)

STAGE_NAMES = ("answer", "summarizer", "synthesizer", "prompt_rewriter")

ABSTENTION = "The provided context does not contain enough information to answer this question."
ANSWER_INSTRUCTION = "Answer the question using only the numbered context passages provided."
SUMMARY_INSTRUCTION = "Summarise the following section of a research paper."
SYNTHESIS_INSTRUCTION = "Merge the following section summaries into a continuous account."
REWRITE_INSTRUCTION = "Rewrite the question so that it is grammatically correct and correctly spelled."

STOPWORDS = frozenset(
    "a an the of in on at to for with by and or as is are was were be been being this "
    "that these those it its from into we our their can may such using used also than "
    "then which who what where when how not but if each other more most some".split()
)

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["questions"],
}

TEACHER_QA_SYSTEM = (
    "You build supervised training data for a scientific question answering model. "
    "You read one excerpt from a research paper and produce self-contained questions "
    "that are fully answerable from that excerpt alone. Every question must make sense "
    "to a reader who has never seen the excerpt, so never refer to the passage, the "
    "text, or the excerpt. If the excerpt is a reference list, a table of symbols, an "
    "acknowledgement, a header, or otherwise carries no substantive claim, return an "
    "empty list."
)


def writer_path(output_dir: Path, writer_name: str) -> Path:
    agent = writer_name.rsplit("_", 1)[0]
    return output_dir / agent / f"{writer_name}.jsonl"


def derive_rng(seed: int, key: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


class ShutdownSignal:
    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        tqdm.write("\nShutdown requested. Draining in-flight requests and saving checkpoint.")
        tqdm.write("Press Ctrl+C again to abort immediately without saving.")


def serialise_rng(rng: random.Random) -> dict:
    version, internal, gauss = rng.getstate()
    return {"version": version, "internal": list(internal), "gauss": gauss}


def restore_rng(rng: random.Random, payload: dict) -> None:
    rng.setstate((payload["version"], tuple(payload["internal"]), payload["gauss"]))


class RunState:
    def __init__(self, path: Path, rng: random.Random):
        self.path = path
        self.rng = rng
        self.resumed = False
        self.used: set[str] = set()
        self.stages = {name: {"accepted": 0, "done": False} for name in STAGE_NAMES}
        self.questions: list[str] = []
        self.bank: dict[str, list[str]] = defaultdict(list)
        self.synthesised_sources: set[str] = set()
        self.rewriter_seen: set[str] = set()
        self.writer_counts = {name: 0 for name in WRITER_NAMES}
        self.skip_count = 0

    def load(self) -> None:
        if not self.path.exists():
            return

        with open(self.path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.used = set(payload["used"])
        self.stages = payload["stages"]
        self.questions = payload["questions"]
        self.bank = defaultdict(list, payload["bank"])
        self.synthesised_sources = set(payload["synthesised_sources"])
        self.rewriter_seen = set(payload["rewriter_seen"])
        self.writer_counts = payload["writer_counts"]
        self.skip_count = payload["skip_count"]
        restore_rng(self.rng, payload["rng_state"])
        self.resumed = True

    def save(self) -> None:
        payload = {
            "version": 2,
            "rng_state": serialise_rng(self.rng),
            "used": sorted(self.used),
            "stages": self.stages,
            "questions": self.questions,
            "bank": dict(self.bank),
            "synthesised_sources": sorted(self.synthesised_sources),
            "rewriter_seen": sorted(self.rewriter_seen),
            "writer_counts": self.writer_counts,
            "skip_count": self.skip_count,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")

        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

        os.replace(temporary, self.path)

    def stage_done(self, name: str) -> bool:
        return self.stages[name]["done"]

    def mark_done(self, name: str) -> None:
        self.stages[name]["done"] = True


class JsonlWriter:
    def __init__(self, path: Path, resume_count: int = 0):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.count = self._prepare(path, resume_count)
        self.handle = open(path, "a", encoding="utf-8")

    def _prepare(self, path: Path, resume_count: int) -> int:
        if resume_count <= 0 or not path.exists():
            open(path, "w", encoding="utf-8").close()
            return 0

        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()

        kept = lines[:resume_count]
        with open(path, "w", encoding="utf-8") as handle:
            handle.writelines(kept)

        return len(kept)

    def write(self, record: dict) -> None:
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1

    def flush(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.close()


class TeacherAgent(Agent):
    def generate_text(self, system_prompt, user_prompt, temperature=None,
                       max_tokens=None, extra_args=None):
        payload = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        settings = self.format_generation_args()
        if temperature is not None:
            settings["temperature"] = temperature
        if max_tokens is not None:
            settings["max_tokens"] = max_tokens

        for _ in range(CONFIG["REQUEST_RETRIES"] + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.served_model_name,
                    messages=payload,
                    extra_body=extra_args or {},
                    **settings,
                )
            except Exception:
                continue

            self.last_usage = response.usage
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()

        return None

    def generate_json(self, system_prompt, user_prompt, schema, temperature=None):
        raw = self.generate_text(
            system_prompt, user_prompt, temperature=temperature,
            extra_args={"structured_outputs": {"json": schema}},
        )
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None


class ThreadLocalAgent:
    def __init__(self, yaml_config: str, factory=Agent):
        self._yaml_config = yaml_config
        self._factory = factory
        self._local = threading.local()
        template = factory(yaml_config=yaml_config)
        self.agent_system_prompt = template.agent_system_prompt
        self._local.instance = template

    @property
    def agent(self):
        instance = getattr(self._local, "instance", None)
        if instance is None:
            instance = self._factory(yaml_config=self._yaml_config)
            self._local.instance = instance
        return instance


def load_prompt_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["system_prompt"].strip()


def longest_overlap(left: str, right: str, limit: int = 600) -> int:
    window = min(limit, len(left), len(right))
    for size in range(window, 20, -1):
        if left.endswith(right[:size]):
            return size
    return 0


def merge_chunks(chunks: list[dict]) -> str:
    merged = chunks[0]["chunk_text"]
    for chunk in chunks[1:]:
        text = chunk["chunk_text"]
        merged = merged + " " + text[longest_overlap(merged, text):]
    return re.sub(r"\s+", " ", merged).strip()


def content_terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOPWORDS}


def length_balanced(chosen: str, rejected: str, max_ratio: float) -> bool:
    left = max(1, len(chosen.split()))
    right = max(1, len(rejected.split()))
    return max(left, right) / min(left, right) <= max_ratio


class QualityGate:
    BANNED_PHRASES = (
        "the passage", "this passage", "the chunk", "the text above", "the excerpt",
        "the document above", "according to the text", "the provided context",
        "the above", "the following text", "the section",
    )
    REFERENCE_HINTS = ("et al.", "doi:", "arxiv:", "pp.", "vol.", "proceedings of", "isbn")

    def __init__(self, min_words: int, min_alpha_ratio: float, min_grounding: float):
        self.min_words = min_words
        self.min_alpha_ratio = min_alpha_ratio
        self.min_grounding = min_grounding

    def accept_context(self, text: str) -> tuple[bool, str]:
        if len(text.split()) < self.min_words:
            return False, "too_short"
        if self._alpha_ratio(text) < self.min_alpha_ratio:
            return False, "symbol_heavy"
        if self._looks_like_references(text):
            return False, "reference_list"
        if text.count(".") < 2:
            return False, "no_sentence_structure"
        return True, "ok"

    def accept_question(self, question: str) -> tuple[bool, str]:
        stripped = question.strip()
        if len(stripped.split()) < 5:
            return False, "question_too_short"
        if not stripped.endswith("?"):
            return False, "not_interrogative"
        lowered = stripped.lower()
        if any(phrase in lowered for phrase in self.BANNED_PHRASES):
            return False, "context_dependent_question"
        return True, "ok"

    def accept_answer(self, answer: str, context: str) -> tuple[bool, str]:
        if len(answer.split()) < 8:
            return False, "answer_too_short"
        if not self._numbers_supported(answer, context):
            return False, "unsupported_numbers"
        if self._grounding(answer, context) < self.min_grounding:
            return False, "weak_grounding"
        return True, "ok"

    def accept_summary(self, summary: str, source: str) -> tuple[bool, str]:
        if len(summary.split()) < 30:
            return False, "summary_too_short"
        if len(summary.split()) > len(source.split()) * 0.6:
            return False, "insufficient_compression"
        if not self._numbers_supported(summary, source):
            return False, "unsupported_numbers"
        if self._grounding(summary, source) < self.min_grounding:
            return False, "weak_grounding"
        return True, "ok"

    def _alpha_ratio(self, text: str) -> float:
        clean = sum(1 for ch in text if ch.isalpha() or ch.isspace())
        return clean / max(1, len(text))

    def _looks_like_references(self, text: str) -> bool:
        lowered = text.lower()
        hits = sum(lowered.count(hint) for hint in self.REFERENCE_HINTS)
        per_hundred_words = hits / max(1.0, len(text.split()) / 100)
        return per_hundred_words >= 2.0

    def _numbers_supported(self, candidate: str, source: str) -> bool:
        found = set(re.findall(r"\d+(?:\.\d+)?", candidate))
        available = set(re.findall(r"\d+(?:\.\d+)?", source))
        return found.issubset(available)

    def _grounding(self, candidate: str, source: str) -> float:
        terms = content_terms(candidate)
        if not terms:
            return 0.0
        return len(terms & content_terms(source)) / len(terms)


class ChunkPool:
    def __init__(self, path: Path, gate: QualityGate, rng: random.Random,
                 state: RunState, skip_writer: JsonlWriter):
        self.gate = gate
        self.rng = rng
        self.state = state
        self.skip_writer = skip_writer
        self.records: dict[str, dict] = {}
        self.by_source: dict[str, list[str]] = defaultdict(list)
        self._window_starts: dict[str, list[int]] = {}
        self._load(path)
        self._all_keys = list(self.records)
        self._chunk_queue = [key for key in self.records if key not in state.used]
        self.rng.shuffle(self._chunk_queue)

    def _load(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                self.records[record["chunk_id"]] = record
                self.by_source[record["source"]].append(record["chunk_id"])
        for source in self.by_source:
            self.by_source[source].sort(key=lambda key: self.records[key]["ordinal"])

    def sources(self) -> list[str]:
        names = list(self.by_source)
        self.rng.shuffle(names)
        return names

    def draw_chunk(self) -> dict | None:
        while self._chunk_queue:
            key = self._chunk_queue.pop()
            if key in self.state.used:
                continue
            self.state.used.add(key)
            record = self.records[key]
            accepted, reason = self.gate.accept_context(record["chunk_text"])
            if not accepted:
                self.log_skip(key, reason)
                continue
            return record
        return None

    def draw_window(self, source: str, size: int) -> tuple[list[dict], str] | None:
        starts = self._starts_for(source, size)
        while starts:
            start = starts.pop()
            keys = self.by_source[source][start:start + size]
            if len(keys) < size or any(key in self.state.used for key in keys):
                continue
            self.state.used.update(keys)
            window = [self.records[key] for key in keys]
            text = merge_chunks(window)
            accepted, reason = self.gate.accept_context(text)
            if not accepted:
                self.log_skip(keys[0], reason)
                continue
            return window, text
        return None

    def random_context(self, exclude_source: str, rng: random.Random) -> str | None:
        for _ in range(25):
            record = self.records[rng.choice(self._all_keys)]
            if record["source"] == exclude_source:
                continue
            accepted, _ = self.gate.accept_context(record["chunk_text"])
            if accepted:
                return record["chunk_text"]
        return None

    def _starts_for(self, source: str, size: int) -> list[int]:
        if source not in self._window_starts:
            total = len(self.by_source[source])
            starts = list(range(0, max(0, total - size + 1), size))
            self.rng.shuffle(starts)
            self._window_starts[source] = starts
        return self._window_starts[source]

    def log_skip(self, key: str, reason: str) -> None:
        self.skip_writer.write({"chunk_id": key, "reason": reason})
        self.state.skip_count += 1


class PromptCorruptor:
    ADJACENT = {
        "a": "qsz", "b": "vgn", "c": "xdv", "d": "sfce", "e": "wrds", "f": "dgrv",
        "g": "fhtb", "h": "gjyn", "i": "uok", "j": "hkun", "k": "jlim", "l": "kop",
        "m": "njk", "n": "bmhj", "o": "ipl", "p": "ol", "q": "wa", "r": "etf",
        "s": "adwx", "t": "ryg", "u": "yij", "v": "cfb", "w": "qes", "x": "zsc",
        "y": "tuh", "z": "asx",
    }
    FILLERS = ("um", "so", "hey", "ok so", "quick question", "i wanna know", "pls")
    CHATSPEAK = {
        "you": "u", "are": "r", "please": "pls", "because": "bc", "with": "w/",
        "your": "ur", "about": "abt", "between": "btwn", "for": "4",
    }
    CONTRACTIONS = {
        "what's": "whats", "doesn't": "doesnt", "don't": "dont", "it's": "its",
        "how's": "hows", "isn't": "isnt", "can't": "cant", "won't": "wont",
    }

    def __init__(self):
        self.operations = (
            self._keyboard_typo, self._drop_character, self._transpose,
            self._double_character, self._lowercase, self._strip_terminal,
            self._add_filler, self._break_contraction, self._chatspeak,
            self._double_space,
        )

    def corrupt(self, text: str, rng: random.Random) -> str | None:
        for _ in range(6):
            candidate = text
            for operation in rng.sample(self.operations, rng.randint(2, 4)):
                candidate = operation(candidate, rng)
            if candidate != text and candidate.strip():
                return candidate
        return None

    def _pick_index(self, text: str, rng: random.Random) -> int | None:
        positions = [i for i, ch in enumerate(text) if ch.lower() in self.ADJACENT]
        return rng.choice(positions) if positions else None

    def _keyboard_typo(self, text: str, rng: random.Random) -> str:
        index = self._pick_index(text, rng)
        if index is None:
            return text
        replacement = rng.choice(self.ADJACENT[text[index].lower()])
        return text[:index] + replacement + text[index + 1:]

    def _drop_character(self, text: str, rng: random.Random) -> str:
        index = self._pick_index(text, rng)
        return text if index is None else text[:index] + text[index + 1:]

    def _transpose(self, text: str, rng: random.Random) -> str:
        index = self._pick_index(text, rng)
        if index is None or index + 1 >= len(text):
            return text
        return text[:index] + text[index + 1] + text[index] + text[index + 2:]

    def _double_character(self, text: str, rng: random.Random) -> str:
        index = self._pick_index(text, rng)
        return text if index is None else text[:index + 1] + text[index] + text[index + 1:]

    def _lowercase(self, text: str, rng: random.Random) -> str:
        return text.lower()

    def _strip_terminal(self, text: str, rng: random.Random) -> str:
        return text.rstrip("?.! ")

    def _add_filler(self, text: str, rng: random.Random) -> str:
        return f"{rng.choice(self.FILLERS)} {text}"

    def _break_contraction(self, text: str, rng: random.Random) -> str:
        for full, broken in self.CONTRACTIONS.items():
            if full in text.lower():
                return re.sub(full, broken, text, flags=re.IGNORECASE)
        return text

    def _chatspeak(self, text: str, rng: random.Random) -> str:
        words = text.split()
        for index, word in enumerate(words):
            stripped = word.lower().strip(".,?!")
            if stripped in self.CHATSPEAK and rng.random() < 0.6:
                words[index] = self.CHATSPEAK[stripped]
        return " ".join(words)

    def _double_space(self, text: str, rng: random.Random) -> str:
        words = text.split()
        if len(words) < 3:
            return text
        index = rng.randrange(1, len(words))
        return " ".join(words[:index]) + "  " + " ".join(words[index:])


class RejectedBuilder:
    def __init__(self, teacher: ThreadLocalAgent, pool: ChunkPool):
        self.teacher = teacher
        self.pool = pool

    def for_answer(self, question: str, chosen: str, source: str, rng: random.Random) -> str | None:
        strategy = rng.choices(
            ["off_context", "unsupported", "numeric_drift", "hedged"],
            weights=[0.35, 0.35, 0.20, 0.10],
        )[0]

        if strategy == "off_context":
            distractor = self.pool.random_context(exclude_source=source, rng=rng)
            if not distractor:
                return None
            return self.teacher.agent.generate_text(
                "You answer questions using only the context you are given, even when it "
                "is a poor match for the question. Answer confidently in two or three "
                "sentences. Never say the context is insufficient.",
                f"CONTEXT:\n{distractor}\n\nQUESTION:\n{question}",
                temperature=0.7,
                max_tokens=300,
            )

        if strategy == "unsupported":
            return self.teacher.agent.generate_text(
                "You rewrite an answer so that it keeps the same length and tone but "
                "introduces one confident factual claim that the source never made, such "
                "as an invented benchmark score, dataset, or citation. Output only the "
                "rewritten answer.",
                f"ANSWER:\n{chosen}",
                temperature=0.8,
                max_tokens=400,
            )

        if strategy == "numeric_drift":
            return self._drift_numbers(chosen, rng)

        return self.teacher.agent.generate_text(
            "You rewrite an answer so that it becomes evasive. Keep roughly the same "
            "length, but replace every specific fact with a vague generality and hedge "
            "each remaining claim. Output only the rewritten answer.",
            f"ANSWER:\n{chosen}",
            temperature=0.7,
            max_tokens=400,
        )

    def for_summary(self, chosen: str, source: str, rng: random.Random) -> str | None:
        strategy = rng.choices(
            ["verbatim", "meta_commentary", "hallucinated"],
            weights=[0.35, 0.30, 0.35],
        )[0]

        if strategy == "verbatim":
            sentences = re.split(r"(?<=[.!?])\s+", source)
            budget = max(3, len(chosen.split()) // 18)
            extract = " ".join(sentences[:budget]).strip()
            return extract if extract else None

        if strategy == "meta_commentary":
            return self.teacher.agent.generate_text(
                "You rewrite a summary so that every sentence is framed as commentary "
                "about the source document rather than a direct statement of its content. "
                "Begin with a phrase such as 'This section discusses'. Output only the "
                "rewritten summary.",
                f"SUMMARY:\n{chosen}",
                temperature=0.7,
                max_tokens=400,
            )

        return self.teacher.agent.generate_text(
            "You rewrite a summary so that it keeps the same length but adds two claims "
            "that the source never made, such as background context or a comparison to "
            "other work. Output only the rewritten summary.",
            f"SUMMARY:\n{chosen}",
            temperature=0.8,
            max_tokens=400,
        )

    def for_synthesis(self, chosen: str, summaries: list[str], rng: random.Random) -> str | None:
        strategy = rng.choices(
            ["bulleted", "concatenated", "contradiction"],
            weights=[0.35, 0.35, 0.30],
        )[0]

        if strategy == "bulleted":
            return self.teacher.agent.generate_text(
                "You convert a piece of continuous prose into a bulleted list with a "
                "heading above each group of bullets. Preserve all content. Output only "
                "the list.",
                f"PROSE:\n{chosen}",
                temperature=0.4,
                max_tokens=700,
            )

        if strategy == "concatenated":
            connectives = ["Additionally,", "Furthermore,", "Moreover,", "In addition,"]
            parts = [summaries[0]]
            for summary in summaries[1:]:
                parts.append(f"{rng.choice(connectives)} {summary}")
            return " ".join(parts)

        return self.teacher.agent.generate_text(
            "You rewrite a passage so that it keeps the same length and structure but "
            "introduces one statement that contradicts an earlier statement in the same "
            "passage. Output only the rewritten passage.",
            f"PASSAGE:\n{chosen}",
            temperature=0.8,
            max_tokens=700,
        )

    def _drift_numbers(self, text: str, rng: random.Random) -> str | None:
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if not numbers:
            return None
        target = rng.choice(numbers)
        if "." in target:
            drifted = f"{float(target) * rng.uniform(1.2, 1.8):.2f}"
        else:
            drifted = str(int(int(target) * rng.uniform(1.3, 2.0)) + 1)
        return text.replace(target, drifted, 1)


def run_concurrent(draw, process, emit, on_skip, target, initial, desc,
                    shutdown, checkpoint, workers):
    accepted = initial
    progress = tqdm(total=target, initial=initial, desc=desc)
    since_checkpoint = 0
    exhausted = False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}

        def fill() -> None:
            nonlocal exhausted
            while (
                not exhausted
                and not shutdown.requested
                and len(pending) < workers
                and accepted + len(pending) < target
            ):
                item = draw()
                if item is None:
                    exhausted = True
                    return
                pending[executor.submit(process, item)] = item

        fill()

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)

            for future in done:
                item = pending.pop(future)

                try:
                    payload, reason = future.result()
                except Exception as error:
                    on_skip(item, f"worker_error:{type(error).__name__}")
                    continue

                if payload is None:
                    on_skip(item, reason)
                    continue

                emit(payload)
                accepted += 1
                progress.update(1)

                since_checkpoint += 1
                if since_checkpoint >= CONFIG["CHECKPOINT_EVERY"]:
                    checkpoint()
                    since_checkpoint = 0

            if accepted < target and not shutdown.requested:
                fill()

    progress.close()
    return accepted, exhausted


class AnswerGenerator:
    def __init__(self, pool, teacher, answer_agent, gate, rejector, state, shutdown, checkpoint):
        self.pool = pool
        self.teacher = teacher
        self.answer_agent = answer_agent
        self.gate = gate
        self.rejector = rejector
        self.state = state
        self.shutdown = shutdown
        self.checkpoint = checkpoint
        self.system_prompt = answer_agent.agent_system_prompt

    def run(self, budget: int, sft: JsonlWriter, dpo: JsonlWriter) -> None:
        if self.state.stage_done("answer"):
            return

        def emit(payload: dict) -> None:
            for record in payload["sft"]:
                sft.write(record)
            for record in payload["dpo"]:
                dpo.write(record)
            self.state.questions.extend(payload["questions"])
            self.state.stages["answer"]["accepted"] += 1

        def on_skip(chunk: dict, reason: str) -> None:
            self.pool.log_skip(chunk["chunk_id"], reason)

        accepted, exhausted = run_concurrent(
            draw=self.pool.draw_chunk,
            process=self.process,
            emit=emit,
            on_skip=on_skip,
            target=budget,
            initial=self.state.stages["answer"]["accepted"],
            desc="answer",
            shutdown=self.shutdown,
            checkpoint=self.checkpoint,
            workers=CONFIG["CONCURRENCY"],
        )

        self.state.stages["answer"]["accepted"] = accepted

        if not self.shutdown.requested:
            self.state.mark_done("answer")

        self.checkpoint()

    def process(self, chunk: dict) -> tuple[dict | None, str]:
        rng = derive_rng(CONFIG["SEED"], chunk["chunk_id"])

        payload = self.teacher.agent.generate_json(
            TEACHER_QA_SYSTEM,
            f"EXCERPT:\n{chunk['chunk_text']}",
            QUESTION_SCHEMA,
            temperature=0.3,
        )

        if not payload or not payload.get("questions"):
            return None, "no_questions_generated"

        kept = self._answer_questions(payload["questions"], chunk)
        if not kept:
            return None, "all_pairs_rejected"

        sft_records = []
        dpo_records = []
        questions = []

        for question, answer in kept:
            questions.append(question)
            context = self._build_context(chunk["chunk_text"], chunk["source"], rng)
            user_input = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

            sft_records.append({
                "system": self.system_prompt,
                "instruction": ANSWER_INSTRUCTION,
                "input": user_input,
                "output": answer,
                "meta": {"source": chunk["source"], "chunk_id": chunk["chunk_id"]},
            })

            if rng.random() >= CONFIG["DPO_RATE"]:
                continue

            rejected = self.rejector.for_answer(question, answer, chunk["source"], rng)
            if not rejected or rejected.strip() == answer.strip():
                continue
            if not length_balanced(answer, rejected, CONFIG["MAX_LENGTH_RATIO"]):
                continue

            dpo_records.append({
                "system": self.system_prompt,
                "instruction": ANSWER_INSTRUCTION,
                "input": user_input,
                "chosen": answer,
                "rejected": rejected.strip(),
                "meta": {"source": chunk["source"], "chunk_id": chunk["chunk_id"]},
            })

        if rng.random() < CONFIG["UNANSWERABLE_RATE"]:
            abstention = self._build_abstention(chunk, kept[0][0], rng)
            if abstention:
                sft_records.append(abstention)

        return {"sft": sft_records, "dpo": dpo_records, "questions": questions}, "ok"

    def _answer_questions(self, questions: list[str], chunk: dict) -> list[tuple[str, str]]:
        kept = []
        for question in questions[:CONFIG["PAIRS_PER_CHUNK"]]:
            question = question.strip()
            question_ok, _ = self.gate.accept_question(question)
            if not question_ok:
                continue

            answer = self.answer_agent.agent.generate_response(
                f"CONTEXT:\n[1] {chunk['chunk_text']}\n\nQUESTION:\n{question}"
            )
            if not answer:
                continue

            answer = answer.strip()
            answer_ok, _ = self.gate.accept_answer(answer, chunk["chunk_text"])
            if not answer_ok:
                continue

            kept.append((question, answer))
        return kept

    def _build_context(self, gold: str, source: str, rng: random.Random) -> str:
        blocks = [gold]
        if rng.random() < CONFIG["DISTRACTOR_RATE"]:
            for _ in range(rng.randint(1, 2)):
                distractor = self.pool.random_context(exclude_source=source, rng=rng)
                if distractor:
                    blocks.append(distractor)
        rng.shuffle(blocks)
        return "\n\n".join(f"[{i + 1}] {block}" for i, block in enumerate(blocks))

    def _build_abstention(self, chunk: dict, question: str, rng: random.Random) -> dict | None:
        blocks = []
        for _ in range(2):
            distractor = self.pool.random_context(exclude_source=chunk["source"], rng=rng)
            if distractor:
                blocks.append(distractor)
        if not blocks:
            return None

        context = "\n\n".join(f"[{i + 1}] {block}" for i, block in enumerate(blocks))
        return {
            "system": self.system_prompt,
            "instruction": ANSWER_INSTRUCTION,
            "input": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}",
            "output": ABSTENTION,
            "meta": {"source": chunk["source"], "kind": "abstention"},
        }


class SummarizerGenerator:
    def __init__(self, pool, summarizer_agent, gate, rejector, state, shutdown, checkpoint):
        self.pool = pool
        self.summarizer_agent = summarizer_agent
        self.gate = gate
        self.rejector = rejector
        self.state = state
        self.shutdown = shutdown
        self.checkpoint = checkpoint
        self.system_prompt = summarizer_agent.agent_system_prompt

    def run(self, budget: int, sft: JsonlWriter, dpo: JsonlWriter) -> None:
        if self.state.stage_done("summarizer"):
            return

        window_size = CONFIG["WINDOW_SIZE"]
        target_windows = budget // window_size
        draw = self._make_draw(window_size)

        def emit(payload: dict) -> None:
            sft.write(payload["sft"])
            if payload["dpo"]:
                dpo.write(payload["dpo"])
            self.state.bank[payload["source"]].append(payload["summary"])
            self.state.stages["summarizer"]["accepted"] += 1

        def on_skip(item, reason: str) -> None:
            window, _ = item
            self.pool.log_skip(window[0]["chunk_id"], reason)

        accepted, exhausted = run_concurrent(
            draw=draw,
            process=self.process,
            emit=emit,
            on_skip=on_skip,
            target=target_windows,
            initial=self.state.stages["summarizer"]["accepted"],
            desc="summarizer",
            shutdown=self.shutdown,
            checkpoint=self.checkpoint,
            workers=CONFIG["CONCURRENCY"],
        )

        self.state.stages["summarizer"]["accepted"] = accepted

        if not self.shutdown.requested:
            self.state.mark_done("summarizer")

        self.checkpoint()

    def _make_draw(self, window_size: int):
        def generator():
            for source in self.pool.sources():
                for _ in range(CONFIG["WINDOWS_PER_SOURCE"]):
                    drawn = self.pool.draw_window(source, window_size)
                    if drawn is None:
                        break
                    yield drawn

        iterator = generator()

        def draw():
            return next(iterator, None)

        return draw

    def process(self, item) -> tuple[dict | None, str]:
        window, text = item
        source = window[0]["source"]
        rng = derive_rng(CONFIG["SEED"], window[0]["chunk_id"])

        summary = self.summarizer_agent.agent.generate_response(f"SECTION TEXT:\n{text}")
        if not summary:
            return None, "no_summary_generated"

        summary = summary.strip()
        accepted, reason = self.gate.accept_summary(summary, text)
        if not accepted:
            return None, reason

        user_input = f"SECTION TEXT:\n{text}"
        sft_record = {
            "system": self.system_prompt,
            "instruction": SUMMARY_INSTRUCTION,
            "input": user_input,
            "output": summary,
            "meta": {"source": source},
        }

        dpo_record = None
        if rng.random() < CONFIG["DPO_RATE"]:
            rejected = self.rejector.for_summary(summary, text, rng)
            if (
                rejected
                and rejected.strip() != summary.strip()
                and length_balanced(summary, rejected, CONFIG["MAX_LENGTH_RATIO"])
            ):
                dpo_record = {
                    "system": self.system_prompt,
                    "instruction": SUMMARY_INSTRUCTION,
                    "input": user_input,
                    "chosen": summary,
                    "rejected": rejected.strip(),
                    "meta": {"source": source},
                }

        return {
            "source": source,
            "summary": summary,
            "sft": sft_record,
            "dpo": dpo_record,
        }, "ok"


class SynthesizerGenerator:
    def __init__(self, synthesizer_agent, rejector, state, shutdown, checkpoint):
        self.synthesizer_agent = synthesizer_agent
        self.rejector = rejector
        self.state = state
        self.shutdown = shutdown
        self.checkpoint = checkpoint
        self.system_prompt = synthesizer_agent.agent_system_prompt

    def run(self, sft: JsonlWriter, dpo: JsonlWriter) -> None:
        if self.state.stage_done("synthesizer"):
            return

        group_size = CONFIG["SYNTHESIS_GROUP"]
        eligible = [
            source for source, items in sorted(self.state.bank.items())
            if len(items) >= group_size and source not in self.state.synthesised_sources
        ]
        iterator = iter(eligible)

        def draw():
            return next(iterator, None)

        def emit(payload: dict) -> None:
            for record in payload["sft"]:
                sft.write(record)
            for record in payload["dpo"]:
                dpo.write(record)
            self.state.synthesised_sources.add(payload["source"])
            self.state.stages["synthesizer"]["accepted"] += 1

        def on_skip(source: str, reason: str) -> None:
            self.state.synthesised_sources.add(source)

        completed = self.state.stages["synthesizer"]["accepted"]
        accepted, exhausted = run_concurrent(
            draw=draw,
            process=self.process,
            emit=emit,
            on_skip=on_skip,
            target=len(eligible) + completed,
            initial=completed,
            desc="synthesizer",
            shutdown=self.shutdown,
            checkpoint=self.checkpoint,
            workers=CONFIG["CONCURRENCY"],
        )

        self.state.stages["synthesizer"]["accepted"] = accepted

        if not self.shutdown.requested:
            self.state.mark_done("synthesizer")

        self.checkpoint()

    def process(self, source: str) -> tuple[dict | None, str]:
        group_size = CONFIG["SYNTHESIS_GROUP"]
        rng = derive_rng(CONFIG["SEED"], f"synthesis:{source}")

        summaries = list(self.state.bank[source])
        rng.shuffle(summaries)

        sft_records = []
        dpo_records = []

        for start in range(0, len(summaries) - group_size + 1, group_size):
            group = summaries[start:start + group_size]
            user_input = "SUMMARIES:\n" + "\n".join(
                f"[{i + 1}] {item}" for i, item in enumerate(group)
            )

            synthesis = self.synthesizer_agent.agent.generate_response(user_input)
            if not synthesis or len(synthesis.split()) < 80:
                continue

            synthesis = synthesis.strip()

            sft_records.append({
                "system": self.system_prompt,
                "instruction": SYNTHESIS_INSTRUCTION,
                "input": user_input,
                "output": synthesis,
                "meta": {"source": source},
            })

            if rng.random() >= CONFIG["DPO_RATE"]:
                continue

            rejected = self.rejector.for_synthesis(synthesis, group, rng)
            if not rejected or rejected.strip() == synthesis.strip():
                continue
            if not length_balanced(synthesis, rejected, CONFIG["MAX_LENGTH_RATIO"]):
                continue

            dpo_records.append({
                "system": self.system_prompt,
                "instruction": SYNTHESIS_INSTRUCTION,
                "input": user_input,
                "chosen": synthesis,
                "rejected": rejected.strip(),
                "meta": {"source": source},
            })

        if not sft_records:
            return None, "no_synthesis_generated"

        return {"source": source, "sft": sft_records, "dpo": dpo_records}, "ok"


class RewriterGenerator:
    def __init__(self, corruptor, system_prompt, state, shutdown, checkpoint):
        self.corruptor = corruptor
        self.system_prompt = system_prompt
        self.state = state
        self.shutdown = shutdown
        self.checkpoint = checkpoint

    def run(self, limit: int, sft: JsonlWriter) -> None:
        if self.state.stage_done("prompt_rewriter"):
            return

        pending = [
            question.strip() for question in self.state.questions[:limit]
            if question.strip() not in self.state.rewriter_seen
        ]
        completed = self.state.stages["prompt_rewriter"]["accepted"]
        progress = tqdm(total=len(pending) + completed, initial=completed, desc="prompt_rewriter")
        since_checkpoint = 0

        for clean in pending:
            if self.shutdown.requested:
                break

            rng = derive_rng(CONFIG["SEED"], f"rewrite:{clean}")
            corrupted = self.corruptor.corrupt(clean, rng)
            if not corrupted:
                continue

            self.state.rewriter_seen.add(clean)

            sft.write({
                "system": self.system_prompt,
                "instruction": REWRITE_INSTRUCTION,
                "input": corrupted,
                "output": clean,
                "meta": {"kind": "synthetic_corruption"},
            })

            completed += 1
            self.state.stages["prompt_rewriter"]["accepted"] = completed
            progress.update(1)

            since_checkpoint += 1
            if since_checkpoint >= CONFIG["CHECKPOINT_EVERY"]:
                self.checkpoint()
                since_checkpoint = 0

        progress.close()

        if not self.shutdown.requested:
            self.state.mark_done("prompt_rewriter")

        self.checkpoint()



def report_progress(state: RunState, writers: dict[str, JsonlWriter]) -> None:
    chunk_budget = CONFIG["CHUNK_BUDGET"]
    window_size = CONFIG["WINDOW_SIZE"]

    sampled = {
        "answer": state.stages["answer"]["accepted"],
        "summarizer": state.stages["summarizer"]["accepted"] * window_size,
    }

    rows = [
        ("answer", writers["answer_sft"].count + writers["answer_dpo"].count,
         sampled["answer"], chunk_budget["answer"]),
        ("summarizer", writers["summarizer_sft"].count + writers["summarizer_dpo"].count,
         sampled["summarizer"], chunk_budget["summarizer"]),
        ("synthesizer", writers["synthesizer_sft"].count + writers["synthesizer_dpo"].count,
         None, None),
        ("prompt_rewriter", writers["prompt_rewriter_sft"].count, None, None),
    ]

    total_records = sum(row[1] for row in rows)
    total_sampled = sum(sampled.values())
    total_budget = sum(chunk_budget.values())

    print()
    print(f"{'stage':<18}{'records':>12}{'sampled / budget':>26}{'status':>12}")
    print("-" * 68)

    for name, records, done, budget in rows:
        sample_cell = "derived" if budget is None else f"{done:,} / {budget:,}"
        status = "done" if state.stage_done(name) else "partial"
        print(f"{name:<18}{records:>12,}{sample_cell:>26}{status:>12}")

    print("-" * 68)
    print(f"{'TOTAL':<18}{total_records:>12,}{f'{total_sampled:,} / {total_budget:,}':>26}")
    print()
    print(f"Chunks consumed: {len(state.used):,}")
    print(f"Chunks skipped:  {state.skip_count:,}")
    print(f"Concurrency:     {CONFIG['CONCURRENCY']}")


def main() -> None:
    rng = random.Random(CONFIG["SEED"])
    state = RunState(CONFIG["CHECKPOINT_FILE"], rng)
    state.load()
    shutdown = ShutdownSignal()

    output_dir = CONFIG["OUTPUT_DIR"]
    output_dir.mkdir(parents=True, exist_ok=True)

    writers = {
        name: JsonlWriter(writer_path(output_dir, name), state.writer_counts.get(name, 0))
        for name in WRITER_NAMES
    }
    skip_writer = JsonlWriter(CONFIG["SKIP_LOG"], state.skip_count)

    def checkpoint() -> None:
        for name, writer in writers.items():
            writer.flush()
            state.writer_counts[name] = writer.count
        skip_writer.flush()
        state.save()

    if state.resumed:
        print(f"Resumed from {CONFIG['CHECKPOINT_FILE']}")
        report_progress(state, writers)
    else:
        print("Starting a fresh run.")

    gate = QualityGate(CONFIG["MIN_WORDS"], CONFIG["MIN_ALPHA_RATIO"], CONFIG["MIN_GROUNDING"])
    pool = ChunkPool(CONFIG["CHUNK_FILE"], gate, rng, state, skip_writer)

    teacher = ThreadLocalAgent(str(CONFIG["TEACHER_AGENT_CONFIG"]), factory=TeacherAgent)
    answer_agent = ThreadLocalAgent(str(CONFIG["ANSWER_AGENT_CONFIG"]))
    summarizer_agent = ThreadLocalAgent(str(CONFIG["SUMMARIZER_AGENT_CONFIG"]))
    synthesizer_agent = ThreadLocalAgent(str(CONFIG["SYNTHESIZER_AGENT_CONFIG"]))
    rewriter_prompt = load_prompt_text(CONFIG["PROMPT_REWRITER_AGENT_CONFIG"])

    rejector = RejectedBuilder(teacher, pool)
    corruptor = PromptCorruptor()

    AnswerGenerator(
        pool, teacher, answer_agent, gate, rejector, state, shutdown, checkpoint
    ).run(CONFIG["CHUNK_BUDGET"]["answer"], writers["answer_sft"], writers["answer_dpo"])

    if not shutdown.requested:
        SummarizerGenerator(
            pool, summarizer_agent, gate, rejector, state, shutdown, checkpoint
        ).run(CONFIG["CHUNK_BUDGET"]["summarizer"], writers["summarizer_sft"], writers["summarizer_dpo"])

    if not shutdown.requested:
        SynthesizerGenerator(
            synthesizer_agent, rejector, state, shutdown, checkpoint
        ).run(writers["synthesizer_sft"], writers["synthesizer_dpo"])

    if not shutdown.requested:
        RewriterGenerator(
            corruptor, rewriter_prompt, state, shutdown, checkpoint
        ).run(CONFIG["REWRITER_MAX_RECORDS"], writers["prompt_rewriter_sft"])

    checkpoint()
    report_progress(state, writers)

    for writer in writers.values():
        writer.close()
    skip_writer.close()

    if shutdown.requested:
        print("Stopped early. Rerun the same command to resume from this checkpoint.")
    elif all(state.stage_done(name) for name in STAGE_NAMES):
        print("All stages complete.")


if __name__ == "__main__":
    main()