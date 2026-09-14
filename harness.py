"""Prompts, parsers, response cache, and the offline mock client.

evaluate.py doesn't know anything about prompt text or about a specific API. It just
builds a prompt, hands it to a client, and parses whatever comes back. The live client
lives in providers.py.
"""
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import NamedTuple

_NOTIFIED = set()


def notify(msg):
    """Print an operational notice to stderr, once per distinct message.

    stdout carries the report and must stay pipeable. PROMPTBENCH_QUIET silences these.
    """
    if msg in _NOTIFIED or os.environ.get("PROMPTBENCH_QUIET"):
        return
    _NOTIFIED.add(msg)
    print(msg, file=sys.stderr)


CATEGORIES = ["billing", "technical", "account", "other"]
CATLIST = ", ".join(CATEGORIES)
SYSTEM = "You are a support ticket triage classifier."

# Held out of dataset.csv. Reusing scored items as demonstrations inflates few_shot.
FEWSHOT = [
    ("My renewal charged me before the trial ended.", "billing"),
    ("The mobile app freezes on the login screen.", "technical"),
    ("I need to reset the security questions on my profile.", "account"),
]


def zero_shot(msg):
    return (f"Classify this support message into exactly one of: {CATLIST}.\n"
            f"Reply with the category word only, nothing else.\n\nMessage: {msg}")


def few_shot(msg):
    shots = "\n".join(f"Message: {m}\nCategory: {c}" for m, c in FEWSHOT)
    return (f"Classify each support message into exactly one of: {CATLIST}.\n\n"
            f"{shots}\n\nMessage: {msg}\nCategory:")


def few_shot_instruct(msg):
    """few_shot's examples in instruction form. Control for prompt format vs examples."""
    shots = "\n".join(f"Message: {m}\nCategory: {c}" for m, c in FEWSHOT)
    return (f"Here are three examples of support messages and their categories.\n\n{shots}\n\n"
            f"Now classify this message into exactly one of: {CATLIST}.\n"
            f"Reply with the category word only, nothing else.\n\nMessage: {msg}")


def chain_of_thought(msg):
    return (f"Classify this support message into exactly one of: {CATLIST}.\n"
            f"Think step by step in at most three short sentences, then output a final "
            f"line of exactly 'ANSWER: <category>'.\n\nMessage: {msg}")


def strict_json(msg):
    return (f"Classify this support message into exactly one of: {CATLIST}.\n"
            f'Respond with only a JSON object of the form {{"category": "<one of the four>"}}. '
            f"No markdown fences, no prose.\n\nMessage: {msg}")


# Parsers return (label_or_None, status). status is ok, hallucinated, malformed or empty.

def _classify_token(tok):
    tok = tok.strip().strip(".,\"'`*").lower()
    if not tok:
        return None, "malformed"
    if tok in CATEGORIES:
        return tok, "ok"
    # Short unknown token = invented label. Longer = format ignored. Separate buckets.
    return (tok, "hallucinated") if len(tok.split()) <= 2 else (tok, "malformed")


def parse_plain(text):
    if not text.strip():
        return None, "empty"
    last = [ln for ln in text.strip().splitlines() if ln.strip()][-1]
    last = re.sub(r"^\s*(category|answer)\s*:\s*", "", last, flags=re.I)
    return _classify_token(last)


def parse_cot(text):
    if not text.strip():
        return None, "empty"
    hits = re.findall(r"ANSWER\s*:\s*(\w[\w\- ]*)", text, flags=re.I)
    if not hits:
        return None, "malformed"
    return _classify_token(hits[-1])


def parse_json(text):
    text = text.strip()
    if not text:
        return None, "empty"
    # Fences are stripped before parsing. Leniency on packaging only; malformed
    # structure still fails. Note this favours strict_json in the comparison.
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None, "malformed"
    if not isinstance(obj, dict) or "category" not in obj:
        return None, "malformed"
    return _classify_token(str(obj["category"]))


class Strategy(NamedTuple):
    build: object
    parse: object
    max_tokens: int
    stop: tuple = ()   # sequences that end generation, for completion-style prompts


# Add a strategy by adding a row here.
STRATEGIES = {
    "zero_shot": Strategy(zero_shot, parse_plain, 16),
    # few_shot's prompt ends with "Category:", so generation continues into invented
    # examples unless stopped. Without these, llama3.2 scores 0.
    "few_shot": Strategy(few_shot, parse_plain, 32, ("\nMessage:", "\n\n")),
    "few_shot_instruct": Strategy(few_shot_instruct, parse_plain, 16),
    "chain_of_thought": Strategy(chain_of_thought, parse_cot, 300),
    "strict_json": Strategy(strict_json, parse_json, 64),
}


@dataclass
class Completion:
    text: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""
    cached: bool = False
    attempts: int = 1


class ResponseCache:
    """Disk cache, one file per response. Key covers every input affecting the output."""

    def __init__(self, directory=".cache"):
        self.dir = directory
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    def key(*parts):
        raw = "|".join(str(p) for p in parts).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:32]

    def _path(self, key):
        return os.path.join(self.dir, f"{key}.json")

    def get(self, key):
        try:
            with open(self._path(key), encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        return Completion(text=d["text"], input_tokens=d.get("input_tokens", 0),
                          output_tokens=d.get("output_tokens", 0), cached=True)

    def put(self, key, completion):
        tmp = self._path(key) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"text": completion.text, "input_tokens": completion.input_tokens,
                       "output_tokens": completion.output_tokens}, f)
        os.replace(tmp, self._path(key))  # atomic, so no reader sees half a file


class MockClient:
    """Offline client for --dry-run.

    Keyword-classifies the message, formats per strategy, and injects all four failure
    modes on a seeded schedule. Accuracy reflects the keyword table below, not a model.
    """

    KEYWORDS = {  # order matters: billing wins on "money left my bank account"
        "billing": ["charge", "invoice", "payment", "bill", "price", "pricing", "refund",
                    "receipt", "card", "subscription"],
        "technical": ["crash", "error", "load", "broken", "sync", "export", "timeout",
                      "search", "notification", "webhook"],
        "account": ["password", "account", "email address", "two factor", "workspace",
                    "teammates", "signed in", "log in", "login", "device"],
    }

    def __init__(self, seed=7, latency=0.004):
        self.seed, self.latency = seed, latency
        self.reset()

    def reset(self, run=0):
        """Reseed. Same schedule for every strategy in a run, different across runs.

        Identical rows across strategies in a dry run are therefore expected.
        """
        self.rng = random.Random(self.seed + run)

    def _guess(self, msg):
        low = msg.lower()
        for cat, words in self.KEYWORDS.items():
            if any(w in low for w in words):
                return cat
        return "other"

    @staticmethod
    def _format(label, strategy):
        if strategy == "chain_of_thought":
            return f"The message points at a {label} concern.\nANSWER: {label}"
        if strategy == "strict_json":
            return json.dumps({"category": label})
        return label

    def complete(self, system, user, strategy, message, max_tokens, cache_key=None, stop=()):
        # Keyword match runs on `message`, not `user`: the prompt contains few_shot's
        # demonstrations, which would otherwise match.
        t0 = time.perf_counter()
        time.sleep(self.latency)
        label, roll = self._guess(message), self.rng.random()
        if roll < 0.04:
            text = ""
        elif roll < 0.09:
            # Formatted per strategy, or parse_cot and parse_json would read it as
            # malformed and never reach the hallucinated bucket.
            text = self._format("billing-and-payments", strategy)
        elif roll < 0.14:
            text = '{"category": ' if strategy == "strict_json" else "I am not sure about this one."
        elif roll < 0.19:
            wrong = CATEGORIES[(CATEGORIES.index(label) + 1) % len(CATEGORIES)]
            text = self._format(wrong, strategy)
        else:
            text = self._format(label, strategy)
        # Token counts left at zero; they feed the cost column and must not be invented.
        return Completion(text=text, latency_ms=(time.perf_counter() - t0) * 1000.0)
