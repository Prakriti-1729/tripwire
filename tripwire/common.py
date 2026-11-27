"""Shared paths and small I/O helpers."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ACTS = ROOT / "acts"
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_GUARD = "ibm-granite/granite-guardian-3.2-3b-a800m"
# The datasets that extract_acts.py processes (data/<name>.jsonl).
DATASETS = ["plain", "wrapped_test", "wrapped_aug", "xstest"]


def slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model_name.split("/")[-1].lower()).strip("-")


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def results_dir(model_name: str) -> Path:
    d = RESULTS / slug(model_name)
    d.mkdir(parents=True, exist_ok=True)
    return d
