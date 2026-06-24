"""Shared helpers: paths, tesseract config, IO."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import pytesseract

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_DIR = PROJECT_ROOT / "eval"
OUTPUTS_DIR.mkdir(exist_ok=True)


def configure_tesseract() -> None:
    """Point pytesseract at the system binary in a portable way.

    Order of precedence:
      1. TESSERACT_CMD env var (explicit full path)
      2. `tesseract` already on PATH (no-op)
      3. Common Windows install location
    """
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and Path(env_path).exists():
        pytesseract.pytesseract.tesseract_cmd = env_path
        return

    # On Linux/macOS, if tesseract is on PATH, pytesseract finds it by default.
    win_default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if win_default.exists():
        pytesseract.pytesseract.tesseract_cmd = str(win_default)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
