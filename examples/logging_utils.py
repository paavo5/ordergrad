#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def infer_data_dir_from_output(output_path: Path, fallback: str = "examples/data") -> Path:
    parts = list(output_path.parts)
    if "artifacts" in parts:
        i = parts.index("artifacts")
        if i + 1 < len(parts):
            tag = parts[i + 1]
            return Path(*parts[:i], "data", tag) if i > 0 else Path("examples/data") / tag
    return Path(fallback)


def save_metadata_json(
    *,
    experiment: str,
    tag: str,
    setup: dict[str, Any],
    artifacts: dict[str, Any],
    data_dir: Path,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "experiment": experiment,
        "tag": tag,
        "setup": setup,
        "artifacts": artifacts,
        "cli": {
            "argv": sys.argv,
        },
    }
    p = data_dir / f"{experiment}__{tag}.json"
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved: {p}")
    return p
