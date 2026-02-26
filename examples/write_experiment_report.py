#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\\textbackslash{}",
        "&": r"\\&",
        "%": r"\\%",
        "$": r"\\$",
        "#": r"\\#",
        "_": r"\\_",
        "{": r"\\{",
        "}": r"\\}",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
    }
    out = text
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


def fig_block(path: str, caption: str, label: str) -> str:
    return f"""\\begin{{figure}}[t]
    \\centering
    \\includegraphics[width=0.95\\linewidth]{{{path}}}
    \\caption{{{latex_escape(caption)}}}
    \\label{{{label}}}
\\end{{figure}}
"""


def _format_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, list):
        if not v:
            return "[]"
        if len(v) <= 6:
            return "[" + ", ".join(_format_value(x) for x in v) + "]"
        head = ", ".join(_format_value(x) for x in v[:3])
        tail = ", ".join(_format_value(x) for x in v[-2:])
        return f"[{head}, ..., {tail}] (n={len(v)})"
    return str(v)


def _caption_from_metadata(meta: dict[str, Any]) -> str:
    experiment = str(meta.get("experiment", "unknown_experiment"))
    tag = str(meta.get("tag", "unknown_tag"))
    setup = meta.get("setup", {})
    if not isinstance(setup, dict):
        setup = {}

    key_priority = [
        "a",
        "k",
        "k_grid",
        "k_list",
        "quantile",
        "method",
        "methods",
        "N",
        "num_arms",
        "dim",
        "reward_mode",
        "objective",
        "dist",
        "num_mc",
        "t_grid",
        "seed",
    ]

    used: set[str] = set()
    pieces: list[str] = []
    for key in key_priority:
        if key in setup:
            used.add(key)
            pieces.append(f"{key}={_format_value(setup[key])}")

    for key in sorted(setup.keys()):
        if key not in used:
            pieces.append(f"{key}={_format_value(setup[key])}")

    details = "; ".join(pieces) if pieces else "No setup metadata found"
    return (
        f"Experiment '{experiment}' (tag='{tag}'). "
        f"Automatically extracted settings: {details}."
    )


def _load_metadata_maps(data_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_artifact_name: dict[str, dict[str, Any]] = {}
    by_tag_stem: dict[str, dict[str, Any]] = {}

    for jp in sorted(data_dir.glob("*.json")):
        try:
            meta = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue

        tag = meta.get("tag")
        exp = meta.get("experiment")
        if isinstance(tag, str) and isinstance(exp, str):
            by_tag_stem[f"{exp}__{tag}"] = meta

        artifacts = meta.get("artifacts", {})
        if isinstance(artifacts, dict):
            for _, value in artifacts.items():
                if isinstance(value, str) and value:
                    by_artifact_name[Path(value).name] = meta

    return by_artifact_name, by_tag_stem


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a LaTex report that includes generated experiment figures with metadata-rich captions.")
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--data-dir", default=None, help="Directory with experiment JSON metadata (defaults to examples/data/<artifact_dir_name> when available).")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    art = Path(args.art_dir)
    out = Path(args.output) if args.output else art / "report.tex"

    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
    else:
        data_dir = art.parent.parent / "data" / art.name

    by_artifact_name: dict[str, dict[str, Any]] = {}
    by_tag_stem: dict[str, dict[str, Any]] = {}
    if data_dir.exists():
        by_artifact_name, by_tag_stem = _load_metadata_maps(data_dir)

    figures = []

    pdfs = sorted(art.glob("*.pdf"))
    pngs = sorted(art.glob("*.png"))
    stems_with_pdf = {p.stem for p in pdfs}
    selected = list(pdfs) + [p for p in pngs if p.stem not in stems_with_pdf]

    for p in selected:
        meta = by_artifact_name.get(p.name) or by_tag_stem.get(p.stem)
        if meta is None:
            cap = (
                f"Auto-included figure '{p.name}'. "
                "No matching metadata JSON was found for this artifact."
            )
        else:
            cap = _caption_from_metadata(meta)
        label = f"fig:auto:{p.stem}"
        figures.append(fig_block(p.name, cap, label))

    tex = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{hyperref}
\title{OrderGrad Experiment Report}
\author{Auto-generated by run\_all\_experiments.sh}
\date{\today}
\begin{document}
\maketitle
\section*{Overview}
This document collects generated experiment figures and uses per-experiment JSON metadata to automatically enrich each caption with concrete run settings.
"""
    tex += "\n".join(figures)
    tex += "\n\\end{document}\n"

    out.write_text(tex, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
