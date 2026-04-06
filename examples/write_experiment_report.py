#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


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


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return True
    return False


def _infer_lstat_summary(setup: dict[str, Any]) -> str:
    a_value = setup.get("a")
    if not _is_missing(a_value):
        return f"L-statistic preset/weights: a={_format_value(a_value)}"

    methods = setup.get("methods")
    method = setup.get("method")
    candidates: list[str] = []
    if isinstance(methods, list):
        candidates.extend(str(m) for m in methods)
    elif isinstance(methods, str):
        candidates.extend([m.strip() for m in methods.split(",") if m.strip()])
    if isinstance(method, str):
        candidates.append(method)

    if candidates:
        lstat_like = [m for m in candidates if any(k in m.lower() for k in ["quantile", "harrelldavis", "topm", "botm", "trim", "winsor", "lmoment", "gmd", "rank", "tailmean"]) ]
        if lstat_like:
            return f"L-statistic estimator methods: {_format_value(lstat_like)}"

    if "arm_rank" in setup:
        return "No L-statistic preset provided; this figure is configured by a fixed order-stat rank (arm_rank)"

    return "No explicit L-statistic preset metadata was provided for this run."


def _normalize_setup_for_caption(setup: dict[str, Any]) -> dict[str, Any]:
    out = dict(setup)
    # Some experiments store objective name instead of reward_mode.
    if 'reward_mode' not in out and 'objective' in out:
        out['reward_mode'] = out['objective']
    return out


def _caption_from_metadata(meta: dict[str, Any]) -> str:
    experiment = str(meta.get("experiment", "unknown_experiment"))
    tag = str(meta.get("tag", "unknown_tag"))
    setup = meta.get("setup", {})
    if not isinstance(setup, dict):
        setup = {}
    setup = _normalize_setup_for_caption(setup)

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
            if key == "a" and _is_missing(setup[key]):
                used.add(key)
                continue
            used.add(key)
            pieces.append(f"{key}={_format_value(setup[key])}")

    for key in sorted(setup.keys()):
        if key not in used:
            if key == "a" and _is_missing(setup[key]):
                continue
            pieces.append(f"{key}={_format_value(setup[key])}")

    details = "; ".join(pieces) if pieces else "No setup metadata found"
    return (
        f"Experiment '{experiment}' (tag='{tag}'). "
        f"Automatically extracted settings: {details}. "
        f"{_infer_lstat_summary(setup)}."
    )


def _load_metadata_maps(data_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_artifact_name: dict[str, dict[str, Any]] = {}
    by_tag_stem: dict[str, dict[str, Any]] = {}
    all_meta: list[dict[str, Any]] = []

    for jp in sorted(data_dir.glob("*.json")):
        try:
            meta = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        all_meta.append(meta)

        tag = meta.get("tag")
        exp = meta.get("experiment")
        if isinstance(tag, str) and isinstance(exp, str):
            by_tag_stem[f"{exp}__{tag}"] = meta

        artifacts = meta.get("artifacts", {})
        if isinstance(artifacts, dict):
            for _, value in artifacts.items():
                if isinstance(value, str) and value:
                    by_artifact_name[Path(value).name] = meta

    return by_artifact_name, by_tag_stem, all_meta


def _meta_matches_stem(meta: dict[str, Any], stem: str) -> bool:
    tag = meta.get("tag")
    exp = meta.get("experiment")
    if isinstance(tag, str) and tag == stem:
        return True
    if isinstance(tag, str) and stem.endswith(tag):
        return True
    if isinstance(tag, str) and isinstance(exp, str) and f"{exp}__{tag}" == stem:
        return True
    return False


def _summarize_combined_metadata(metas: list[dict[str, Any]], figure_stem: str) -> str:
    if not metas:
        return (
            f"Auto-included figure '{figure_stem}'. "
            "No matching metadata JSON was found for this artifact."
        )

    experiment = str(metas[0].get("experiment", "unknown_experiment"))
    tags = [str(m.get("tag", "unknown_tag")) for m in metas]
    setups = [_normalize_setup_for_caption(m.get("setup", {})) if isinstance(m.get("setup", {}), dict) else {} for m in metas]

    all_keys = sorted({k for s in setups for k in s.keys()})
    shared: dict[str, Any] = {}
    varying: dict[str, list[Any]] = {}
    for key in all_keys:
        vals = [s.get(key) for s in setups if key in s]
        if not vals:
            continue
        first = vals[0]
        if all(v == first for v in vals):
            shared[key] = first
        else:
            uniq: list[Any] = []
            for v in vals:
                if v not in uniq:
                    uniq.append(v)
            varying[key] = uniq

    shared_keys = ["a", "k", "N", "num_mc", "seed", "objective", "reward_mode", "prob_mode", "dist", "quantile", "method", "methods"]
    shared_parts = [f"{k}={_format_value(shared[k])}" for k in shared_keys if k in shared]
    shared_parts.extend(f"{k}={_format_value(shared[k])}" for k in sorted(shared) if k not in shared_keys)

    varying_keys = ["dim", "num_arms", "k", "k_grid", "k_list", "tag"]
    varying_parts = [f"{k}={_format_value(varying[k])}" for k in varying_keys if k in varying]
    varying_parts.extend(f"{k}={_format_value(varying[k])}" for k in sorted(varying) if k not in varying_keys)

    lstat_notes = [_infer_lstat_summary(s) for s in setups]
    uniq_notes: list[str] = []
    for note in lstat_notes:
        if note not in uniq_notes:
            uniq_notes.append(note)

    return (
        f"Combined figure assembled from {len(metas)} runs of experiment '{experiment}' "
        f"(tags={_format_value(tags)}). "
        f"Shared settings: {'; '.join(shared_parts) if shared_parts else 'none identified'}. "
        f"Varying settings across source runs: {'; '.join(varying_parts) if varying_parts else 'none identified'}. "
        f"L-statistic metadata summary across source runs: {' | '.join(uniq_notes)}."
    )


def _find_metadata_for_figure(stem: str, name: str, by_artifact_name: dict[str, dict[str, Any]], by_tag_stem: dict[str, dict[str, Any]], all_meta: list[dict[str, Any]]) -> str:
    direct = by_artifact_name.get(name) or by_tag_stem.get(stem)
    if direct is not None:
        return _caption_from_metadata(direct)

    if stem.endswith("_arms"):
        base = stem[: -len("_arms")]
        base_meta = by_tag_stem.get(f"monte_carlo_accuracy__{base}") or by_tag_stem.get(base)
        if base_meta is not None:
            return _caption_from_metadata(base_meta) + " This specific panel is the per-arm detail view."

    if "varydim_combined" in stem:
        metas = [m for m in all_meta if isinstance(m.get("tag"), str) and str(m.get("tag")).startswith("snr_cont_fixN_varydim_")]
        return _summarize_combined_metadata(metas, stem)

    if "varyarms_combined" in stem:
        metas = [m for m in all_meta if isinstance(m.get("tag"), str) and str(m.get("tag")).startswith("snr_multiarm_fixN_varyarms_")]
        return _summarize_combined_metadata(metas, stem)

    fuzzy = [m for m in all_meta if _meta_matches_stem(m, stem)]
    if len(fuzzy) == 1:
        return _caption_from_metadata(fuzzy[0])
    if len(fuzzy) > 1:
        return _summarize_combined_metadata(fuzzy, stem)

    return (
        f"Auto-included figure '{name}'. "
        "No matching metadata JSON was found for this artifact."
    )


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
    all_meta: list[dict[str, Any]] = []
    if data_dir.exists():
        by_artifact_name, by_tag_stem, all_meta = _load_metadata_maps(data_dir)

    figures = []

    pdfs = sorted(art.glob("*.pdf"))
    pngs = sorted(art.glob("*.png"))
    stems_with_pdf = {p.stem for p in pdfs}
    selected = list(pdfs) + [p for p in pngs if p.stem not in stems_with_pdf]
    # Avoid self-inclusion cycles and stale build artifacts.
    selected = [p for p in selected if p.stem != "report" and not p.name.startswith("report.")]

    for p in selected:
        cap = _find_metadata_for_figure(p.stem, p.name, by_artifact_name, by_tag_stem, all_meta)
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
