#!/usr/bin/env python3
"""Summarize output equivalence/quality across normal, folded-base, and plugin engines."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_text(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip().lower())


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()


def token_set_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"[A-Za-z0-9_./+-]+", norm_text(a)))
    tb = set(re.findall(r"[A-Za-z0-9_./+-]+", norm_text(b)))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def repetition_score(text: str) -> float:
    toks = re.findall(r"\S+", text or "")
    if len(toks) < 8:
        return 0.0
    bigrams = list(zip(toks, toks[1:]))
    if not bigrams:
        return 0.0
    counts = defaultdict(int)
    for bg in bigrams:
        counts[bg] += 1
    return max(counts.values()) / max(1, len(bigrams))


def expected_pass(text: str, expected: Iterable[str]) -> bool:
    expected = [str(x).strip() for x in expected if str(x).strip()]
    if not expected:
        return True
    low = (text or "").lower()
    return all(x.lower() in low for x in expected)


def garbage_flags(text: str) -> List[str]:
    flags: List[str] = []
    stripped = (text or "").strip()
    if not stripped:
        flags.append("empty")
    if "�" in stripped:
        flags.append("replacement_char")
    if len(stripped) > 0:
        unique_ratio = len(set(stripped)) / max(1, len(stripped))
        if len(stripped) > 80 and unique_ratio < 0.08:
            flags.append("low_char_diversity")
    rep = repetition_score(stripped)
    if rep > 0.20:
        flags.append(f"repetitive_bigram_{rep:.2f}")
    return flags


def pick(rows: List[Dict[str, Any]], target: str) -> Dict[str, Dict[str, Any]]:
    return {str(r["prompt_id"]): r for r in rows if r.get("target") == target}


def pct(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", required=True)
    ap.add_argument("--folded-base", required=True)
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--similarity-threshold", type=float, default=0.60)
    args = ap.parse_args()

    normal_rows = load_jsonl(args.normal)
    folded_rows = load_jsonl(args.folded_base)
    plugin_rows = load_jsonl(args.plugin)

    normal = pick(normal_rows, "normal") or {str(r["prompt_id"]): r for r in normal_rows}
    folded = pick(folded_rows, "folded_base") or {str(r["prompt_id"]): r for r in folded_rows}
    plugin = pick(plugin_rows, "plugin") or {str(r["prompt_id"]): r for r in plugin_rows}

    prompt_ids = sorted(set(normal) | set(folded) | set(plugin))
    rows: List[Dict[str, Any]] = []

    for pid in prompt_ids:
        n = normal.get(pid, {})
        b = folded.get(pid, {})
        c = plugin.get(pid, {})
        nt = n.get("text", "")
        bt = b.get("text", "")
        ct = c.get("text", "")
        expected = c.get("expected_contains") or b.get("expected_contains") or n.get("expected_contains") or []
        row = {
            "prompt_id": pid,
            "normal_success": bool(n.get("success")),
            "folded_base_success": bool(b.get("success")),
            "plugin_success": bool(c.get("success")),
            "normal_tokens": n.get("completion_tokens", 0),
            "folded_base_tokens": b.get("completion_tokens", 0),
            "plugin_tokens": c.get("completion_tokens", 0),
            "plugin_vs_normal_char_similarity": sim(ct, nt),
            "plugin_vs_folded_base_char_similarity": sim(ct, bt),
            "folded_base_vs_normal_char_similarity": sim(bt, nt),
            "plugin_vs_normal_token_overlap": token_set_overlap(ct, nt),
            "plugin_vs_folded_base_token_overlap": token_set_overlap(ct, bt),
            "folded_base_vs_normal_token_overlap": token_set_overlap(bt, nt),
            "normal_expected_pass": expected_pass(nt, expected),
            "folded_base_expected_pass": expected_pass(bt, expected),
            "plugin_expected_pass": expected_pass(ct, expected),
            "normal_garbage_flags": ";".join(garbage_flags(nt)),
            "folded_base_garbage_flags": ";".join(garbage_flags(bt)),
            "plugin_garbage_flags": ";".join(garbage_flags(ct)),
            "normal_text_preview": nt[:240].replace("\n", "\\n"),
            "folded_base_text_preview": bt[:240].replace("\n", "\\n"),
            "plugin_text_preview": ct[:240].replace("\n", "\\n"),
        }
        row["plugin_close_to_normal"] = row["plugin_vs_normal_char_similarity"] >= args.similarity_threshold or row["plugin_vs_normal_token_overlap"] >= 0.45
        row["plugin_close_to_folded_base"] = row["plugin_vs_folded_base_char_similarity"] >= args.similarity_threshold or row["plugin_vs_folded_base_token_overlap"] >= 0.45
        rows.append(row)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["prompt_id"])
        writer.writeheader()
        writer.writerows(rows)

    n_prompts = len(rows)
    plugin_expected = sum(1 for r in rows if r["plugin_expected_pass"])
    normal_expected = sum(1 for r in rows if r["normal_expected_pass"])
    folded_expected = sum(1 for r in rows if r["folded_base_expected_pass"])
    plugin_garbage = sum(1 for r in rows if r["plugin_garbage_flags"])
    normal_garbage = sum(1 for r in rows if r["normal_garbage_flags"])
    folded_garbage = sum(1 for r in rows if r["folded_base_garbage_flags"])
    plugin_close_normal = sum(1 for r in rows if r["plugin_close_to_normal"])
    plugin_close_folded = sum(1 for r in rows if r["plugin_close_to_folded_base"])

    avg_pn = sum(float(r["plugin_vs_normal_char_similarity"]) for r in rows) / max(1, n_prompts)
    avg_pb = sum(float(r["plugin_vs_folded_base_char_similarity"]) for r in rows) / max(1, n_prompts)
    avg_bn = sum(float(r["folded_base_vs_normal_char_similarity"]) for r in rows) / max(1, n_prompts)

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# NVFP4 three-way output quality sanity report\n\n")
        f.write("This report checks whether the optimized plugin engine produces non-empty, non-garbage, broadly comparable outputs. It is a sanity check, not a full benchmark-suite accuracy evaluation.\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Prompts: {n_prompts}\n")
        f.write(f"- Expected-keyword pass: normal={normal_expected}/{n_prompts} ({pct(normal_expected,n_prompts):.1f}%), folded_base={folded_expected}/{n_prompts} ({pct(folded_expected,n_prompts):.1f}%), plugin={plugin_expected}/{n_prompts} ({pct(plugin_expected,n_prompts):.1f}%)\n")
        f.write(f"- Garbage flags: normal={normal_garbage}, folded_base={folded_garbage}, plugin={plugin_garbage}\n")
        f.write(f"- Plugin close to normal: {plugin_close_normal}/{n_prompts} ({pct(plugin_close_normal,n_prompts):.1f}%)\n")
        f.write(f"- Plugin close to folded_base: {plugin_close_folded}/{n_prompts} ({pct(plugin_close_folded,n_prompts):.1f}%)\n")
        f.write(f"- Avg char similarity: plugin-vs-normal={avg_pn:.3f}, plugin-vs-folded_base={avg_pb:.3f}, folded_base-vs-normal={avg_bn:.3f}\n\n")
        f.write("## Per-prompt previews\n\n")
        for r in rows:
            f.write(f"### {r['prompt_id']}\n\n")
            f.write(f"- similarity: C/A={float(r['plugin_vs_normal_char_similarity']):.3f}, C/B={float(r['plugin_vs_folded_base_char_similarity']):.3f}, B/A={float(r['folded_base_vs_normal_char_similarity']):.3f}\n")
            f.write(f"- expected pass: A={r['normal_expected_pass']}, B={r['folded_base_expected_pass']}, C={r['plugin_expected_pass']}\n")
            if r["plugin_garbage_flags"] or r["normal_garbage_flags"] or r["folded_base_garbage_flags"]:
                f.write(f"- garbage flags: A={r['normal_garbage_flags']}, B={r['folded_base_garbage_flags']}, C={r['plugin_garbage_flags']}\n")
            f.write("\n")
            f.write(f"**A normal:** {r['normal_text_preview']}\n\n")
            f.write(f"**B folded base:** {r['folded_base_text_preview']}\n\n")
            f.write(f"**C plugin:** {r['plugin_text_preview']}\n\n")

    print(f"CSV written to {out_csv}")
    print(f"Markdown report written to {out_md}")
    print(f"Plugin expected-keyword pass: {plugin_expected}/{n_prompts}")
    print(f"Plugin garbage flags: {plugin_garbage}")
    print(f"Avg plugin-vs-normal similarity: {avg_pn:.3f}")


if __name__ == "__main__":
    main()
