"""Turn results/raw/*.jsonl into a single readable comparison report.

Writes results/REPORT.md. Safe to run with only some of the runs present --
missing sections are simply skipped and noted.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "results" / "raw"
OUT = REPO_ROOT / "results" / "REPORT.md"


def read_jsonl(path: Path) -> list:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        print("[warn] could not read " + path.name + ": " + str(exc), file=sys.stderr)
    return rows


def md_table(headers: list, rows: list) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return str(round(x, nd))
    return str(x)


# ---------------------------------------------------------------------------
def section_host() -> str:
    parts = ["## 1. Machine\n"]
    found = False
    for name in ("host_gpu.json", "host_llamacpp.json"):
        p = RAW / name
        if not p.exists():
            continue
        found = True
        info = json.loads(p.read_text(encoding="utf-8"))
        parts.append("**" + name.replace("host_", "").replace(".json", "") + " run**\n")
        parts.append("- Host: `" + str(info.get("hostname")) + "` (" + str(info.get("os")) + ")")
        parts.append("- CPU: " + str(info.get("cpu")))
        parts.append("- Cores: " + str(info.get("physical_cores")) + " physical / "
                     + str(info.get("logical_cores")) + " logical")
        for g in info.get("gpus", []):
            parts.append(
                "- GPU " + str(g["index"]) + ": " + g["name"] + " -- "
                + str(g["memory_total_mib"]) + " MiB total, "
                + str(g["memory_used_mib_at_start"]) + " MiB already in use at start "
                + "(driver " + str(g["driver"]) + ")"
            )
        parts.append("")
    if not found:
        parts.append("_No host metadata found. Run a benchmark first._\n")
    return "\n".join(parts)


def section_track_a() -> str:
    """llama.cpp, identical GGUF, CPU vs GPU."""
    rows = read_jsonl(RAW / "llamacpp.jsonl")
    parts = ["## 2. Track A -- controlled CPU vs GPU (llama.cpp, identical Q4_K_M weights)\n"]
    if not rows:
        parts.append("_Not run yet: `python bench/bench_llamacpp.py --gguf ...`_\n")
        return "\n".join(parts)

    parts.append(
        "Same engine, same weights, same tokenizer. Only `-ngl` differs, so the "
        "ratio below is a clean hardware measurement.\n"
    )

    by = defaultdict(dict)
    for r in rows:
        by[r["case"]][r["device"]] = r

    table = []
    for case in ["pp512", "pp4096", "tg128", "tg512"]:
        entry = by.get(case)
        if not entry:
            continue
        cpu = entry.get("cpu")
        gpu = entry.get("gpu")
        cpu_ts = cpu["tok_s"] if cpu else None
        gpu_ts = gpu["tok_s"] if gpu else None
        speedup = round(gpu_ts / cpu_ts, 1) if (cpu_ts and gpu_ts and cpu_ts > 0) else None
        phase = "prefill" if case.startswith("pp") else "decode"
        table.append([
            case, phase, fmt(cpu_ts), fmt(gpu_ts),
            (str(speedup) + "x") if speedup else "-",
        ])
    parts.append(md_table(
        ["Case", "Phase", "CPU tok/s", "GPU tok/s", "GPU speedup"], table))

    # Resource use
    parts.append("\n**Resource use**\n")
    res = []
    for device in ("cpu", "gpu"):
        sub = [r for r in rows if r["device"] == device]
        if not sub:
            continue
        vram = [r["telemetry"].get("peak_vram_mib") for r in sub
                if r["telemetry"].get("peak_vram_mib")]
        rss = [r["telemetry"].get("peak_host_rss_mib") for r in sub
               if r["telemetry"].get("peak_host_rss_mib")]
        pwr = [r["telemetry"].get("mean_power_w") for r in sub
               if r["telemetry"].get("mean_power_w")]
        res.append([
            device,
            fmt(max(vram) if vram else None, 0),
            fmt(max(rss) if rss else None, 0),
            fmt(statistics.fmean(pwr) if pwr else None, 0),
        ])
    parts.append(md_table(
        ["Device", "Peak VRAM (MiB)", "Peak host RSS (MiB)", "Mean GPU power (W)"], res))
    parts.append("")
    return "\n".join(parts)


def section_track_b() -> str:
    """transformers bf16 GPU."""
    rows = read_jsonl(RAW / "gpu_transformers.jsonl")
    parts = ["## 3. Track B -- best realistic GPU deployment (transformers, bf16, 2x A40)\n"]
    if not rows:
        parts.append("_Not run yet: `python bench/bench_hf_gpu.py`_\n")
        return "\n".join(parts)

    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(r)

    table = []
    for case, rs in by_case.items():
        prefill = [r["prefill_tok_s"] for r in rs if r["prefill_tok_s"]]
        decode = [r["decode_tok_s"] for r in rs if r["decode_tok_s"]]
        ttft = [r["ttft_s"] for r in rs if r["ttft_s"]]
        vram = [r["telemetry"].get("peak_vram_mib") for r in rs
                if r["telemetry"].get("peak_vram_mib")]
        table.append([
            case,
            rs[0]["prompt_tokens"],
            rs[0]["generated_tokens"],
            fmt(statistics.fmean(prefill) if prefill else None),
            fmt(statistics.fmean(decode) if decode else None),
            fmt(statistics.fmean(ttft) if ttft else None, 3),
            fmt(max(vram) if vram else None, 0),
        ])
    parts.append(md_table(
        ["Case", "Prompt tok", "Gen tok", "Prefill tok/s", "Decode tok/s",
         "TTFT (s)", "Peak VRAM (MiB)"], table))
    parts.append("")
    return "\n".join(parts)


def section_uzbek() -> str:
    files = sorted(RAW.glob("uzbek_eval_*.jsonl"))
    parts = ["## 4. Uzbek language quality\n"]
    if not files:
        parts.append("_Not run yet: `python eval/run_uzbek_eval.py --backend hf --tag gpu-bf16`_\n")
        return "\n".join(parts)

    parts.append(
        "Identical weights produce identical text regardless of device, so the "
        "comparison that matters here is **precision**, not CPU vs GPU: how much "
        "Uzbek quality the Q4 quantisation costs versus bf16.\n"
    )

    # Headline table, one row per run tag.
    head = []
    per_tag_rows = {}
    for f in files:
        rows = read_jsonl(f)
        if not rows:
            continue
        tag = rows[0]["tag"]
        per_tag_rows[tag] = rows
        uz_rows = [r for r in rows if r["scores"].get("uzbek_ok") is not None]
        chrf = [r["scores"]["chrf"] for r in rows if r["scores"].get("chrf") is not None]
        req = [r["scores"]["required_coverage"] for r in rows
               if r["scores"].get("required_coverage") is not None]
        script_ok = [r["scores"]["script_ok"] for r in rows
                     if r["scores"].get("script_ok") is not None]
        head.append([
            tag,
            rows[0].get("precision"),
            len(rows),
            fmt(100 * sum(1 for r in uz_rows if r["scores"]["uzbek_ok"]) / len(uz_rows), 1)
            if uz_rows else "-",
            fmt(100 * sum(1 for s in script_ok if s) / len(script_ok), 1) if script_ok else "-",
            fmt(statistics.fmean(chrf), 1) if chrf else "-",
            fmt(100 * statistics.fmean(req), 1) if req else "-",
            fmt(statistics.fmean([r["scores"].get("known_word_rate", 0) for r in rows]) * 100, 1),
            fmt(statistics.fmean([r["scores"].get("unseen_trigram_rate", 0) for r in rows]) * 100, 2),
            sum(r["scores"].get("turkish_char_count", 0) for r in rows),
        ])
    parts.append(md_table(
        ["Run", "Precision", "Items", "Uzbek-OK %", "Right script %", "Mean chrF",
         "Required-fact %", "Known-word %", "Unseen-trigram %", "Turkish chars"],
        head))

    parts.append(
        "\n- **Uzbek-OK %** -- right script, zero Turkish drift, <10% English leakage, no looping.\n"
        "- **chrF** -- character F-score against this suite's own references "
        "(translation + transliteration items only). Comparative, not a WMT number.\n"
        "- **Unseen-trigram %** -- share of character trigrams absent from real Uzbek. "
        "Rises sharply when a model emits Uzbek-shaped non-words.\n"
        "- **Turkish chars** -- count of `ç ş ğ ı ö ü`, which do not exist in Uzbek Latin. "
        "Any nonzero value means the decoder slid toward Turkish.\n"
    )

    # Per-category breakdown for each tag.
    for tag, rows in per_tag_rows.items():
        parts.append("\n### 4." + str(list(per_tag_rows).index(tag) + 1) + " " + tag + " by category\n")
        by_cat = defaultdict(list)
        for r in rows:
            by_cat[r["category"]].append(r)
        cat_table = []
        for cat, rs in sorted(by_cat.items()):
            ok = [r for r in rs if r["scores"].get("uzbek_ok") is not None]
            chrf = [r["scores"]["chrf"] for r in rs if r["scores"].get("chrf") is not None]
            cat_table.append([
                cat,
                len(rs),
                fmt(100 * sum(1 for r in ok if r["scores"]["uzbek_ok"]) / len(ok), 0)
                if ok else "-",
                fmt(statistics.fmean(chrf), 1) if chrf else "-",
                fmt(statistics.fmean([r["scores"].get("known_word_rate", 0) for r in rs]) * 100, 1),
            ])
        parts.append(md_table(
            ["Category", "N", "Uzbek-OK %", "Mean chrF", "Known-word %"], cat_table))

        flagged = [r for r in rows if r["scores"].get("uzbek_ok") is False
                   or r["scores"].get("script_ok") is False]
        if flagged:
            parts.append("\n**Flagged items (" + str(len(flagged)) + ")**\n")
            for r in flagged[:10]:
                sc = r["scores"]
                reasons = []
                if sc.get("script_ok") is False:
                    reasons.append("wrong script (" + str(sc.get("script")) + ")")
                if sc.get("turkish_char_count"):
                    reasons.append("Turkish chars: " + "".join(sc.get("turkish_chars_found", [])))
                if sc.get("english_stopword_rate", 0) >= 0.10:
                    reasons.append("English leakage " + fmt(sc["english_stopword_rate"] * 100, 0) + "%")
                if sc.get("repetition_rate", 0) >= 0.25:
                    reasons.append("looping " + fmt(sc["repetition_rate"] * 100, 0) + "%")
                snippet = (r["output"] or "")[:160].replace("\n", " ")
                parts.append("- `" + r["id"] + "` -- " + ("; ".join(reasons) or "composite fail")
                             + "\n  > " + snippet + ("..." if len(r["output"] or "") > 160 else ""))
        parts.append("")

    parts.append(
        "\n> These are automated proxies. Before drawing a conclusion about Uzbek "
        "fluency, read `results/raw/uzbek_eval_*.jsonl` -- the full generations are "
        "stored there -- or have a native speaker rate a sample.\n"
    )
    return "\n".join(parts)


def main():
    if not RAW.exists():
        sys.exit("No results directory. Run the benchmarks first.")

    doc = [
        "# Gemma 4 26B-A4B -- CPU vs GPU, and Uzbek quality\n",
        "_Generated by `analyze/make_report.py`. Model: `google/gemma-4-26B-A4B-it` "
        "(25.2B total / 3.8B active, MoE, 128 experts top-8)._\n",
        section_host(),
        section_track_a(),
        section_track_b(),
        section_uzbek(),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(doc), encoding="utf-8")
    print("[done] wrote " + str(OUT))


if __name__ == "__main__":
    main()
