# Gemma 4 26B — CPU vs GPU benchmark + Uzbek quality evaluation

Benchmarks **`google/gemma-4-26B-A4B-it`** on CPU and GPU, and measures how well it
actually handles **Uzbek**.

Built for `gpu-sqb2` (2× NVIDIA A40 46 GB, driver 595.84, CUDA 13.2), but nothing is
hard-coded to that box.

---

## The one thing to know first

Gemma 4 26B is **not a dense 26B model**. It is a Mixture-of-Experts:

| | |
|---|---|
| Total parameters | 25.2 B |
| **Active per token** | **3.8 B** |
| Experts | 128, top-8 routed + 1 shared |
| Layers | 30 |
| Context | 256 K |
| Vocabulary | 262 K |
| Modality | text + image in, text out |

This changes the whole CPU-vs-GPU question. You must **store** 26 B parameters,
but you only **compute** 3.8 B per token. So:

- **Memory** cost is dense-26B-sized → this is what constrains you.
- **Compute** cost is ~4B-sized → this is why CPU inference is genuinely usable here,
  which it would not be for a dense 26 B model.

Expect the CPU/GPU gap to be **large for prefill** (compute-bound, GPU wins hard) and
**much smaller for decode** (memory-bandwidth-bound, where a many-channel server CPU
holds up far better). Confirming that split is the main point of the benchmark.

---

## What fits where

| Configuration | Size on disk/RAM | Fits 1× A40 (45 GB)? | Fits 2× A40? |
|---|---|---|---|
| bf16 safetensors | ~50 GB | ❌ no | ✅ yes |
| int8 (bitsandbytes) | ~26 GB | ✅ yes | ✅ yes |
| nf4 (bitsandbytes) | ~14 GB | ✅ yes | ✅ yes |
| Q4_K_M GGUF | ~15 GB | ✅ yes | ✅ yes |

**Your A40s are shared.** The `nvidia-smi` snapshot showed someone else's `python3`
holding 5 877 MiB on GPU 0 and 1 455 MiB on GPU 1. Every script here defaults to
leaving **6 GB free per card** (`--reserve-mib 6000`) so you do not OOM their job.
Check the current state before you start:

```bash
nvidia-smi
```

---

## Two tracks, and why

Comparing "GPU in bf16" against "CPU in Q4" would conflate two variables — hardware
*and* quantisation — and the number you got back would mean nothing. So this measures
both separately:

- **Track A — controlled.** llama.cpp, identical Q4_K_M weights, identical engine.
  Only `-ngl` changes. The CPU/GPU ratio here is a **clean hardware measurement**.
- **Track B — realistic.** transformers bf16 sharded over both A40s. This is the best
  the GPU can actually do in deployment, and it is what you would ship.

Read Track A to answer *"how much faster is the GPU?"*, and Track B to answer
*"what do I actually get if I deploy on the A40s?"*

---

## Setup

```bash
git clone https://github.com/isoqovjorabek2/Locall_llm.git
cd Locall_llm
```

Check you have the room first — you need **~70 GB** of disk for both weight sets:

```bash
df -h ~ && free -h && nproc
```

Check the toolchain before anything else — it tells you exactly what is missing:

```bash
bash scripts/setup_server.sh preflight
```

Then:

```bash
bash scripts/setup_server.sh      # venv + torch(cu128) + deps, then builds llama.cpp
bash scripts/download_models.sh   # bf16 safetensors (~52 GB) + Q4_K_M GGUF (~15 GB)
```

**No root needed.** If `cmake` or `ninja` are missing, they are installed from PyPI —
those wheels ship real binaries, so a locked-down server is fine. llama.cpp is built
for `sm_86` only (A40 is Ampere), which keeps the compile to minutes rather than an hour.

Run stages separately if you prefer: `bash scripts/setup_server.sh python` /
`... llamacpp` / `... prebuilt`, and `bash scripts/download_models.sh hf` / `... gguf`.

### Two things pip cannot fix

**A C++ compiler.** If preflight reports no `g++`/`clang++`, check for a module system
first (`module avail gcc && module load gcc`); otherwise you need an admin. In the
meantime, `bash scripts/setup_server.sh prebuilt` fetches official CPU-only binaries so
the CPU track and the Q4 Uzbek eval still run.

**The CUDA toolkit.** `nvidia-smi` reporting "CUDA 13.2" is the **driver's** runtime
version — it does *not* mean `nvcc` is installed, and `GGML_CUDA=ON` needs the real
toolkit. Try `module avail cuda && module load cuda`. Without it the build falls back to
CPU-only, which costs you Track A's GPU half; Track A's CPU half and all of Track B
(transformers, which uses pip-installed CUDA via torch and needs no `nvcc`) still work.

> There are no prebuilt **Linux CUDA** llama.cpp binaries published upstream — only CPU,
> Vulkan, and SYCL — so GPU llama.cpp genuinely requires compiling from source.

> **transformers version matters.** Gemma 4 needs a build that knows the `gemma4`
> architecture and exposes `Gemma4ForConditionalGeneration`. If loading fails with an
> unknown-model-type error, `pip install -U transformers` and retry.

---

## Run it

Use tmux. This is a multi-hour job and an SSH drop should not kill it.

```bash
tmux new -s gemma
bash scripts/run_all.sh
```

That runs both benchmark tracks, both Uzbek evals (bf16 and Q4), and writes
`results/REPORT.md`. Detach with `Ctrl-b d`, reattach with `tmux attach -t gemma`.

### Or run pieces individually

```bash
# Track A — same GGUF on CPU and GPU
python bench/bench_llamacpp.py --gguf models/*.gguf --devices cpu,gpu --gpu-index 1

# Track B — bf16 across both A40s
python bench/bench_hf_gpu.py --gpus 0,1 --reserve-mib 6000

# Track B on a single card (needs quantisation — bf16 will not fit)
python bench/bench_hf_gpu.py --gpus 1 --load-4bit

# Uzbek eval, bf16
python eval/run_uzbek_eval.py --backend hf --tag gpu-bf16

# Uzbek eval against anything llama-server is serving
llama-server -m models/*.gguf -ngl 0 -c 4096 --port 8080 &
python eval/run_uzbek_eval.py --backend server --url http://127.0.0.1:8080 --tag cpu-q4km

# Rebuild the report from whatever results exist
python analyze/make_report.py
```

---

## The Uzbek evaluation

30 items across 11 categories: translation both directions, Latin/Cyrillic script
control, transliteration, morphology (case declension, verb conjugation, possessive
chains), instruction-following, Uzbekistan knowledge, free generation, summarisation,
arithmetic reasoning, and code-switching resistance.

**Device does not affect quality — precision does.** The same weights produce the same
text on CPU or GPU. So the comparison that matters is **bf16 vs Q4_K_M**, and Uzbek is
exactly the kind of low-resource language where quantisation damage shows up first,
long before it is visible in English.

### What gets measured

Every generation is scored for:

| Signal | Catches |
|---|---|
| `script_ok` | Answered in Cyrillic when asked for Latin (or vice versa) |
| `turkish_char_count` | Drift into Turkish/Azeri — Uzbek Latin has **no** `ç ş ğ ı ö ü` |
| `english_stopword_rate` | Silently answering in English |
| `russian_*` | Russian bleeding into Uzbek Cyrillic |
| `unseen_trigram_rate` | **Uzbek-shaped non-words** — character trigrams absent from real Uzbek |
| `known_word_rate` | Recognisable Uzbek after stripping agglutinative suffixes |
| `suffix_rate` | Actually agglutinating, vs bare stems in Uzbek-flavoured word salad |
| `repetition_rate` | Looping / degeneration |
| `chrf` | Character F-score vs this suite's references (translation + transliteration) |
| `required_coverage` | Did the required fact or morphological form appear |

`unseen_trigram_rate` is the one to watch if you have seen a quantised model emit
plausible-looking Uzbek gibberish — it separates real Uzbek from non-words by roughly
3–4× in the test fixtures.

Verify the scorers still discriminate before trusting a run:

```bash
python tests/test_scoring.py
```

### Reading the results honestly

These are **automated proxies, not a native speaker.** Two limits worth stating plainly:

1. The reference corpus (~4 200 characters) and lexicon (~350 words) are small, so
   `unseen_trigram_rate` and `known_word_rate` are meaningful **compared between runs**,
   not as absolute scores. Real Uzbek baselines around 0.15 unseen-trigram, not 0.
2. `chrf` is computed against references written for this suite, so it is a relative
   signal between configurations — not a WMT-comparable number.

Full generations are saved to `results/raw/uzbek_eval_*.jsonl`. Before concluding
anything about Uzbek fluency, read them, or have a native speaker rate a sample.

---

## Layout

```
bench/
  common.py            telemetry sampling, timing, shared record schema
  bench_hf_gpu.py      Track B — transformers bf16/int8/nf4 on the A40s
  bench_llamacpp.py    Track A — same GGUF on CPU and GPU via llama-bench
eval/
  uzbek_suite.json     30 test items + Uzbek reference corpus
  uz_scoring.py        the scorers described above
  run_uzbek_eval.py    runner for either backend
analyze/
  make_report.py       results/raw/*.jsonl → results/REPORT.md
scripts/
  setup_server.sh      venv, torch, CUDA llama.cpp build
  download_models.sh   safetensors + GGUF
  run_all.sh           full pipeline
tests/
  test_scoring.py      scorer discrimination checks
```

Model weights and raw `.jsonl` dumps are gitignored; `results/REPORT.md` is not, so you
can commit a finished report.

---

## Troubleshooting

**CUDA OOM on load.** bf16 needs both cards. Either use `--gpus 0,1`, or drop to
`--load-4bit` on one. If someone else's job grew, raise `--reserve-mib`.

**`KeyError: 'gemma4'` or unknown model type.** transformers is too old:
`pip install -U transformers`.

**`cmake: command not found`.** Fixed automatically — `setup_server.sh` installs cmake
and ninja from PyPI. If they land somewhere off `PATH`:

```bash
export PATH="$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
```

**CMake complains about minimum-version compatibility.** CMake 4 dropped support for
projects declaring `cmake_minimum_required` below 3.5. Pin back a major version:
`pip install "cmake<4"`.

**llama.cpp built but no GPU offload.** A CPU-only build *accepts* `-ngl` and silently
ignores it, so the "GPU" numbers would really be CPU numbers. `bench_llamacpp.py` guards
against this: it samples VRAM before and during each GPU run and errors loudly if memory
never moved, and `make_report.py` stamps a warning on the table rather than letting bad
numbers through. If it fires, rebuild with the toolkit loaded:
`module load cuda && bash scripts/setup_server.sh llamacpp`.

**CPU run is far slower than expected.** llama.cpp degrades past *physical* cores;
the scripts default to physical count via `lscpu`. Override with `--threads N`.

**Everything is slow and `nvidia-smi` shows another big job.** You are sharing the box.
Coordinate, or pin yourself to the emptier card with `--gpus 1` / `--gpu-index 1`.
