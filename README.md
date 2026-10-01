# DALSA: Difficulty-Aware Long-to-Short LLM Reasoning with Model Merging

Code for the AACL-IJCNLP 2026 paper. DALSA predicts the difficulty of a prompt from the short
model's last prompt hidden state and turns it into a per-instance merge strength
`λ(x) = λmin + (λmax − λmin) · p̃(x)`. It then injects `(1 − t_l) · λ(x) · (θ_long − θ_short)`
into the short-reasoning model, where `t_l` is a layer-wise activation-consensus factor.
For fast inference, `λ(x)` is rounded to the nearest of a few buckets, and each bucket is served
as one merged vLLM checkpoint.

## Setup

Python 3.12 and CUDA GPUs are required.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests   # CPU-only checks
```

The models (Qwen2.5-Math, DeepSeek-R1-Distill, Llama-3.1) and datasets are downloaded from the
Hugging Face Hub on first use.

## Layout

```text
src/dalsa/        data loading, prompts, answer checking, difficulty probe, model merging
scripts/
  run_pipeline.sh         end-to-end pipeline (probe -> baselines -> DALSA -> summary)
  generate_rollouts.py    K short-model rollouts per training problem (probe labels)
  extract_h0.py           last prompt hidden state of the short model
  train_estimator.py      difficulty probe (linear, soft-failure regression)
  split_h0_holdout.py     stratified train/validation split of the probe data
  evaluate_vllm.py        native short/long baselines
  evaluate_adaptive_vllm.py  Simple Routing, fixed merges (TA, ACM-TA, TIES, DARE, FIM-TA), DALSA
  merge_h0_shards.py, merge_eval_shards.py   merge data-parallel shards
  make_calibration_texts.py  calibration pool for the CSQA/ARC-Challenge diagnostic
  summarize.py            per-benchmark / family averages, routing threshold curve, bucket stats
tests/            unit tests (answer checking, datasets, lambda remapping, summarize)
```

## Quick start

```bash
PAIR=qwen1.5b GPUS=0,1,2,3 OUT=runs/qwen1.5b bash scripts/run_pipeline.sh
```

This trains the difficulty probe, runs the native baselines, Simple Routing, TA, ACM-TA and DALSA
`[0.1, 0.7]` on GSM8K, MATH500, AMC23 and AIME24/25/26, and prints a table such as

```text
| run | aime24_26 | amc23 | gsm8k | math500 | avg |
| dalsa_max0p7 | ... accuracy (avg generated tokens) ... |
```

`OUT/summary.json` contains per-dataset statistics in `datasets`, benchmark-family averages in
`families`, and DALSA bucket counts/mean exact lambda in `buckets`. Every stage reads from and
writes to `OUT`, so stages can be rerun with `STAGES="..."`. Use a different `OUT` for different
experiment settings: rerunning replaces outputs for the selected stages. Useful knobs:

| variable | default | meaning |
|---|---|---|
| `PAIR` | `qwen1.5b` | `qwen1.5b`, `qwen7b`, `llama8b` |
| `GPUS` | `0` | one data-parallel shard per listed GPU |
| `STAGES` | `probe native routing fixed dalsa summary` | also `noconsensus`, `static`, `dalsa_fim` |
| `MAX_TOKENS` | `3840` | generation limit of merged models and routing (context is 4096) |
| `LAMBDA_MAXES` | `0.7` | DALSA ranges `[0.1, X]` to evaluate |
| `ROLLOUT_MAX_TOKENS` | `2048` | generation cap when creating probe labels |
| `ROLLOUT_TEMPERATURE` | `0.7` | label-generation temperature; use 0.6 for the 1.5B main recipe |
| `NATIVE_LONG_MAX_MODEL_LEN` | `32768` | native-long context limit, separate from its 32768-token generation cap |
| `NUM_GENERATIONS` | `1` | samples per prompt |
| `PROBE_VAL_RATIO` | empty | train the probe on a stratified `1 − ratio` split |
| `KEEP_MERGED` | `0` | keep merged checkpoints (≈15 GB each at 7B) |
| `LIMIT` | empty | a seed-42 random subset of N examples per dataset (quick smoke test) |

## Reproducing the paper

The table and figure numbers below refer to the Opus camera-ready paper. Run commands from the
repository root; add `GPUS=...` as needed. Evaluation uses temperature 0.6, top-p 0.95 and seed 42;
the native short baseline uses greedy decoding. Probe-label sampling is configured separately.

These commands regenerate the experiments, not their exact published numbers. The public answer
checker is versioned `dalsa-v2` and corrects historical scoring behavior; both probe labels and
reported accuracy can therefore differ. The paper also separates regenerated controllers and
calibration from main-table runs, with different generation caps at 7B. Those comparisons do not
provide a bound on sampling variance. Keep each suite in its own output directory.

`dalsa-v2` fixes explicit arithmetic, undefined/non-finite answers, AIME integer validation, and
symbolic-function normalization. Symbolic comparison has conservative expression-length and
exponent limits; it is not a general-purpose mathematical equivalence checker. New results carry
`scorer_version`; the summarizer rejects mixtures of new and unversioned historical scores.

| Paper | Command |
|---|---|
| Table 1 (1.5B) | `PAIR=qwen1.5b OUT=runs/main_qwen1.5b ROLLOUT_MAX_TOKENS=3072 ROLLOUT_TEMPERATURE=0.6 bash scripts/run_pipeline.sh` (Simple Routing = row `tau=0.5`) |
| Table 2 (7B) | `PAIR=qwen7b OUT=runs/main_qwen7b MAX_TOKENS=3072 PROBE_VAL_RATIO=0.3 NATIVE_LONG_MAX_MODEL_LEN=65536 bash scripts/run_pipeline.sh` |
| Tables 3, 9 (TIES / DARE / FIM-TA) | `PAIR=qwen1.5b OUT=runs/static_qwen1.5b STAGES="static summary" bash scripts/run_pipeline.sh`; for 7B change both `PAIR=qwen7b` and `OUT=runs/static_qwen7b` |
| Tables 3, 10, 11, Figure 4 (routing curve, merge range, consensus ablation) | `PAIR=qwen7b OUT=runs/regen_qwen7b STAGES="probe routing dalsa noconsensus dalsa_fim summary" LAMBDA_MAXES="0.3 0.5 0.7 0.9" bash scripts/run_pipeline.sh`; for 1.5B change both `PAIR=qwen1.5b` and `OUT=runs/regen_qwen1.5b` |
| Table 11, ×4 rows | see the repeated-run command below |
| Table 12 (Llama) | `PAIR=llama8b OUT=runs/regen_llama8b STAGES="probe routing dalsa noconsensus summary" LAMBDA_MAXES="0.7 0.9" bash scripts/run_pipeline.sh` (short/long rows = `routing_all_short` / `routing_all_long`) |
| Table 8 (CSQA / ARC-Challenge) | see below |
| Table 4, Figure 3 (buckets, λ profile) | combine `buckets` with `families` in `runs/main_qwen7b/summary.json` → `dalsa_max0p7`; sum the three AIME bucket counts and average their mean exact lambdas for AIME24–26 |

The 65536-token native-long context in the 7B command matches the recorded AMC23 and AIME25/26
settings; available records do not establish this as the original context for every 7B baseline
cell. The default remains 32768 for other recipes.

For Table 11's four-generation rows, first finish the regenerated 7B suite above, then copy its
probe and consensus calibration into a new output directory:

```bash
mkdir runs/regen_qwen7b_x4
cp -R runs/regen_qwen7b/probe runs/regen_qwen7b_x4/probe
cp runs/regen_qwen7b/calibration_acm.pt runs/regen_qwen7b_x4/calibration_acm.pt
PAIR=qwen7b OUT=runs/regen_qwen7b_x4 NUM_GENERATIONS=4 STAGES="dalsa noconsensus summary" \
LAMBDA_MAXES="0.7 0.9" bash scripts/run_pipeline.sh
```

**Table 8 (commonsense diagnostic).** One probe per dataset and a shared 128-prompt
calibration pool:

```bash
python scripts/make_calibration_texts.py --output runs/cs_calibration.jsonl
for ds in csqa arc; do
  PAIR=qwen1.5b OUT=runs/commonsense_$ds TRAIN_DATASET=$ds EVAL_DATASETS=$ds MAX_TOKENS=2048 ROLLOUT_MAX_TOKENS=1024 \
  NATIVE_SHORT_MAX_TOKENS=2048 CALIBRATION_DATASET=runs/cs_calibration.jsonl CALIBRATION_SAMPLES=128 \
  bash scripts/run_pipeline.sh
done
```

**Table 5 (long-base diagnostic, 7B).** This uses an earlier hard-label architecture without
quantile remapping and 11 buckets. Train a separate probe from the 7B main recipe's full H0:

```bash
H0_7B=runs/main_qwen7b/probe/h0.pt
python scripts/train_estimator.py --train "$H0_7B" --num-rollouts 3 --output runs/longbase_qwen7b/probe_hard --network residual_mlp_v3 --target-mode hard_bce
for base in short long; do
  [ "$base" = short ] && L=4096 T=3072 || L=32768 T=16384
  python scripts/evaluate_adaptive_vllm.py --method dalsa_bucketed --prompt-mode shared \
    --short-model Qwen/Qwen2.5-Math-7B-Instruct --long-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --datasets gsm8k,math500,aime24,aime25,aime26 --estimator runs/longbase_qwen7b/probe_hard \
    --lambda-score-remap none --lambda-values 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 --min-lambda 0 --max-lambda 1 \
    --merge-base "$base" --max-model-len "$L" --max-tokens "$T" --temperature 0.6 --top-p 0.95 \
    --calibration runs/main_qwen7b/calibration_acm.pt --output-dir "runs/longbase_qwen7b/$base"
done
```

**Tables 6–7 (probe architecture, 1.5B).** Table 6 trains two hard-label probes on each dataset's
70% training partition and evaluates them on its 30% held-out partition. Use the **1.5B** GSM8K H0
from Table 1; create separate MATH500 training H0 with the 1.5B short model:

```bash
mkdir -p runs/probes_qwen1.5b
python scripts/generate_rollouts.py --dataset ricdomolm/math-500 --split train \
  --model Qwen/Qwen2.5-Math-1.5B-Instruct --prompt-mode shared --engine vllm \
  --num-rollouts 3 --max-tokens 3072 --max-model-len 4096 --temperature 0.6 --top-p 0.95 \
  --output runs/probes_qwen1.5b/math500_rollouts.json
python scripts/extract_h0.py --input runs/probes_qwen1.5b/math500_rollouts.json \
  --model Qwen/Qwen2.5-Math-1.5B-Instruct --prompt-mode saved \
  --output runs/probes_qwen1.5b/math500_h0.pt
H0_GSM8K_1P5B=runs/main_qwen1.5b/probe/h0.pt
H0_MATH500_1P5B=runs/probes_qwen1.5b/math500_h0.pt
for ds in gsm8k math500; do
  [ "$ds" = gsm8k ] && H0_1P5B="$H0_GSM8K_1P5B" || H0_1P5B="$H0_MATH500_1P5B"
  python scripts/split_h0_holdout.py --input "$H0_1P5B" \
    --train-output "runs/probes_qwen1.5b/${ds}_train.pt" --val-output "runs/probes_qwen1.5b/${ds}_val.pt" --val-ratio 0.3 --seed 42
  for net in residual_mlp_v3 linear; do
    python scripts/train_estimator.py --train "runs/probes_qwen1.5b/${ds}_train.pt" \
      --val "runs/probes_qwen1.5b/${ds}_val.pt" --num-rollouts 3 \
      --output "runs/probes_qwen1.5b/${ds}_holdout_$net" --network "$net" --target-mode hard_bce --threshold 0.5
  done
done
```

Table 7 instead trains each architecture on **all GSM8K training H0**, not the Table 6 holdout
partition, then evaluates it on `gsm8k,math500,aime24`:

```bash
H0_GSM8K_1P5B=runs/main_qwen1.5b/probe/h0.pt
for net in residual_mlp_v3 linear; do
  python scripts/train_estimator.py --train "$H0_GSM8K_1P5B" --num-rollouts 3 \
    --output "runs/probes_qwen1.5b/gsm8k_full_$net" --network "$net" --target-mode hard_bce --threshold 0.5
  python scripts/evaluate_adaptive_vllm.py --method dalsa_bucketed --prompt-mode shared \
    --short-model Qwen/Qwen2.5-Math-1.5B-Instruct --long-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --datasets gsm8k,math500,aime24 --estimator "runs/probes_qwen1.5b/gsm8k_full_$net" \
    --lambda-score-remap none --min-lambda 0.1 --max-lambda 0.5 --lambda-buckets 5 \
    --max-tokens 3840 --max-model-len 4096 --temperature 0.6 --top-p 0.95 \
    --calibration runs/main_qwen1.5b/calibration_acm.pt --output-dir "runs/probes_qwen1.5b/eval_$net"
done
```

New H0 files preserve rollout count and scorer metadata. For a legacy H0 file without rollout-count
metadata, `--num-rollouts 3` declares its known K; a conflicting existing K is rejected. The CLI
updates only the loaded data in memory, leaving the source H0 archive unchanged. It does not
rescore legacy labels or turn them into `dalsa-v2` labels.

**Differences from the original runs.**
- The main-table probes in Tables 1–2 come from earlier runs. Rerunning regenerates them; the
  regenerated suite in Appendix B is a separate experiment, not an estimate of rerun variance.
- The 7B main probe was trained in the same process as other probe variants. The split,
  objective and training hyperparameters follow that recipe, but regenerated labels and random
  initialisation are not bit-identical.
- The Llama pair uses the public mirror `NousResearch/Meta-Llama-3.1-8B-Instruct`. Set
  `LLAMA_SHORT=meta-llama/Llama-3.1-8B-Instruct` to use the gated original.

Calibration caches are checked against their requested settings. For an incompatible or legacy
cache whose settings cannot be verified, use a new cache path or `OUT` and regenerate it. Do not
copy old private-run caches into a new public experiment.

A release license has not yet been selected; no license grant is included in this tree.

## Citation

```bibtex
@inproceedings{min2026dalsa,
  title     = {{DALSA}: Difficulty-Aware Long-to-Short {LLM} Reasoning with Model Merging},
  author    = {Min, Dongwook and Kang, Jeonhwi and Jeon, Donghyeon and Kang, Inho and Na, Seung-Hoon},
  booktitle = {Proceedings of AACL-IJCNLP 2026},
  year      = {2026}
}
```
