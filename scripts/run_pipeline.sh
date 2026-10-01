#!/usr/bin/env bash
# DALSA end-to-end pipeline for one short/long model pair.
#
#   PAIR=qwen7b GPUS=0,1 OUT=runs/qwen7b bash scripts/run_pipeline.sh
#
# Stages (STAGES, space-separated; default "probe native routing fixed dalsa summary"):
#   probe       short-model rollouts on TRAIN_DATASET -> prompt hidden states -> linear soft-failure probe
#   native      native short/long baselines (Tables 1-2, first two rows)
#   routing     capped always-short / always-long runs; the Simple Routing threshold curve
#               (tau = 0.1..0.9, incl. the default tau = 0.5) is composed from them by summarize.py
#   fixed       TA and ACM-TA with lambda = 0.7
#   dalsa       DALSA with lambda in [0.1, X] for every X in LAMBDA_MAXES
#   noconsensus DALSA [0.1, 0.7] without the (1 - t_l) consensus term
#   static      TIES / DARE / FIM-TA proxy for lambda in {0.3, 0.5, 0.7}
#   dalsa_fim   DALSA-on-FIM for lambda in [0.1, 0.7] and [0.1, 0.9]
#   summary     aggregate every run under OUT into OUT/summary.json and a markdown table
# Every evaluation runs data-parallel over the GPUs in GPUS (one shard per GPU).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
PAIR=${PAIR:-qwen1.5b}
case "$PAIR" in
  qwen1.5b) SHORT=Qwen/Qwen2.5-Math-1.5B-Instruct; LONG=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B ;;
  qwen7b)   SHORT=Qwen/Qwen2.5-Math-7B-Instruct;   LONG=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B ;;
  llama8b)  SHORT=${LLAMA_SHORT:-NousResearch/Meta-Llama-3.1-8B-Instruct}; LONG=deepseek-ai/DeepSeek-R1-Distill-Llama-8B ;;
  *) echo "Unknown PAIR=$PAIR (qwen1.5b | qwen7b | llama8b)" >&2; exit 2 ;;
esac

GPUS=${GPUS:-0}
IFS=, read -r -a GPU_LIST <<< "$GPUS"
N=${#GPU_LIST[@]}
OUT=${OUT:-runs/$PAIR}
STAGES=${STAGES:-probe native routing fixed dalsa summary}

TRAIN_DATASET=${TRAIN_DATASET:-gsm8k}
EVAL_DATASETS=${EVAL_DATASETS:-gsm8k,math500,amc23,aime24,aime25,aime26}
MAX_TOKENS=${MAX_TOKENS:-3840}            # paper: 3840; the 7B main results (Table 2) used 3072; CSQA/ARC 2048
MAX_MODEL_LEN=4096                        # context of the short-base merged models
NUM_GENERATIONS=${NUM_GENERATIONS:-1}     # 4 for the repeated 7B rows in Table 11
ROLLOUT_MAX_TOKENS=${ROLLOUT_MAX_TOKENS:-2048}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}
PROBE_VAL_RATIO=${PROBE_VAL_RATIO:-}      # 0.3: train the probe on a stratified 70% split (7B main results)
LAMBDA_MAXES=${LAMBDA_MAXES:-0.7}         # "0.3 0.5 0.7 0.9" for the merge-range sweep
CALIBRATION_DATASET=${CALIBRATION_DATASET:-simplescaling/s1K}
CALIBRATION_SAMPLES=${CALIBRATION_SAMPLES:-20}
KEEP_MERGED=${KEEP_MERGED:-0}             # merged checkpoints are large (~15 GB each at 7B)
LIMIT=${LIMIT:-}                          # e.g. 8: only the first N examples per dataset (smoke test)
NATIVE_SHORT_MAX_TOKENS=${NATIVE_SHORT_MAX_TOKENS:-3072}  # 2048 for CSQA/ARC
NATIVE_LONG_MAX_MODEL_LEN=${NATIVE_LONG_MAX_MODEL_LEN:-32768}
NATIVE_TP=${NATIVE_TP:-1}                 # tensor parallelism for the native baselines (must divide the head count)

PROBE=$OUT/probe
EST=$PROBE/estimator
REF_H0=$PROBE/h0.pt
CACHE=$OUT/merged_vllm
mkdir -p "$OUT" "$PROBE"

lambdas_up_to() { "$PY" -c "import sys; x=float(sys.argv[1]); print(','.join(f'{v/10:.1f}' for v in range(1, round(x*10)+1)))" "$1"; }

# run_eval NAME ARGS... : sharded evaluate_adaptive_vllm.py run, merged into OUT/NAME
run_eval() {
  local name=$1; shift
  local common=(--datasets "$EVAL_DATASETS" --short-model "$SHORT" --long-model "$LONG"
    --max-tokens "$MAX_TOKENS" --temperature 0.6 --top-p 0.95 --num-generations "$NUM_GENERATIONS"
    --max-model-len "$MAX_MODEL_LEN" --merge-base short --merged-model-cache "$CACHE" ${LIMIT:+--limit "$LIMIT"} "$@")
  echo "[$(date -Is)] $name"
  # build merged checkpoints once, before the shards start (merge methods only)
  if [[ " $* " != *" routing "* ]]; then
    CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} "$PY" scripts/evaluate_adaptive_vllm.py "${common[@]}" \
      --materialize-only --output-dir "$OUT/$name/prepare"
  fi
  local pids=() shard_dirs=() i
  for i in $(seq 0 $((N - 1))); do
    shard_dirs+=("$OUT/$name/shards/$i")
    CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} "$PY" scripts/evaluate_adaptive_vllm.py "${common[@]}" \
      --num-shards "$N" --shard-index "$i" --output-dir "$OUT/$name/shards/$i" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  "$PY" scripts/merge_eval_shards.py --inputs "${shard_dirs[@]}" --output-dir "$OUT/$name"
  [[ "$KEEP_MERGED" == 1 ]] || rm -rf "$CACHE"
}

controller=(--estimator "$EST" --h0-batch-size 32 --h0-max-length 2048
  --lambda-score-remap quantile --lambda-remap-reference-h0 "$REF_H0")
acm=(--calibration "$OUT/calibration_acm.pt" --calibration-dataset "$CALIBRATION_DATASET"
  --calibration-samples "$CALIBRATION_SAMPLES" --mi-clusters 8 --theta 1.0)
fim=(--calibration "$OUT/calibration_fim.pt" --fim-random-samples 8 --fim-seq-len 128)

for stage in $STAGES; do
  case "$stage" in
  probe)
    pids=()
    h0_shards=()
    for i in $(seq 0 $((N - 1))); do
      h0_shards+=("$PROBE/h0_$i.pt")
      (
        export CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]}
        "$PY" scripts/generate_rollouts.py --dataset "$TRAIN_DATASET" --split train --model "$SHORT" \
          --prompt-mode shared --engine vllm --num-rollouts 3 --max-tokens "$ROLLOUT_MAX_TOKENS" \
          --temperature "$ROLLOUT_TEMPERATURE" --top-p 0.95 --max-model-len "$MAX_MODEL_LEN" \
          --num-shards "$N" --shard-index "$i" --output "$PROBE/rollouts_$i.json" ${LIMIT:+--limit "$LIMIT"}
        "$PY" scripts/extract_h0.py --input "$PROBE/rollouts_$i.json" --model "$SHORT" \
          --output "$PROBE/h0_$i.pt" --prompt-mode saved --batch-size 32 --max-length 2048
      ) &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    "$PY" scripts/merge_h0_shards.py --inputs "${h0_shards[@]}" --output "$REF_H0"
    train=$REF_H0
    if [[ -n "$PROBE_VAL_RATIO" ]]; then
      "$PY" scripts/split_h0_holdout.py --input "$REF_H0" --train-output "$PROBE/h0_train.pt" \
        --val-output "$PROBE/h0_val.pt" --val-ratio "$PROBE_VAL_RATIO" --seed 42
      train=$PROBE/h0_train.pt
    fi
    CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} "$PY" scripts/train_estimator.py --train "$train" --output "$EST" \
      --network linear --target-mode soft_fail_regression --epochs 30 --batch-size 64 --lr 1e-4 --seed 42
    ;;
  native)
    # Tables 1-2: the short model is decoded greedily; the long model is sampled with its full 32K budget
    CUDA_VISIBLE_DEVICES=$GPUS "$PY" scripts/evaluate_vllm.py --datasets "$EVAL_DATASETS" --model "$SHORT" \
      --prompt-mode native --max-tokens "$NATIVE_SHORT_MAX_TOKENS" --max-model-len 4096 --temperature 0.0 --top-p 1.0 \
      --tensor-parallel-size "$NATIVE_TP" --output-dir "$OUT/native_short" ${LIMIT:+--limit "$LIMIT"}
    CUDA_VISIBLE_DEVICES=$GPUS "$PY" scripts/evaluate_vllm.py --datasets "$EVAL_DATASETS" --model "$LONG" \
      --prompt-mode native --max-tokens 32768 --max-model-len "$NATIVE_LONG_MAX_MODEL_LEN" --temperature 0.6 --top-p 0.95 \
      --tensor-parallel-size "$NATIVE_TP" --output-dir "$OUT/native_long" ${LIMIT:+--limit "$LIMIT"}
    ;;
  routing)
    run_eval routing_all_long  --method routing --prompt-mode native --routing-threshold 0.0  "${controller[@]}"
    run_eval routing_all_short --method routing --prompt-mode native --routing-threshold 1.01 "${controller[@]}"
    ;;
  fixed)
    run_eval ta     --method fixed_merge --prompt-mode shared --fixed-lambda 0.7 --merge-scope full_model
    run_eval acm_ta --method fixed_merge --prompt-mode shared --fixed-lambda 0.7 "${acm[@]}"
    ;;
  dalsa)
    for x in $LAMBDA_MAXES; do
      run_eval "dalsa_max${x/./p}" --method dalsa_bucketed --prompt-mode shared "${controller[@]}" "${acm[@]}" \
        --min-lambda 0.1 --max-lambda "$x" --lambda-values "$(lambdas_up_to "$x")"
    done
    ;;
  noconsensus)
    run_eval dalsa_noconsensus_max0p7 --method dalsa_bucketed --prompt-mode shared "${controller[@]}" "${acm[@]}" \
      --min-lambda 0.1 --max-lambda 0.7 --lambda-values "$(lambdas_up_to 0.7)" --disable-layer-consensus
    ;;
  static)
    for lam in 0.3 0.5 0.7; do
      run_eval "ties_${lam/./p}" --method fixed_merge --prompt-mode shared --fixed-lambda "$lam" --merge-scope ties_full --ties-density 0.2 --seed 42
      run_eval "dare_${lam/./p}" --method fixed_merge --prompt-mode shared --fixed-lambda "$lam" --merge-scope dare_full --dare-drop-rate 0.5 --seed 42
      run_eval "fim_${lam/./p}"  --method fixed_merge --prompt-mode shared --fixed-lambda "$lam" --merge-scope fim_full "${fim[@]}"
    done
    ;;
  dalsa_fim)
    for x in 0.7 0.9; do
      run_eval "dalsa_fim_max${x/./p}" --method dalsa_bucketed --prompt-mode shared "${controller[@]}" \
        --merge-scope fim_full "${fim[@]}" --min-lambda 0.1 --max-lambda "$x" --lambda-values "$(lambdas_up_to "$x")"
    done
    ;;
  summary)
    "$PY" scripts/summarize.py "$OUT"
    ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
  esac
done
