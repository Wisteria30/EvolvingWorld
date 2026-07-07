#!/bin/bash
# ==============================================================
# EvolvingWorld - Unified Training Launcher
# ==============================================================
#
# Usage:
#   bash training/run.sh --model model_a --mode full [OPTIONS]
#
# Required:
#   --model model_a|model_b    Which model to train
#   --mode  full|balanced|custom  Data mixing mode
#
# Options:
#   --with-tulu3               Mix Tulu3 general-domain data
#   --gpu N                    GPU index (sets CUDA_VISIBLE_DEVICES)
#   --model-name-or-path PATH  Override base model
#   --config PATH              Override train_config.yaml
#   --accelerate-config PATH   Override accelerate.yaml
#   Extra args forwarded to llamafactory-cli train
#
# Examples:
#   bash training/run.sh --model model_a --mode full --gpu 0
#   bash training/run.sh --model model_b --mode balanced --with-tulu3 --gpu 5
#   bash training/run.sh --model model_a --mode full --model-name-or-path Qwen/Qwen2.5-7B-Instruct
# ==============================================================

set -euo pipefail

source "training/common_training.sh"

# ── Defaults ──────────────────────────────────────────────────────
CONFIG_PATH="training/train_config.yaml"
ACCELERATE_CONFIG_PATH="training/accelerate.yaml"
MODEL_NAME_OR_PATH_OVERRIDE=""
WITH_TULU3=false
GPU_ID=""
# Training hyperparameters (override train_config.yaml defaults)
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=64
PER_DEVICE_EVAL_BATCH_SIZE=1
PACKING=false

# ── Parse arguments ───────────────────────────────────────────────
MODEL=""
MODE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)           MODEL="$2";           shift 2 ;;
        --mode)            MODE="$2";            shift 2 ;;
        --with-tulu3)      WITH_TULU3=true;      shift ;;
        --gpu)             GPU_ID="$2";          shift 2 ;;
        --model-name-or-path) MODEL_NAME_OR_PATH_OVERRIDE="$2"; shift 2 ;;
        --config)          CONFIG_PATH="$2";     shift 2 ;;
        --accelerate-config) ACCELERATE_CONFIG_PATH="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 --model <model_a|model_b> --mode <full|balanced|custom> [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --model model_a|model_b"
            echo "  --mode  full|balanced|custom"
            echo ""
            echo "Options:"
            echo "  --with-tulu3              Mix Tulu3 general-domain data"
            echo "  --gpu N                   GPU index"
            echo "  --model-name-or-path PATH Override base model"
            echo "  --config PATH             Override train_config.yaml"
            echo "  --accelerate-config PATH  Override accelerate config"
            exit 0
            ;;
        *)                  break ;;  # remaining args → llamafactory
    esac
done

if [ -z "${MODEL}" ] || [ -z "${MODE}" ]; then
    echo "ERROR: --model and --mode are required."
    echo "Usage: $0 --model <model_a|model_b> --mode <full|balanced|custom> [OPTIONS]"
    exit 1
fi

if [[ "${MODEL}" != "model_a" && "${MODEL}" != "model_b" ]]; then
    echo "ERROR: --model must be model_a or model_b, got: ${MODEL}"
    exit 1
fi

if [[ "${MODE}" != "full" && "${MODE}" != "balanced" && "${MODE}" != "custom" ]]; then
    echo "ERROR: --mode must be full, balanced, or custom, got: ${MODE}"
    exit 1
fi

# ── GPU binding ───────────────────────────────────────────────────
if [ -n "${GPU_ID}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

# ── Build extra args ──────────────────────────────────────────────
EXTRA_ARGS=(
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --packing "${PACKING}"
    "$@"
)

# ── Launch ────────────────────────────────────────────────────────
run_training_workflow \
    "${MODEL}" \
    "${MODE}" \
    "${WITH_TULU3}" \
    "${CONFIG_PATH}" \
    "${ACCELERATE_CONFIG_PATH}" \
    "${MODEL_NAME_OR_PATH_OVERRIDE}" \
    "${EXTRA_ARGS[@]}"
