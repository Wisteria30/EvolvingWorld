#!/bin/bash
# Batch evaluation: evaluate specified simulation outputs with a judge model in order
set -e

INPUT_BASE="simulation/outputs"
OUTPUT_BASE="evaluation/results"
JUDGE_MODEL="gemini-2.5-pro"
NUM_WORKERS=100
RESUME=false
SAMPLE_RATIO=1.0

# Parse --judge argument
MODELS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --judge)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --sample-ratio|--sample_ratio)
            SAMPLE_RATIO="$2"
            shift 2
            ;;
        *)
            MODELS+=("$1")
            shift
            ;;
    esac
done

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Usage: $0 [--judge <judge_model>] [--resume] <model_dir_1> [model_dir_2] ..."
    echo ""
    echo "Options:"
    echo "  --judge <model>    Specify judge model (default: gemini-2.5-pro)"
    echo "  --workers <num>    Parallel worker count (default: 100)"
    echo "  --sample-ratio <r> Test sample ratio 0.0-1.0 (default: 1.0, all samples)"
    echo "  --resume           Skip completed samples, only rerun missing or failed"
    echo ""
    echo "Example: $0 --judge gpt-4o gpt-4o claude-opus-4-6"
    echo "       $0 --workers 50 --resume gpt-4o claude-opus-4-6"
    echo ""
    echo "Available model directories:"
    ls -1 "${INPUT_BASE}" | grep -v '^\.' | sed 's/^/  /'
    exit 1
fi

#  Generate output directory suffix from judge model name (replace special chars with underscores)
JUDGE_SUFFIX=$(echo "${JUDGE_MODEL}" | tr '.-' '__')

TOTAL=${#MODELS[@]}
CURRENT=0
FAILED=()

echo "=========================================="
echo " Batch evaluation start"
echo " Judge Model: ${JUDGE_MODEL}"
echo " Workers: ${NUM_WORKERS}"
echo " Sample Ratio: ${SAMPLE_RATIO}"
echo " Resume: ${RESUME}"
echo " Models to evaluate: ${TOTAL}"
echo "=========================================="

for MODEL in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    INPUT_DIR="${INPUT_BASE}/${MODEL}"
    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}_${JUDGE_SUFFIX}_final"

    echo ""
    echo "------------------------------------------"
    echo " [${CURRENT}/${TOTAL}] Evaluating: ${MODEL}"
    echo " Input: ${INPUT_DIR}"
    echo " Output: ${OUTPUT_DIR}"
    echo "------------------------------------------"

    if [ ! -d "${INPUT_DIR}" ]; then
        echo " ⚠ Input directory not found, skipping"
        FAILED+=("${MODEL} (directory not found)")
        continue
    fi

    RESUME_FLAG=""
    if [ "${RESUME}" = true ]; then
        RESUME_FLAG="--resume"
    fi

    RATIO_FLAG=""
    if awk "BEGIN{exit !(${SAMPLE_RATIO} < 1.0)}"; then
        RATIO_FLAG="--sample_ratio ${SAMPLE_RATIO}"
    fi

    if python "evaluation/main.py" \
        --input_dir "${INPUT_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --judge_model "${JUDGE_MODEL}" \
        --num_workers "${NUM_WORKERS}" \
        ${RATIO_FLAG} \
        ${RESUME_FLAG}; then
        echo " ✓ ${MODEL} evaluation complete"
    else
        echo " ✗ ${MODEL} evaluation failed (exit code: $?)"
        FAILED+=("${MODEL}")
    fi
done

echo ""
echo "=========================================="
echo " Batch evaluation end"
echo " Succeeded: $((TOTAL - ${#FAILED[@]}))/${TOTAL}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo " Failed:"
    for F in "${FAILED[@]}"; do
        echo "   - ${F}"
    done
fi
echo "=========================================="
