#!/bin/bash

set -euo pipefail

resolve_path() {
    python3 -c "from pathlib import Path; print(Path('$1').expanduser().resolve())"
}

read_config_value() {
    local config_path="$1"
    local key="$2"
    local default_value="$3"
    python3 -c "
import yaml
with open('${config_path}') as f:
    c = yaml.safe_load(f) or {}
print(c.get('${key}', '${default_value}'))
"
}

get_model_tasks() {
    local model="$1"
    if [ "${model}" = "model_a" ]; then
        echo "scene_cast location_scenario next_character world_update"
    else
        echo "interaction_gen character_update motivation_update"
    fi
}

infer_template_from_model_path() {
    local model_path
    model_path=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    case "${model_path}" in
        *qwen3-vl*|*qwen3_vl*|*qwen3-omni*|*qwen3_omni*)
            echo ""
            ;;
        *qwen3*instruct*|*qwen3*chat*|*qwen3*nothink*)
            echo "qwen3_nothink"
            ;;
        *qwen3*thinking*|*qwen3*base*|*qwen3*)
            echo "qwen3"
            ;;
        *qwen*)
            echo "qwen"
            ;;
        *llama-3*|*meta-llama*|*llama3*)
            echo "llama3"
            ;;
        *)
            echo ""
            ;;
    esac
}

make_model_tag() {
    local model_path="$1"
    local tag="${model_path##*/}"
    tag=$(printf '%s' "${tag}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')
    echo "${tag:-model}"
}

run_training_workflow() {
    local model="$1"
    local mode="$2"
    local with_tulu3="$3"
    local config_path="$4"
    local accelerate_config_path="$5"
    local model_name_or_path_override="$6"
    shift 6
    local extra_args=("$@")

    config_path="$(resolve_path "${config_path}")"
    if [ ! -f "${config_path}" ]; then
        echo "ERROR: Config file not found: ${config_path}"
        exit 1
    fi

    if [ -n "${accelerate_config_path}" ]; then
        accelerate_config_path="$(resolve_path "${accelerate_config_path}")"
        if [ ! -f "${accelerate_config_path}" ]; then
            echo "ERROR: Accelerate config file not found: ${accelerate_config_path}"
            exit 1
        fi
    fi

    local model_path
    model_path=$(read_config_value "${config_path}" model_name_or_path "meta-llama/Llama-3.1-8B-Instruct")
    if [ -n "${model_name_or_path_override}" ]; then
        model_path="${model_name_or_path_override}"
    fi

    local template
    template=$(read_config_value "${config_path}" template "llama3")
    local inferred_template
    inferred_template=$(infer_template_from_model_path "${model_path}")
    if [ -n "${model_name_or_path_override}" ] && [ -n "${inferred_template}" ]; then
        template="${inferred_template}"
    fi

    local model_tag
    model_tag=$(make_model_tag "${model_path}")

    echo "============================================================"
    echo " EvolvingWorld Training Pipeline"
    echo "============================================================"
    echo " Model:      ${model}"
    echo " Mode:       ${mode}"
    echo " Base:       ${model_path}"
    echo " Template:   ${template}"
    echo " Tulu3:      ${with_tulu3}"
    echo " Config:     ${config_path}"
    if [ -n "${accelerate_config_path}" ]; then
        echo " Accelerate: ${accelerate_config_path}"
    fi
    echo "============================================================"

    echo ""
    echo "[Step 1] Preparing training data..."

    local prepare_cmd=(
        python3 "training/prepare_data.py"
        --model "${model}"
        --mode "${mode}"
    )

    local tulu3_ratio=""
    if [ "${with_tulu3}" = "true" ]; then
        local tulu3_path
        tulu3_path=$(python3 -c "
import yaml
from pathlib import Path
with open('${config_path}') as f:
    c = yaml.safe_load(f) or {}
p = c.get('tulu3_path', '') or ''
if p and p != 'null' and p != 'None':
    print(str(Path(p).expanduser().resolve()))
else:
    print('')
")
        tulu3_ratio=$(python3 -c "
import yaml
with open('${config_path}') as f:
    c = yaml.safe_load(f) or {}
print(c.get('tulu3_ratio', 1.0))
")
        if [ -z "${tulu3_path}" ] || [ "${tulu3_path}" = "None" ] || [ "${tulu3_path}" = "null" ]; then
            echo "ERROR: --with-tulu3 specified but tulu3_path is not set in config!"
            echo "Please set tulu3_path in ${config_path}"
            exit 1
        fi
        prepare_cmd+=(--tulu3_path "${tulu3_path}" --tulu3_ratio "${tulu3_ratio}")
    fi

    if [ "${mode}" = "custom" ]; then
        local custom_ratios
        custom_ratios=$(python3 -c "
import json
import yaml
with open('${config_path}') as f:
    c = yaml.safe_load(f) or {}
print(json.dumps(c.get('custom_ratios', {}).get('${model}', {})))
")
        prepare_cmd+=(--custom_ratios "${custom_ratios}")
    fi

    local data_dir
    data_dir=$(python3 -c "
import yaml
from pathlib import Path
with open('${config_path}') as f:
    c = yaml.safe_load(f) or {}
p = c.get('data_dir', 'dataset/train')
print(Path(p).expanduser().resolve())
")

    local data_output_base
    data_output_base=$(python3 -c "
import yaml
from pathlib import Path
with open('${config_path}') as f:
    c = yaml.safe_load(f) or {}
p = c.get('data_output_dir', 'training/prepared_data')
print(Path(p).expanduser().resolve())
")

    local seed
    seed=$(python3 -c "
import yaml
with open('${config_path}') as f:
    c = yaml.safe_load(f) or {}
print(c.get('seed', 42))
")

    local eval_holdout_ratio
    eval_holdout_ratio=$(python3 -c "
import yaml
with open('${config_path}') as f:
    c = yaml.safe_load(f) or {}
print(c.get('eval_holdout_ratio', 0.0))
")

    local prepare_cutoff_len
    prepare_cutoff_len=$(python3 -c "
import yaml
with open('${config_path}') as f:
    c = yaml.safe_load(f) or {}
v = c.get('cutoff_len', None)
print(v if v is not None else '')
")

    local prepare_model_name_or_path="${model_path}"

    local output_dir_name="${model}_${mode}_${model_tag}"
    if [ "${with_tulu3}" = "true" ]; then
        output_dir_name="${output_dir_name}_tulu3_${tulu3_ratio}"
    fi
    local output_dir="${data_output_base}/${output_dir_name}"
    mkdir -p "${output_dir}"

    prepare_cmd+=(
        --data_dir "${data_dir}"
        --output_dir "${output_dir}"
        --eval_holdout_ratio "${eval_holdout_ratio}"
        --seed "${seed}"
    )

    if [ -n "${prepare_cutoff_len}" ]; then
        prepare_cmd+=(--cutoff_len "${prepare_cutoff_len}")
    fi
    if [ -n "${prepare_model_name_or_path}" ]; then
        prepare_cmd+=(--model_name_or_path "${prepare_model_name_or_path}")
    fi

    printf 'Running:'
    printf ' %q' "${prepare_cmd[@]}"
    printf '\n'
    "${prepare_cmd[@]}"

    echo ""
    echo "[Step 2] Launching LLaMA-Factory training..."

    local tulu3_suffix=""
    if [ "${with_tulu3}" = "true" ]; then
        tulu3_suffix="_tulu3_${tulu3_ratio}"
    fi

    local dataset_file="${output_dir}/${model}_${mode}${tulu3_suffix}_sharegpt.jsonl"
    if [ ! -f "${dataset_file}" ]; then
        echo "ERROR: Prepared data not found: ${dataset_file}"
        exit 1
    fi

    local stage
    stage=$(read_config_value "${config_path}" stage "sft")
    local finetuning_type
    finetuning_type=$(read_config_value "${config_path}" finetuning_type "lora")
    local lora_rank
    lora_rank=$(read_config_value "${config_path}" lora_rank 64)
    local lora_alpha
    lora_alpha=$(read_config_value "${config_path}" lora_alpha 128)
    local lora_dropout
    lora_dropout=$(read_config_value "${config_path}" lora_dropout 0.05)
    local lora_target
    lora_target=$(read_config_value "${config_path}" lora_target "all")
    local cutoff_len
    cutoff_len=$(read_config_value "${config_path}" cutoff_len 8192)
    local packing
    packing=$(read_config_value "${config_path}" packing false)
    local num_epochs
    num_epochs=$(read_config_value "${config_path}" num_train_epochs 3.0)
    local batch_size
    batch_size=$(read_config_value "${config_path}" per_device_train_batch_size 2)
    local grad_accum
    grad_accum=$(read_config_value "${config_path}" gradient_accumulation_steps 8)
    local learning_rate
    learning_rate=$(read_config_value "${config_path}" learning_rate "2.0e-5")
    local lr_scheduler
    lr_scheduler=$(read_config_value "${config_path}" lr_scheduler_type "cosine")
    local warmup_ratio
    warmup_ratio=$(read_config_value "${config_path}" warmup_ratio 0.1)
    local weight_decay
    weight_decay=$(read_config_value "${config_path}" weight_decay 0.01)
    local max_grad_norm
    max_grad_norm=$(read_config_value "${config_path}" max_grad_norm 1.0)
    local bf16
    bf16=$(read_config_value "${config_path}" bf16 true)
    local grad_ckpt
    grad_ckpt=$(read_config_value "${config_path}" gradient_checkpointing true)
    local flash_attn
    flash_attn=$(read_config_value "${config_path}" flash_attn "fa2")
    local logging_steps
    logging_steps=$(read_config_value "${config_path}" logging_steps 10)
    local save_steps
    save_steps=$(read_config_value "${config_path}" save_steps 500)
    local save_total_limit
    save_total_limit=$(read_config_value "${config_path}" save_total_limit 3)
    local report_to
    report_to=$(read_config_value "${config_path}" report_to "none")
    local preproc_workers
    preproc_workers=$(read_config_value "${config_path}" preprocessing_num_workers 16)
    local deepspeed
    deepspeed=$(read_config_value "${config_path}" deepspeed "None")
    local train_output_base
    train_output_base=$(read_config_value "${config_path}" train_output_dir "training/outputs")
    if [ -z "${train_output_base}" ] || [ "${train_output_base}" = "None" ] || [ "${train_output_base}" = "null" ]; then
        train_output_base="training/outputs"
    fi
    train_output_base=$(python3 -c "from pathlib import Path; print(Path('${train_output_base}').expanduser().resolve())")

    if [ "${deepspeed}" != "None" ] && [ "${deepspeed}" != "null" ]; then
        deepspeed=$(python3 -c "from pathlib import Path; print(Path('${deepspeed}').expanduser().resolve())")
        if [ ! -f "${deepspeed}" ]; then
            echo "ERROR: DeepSpeed config not found: ${deepspeed}"
            exit 1
        fi
    fi

    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local train_output_dir="${train_output_base}/${model}_${mode}_${model_tag}${tulu3_suffix}_${timestamp}"
    mkdir -p "${train_output_dir}"

    local llamafactory_cli_bin
    llamafactory_cli_bin=$(command -v llamafactory-cli || true)
    if [ -z "${llamafactory_cli_bin}" ]; then
        echo "ERROR: llamafactory-cli not found in PATH"
        exit 1
    fi

    local cmd=()
    if [ -n "${accelerate_config_path}" ]; then
        cmd=(accelerate launch --config_file "${accelerate_config_path}" "${llamafactory_cli_bin}" train)
    else
        cmd=("${llamafactory_cli_bin}" train)
    fi

    cmd+=(
        --do_train
        --model_name_or_path "${model_path}"
        --stage "${stage}"
        --finetuning_type "${finetuning_type}"
        --template "${template}"
        --cutoff_len "${cutoff_len}"
        --packing "${packing}"
        --preprocessing_num_workers "${preproc_workers}"
        --dataset_dir "${output_dir}"
        --dataset "evolvingworld_${model}"
        --num_train_epochs "${num_epochs}"
        --per_device_train_batch_size "${batch_size}"
        --gradient_accumulation_steps "${grad_accum}"
        --learning_rate "${learning_rate}"
        --lr_scheduler_type "${lr_scheduler}"
        --warmup_ratio "${warmup_ratio}"
        --weight_decay "${weight_decay}"
        --max_grad_norm "${max_grad_norm}"
        --bf16 "${bf16}"
        --gradient_checkpointing "${grad_ckpt}"
        --flash_attn "${flash_attn}"
        --logging_steps "${logging_steps}"
        --save_steps "${save_steps}"
        --output_dir "${train_output_dir}"
        --report_to "${report_to}"
        --overwrite_output_dir
    )

    if [ -n "${save_total_limit}" ] && [ "${save_total_limit}" != "None" ] && [ "${save_total_limit}" != "null" ]; then
        cmd+=(--save_total_limit "${save_total_limit}")
    fi

    if [ "${finetuning_type}" = "lora" ]; then
        cmd+=(
            --lora_rank "${lora_rank}"
            --lora_alpha "${lora_alpha}"
            --lora_dropout "${lora_dropout}"
            --lora_target "${lora_target}"
        )
    fi

    if [ "${deepspeed}" != "None" ] && [ "${deepspeed}" != "null" ]; then
        cmd+=(--deepspeed "${deepspeed}")
    fi

    local do_eval
    do_eval=$(read_config_value "${config_path}" do_eval false)
    if [ "${do_eval}" = "True" ] || [ "${do_eval}" = "true" ]; then
        local eval_strategy
        eval_strategy=$(read_config_value "${config_path}" eval_strategy "steps")
        local eval_steps
        eval_steps=$(read_config_value "${config_path}" eval_steps 500)
        local eval_on_each_task
        eval_on_each_task=$(python3 -c "
import yaml
with open('${config_path}') as f:
    c = yaml.safe_load(f) or {}
v = c.get('eval_on_each_task', c.get('eval_on_each_dataset', True))
print(v)
")

        local eval_datasets=()
        if [ "${eval_on_each_task}" = "True" ] || [ "${eval_on_each_task}" = "true" ]; then
            local task_name
            for task_name in $(get_model_tasks "${model}"); do
                local task_eval_file="${output_dir}/${model}_${task_name}_${mode}${tulu3_suffix}_eval.jsonl"
                if [ ! -f "${task_eval_file}" ]; then
                    echo "ERROR: do_eval=true and eval_on_each_task=true but task eval dataset not found: ${task_eval_file}"
                    exit 1
                fi
                eval_datasets+=("evolvingworld_${model}_${task_name}_eval")
            done
        fi

        if [ ${#eval_datasets[@]} -eq 0 ]; then
            echo "ERROR: do_eval=true but no task eval datasets were found."
            echo "Please set eval_holdout_ratio > 0 in ${config_path} and re-run."
            exit 1
        fi

        local eval_dataset_arg
        eval_dataset_arg=$(IFS=,; echo "${eval_datasets[*]}")
        cmd+=(
            --do_eval
            --eval_strategy "${eval_strategy}"
            --eval_dataset "${eval_dataset_arg}"
        )
        if [ "${eval_on_each_task}" = "True" ] || [ "${eval_on_each_task}" = "true" ]; then
            cmd+=(--eval_on_each_dataset true)
        fi
        if [ "${eval_strategy}" = "steps" ]; then
            cmd+=(--eval_steps "${eval_steps}")
        fi
    fi

    if [ ${#extra_args[@]} -gt 0 ]; then
        cmd+=("${extra_args[@]}")
    fi

    echo ""
    echo "Training command:"
    echo "============================================================"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    echo "============================================================"
    echo ""

    printf '%q ' "${cmd[@]}" > "${train_output_dir}/training_command.txt"
    printf '\n' >> "${train_output_dir}/training_command.txt"
    cp "${config_path}" "${train_output_dir}/train_config.yaml"

    "${cmd[@]}"

    echo ""
    echo "============================================================"
    echo " Training completed!"
    echo " Output: ${train_output_dir}"
    echo "============================================================"
}
