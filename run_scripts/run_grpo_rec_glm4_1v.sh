#!/usr/bin/bash


PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
nproc_per_node=${1:-"8"}
nnodes=${2:-"1"}
node_rank=${3:-"0"}
master_addr=${4:-"127.0.0.1"}
master_port=${5:-"15469"}

default_data_paths="${HOME}/datasets/VLM-R1/rec_jsons_processed/refcoco_train.jsonl:"
default_data_paths+="${HOME}/datasets/VLM-R1/rec_jsons_processed/refcocop_train.jsonl:"
default_data_paths+="${HOME}/datasets/VLM-R1/rec_jsons_processed/refcocog_train.jsonl"
data_paths=${6:-"${default_data_paths}"}

default_image_folders="${HOME}/datasets/coco2014/images:"
default_image_folders+="${HOME}/datasets/coco2014/images:"
default_image_folders+="${HOME}/datasets/coco2014/images"
image_folders=${7:-"${default_image_folders}"}

model_path=${8:-"${HOME}/ckpts/GLM-4.1V-9B-Thinking"}
is_reward_customized_from_vlm_module=${9:-"True"}
exp_name=${10:-"GLM-4.1V-9B-Thinking-baseline"}
task_type=${11:-"rec"}
task_type_for_format_reward=${12:-"rec"}
max_steps=${13:-"200"}
debug_mode=${14:-"false"}
wandb_proj_name=${15:"RL-post-training"}

source ${HOME}/.bashrc  # for CentOS
source ${HOME}/depends/anaconda3/etc/profile.d/conda.sh  # for Ubuntu 22.04
conda activate

cd ${PROJECT_ROOT}/src/open-r1-multimodal
export DEBUG_MODE=${debug_mode}  # Enable Debug if you want to see the rollout of model during RL
export WANDB_PROJECT=${wandb_proj_name}

output_dir=${HOME}/outputs/VLM-R1/${exp_name}
if [ -d ${output_dir} ]; then
  rm -rf ${output_dir}
fi
mkdir -p ${output_dir}
export log_path=${output_dir}/output.log


log_func() {
  "$@" 2>&1 | tee --append ${log_path}
}


echo "data paths for training: ${data_paths}" 2>&1 | tee ${log_path}
log_func echo "image folders for training: ${image_folders}"
log_func echo "start GRPO traning at `date +%Y-%m-%d-%H-%M-%S`"

log_func torchrun --nproc_per_node=${nproc_per_node} \
             --nnodes=${nnodes} \
             --node_rank=${node_rank} \
             --master_addr=${master_addr} \
             --master_port=${master_port} \
           src/open_r1/grpo_jsonl.py \
             --use_vllm False \
             --output_dir ${output_dir} \
             --resume_from_checkpoint True \
             --model_name_or_path ${model_path} \
             --data_file_paths ${data_paths} \
             --image_folders ${image_folders} \
             --is_reward_customized_from_vlm_module ${is_reward_customized_from_vlm_module} \
             --task_type ${task_type} \
             --task_type_for_format_reward ${task_type_for_format_reward} \
             --iou_type giou \
             --per_device_train_batch_size 8 \
             --gradient_accumulation_steps 1 \
             --gradient_checkpointing true \
             --temperature 1.0 \
             --top_p 1.0 \
             --top_k 50 \
             --logging_steps 1 \
             --max_steps ${max_steps} \
             --num_train_epochs 2 \
             --learning_rate 1e-6 \
             --vision_learning_rate 1e-6 \
             --projector_learning_rate 1e-6 \
             --bf16 \
             --ddp_timeout 7200 \
             --attn_implementation flash_attention_2 \
             --freeze_vision_modules False \
             --freeze_projector_modules False \
             --freeze_language_modules False \
             --run_name ${exp_name} \
             --data_seed 42 \
             --vision_lora False \
             --language_lora False \
             --lora_r 16 \
             --lora_alpha 32 \
             --lora_dropout 0.05 \
             --save_steps 50 \
             --save_total_limit 1 \
             --num_generations 8 \
             --num_iterations 1 \
             --max_completion_length 2048 \
             --reward_funcs accuracy \
             --beta 0.0 \
             --report_to wandb \
             --run_name ${exp_name} \
             --deepspeed ${PROJECT_ROOT}/src/open-r1-multimodal/local_scripts/zero3.json

log_func echo "end GRPO traning at `date +%Y-%m-%d-%H-%M-%S`"
