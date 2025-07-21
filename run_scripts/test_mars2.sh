#!/usr/bin/bash


source $HOME/.bashrc
source $HOME/depends/anaconda3/etc/profile.d/conda.sh  # for bash5.1 (Ubuntu 22.04)

CKPT_NAME=${1:-"Qwen2.5VL-3B-VLM-R1-REC-500steps-baseline"}
OUTPUT_NAME=${2:-"VLM-R1-Qwen2.5-VL-3B-REC-500steps-baseline-results"}
BSZ=${3:-"128"}
DATA_DIR=${4:-"ICCV-2025-Workshops-MARS2"}
TEST_DATASETS=${5:-"['VG-RS']"}
GPU_DEVICES=${6:-"0,1,2,3,4,5,6,7"}

export CUDA_VISIBLE_DEVICES=${GPU_DEVICES}
IFS="," read -ra GPU_IDS <<< "${GPU_DEVICES}"
NPROC_PER_NODE="${#GPU_IDS[@]}"

cd $HOME/projects/VLM-R1/src/eval/
torchrun --nproc-per-node ${NPROC_PER_NODE} test_mars2.py "${CKPT_NAME}" \
                                                          "${OUTPUT_NAME}" \
                                                          "${BSZ}" \
                                                          "${DATA_DIR}" \
                                                          "${TEST_DATASETS}"
