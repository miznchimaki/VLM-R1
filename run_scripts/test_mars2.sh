#!/usr/bin/bash


source $HOME/.bashrc
source $HOME/depends/anaconda3/etc/profile.d/conda.sh  # for bash5.1 (Ubuntu 22.04)

MODEL_TYPE=${1:-"qwen2_5_vl"}
CKPT_NAME=${2:-"Qwen2.5VL-3B-VLM-R1-REC-500steps"}
OUTPUT_NAME=${3:-"VLM-R1-Qwen2.5-VL-3B-REC-500steps-baseline-results"}
BSZ=${4:-"1"}
DATA_DIR=${5:-"ICCV-2025-Workshops-MARS2"}
TEST_DATASETS=${6:-"['VG-RS']"}
GPU_DEVICES=${7:-"0,1,2,3,4,5,6,7"}

OUTPUT_PATH="${HOME}/outputs/MARS2/${OUTPUT_NAME}"
rm --recursive --force ${OUTPUT_PATH}
if [ ! -e ${OUTPUT_PATH} ]; then
    mkdir -p ${OUTPUT_PATH}
fi
LOG_PATH="${OUTPUT_PATH}/output.log"

export CUDA_VISIBLE_DEVICES=${GPU_DEVICES}
IFS="," read -ra GPU_IDS <<< "${GPU_DEVICES}"
NPROC_PER_NODE="${#GPU_IDS[@]}"

conda activate
cd $HOME/projects/VLM-R1/src/eval/
torchrun --nproc-per-node ${NPROC_PER_NODE} test_mars2.py "${CKPT_NAME}" \
                                                          "${OUTPUT_NAME}" \
                                                          "${BSZ}" \
                                                          "${DATA_DIR}" \
                                                          "${TEST_DATASETS}" 2>&1 | tee ${LOG_PATH}
