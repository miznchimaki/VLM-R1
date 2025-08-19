from transformers import Qwen2_5_VLForConditionalGeneration, Glm4vForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os, sys
import shutil
from pathlib import Path
from datetime import timedelta
import random
from PIL import Image

import torch.distributed as dist

import warnings


warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=7200))
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    return local_rank, world_size, rank

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"
print(f"Process {rank} using {device}")
main_rank = 0
HOME_DIR = os.getenv("HOME", None)


DEFAULT_MODEL_TYPE = "qwen2_5_vl"
try:
    tmp_model_type = sys.argv[1]
    if tmp_model_type.lower() == "none":
        MODEL_TYPE = DEFAULT_MODEL_TYPE
    else:
        MODEL_TYPE = tmp_model_type
except IndexError as _:
    MODEL_TYPE = DEFAULT_MODEL_TYPE


DEFAULT_CKPT_NAME = "Qwen2.5VL-3B-VLM-R1-REC-500steps"
try:
    tmp_ckpt_name = sys.argv[2]
    if tmp_ckpt_name.lower() == "none":
        CKPT_NAME = DEFAULT_CKPT_NAME
    else:
        CKPT_NAME = tmp_ckpt_name
except IndexError as _:
    CKPT_NAME = DEFAULT_CKPT_NAME
MODEL_PATH = Path(HOME_DIR) / 'ckpts' / CKPT_NAME


DEFAULT_OUTPUT_NAME = f"VLM-R1-Qwen2.5-VL-3B-REC-500steps-baseline-results"
try:
    tmp_output_name = sys.argv[3]
    if tmp_output_name.lower() == "none":
        OUTPUT_NAME = DEFAULT_OUTPUT_NAME
    else:
        OUTPUT_NAME = tmp_output_name
except IndexError as _:
    OUTPUT_NAME = DEFAULT_OUTPUT_NAME
OUTPUT_PATH = Path(HOME_DIR) / 'outputs' / 'MARS2' / OUTPUT_NAME


DEFAULT_BSZ = 1
try:
    tmp_bsz = sys.argv[4]
    if tmp_bsz.lower() == "none":
        BSZ = DEFAULT_BSZ
    else:
        try:
            BSZ = int(tmp_bsz)
        except ValueError as _:
            BSZ = DEFAULT_BSZ
except ValueError as _:
    BSZ = DEFAULT_BSZ

DEFAULT_DATA_DIR = 'ICCV-2025-Workshops-MARS2'
try:
    tmp_data_dir = sys.argv[5]
    if tmp_data_dir.lower() == "none":
        DATA_DIR = DEFAULT_DATA_DIR
    else:
        DATA_DIR = tmp_data_dir
except IndexError as _:
    DATA_DIR = DEFAULT_DATA_DIR
DATA_ROOT = Path(HOME_DIR) / 'datasets' / DATA_DIR


# TODO: need modifications
DEFAULT_TEST_DATASETS = ['VG-RS']
try:
    tmp_test_datasets = sys.argv[6]
    if tmp_test_datasets.lower() == "none":
        TEST_DATASETS = DEFAULT_TEST_DATASETS
    else:
        try:
            TEST_DATASETS = eval(tmp_test_datasets)
        except Exception as _:
            TEST_DATASETS = DEFAULT_TEST_DATASETS
except IndexError as _:
    TEST_DATASETS = DEFAULT_TEST_DATASETS

# TODO: need modifications
IMAGE_DIRS = []
for test_ds in TEST_DATASETS:
    IMAGE_DIRS.append(str(DATA_ROOT / test_ds / (test_ds + "-images")))


#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
if MODEL_TYPE in ("qwen2_5_vl", "mimo_vl"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank}, 
    )
elif MODEL_TYPE == "glm4v":
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank}
    )
else:
    raise ValueError(f"invalid specified model type: {MODEL_TYPE}, only qwen2_5_vl, glm4v are supported")


# default processer
if MODEL_TYPE in ("qwen2_5_vl", "mimo_vl"):
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
elif MODEL_TYPE == "glm4v":
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)


def extract_bbox_answer(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    if MODEL_TYPE == "qwen2_5_vl":
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern_1 = r'\{.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]\s*.*\}'
        bbox_pattern_2 = r'.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]\s*.*'
        bbox_pattern_3 = r'.*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]\s*.*'
        bbox_pattern_4 = r'.*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*.*'
    elif MODEL_TYPE == "glm4v":
        answer_tag_pattern = r'<answer><\|begin_of_box\|>(.*?)<\|end_of_box\|></answer>'
        bbox_pattern_1 = r'.*\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]].*'
        bbox_pattern_2 = r'.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\].*'
        bbox_pattern_3 = r'.*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\).*'
        bbox_pattern_4 = r'.*<(\d+),\s*(\d+),\s*(\d+),\s*(\d+)>.*'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        bbox_match_1 = re.search(bbox_pattern_1, content_answer, re.DOTALL)
        bbox_match_2 = re.search(bbox_pattern_2, content_answer, re.DOTALL)
        bbox_match_3 = re.search(bbox_pattern_3, content_answer, re.DOTALL)
        bbox_match_4 = re.search(bbox_pattern_4, content_answer, re.DOTALL)
        if bbox_match_1:
            bbox = [float(bbox_match_1.group(1)), float(bbox_match_1.group(2)), float(bbox_match_1.group(3)), float(bbox_match_1.group(4))]
            return bbox
        if bbox_match_2:
            bbox = [float(bbox_match_2.group(1)), float(bbox_match_2.group(2)), float(bbox_match_2.group(3)), float(bbox_match_2.group(4))]
            return bbox
        if bbox_match_3:
            bbox = [float(bbox_match_3.group(1)), float(bbox_match_3.group(2)), float(bbox_match_3.group(3)), float(bbox_match_3.group(4))]
            return bbox
        if bbox_match_4:
            bbox = [float(bbox_match_4.group(1)), float(bbox_match_4.group(2)), float(bbox_match_4.group(3)), float(bbox_match_4.group(4))]
            return bbox
        return None
    else:
        return None


def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2] - 1, box2[2] - 1)
    inter_y2 = min(box1[3] - 1, box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1)*(inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return float(inter) / union


def resize_bbox(bbox, input_height, input_width, image_height, image_width):
    if MODEL_TYPE in ("qwen2_5_vl", "mimo_vl"):
        bbox[0] = bbox[0] / input_width * image_width
        bbox[1] = bbox[1] / input_height * image_height
        bbox[2] = bbox[2] / input_width * image_width
        bbox[3] = bbox[3] / input_height * image_height
    elif MODEL_TYPE == "glm4v":
        bbox[0] = (bbox[0] / 999 * (input_width- 1)) / input_width * image_width
        bbox[1] = (bbox[1] / 999 * (input_height- 1)) / input_height * image_height
        bbox[2] = (bbox[2] / 999 * (input_width- 1)) / input_width * image_width
        bbox[3] = (bbox[3] / 999 * (input_height- 1)) / input_height * image_height
    return bbox


# TODO: maybe need variable named `num_samples`, maybe or not
num_samples = 2000
for idx, ds in enumerate(TEST_DATASETS):
    if rank == main_rank:
        print(f"Processing {ds}...")
    # TODO: need modifications
    ds_path = os.path.join(str(DATA_ROOT), f"{ds}.json")
    data = json.load(open(ds_path, "r"))
    random.seed(42)
    random.shuffle(data)
    if MODEL_TYPE == "qwen2_5_vl":
        # QUESTION_TEMPLATE = ("Please provide the bounding box coordinates of the region this sentence describes: {query}."
        #                      "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.")
        QUESTION_TEMPLATE = ("First output the thinking process in <think> </think> tages and then output the final answer " 
                             "in <answer> </answer> tags. Please provide the bounding box coordinates of the region "
                             "this sentence describes: {query}.")
    elif MODEL_TYPE == "glm4v":
        QUESTION_TEMPLATE = "Please provide the bounding box coordinates of the region this sentence describes in the format [x_min, y_min, x_max, y_max]: {query}."
    elif MODEL_TYPE == "mimo_vl":
        QUESTION_TEMPLATE = ("Please provide the bounding box coordinates of the region this sentence "
                             "describes in the format [x_min, y_min, x_max, y_max]: {query}.")

    # Split data for distributed evaluation
    per_rank_data = len(data) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
    rank_data = data[start_idx: end_idx]

    messages = []
    for x in rank_data:
        # TODO: need modifications
        image_path = "file://" + os.path.join(IMAGE_DIRS[idx], x['image'])
        message = [
            # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": image_path
                },
                {
                    "type": "text",
                    # TODO: need modifications
                    "text": QUESTION_TEMPLATE.format(query=x['problem'])
                }
            ]
        }]
        messages.append(message)
    rank_outputs = [] # List to store answers for this rank
    all_outputs = []  # List to store all answers

    # Process data
    for i in tqdm(range(0, len(messages), BSZ), disable=rank != main_rank):
        batch_messages = messages[i: i + BSZ]
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=2048, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        batch_output = []
        for i, output_text in enumerate(batch_output_text):
            input_height = int(inputs['image_grid_thw'][i][1] * 14)
            input_width = int(inputs['image_grid_thw'][i][2] * 14)
            image = Image.open(batch_messages[i][0]['content'][0]['image'].split("file://")[1])
            image_width, image_height = image.size
            batch_output.append((output_text, input_height, input_width, image_height, image_width))
        rank_outputs.extend(batch_output)
    print(f"Rank {rank} has finished processing {len(rank_outputs)} examples")

    # Gather all outputs from all ranks
    all_outputs = [None] * len(data)
    rank_results = [(start_idx + i, output) for i, output in enumerate(rank_outputs)]
    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, rank_results)
    assert gathered_results[-1][-1][0] == len(data) - 1
    # The main process will collect all results
    if rank == main_rank:
        for results in gathered_results:
            for idx, output in results:
                assert idx < len(all_outputs)
                all_outputs[idx] = output
        assert all_outputs[-1] is not None
        final_output = []
        correct_number = 0

        for input_example, model_output in zip(data, all_outputs):
            original_output, input_height, input_width, image_height, image_width = model_output
            # TODO: need modifications, maybe no groud truth for evaluation
            ground_truth = input_example['solution']
            model_answer = extract_bbox_answer(original_output)
            resized_model_answer = resize_bbox(model_answer, input_height, input_width, image_height, image_width)
            if model_answer is not None:
                if iou(resized_model_answer, ground_truth) > 0.5:
                    iou_match = True
                    correct_number += 1
                else:
                    iou_match = False
            else:
                iou_match = False
            # Create a result dictionary for this example
            result = {
                'image': input_example['image'],
                'question': input_example['problem'],
                'ground_truth': ground_truth,
                'model_output': original_output,
                'input_size': (input_height, input_width),
                'image_size': (image_height, image_width),
                'extracted_answer': resized_model_answer,
                'correct': int(1) if iou_match else 0,
            }
            final_output.append(result)
        # Calculate and print accuracy
        accuracy = correct_number / len(data) * 100
        print(f"\nAccuracy of {ds}: {accuracy:.2f}%")

        # Save results to a JSON file
        output_path = OUTPUT_PATH / ds / f'{CKPT_NAME}-{ds}-results.json'
        output_dir = output_path.parent
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        with open(output_path, "w") as f:
            json.dump(
                {
                    'accuracy': accuracy,
                    'results': final_output
                }, f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"benchmark {ds} evaluation results have been saved to {output_path}")
        print("-"*100)
    # Synchronize all processes
    dist.barrier()
