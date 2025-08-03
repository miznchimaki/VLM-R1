import re, os, json
import math
from datetime import datetime
from transformers import Glm4vForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
from typing import Any, Union
import torch
from open_r1.vlm_modules.vlm_module import VLMBaseModule
from PIL import Image


def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2] - 1, box2[2] - 1)
    inter_y2 = min(box1[3] - 1, box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return float(inter) / union


def giou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2] - 1, box2[2] - 1)
    inter_y2 = min(box1[3] - 1, box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = float(inter) / union

    c_width = max(box1[2], box2[2]) - min(box1[0], box2[0])
    c_height = max(box1[3], box2[3]) - min(box1[1], box2[1])
    c_area = c_width * c_height
    return iou - (c_area - union) / c_area


def diou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2] - 1, box2[2] - 1)
    inter_y2 = min(box1[3] - 1, box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = float(inter) / union

    c_width = max(box1[2], box2[2]) - min(box1[0], box2[0])
    c_height = max(box1[3], box2[3]) - min(box1[1], box2[1])
    c2 = c_width ** 2 + c_height ** 2
    rho2 = (((box2[2] - box1[2]) + (box2[0] - box1[0])) ** 2 + ((box2[3] - box1[3]) + (box2[1] - box1[1])) ** 2) / 4

    return iou - rho2 / c2


def ciou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2] - 1, box2[2] - 1)
    inter_y2 = min(box1[3] - 1, box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = float(inter) / union

    c_width = max(box1[2], box2[2]) - min(box1[0], box2[0])
    c_height = max(box1[3], box2[3]) - min(box1[1], box2[1])
    c2 = c_width ** 2 + c_height ** 2
    rho2 = (((box2[2] - box1[2]) + (box2[0] - box1[0])) ** 2 + ((box2[3] - box1[3]) + (box2[1] - box1[1])) ** 2) / 4
    w1, h1 = box1[2] - box1[0], box1[3] - box1[0]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    alpha = v/ (v - iou + 1)
    return iou - (rho2 / c2 + v * alpha)


def resize_bbox(bbox, input_height, input_width, image_height, image_width):
    bbox[0] = (bbox[0] / 999 * (input_width - 1)) / input_width * image_width
    bbox[1] = (bbox[1] / 999 * (input_height - 1)) / input_height * image_height
    bbox[2] = (bbox[2] / 999 * (input_width - 1)) / input_width * image_width
    bbox[3] = (bbox[3] / 999 * (input_height - 1)) / input_height * image_height
    return bbox


class Glm4vModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        # return "qwen"
        return "glm-4v"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "glm-4.1v" in model_id.lower():
            model_cls = Glm4vForConditionalGeneration
        elif "glm-4v" in model_id.lower():
            model_cls = AutoModelForCausalLM
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls

    def post_model_init(self, model, processing_class):
        pass

    def get_processing_class(self):
        return AutoProcessor

    def get_vision_modules_keywords(self):  
        return ['visual']

    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []

    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]

    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        if not isinstance(inputs, dict):
            prompts_text = [processing_class.apply_chat_template(input_message["prompt"], add_generation_prompt=True, tokenize=False) for input_message in inputs]
        else:
            prompts_text = processing_class.apply_chat_template(inputs["prompt"], add_generation_prompt=True, tokenize=False)
        return prompts_text

    def prepare_model_inputs(
                             self, 
                             processing_class, 
                             prompts_text, 
                             images, 
                             return_tensors="pt",
                             padding=True, 
                             padding_side="left", 
                             add_special_tokens=False
                            ):
        additional_output = None
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
            additional_output = [{'image_grid_thw': image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']]
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs, additional_output

    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return (
                        "Please provide the bounding box coordinates of the region this "
                        "sentence describes in the format [x_min, y_min, x_max, y_max]: {query}"
                       )
            case "ic":
                return (
                        "First thinks about the reasoning process in the mind and then "
                        "provides the user with the answer to the question: {query}"
                       )
            case "odLength":
                return (
                        "First thinks about the reasoning process in the mind and then "
                        "provides the user with the answer to the question: {query}"
                       )
            case _:
                return (
                        "First thinks about the reasoning process in the mind and then "
                        "provides the user with the answer to the question: {query}"
                       )

    @staticmethod
    def glm4v_format_reward(completions, **kwargs):
        """Check if the GLM-4.1V-Thinking model output matches a specific format."""
        non_verify_pat = r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*"
        verify_pat = r"\s*<think>(.*?)</think>\s*<answer>(.*?)<|begin_of_box|>(.*?)<|end_of_box|>(.*?)</answer>\s*"
        think_pat = r"\s*<think>(.*?)</think>\s*"
        answer_pat = r"\s*<answer>(.*?)</answer>\s*"
        box_pat = r"\s*<\|begin_of_box\|>(*.?)<\|end_of_box\|>\s*"
        rec_pat = r".*?\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\].*?"

        completion_contents = [completion[0]["content"] for completion in completions]
        task_type = kwargs.pop("task_type", None)
        if task_type is not None:
            if task_type.lower() == "non_verify":
                non_verify_matches = [re.search(non_verify_pat, content, re.DOTALL) is not None for content in completion_contents]
                think_matches = [len(re.findall(think_pat, content, re.DOTALL)) == 1 for content in completion_contents]
                answer_matches = [len(re.findall(answer_pat, content, re.DOTALL)) == 1 for content in completion_contents]
                matches = [
                    whole_match & think_match & answer_match 
                    for whole_match, think_match, answer_match 
                    in zip(non_verify_matches, think_matches, answer_matches)
                ]
            elif task_type.lower() in ("verify", "rec"):
                verify_matches = [re.search(verify_pat, content, re.DOTALL) is not None for content in completion_contents]
                think_matches = [len(re.findall(think_pat, content, re.DOTALL)) == 1 for content in completion_contents]
                answer_matches = [len(re.findall(answer_pat, content, re.DOTALL)) == 1 for content in completion_contents]
                box_matches = [len(re.findall(box_pat, content, re.DOTALL)) == 1 for content in completion_contents]
                matches = [
                    verify_match & think_match & answer_match & box_match 
                    for verify_match, think_match, answer_match, box_match 
                    in zip(verify_matches, think_matches, answer_matches, box_matches)
                ]
                if task_type == "rec":
                    rec_matches = [len(re.findall(rec_pat, content, re.DOTALL)) == 1 for content in completion_contents]
                    matches = [rec_match & match for rec_match, match in zip(rec_matches, matches)]
        else:
            raise ValueError(f"When using format reward for GLM-4.1V, param named `task_type` must be passed in")

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]

    @staticmethod
    def glm4v_iou_reward(completions, solution, **kwargs):
        """Calculate IoU or GIoU reward between predicted bounding box from GLM-4.1V model and ground truth bounding box."""
        iou_type = kwargs.pop("iou_type", None)
        if iou_type.lower() not in ("iou", "giou", "diou", "ciou"):
            raise ValueError(f"GLM-4.1V now only support following IoU reward types: (IoU, GIoU, DIoU, CIoU)")
        iou_func = eval(iou_type.lower())
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_pat = r'\s*<answer>(.*?)</answer>\s*'
        rec_box_pat = r'\s*<\|begin_of_box\|>\s*\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]\s*<\|end_of_box\|>\s*'

        for i, (content, sol) in enumerate(zip(contents, solution)):
            image_grid_thw = kwargs.get("image_grid_thw")[i]
            image_path = kwargs.get("image_path")[i][0]
            image = Image.open(image_path)
            image_width, image_height = image.size
            input_height = int(image_grid_thw[1] * 14)
            input_width = int(image_grid_thw[2] * 14)

            sol = re.findall(answer_pat, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_pat, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(rec_box_pat, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        bbox = resize_bbox(bbox, input_height, input_width, image_height, image_width)
                        # TODO: Maybe useful for MARS2 track-1 competition (1)
                        # if iou_func(bbox, sol) > 0.5:
                        #     reward = 1.0
                        # TODO: Maybe usefule for MARS2 track-1 competition (2)
                        # if iou_func(bbox, sol) < 0.5:
                        #     reward = 0.0
                        reward = iou_func(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails

            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[i]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "rec":
                    return Glm4vModule.glm4v_iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return Glm4vModule.glm4v_format_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")
