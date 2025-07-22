import os
import sys
import torch
import json
from threshold import threshold
from chat_gemini import chat
# from chat_qwen import chat
from segment import segment
from create_final_mask import create_unwalkable_map
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

BASE_DIR = "/home/awangas/my_habitat-lab/preprocessing"
FILE_NAME = "sample_0"
USE_QWEN = False

def main():
    input_image_name = os.path.join(BASE_DIR, "data", "top_down", FILE_NAME + ".png")
    output_mask_name = os.path.join(BASE_DIR, "data", "processed", FILE_NAME, "threshold.png")
    with open(os.path.join(BASE_DIR, "data", "processed", FILE_NAME, "spacing.json"), "r") as f:
        spacing = json.load(f)

    MIN_MASK_AREA = 0.01 / (spacing['spacing'] * spacing['spacing'] )
    MAX_MASK_AREA = 100 / (spacing['spacing'] * spacing['spacing'] )
    # segment(BASE_DIR, FILE_NAME, MIN_MASK_AREA, MAX_MASK_AREA)
    #threshold(input_image_name, output_mask_name)
    if USE_QWEN:
        MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        device_map="auto" # 自动将模型分配到可用的GPU上
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        chat(BASE_DIR, FILE_NAME, model, processor)
    else:
        chat(BASE_DIR, FILE_NAME)
    create_unwalkable_map(BASE_DIR, FILE_NAME)

if __name__ == "__main__":
    main()
