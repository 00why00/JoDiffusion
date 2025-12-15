import json
import os

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

image_dir = "../dataset/ADE20K/images"
caption_dir = "../dataset/ADE20K/annotations_caption"
os.makedirs(caption_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
model.to(device)

for split in ["training", "validation"]:
    json_path = os.path.join(caption_dir, f"{split}.json")
    image_list = os.listdir(os.path.join(image_dir, split))
    captions = {}
    for image_name in image_list:
        img_path = os.path.join(image_dir, split, image_name)
        raw_image = Image.open(img_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(f"{image_name}: {generated_text}")
        captions[image_name] = generated_text
    with open(json_path, "w") as f:
        json.dump(captions, f)
