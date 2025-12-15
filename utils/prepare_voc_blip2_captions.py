import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

image_dir = "../dataset/VOC2012/JPEGImages"
caption_dir = "../dataset/VOC2012/ImageSets/Caption"
image_list_dir = "../dataset/VOC2012/ImageSets/Segmentation"
os.makedirs(caption_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
model.to(device)

for split in ["trainaug", "val"]:
    image_list_path = os.path.join(image_list_dir, f"{split}.txt")
    image_list = open(image_list_path).readlines()
    json_path = os.path.join(caption_dir, f"{split}.json")
    captions = {}
    for image_name in tqdm(image_list):
        image_name = image_name.strip()
        image_path = os.path.join(image_dir, image_name + ".jpg")
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(f"{image_name}: {generated_text}")
        captions[image_name] = generated_text
    with open(json_path, "w") as f:
        json.dump(captions, f)
