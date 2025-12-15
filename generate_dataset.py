import argparse
import json
import math
import os
import random

import torch
from accelerate import PartialState
from diffusers import AutoencoderKL
from tqdm import tqdm

from pipelines.modeling_uvit import JoDiffusionModel
from pipelines.pipeline_jodiffusion import JoDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned_model_path", type=str, default=None)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="inference/saved_pipeline/jodiffusion")
    parser.add_argument("--pretrained_label_vae_path", type=str, default="output/dense-label-vae-light-ade20k-semantic-lr1e4")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="ade20k_semantic")
    parser.add_argument("--lightweight_label_vae", action="store_true")
    parser.add_argument("--generate_mode", type=str, default="joint", choices=["text2img", "joint"])
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def generate(args, weight_dtype):
    random.seed(args.seed)
    unet = JoDiffusionModel.from_pretrained(args.finetuned_model_path, torch_dtype=weight_dtype)
    pipeline_kwargs = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "unet": unet, "torch_dtype": weight_dtype,
    }
    if args.lightweight_label_vae:
        from pipelines.modeling_lightweight_vae import LightweightLabelVAE
        pipeline_kwargs["label_vae"] = LightweightLabelVAE.from_pretrained(args.pretrained_label_vae_path, torch_dtype=weight_dtype)
    else:
        pipeline_kwargs["label_vae"] = AutoencoderKL.from_pretrained(args.pretrained_label_vae_path, torch_dtype=weight_dtype)
    pipeline = JoDiffusionPipeline.from_pretrained(**pipeline_kwargs)
    pipeline.set_progress_bar_config(disable=True)

    distributed_state = PartialState()
    pipeline = pipeline.to(distributed_state.device)
    generator = torch.Generator(device=distributed_state.device).manual_seed(args.seed)
    if distributed_state.is_main_process:
        if os.path.exists(f"{args.save_dir}/image"):
            print(f"Directory {args.save_dir}/image already exists, please doubleclick it.")
        os.makedirs(f"{args.save_dir}/image", exist_ok=True)
        os.makedirs(f"{args.save_dir}/label", exist_ok=True)
        print(f"Generate args: {args}")

    # prepare dataset to args.num_images
    assert args.dataset_name.split("_")[0] in args.finetuned_model_path
    assert args.dataset_name.split("_")[0] in args.pretrained_label_vae_path
    if args.dataset_name == "ade20k_semantic":
        caption_path = "../dataset/ADE20K/annotations_caption/training.json"
    elif args.dataset_name == "voc_semantic":
        caption_path = "../dataset/VOC2012/ImageSets/Caption/trainaug.json"
    elif args.dataset_name == "coco_semantic":
        caption_path = "../dataset/COCO/annotations/captions_train2017.json"
    else:
        raise ValueError(f"Unknown dataset {args.dataset_name}")

    caption = json.load(open(caption_path, "r"))
    if args.dataset_name == "coco_semantic":
        caption_list = [c["caption"] for c in caption["annotations"]]
    else:
        caption_list = list(caption.values())
        caption_list = caption_list * math.ceil(args.num_images / len(caption_list))

    random.shuffle(caption_list)
    caption_list = caption_list[:args.num_images]
    assert len(caption_list) == args.num_images
    caption_dataset = [[idx, c] for idx, c in enumerate(caption_list)]

    with distributed_state.split_between_processes(caption_dataset) as split_caption:
        for idx, prompt in tqdm(split_caption):
            img_path = f"{args.save_dir}/image/{idx}.jpg"
            lbl_path = f"{args.save_dir}/label/{idx}.png"
            if os.path.isfile(img_path) and os.path.isfile(lbl_path):
                continue
            sample = pipeline(
                mode=args.generate_mode,
                prompt=prompt,
                num_inference_steps=50,
                generator=generator,
                ignore_label=0,
                use_color_map=False
            )
            image = sample.images[0]
            label = sample.labels[0]
            image.save(img_path)
            label.save(lbl_path)

_args = parse_args()
generate(_args, torch.float16)
