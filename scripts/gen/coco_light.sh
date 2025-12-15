#!/usr/bin bash

export MODEL_NAME="inference/saved_pipeline/jodiffusion"

accelerate launch --mixed_precision="fp16" --multi_gpu --num_processes 4 generate_dataset.py \
  --pretrained_model_name_or_path="inference/saved_pipeline/jodiffusion" \
  --pretrained_label_vae_path="output/dense-label-vae-light-coco-semantic-lr1e4" \
  --finetuned_model_path="output/jodiffusion-vae-light-coco-semantic-joint-5e5" \
  --dataset_name="coco_semantic" --generate_mode="text2img" --num_images 80000 --lightweight_label_vae \
  --save_dir="../dataset/ILGeneration/coco_semantic_light_text2img"
