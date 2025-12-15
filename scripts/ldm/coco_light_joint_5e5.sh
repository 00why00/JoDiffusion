#!/usr/bin bash

export MODEL_NAME="inference/saved_pipeline/jodiffusion"

accelerate launch --mixed_precision="fp16" --multi_gpu --num_processes 4 train_ldm.py \
  --pretrained_model_name_or_path=$MODEL_NAME --dataset_name="coco_semantic" --caption_column="coco" \
  --pretrained_label_vae_path="output/dense-label-vae-light-coco-semantic-lr1e4" \
  --resolution=512 --random_flip --center_crop --mixed_precision="fp16" --use_8bit_adam --lightweight_label_vae \
  --train_batch_size=1 --gradient_accumulation_steps=16 --num_train_epochs=20 --noise_type="joint" \
  --learning_rate=5e-5 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --checkpointing_steps=1000 --checkpoints_total_limit=1 --validation_epochs=1 \
  --output_dir="output/jodiffusion-vae-light-coco-semantic-joint-5e5"
