#!/usr/bin bash

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=6000

accelerate launch --mixed_precision="fp16" --multi_gpu --num_processes 4 train_ae.py \
  --dataset_name="coco_semantic" --resolution=512 --mixed_precision="fp16" --lightweight_label_vae \
  --train_batch_size=2 --gradient_accumulation_steps=4 --num_train_epochs=20 \
  --learning_rate=1e-4 --adam_weight_decay 0.05 --max_grad_norm=3 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --checkpointing_steps=1000 --checkpoints_total_limit=1 --validation_epochs=1 \
  --output_dir="output/dense-label-vae-light-coco-semantic-lr1e4"
