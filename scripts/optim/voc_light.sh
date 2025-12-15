#!/usr/bin bash

python optimize_mask.py --area_threshold 20 --num_workers 100 \
  --generated_mask_dir "../dataset/ILGeneration/voc_semantic_light_blip2_text2img/label" \
  --save_dir "../dataset/ILGeneration/voc_semantic_light_blip2_text2img/label_optimized_20"

python optimize_mask.py --area_threshold 50 --num_workers 100 \
  --generated_mask_dir "../dataset/ILGeneration/voc_semantic_light_blip2_text2img/label" \
  --save_dir "../dataset/ILGeneration/voc_semantic_light_blip2_text2img/label_optimized_50"

python optimize_mask.py --area_threshold 100 --num_workers 100 \
  --generated_mask_dir "../dataset/ILGeneration/voc_semantic_light_blip2_text2img/label" \
  --save_dir "../dataset/ILGeneration/voc_semantic_light_blip2_text2img/label_optimized_100"
