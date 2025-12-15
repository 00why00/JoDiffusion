import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy import stats
from scipy.ndimage import label, binary_dilation, generate_binary_structure
from tqdm import tqdm

from utils.utils import encode_seg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--area_threshold", type=int, default=20)
    parser.add_argument("--generated_mask_dir", type=str, default="../dataset/ILGeneration/coco_semantic_light_text2img/label")
    parser.add_argument("--save_dir", type=str, default="../dataset/ILGeneration/coco_semantic_light_text2img/optimized_label_p20")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num_workers", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    return args

def process_mask(mask_name, args):
    mask_path = os.path.join(args.generated_mask_dir, mask_name)
    mask = np.array(Image.open(mask_path))

    if args.visualize:
        encoded_mask = encode_seg(mask[np.newaxis, :, :])
        plt.imshow(encoded_mask[0])
        plt.show()

    label_set = np.unique(mask)
    label_set = label_set[label_set != 0]
    for label_idx in label_set:
        label_mask, nums = label(mask == label_idx)
        for target_idx in range(1, nums + 1):
            target_mask = label_mask == target_idx
            if target_mask.sum() < args.area_threshold:
                dilated_mask = binary_dilation(target_mask, structure=generate_binary_structure(2, 2))
                surrounding_mask = dilated_mask & ~ target_mask
                surrounding_pixel = mask[surrounding_mask]
                mode = int(stats.mode(surrounding_pixel)[0])
                mask[target_mask] = mode

    if args.visualize:
        encoded_mask = encode_seg(mask[np.newaxis, :, :])
        plt.imshow(encoded_mask[0])
        plt.show()
    else:
        Image.fromarray(mask).save(os.path.join(args.save_dir, mask_name))

def optimize_mask(args):
    mask_names = os.listdir(args.generated_mask_dir)
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(process_mask, [mask_name for mask_name in mask_names], [args] * len(mask_names)),
                  total=len(mask_names)))

_args = parse_args()
optimize_mask(_args)
