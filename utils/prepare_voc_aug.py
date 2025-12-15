import argparse
import os
from functools import partial

import numpy as np
from PIL import Image
from scipy.io import loadmat

AUG_LEN = 10582


def convert_mat(mat_file, in_dir, out_dir):
    data = loadmat(os.path.join(in_dir, mat_file))
    mask = data['GTcls'][0]['Segmentation'][0].astype(np.uint8)
    seg_filename = os.path.join(out_dir, mat_file.replace('.mat', '.png'))
    Image.fromarray(mask).save(seg_filename, 'PNG')


def generate_aug_list(merged_list, excluded_list):
    return list(set(merged_list) - set(excluded_list))


def parse_args():
    parser = argparse.ArgumentParser(description='Convert PASCAL VOC annotations to mmsegmentation format')
    parser.add_argument('--devkit_path', default='dataset', help='pascal voc devkit path')
    parser.add_argument('--aug_path', default='../dataset/VOC_AUG', help='pascal voc aug path')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    devkit_path = args.devkit_path
    aug_path = args.aug_path
    if args.out_dir is None:
        out_dir = os.path.join(devkit_path, 'VOC2012', 'SegmentationClassAug')
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    in_dir = os.path.join(aug_path, 'cls')

    for mat_file in os.listdir(in_dir):
        convert_mat(mat_file, in_dir, out_dir)

    full_aug_list = []
    with open(os.path.join(aug_path, 'train.txt')) as f:
        full_aug_list += [line.strip() for line in f]
    with open(os.path.join(aug_path, 'val.txt')) as f:
        full_aug_list += [line.strip() for line in f]

    with open(os.path.join(devkit_path, 'VOC2012/ImageSets/Segmentation', 'train.txt')) as f:
        ori_train_list = [line.strip() for line in f]
    with open(os.path.join(devkit_path, 'VOC2012/ImageSets/Segmentation', 'val.txt')) as f:
        val_list = [line.strip() for line in f]

    aug_train_list = generate_aug_list(ori_train_list + full_aug_list, val_list)
    assert len(aug_train_list) == AUG_LEN, 'len(aug_train_list) != {}'.format(AUG_LEN)

    with open(os.path.join(devkit_path, 'VOC2012/ImageSets/Segmentation', 'trainaug.txt'), 'w') as f:
        f.writelines(line + '\n' for line in aug_train_list)

    aug_list = generate_aug_list(full_aug_list, ori_train_list + val_list)
    assert len(aug_list) == AUG_LEN - len(ori_train_list), 'len(aug_list) != {}'.format(AUG_LEN - len(ori_train_list))
    with open(os.path.join(devkit_path, 'VOC2012/ImageSets/Segmentation', 'aug.txt'), 'w') as f:
        f.writelines(line + '\n' for line in aug_list)

    print('Done!')


if __name__ == '__main__':
    main()