# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import math
import os
import os.path as osp

import mmcv
import numpy as np
from mmengine.utils import ProgressBar


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert levir-cd dataset to mmsegmentation format')
    parser.add_argument('--dataset_path', default='/mnt/search01/dataset/cky_data/WHU/', help='potsdam folder path')
    parser.add_argument('-o', '--out_dir', default='/data/kyanchen/WHU/', help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=512)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=512)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_folder = args.dataset_path
    png_files = glob.glob(
        os.path.join(input_folder, '**/label/*.tif'), recursive=True)
    output_folder = args.out_dir
    prog_bar = ProgressBar(len(png_files))
    for png_file in png_files:
        new_path = os.path.join(
            output_folder,
            os.path.relpath(os.path.dirname(png_file), input_folder))
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        image = mmcv.imread(png_file)

        image[image < 128] = 0
        image[image >= 128] = 1
        image = image[:, :, 0]
        if image.max() > 1:
            print(np.bincount(image.flatten()))
        file_name = os.path.basename(png_file)
        mmcv.imwrite(image.astype(np.uint8), os.path.join(new_path, file_name).replace('.tif', '.png'))
        prog_bar.update()



if __name__ == '__main__':
    main()
