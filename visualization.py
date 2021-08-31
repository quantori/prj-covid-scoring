import argparse
import os
import re

import cv2

from tools.utils import create_vis


def sort_filenames(filename):
    list_of_numbers = re.findall(r'\d+', filename)
    concatenated_nums = ''.join(list_of_numbers)
    return int(concatenated_nums)

def main(args):
    mask_dir = os.path.join(args.images_dir, 'mask')
    prob_mask_filenames = [x for x in os.listdir(mask_dir) if 'Probability' in x]
    prob_mask_filenames.sort(key=sort_filenames)
    num1, num2 = re.findall(r'\d+', prob_mask_filenames[-1])
    num1, num2 = int(num1), int(num2)//10

    for _num2 in range(num2):
        gt_mask_filename = "Probability maps_0_{}0.mask.png".format(_num2)
        gt_mask = cv2.imread(os.path.join(mask_dir, gt_mask_filename), 0)

        pred_masks = []
        for _num1 in range(num1):
            pred_mask_filename = "Probability maps_{}_{}.png".format(_num1, _num2)
            pred_mask = cv2.imread(os.path.join(args.images_dir, pred_mask_filename), 0)
            pred_masks.append(pred_mask)

        create_vis(pred_masks, gt_mask, output_path_map='output_path_map_{}.avi'.format(_num2),
                   output_path_mask='output_path_mask_{}.avi'.format(_num2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str)
    args = parser.parse_args()
    main(args)