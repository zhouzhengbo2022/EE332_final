import argparse
import cv2
import os
import numpy as np
import numba as nb
from font_mask import create_mask
from synthesis import *
from tracking import *

def validate_args(args):
    wh, ww = args.window_height, args.window_width
    if wh < 3 or ww < 3:
        raise ValueError('window_size must be greater than or equal to (3,3).')

    if args.kernel_size <= 1:
        raise ValueError('kernel size must be greater than 1.')

    if args.kernel_size % 2 == 0:
        raise ValueError('kernel size must be odd.')

    if args.kernel_size > min(wh, ww):
        raise ValueError('kernel size must be less than or equal to the smaller window_size dimension.')

def parse_args():
    parser = argparse.ArgumentParser(description='Smart Erasor')
    parser.add_argument('--template_path', type=str, required=False, default='./target.png', help='Path to the template sample')
    parser.add_argument('--in_path', type=str, required=False, default='./Original.avi', help='the path of input video')
    parser.add_argument('--out_path', type=str, required=False, default='./Output.avi', help='the path of erased video')

    # parameters for texture impainting
    parser.add_argument('--kernel_size', type=int, required=False, default=11, help='One dimension of the square synthesis kernel')
    parser.add_argument('--visualize', required=False, action='store_true', help='Visualize the synthesis process')


    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    template = cv2.imread(args.template_path)
    if template is None:
        raise ValueError('Unable to read image from template_path.')

    target = template[0:50, 278:292, :]
    H, W = template.shape[0], template.shape[1]
    height, width = target.shape[0], target.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cap = cv2.VideoCapture(args.in_path)
    out = cv2.VideoWriter(args.out_path, fourcc, 15.0, (320, 240))

    while(1):
        ret, frame = cap.read()
        target_col, frame_img = track(H, W, height, width, ret, frame, target)

        frame_mask = create_mask(frame_img, np.array([0, 0, 0]), np.array([25, 255, 255]), np.ones((3, 3), np.uint8), np.ones((3, 3), np.uint8), 1)
        hole_index = np.where(frame_mask == 255)

        for hole_row, hole_col in zip(hole_index[0], hole_index[1]):
            if hole_col > target_col:
                frame_img[hole_row, hole_col] = 255

        frame_out = synthesize_texture(original_sample=frame_img, 
                                             kernel_size=args.kernel_size, 
                                             visualize=args.visualize)

        out.write(frame_out)
        if cv2.waitKey(100) & 0xff == ord('q'):
            break

if __name__ == '__main__':
    main()