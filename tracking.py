import cv2
import numpy as np
import os

def track(h, w, height, width, ret, frame, target):
    minimum = float('inf')
    for row in range(0, h - height, 1):
        for col in range(0, w - width, 1):
            # ssd
            result = np.sum(np.square(frame[row:row + height, col:col + width, :] - target))
            if result < minimum:
                start_row, start_col = row, col
                minimum = result
    image_out = frame.copy()
    # image_out = cv2.rectangle(image_out, (start_col, start_row), (start_col + width, start_row + height,), (0, 255, 0), 2)
    # output of the coordinate
    coordinate_col = start_col + width / 2

    return coordinate_col, image_out
