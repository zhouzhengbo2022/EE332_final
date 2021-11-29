import cv2
import numpy as np
import os

def create_mask(img, lower_bound, upper_bound, kernelOpen, kernelClose, eraser = 0):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = imgHSV.shape[0]
    mask = cv2.inRange(imgHSV,lower_bound,upper_bound)
    maskOpen = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    if eraser == True:
        for x in range(h):
            if x<64 or x>219:
                maskClose[x] = np.zeros(maskClose[x].shape[0])
    return maskClose