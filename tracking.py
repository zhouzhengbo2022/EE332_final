import cv2
import numpy as np
import os

#capture the object
image_ori = cv2.imread('target.png')
#default param
h,w,c = image_ori.shape
target = image_ori[0:50,278:292,:]
#cv2.imshow('target',target)
cv2.waitKey(0)
height = 50
width = 14
bbox = (55,20,34,46)
#image_out = cv2.rectangle(image_out,(start_x,start_y),(start_x+width,start_y+height,),(0,255,0),2)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
cap = cv2.VideoCapture('Original.avi')
out = cv2.VideoWriter('output_Tracking.avi', fourcc, 15.0, (320,240))
while(1):
    ret, frame = cap.read()
    print(ret)
    if not ret:
        break
    minimum = float('inf')
    for i in range(0, h - height, 1):
        for j in range(0, w - width, 1):
            # ssd
            result = np.sum(np.square(frame[i:i + height, j:j + width, :] - target))
            if result < minimum:
                start_y, start_x = i, j
                minimum = result
    image_out = frame.copy()
    image_out = cv2.rectangle(image_out, (start_x, start_y), (start_x + width, start_y + height,), (0, 255, 0), 2)
    #output of the coordinate
    coordinate_x, coordinate_y = start_x + width/2, start_y + height/2
    out.write(image_out)
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
cap.release()
out.release()

