"""

Author : Phaneendhra
Date : 10 June 2021

"""


import cv2
import torch
from facenet_pytorch import MTCNN
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(device) Uncomment this to know whether torch is using GPU or CPU

def DetectFace(color, result_list):
    try:
        for result in result_list:
            x, y, w, h = result_list[0]
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            roi = color[y:y+h, x:x+w]
            cv2.rectangle(color,
                          (x, y), (w, h),
                          (0, 155, 255),
                          5)
        return color
    except:
        return color


video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

detector = MTCNN(
                select_largest = True,
                device = device
                )

while True:
    _, color = video_capture.read()
    boxes,faces = detector.detect(color)
    detectFaceMTCNN = DetectFace(color, boxes)
    detectFaceMTCNN = cv2.flip(color,1)  #flips the image (symmetric)
    cv2.imshow('Video', detectFaceMTCNN)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
