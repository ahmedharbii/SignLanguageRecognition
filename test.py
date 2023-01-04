import numpy as np
import cv2
import torch

path = "yolov7/best.pt"
model = torch.hub.load("WongKinYiu/yolov7","custom",f"{path}",trust_repo=True)


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('asl_video.mp4')
while True:
    #Waiting for frames from camera
    ret, frame = cap.read()

    #Getting the color frame from the camera
    color_frame = frame

    #converting the color frame into numpy array
    color_image = np.asanyarray(color_frame)

    #Performing inference using our model
    signs = model(color_image)
    # sign = model_onnx(color_image)
    cv2.imshow('Sign Language Detection', np.squeeze(signs.render()))    
    # cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break