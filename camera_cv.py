import cv2
import numpy as np
import time


cap = cv2.VideoCapture(0)
while True:
    #Waiting for frames from camera
    ret, frame = cap.read()

    #Getting the color frame from the camera
    color_frame = frame

    #converting the color frame into numpy array
    color_image = np.asanyarray(color_frame)

    #Performing inference using our model
    # sign = model_onnx(color_image)
    # signs = model(color_image)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break