import numpy as np
import cv2
import torch

path = "best.pt"
model = torch.hub.load("WongKinYiu/yolov7","custom",f"{path}",trust_repo=True)


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('asl_video.mp4')
while True:
    #Waiting for frames from camera
    ret, frame = cap.read()

    #Getting the color frame from the camera
    color_frame = frame

    #converting the color frame into numpy array
    color_image = np.asanyarray(color_frame)

    #Performing inference using our model
    signs = model(color_image)
    cv2.imshow('Sign Language Detection', np.squeeze(signs.render()))    
    # cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Inference on Image
# img = cv2.imread("/home/mrblack/Projects_DL/SignLanguage_YOLOv7/ASL/test/images/A22_jpg.rf.beab2ab0db6a2c06a0725da776ff9d74.jpg")
# signs = model(img)
# cv2.imshow('Sign Language Detection', np.squeeze(signs.render()))
# cv2.waitKey(0)
