import torch
import pyrealsense2 as rs
import numpy as np

#write a code to infer the sign of number using weights from yolov7 model

#Loading Pytorch model to be converted later into ONNX format
model = torch.load('best.pt')

#Create random input of the same size as the input of the model
#This is required to export the model to ONNX
#It is not used when running the model
random_input = torch.randn(1, 3, 640, 640)

#Export the model to ONNX
torch.onnx.export(model, random_input, "yolov7.onnx", export_params=True)


#Tutorial: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/python-tutorial-1-depth.py

#Create a pipeline
pipeline = rs.pipeline()

#Start streaming for realsense
pipeline.start()


while True:
    #Waiting for frames from realsense
    frames = pipeline.wait_for_frames()

    #Getting the color frame from the camera
    color_frame = frames.get_color_frame()

    #converting the color frame into numpy array
    color_image = np.asanarray(color_frame.get_data())

    #Performing inference using our model
    signs = model(color_image)