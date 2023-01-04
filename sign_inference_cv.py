import torch
import cv2
import numpy as np
# from models.experimental import attempt_load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#write a code to infer the sign of number using weights from yolov7 model
# Loading Pytorch model to be converted later into ONNX format
PATH = "yolov7/best.pt"
# opencv_model = torch.hub.load('ultralytics/yolov5', 'custom', path=PATH)  # custom model
# model = torch.hub.load('best.pt')
def loadModel(path:str):
    model = torch.hub.load("WongKinYiu/yolov7","custom",f"{path}",trust_repo=True)
    return model

model = loadModel(PATH)
#Convert the model to ONNX format



# model_onnx = cv2.get_pytorch_onnx_model(model)
print("here")
# model.load_state_dict(torch.load('best.pt'))
#Create random input of the same size as the input of the model
#This is required to export the model to ONNX
#It is not used when running the model
random_input = torch.randn(1, 3, 640, 640)
random_input = random_input.to(device)
# model = model.to(device)
#Export the model to ONNX
# torch.onnx.export(model, random_input, "yolov7.onnx", export_params=True)
# model = torch.jit.load("model.trt")
# model = torch.jit.load("best.onnx")
opencv_net = cv2.dnn.readNetFromONNX("old_best.onnx")


# model = onnx.load("yolov7.onnx")
cap = cv2.VideoCapture(0)
print("a7a")
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