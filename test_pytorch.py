import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os


class ImageClassifier:
    def __init__(self, is_camera=False):
        #Camera or Image upload:
        self.is_camera = is_camera
        #Choose the device:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #Mean and std for normalization:
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        #Model path:
        self.path = "sign_model_100.pth"
        #Image Path (in case you chose image upload method):
        self.PATH_IMAGE = os.path.expanduser("~/Datasets/SignLanguageLetters/0/zero_13.jpg")
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean,self.std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
        }

        

    def load_model(self):
        #Pretrained model:
        self.model = models.resnet18(pretrained=True)
        #Freezing the parameters:
        for param in self.model.parameters():
            param.requires_grad = False
        #Changing the last layer:
        num_features_4 = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features_4, 11) #11 classes
        self.model.to(self.device)
        #Loading the model:
        state_dict = torch.load(self.path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        #Changing the model to evaluation mode:
        self.model.eval()
        #Class labels:
        self.class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    def preprocess_frame(self, frame):
        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame
        frame = cv2.resize(frame, (self.size[1], self.size[2]))

        # Convert the frame to a tensor
        frame = torch.from_numpy(frame).float()

        # Normalize the frame and add a batch dimension
        frame = (frame - torch.tensor(self.mean)) / torch.tensor(self.std)
        frame = frame.unsqueeze(0)

        return frame

    def eval_model(self, image, model):
        with torch.no_grad():
            image = image.to(self.device)
            image = image.view(1, 3, 224, 224)
            output = model(image)
            _, prediction = torch.max(output, 1)
            prediction = prediction.cpu().numpy()[0]
            return prediction

    def get_probability(self, output):
        probability = F.softmax(output, dim=1)[0] * 100
        # print(probability)
        # Compute the probabilities
        probabilities = F.softmax(output, dim=1)
        

        # Get the highest probability
        _, max_index = torch.max(probabilities, dim=1)

        # Get the probability of the highest probability class
        max_probability = probabilities[0, max_index]
        # print)

        print(f'Prediction: {max_index} with probability {max_probability}')

        return probability



    def run(self):
        if self.is_camera:
            self.run_camera()
        else:
            self.run_image()

    def run_camera(self):
        self.load_model()        
        cap = cv2.VideoCapture(0)
        ## cap = cv2.VideoCapture('asl_video.mp4')
        while True:
            #Waiting for frames from camera
            ret, frame = cap.read()

            #Getting the color frame from the camera
            color_frame = self.preprocess_frame(frame)

            #converting the color frame into numpy array
            color_image = np.asanyarray(color_frame)
            color_image = torch.from_numpy(color_image).to(self.device)
            color_image = color_image.view(1, 3, 224, 224)

            #Performing inference using our model
            output = self.model_sign(color_image)
            prediction = self.eval_model(color_image, self.model)
            # _, prediction = torch.max(output, 1)
            # prediction = prediction.cpu().numpy()[0]
            print(prediction)
            self.get_probability(output)
            #can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
            #Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
            output = output.detach().cpu()
            output = output.numpy()
            cv2.imshow('Sign Language Detection', frame)  
            # print(output)  

            # cv2.imshow('Sign Language Detection', np.squeeze(output.render()))    
            # cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def run_image(self):
        #Loading the model
        self.load_model()
        #Getting the image
        # img = cv2.imread(self.PATH_IMAGE)
        img = cv2.imread(self.PATH_IMAGE)
        #Converting to PIL image as pytorch transforms work with PIL images
        img = Image.fromarray(img)
        #Applying the transformations
        transformed_image = self.data_transforms['val'](img)
        #converting the transformed image into numpy array
        transformed_image = np.array(transformed_image)
        transformed_image = torch.from_numpy(transformed_image).to(self.device)
        transformed_image = transformed_image.view(1, 3, 224, 224)
        #Performing inference using our model
        output = self.model(transformed_image)
        prediction = self.eval_model(transformed_image, self.model)
        # _, prediction = torch.max(output, 1)
        # prediction = prediction.cpu().numpy()[0]
        print(prediction)
        self.get_probability(output)
        #can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        #Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
        output = output.detach().cpu()
        output = output.numpy()


if __name__ == '__main__':
    image_classifier = ImageClassifier(is_camera=False)
    image_classifier.run()
