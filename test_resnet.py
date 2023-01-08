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
        #Image size:
        self.size = (3, 224, 224)
        #Model path:
        # self.path = os.path.expanduser("~/Datasets/Trained_Models/SignLanguageModels/sign_model_100.pth")
        self.path = os.path.expanduser("~/Datasets/Trained_Models/SignLanguageModels/sign_model_100.pth")
        #Image Path (in case you chose image upload method):
        self.PATH_IMAGE = os.path.expanduser("~/Datasets/SignLanguageLetters/val/0/zero_13.jpg")
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
        self.model = models.resnet18(weights=None)
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
        self.class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'unkown']

    #Preprocessing the image to convert it to tensor:
    def preprocess_frame(self, frame):
        #Camera:
        if self.is_camera:
            # Convert the frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame
            frame = cv2.resize(frame, (self.size[1], self.size[2]))

            # Convert the frame to a tensor
            frame = torch.from_numpy(frame).float()

            # Normalize the frame and add a batch dimension
            frame = (frame - torch.tensor(self.mean)) / torch.tensor(self.std)
            frame = frame.unsqueeze(0)
        #Image upload:
        else:
            #Converting to PIL image as pytorch transforms work with PIL images
            img = Image.fromarray(frame)
            image = self.data_transforms['val'](img)
            #Adding a batch dimension
            frame = image.unsqueeze(0)
        return frame

    def eval_model(self, image, model):
        with torch.no_grad():
            image = image.to(self.device)
            image = image.view(1, 3, 224, 224)
            output = model(image)

        print("Output: ", output)
        _, prediction = torch.max(output, 1) #Returns max value and index
        prediction = prediction.cpu().numpy()[0].item()
        prediction = self.class_labels[int(prediction)]
        print(prediction)
        probability = F.softmax(output, dim=1)
        probability = probability.cpu().numpy()[0]
        probability = probability[np.argmax(probability)]
        print(probability)
        # print(f'Prediction: {prediction} with probability {probability*100:.2f}%')
        if probability.item()*100 < 10 or prediction == 10:
            print('Unkown')
        else:
            print(f'Prediction: {prediction} with probability {probability*100:.2f}%')

        return prediction, probability

    def get_probability(self, output):
        # probability = F.softmax(output, dim=1)[0] * 100
        # print(probability)
        # Compute the probabilities
        probabilities = F.softmax(output, dim=1)
        

        # Get the highest probability
        _, max_index = torch.max(probabilities, dim=1)

        # Get the probability of the highest probability class
        max_probability = probabilities[0, max_index]
        max_index = max_index.cpu().numpy()[0]
        
        return max_probability.item()

    #Function to show the detection on opencv window:
    def overlay_detection(self, frame, prediction, probability):
        # Get the color
        color = (0, 255, 0)
        # Draw the rectangle
        cv2.rectangle(frame, (0, 0), (300, 60), color, -1)
        # Draw the text
        cv2.putText(frame, f'Prediction: {prediction}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, f'Probability: {probability*100:.2f}%', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame


    def run(self):
        #Flag to check if camera is used:
        if self.is_camera:
            self.run_camera()
        
        #Image upload:
        else:
            self.run_image()

    def run_camera(self):
        self.load_model()        
        cap = cv2.VideoCapture(0)
        ## cap = cv2.VideoCapture('asl_video.mp4')
        while True:
            #Waiting for frames from camera
            ret, frame = cap.read()

            #Getting the color frame from the camera and convert it to tensor:
            color_frame = self.preprocess_frame(frame)

            #Getting the prediction and the probability:
            prediction, probability = self.eval_model(color_frame, self.model)

            #Overlaying the detection on the frame:
            frame = self.overlay_detection(frame, prediction, probability)
            cv2.imshow('Sign Language Detection', frame)  

            #Press q to exit:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def run_image(self):
        #Loading the model
        self.load_model()

        #Getting the image
        img = cv2.imread(self.PATH_IMAGE)

        #Preprocessing the image
        img = self.preprocess_frame(img)

        #Printing the prediction and probability:
        _, _ = self.eval_model(img, self.model)



if __name__ == '__main__':
    image_classifier = ImageClassifier(is_camera=True)
    image_classifier.run()
