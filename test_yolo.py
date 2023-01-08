import numpy as np
import cv2
import torch
import os
import time


class SignLanguageDetection:
    def __init__(self, is_camera=True):
        #Choose whether to use the camera or an image:
        self.is_camera = is_camera
        #Load the model:
        path = os.path.expanduser("~/Datasets/Trained_Models/SignLanguageModels/best.pt")
        self.model = torch.hub.load("WongKinYiu/yolov7","custom",f"{path}",trust_repo=True)
        #Confidence threshold:
        self.confidence_threshold = 0.5
        #Class labels:
        self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        # Creating the dictionary where we will store the alphabets:
        self.alphabets_dict = {}
        #Threshold for alphabets number of occurrences:
        self.alphabets_occur_threshold = 5
        #Command for UJI Butler:
        self.command = str(None)

    def get_prediction_alphabet(self, signs):
        #Get the detections:
        signs = signs.xyxyn[0].cpu().numpy()
        #If there are detections:
        if signs.shape[0] > 0:
            #Get the highest confidence detection - The fifth column is the confidence
            #The array is automatically sorted by confidence
            signs = signs[np.argmax(signs[:, 4])]
            #Get the index of the highest confidence detection
            index = signs[5]
            #Setting the threshold:
            probability = signs[4]
            # print(f'Prediction: {alphabet} with probability {probability:.2f}%')
            if probability < self.confidence_threshold:
                alphabet = None
                return None
            #Get the alphabet from the index
            alphabet = self.classes[int(index)]
            print(f'Prediction: {alphabet}, Probability {probability:.2f}%')
        else:
            alphabet = None
        
        return alphabet

    #Function to save the alphabet in the dictionary:
    def save_alphabet(self, alphabet):
        #If the alphabet is not None:
        if alphabet is not None:
            #If the alphabet is already in the dictionary:
            if alphabet in self.alphabets_dict:
                #If the alphabet is already in the dictionary:
                self.alphabets_dict[alphabet] += 1
            else:
                #If the alphabet is not in the dictionary:
                self.alphabets_dict[alphabet] = 1

    #Function to print the dictionary in the end of the program:
    def print_alphabet(self):
        print(self.alphabets_dict)

    #Function to detect the number of occurrences of each alphabet if it is more than a certain threshold:
    #Then, concatenate the alphabets to form one word:
    #Called at the end of the program
    def concatenate_alphabets(self):
        #Create a list of alphabets:
        alphabets_list = []
        #Iterate through the dictionary:
        for alphabet, occur in self.alphabets_dict.items():
            #If the number of occurrences is more than the predefined threshold:
            if occur > self.alphabets_occur_threshold:
                alphabets_list.append(alphabet)
        #Join the alphabets to form a word:
        word = ''.join(alphabets_list)
        print(f'Word: {word}')
        #Check the occurrence of the word coffee:
        coffee_keywords = ['COFFEE', 'KAFE', 'COFE', 'KAF', 'COE', 'KOF', 'COF', 'KOE', 'CAF']
        for keyword in coffee_keywords:
            # for c in keyword:
            if all(self.alphabets_dict.get(c, 0) > self.alphabets_occur_threshold for c in keyword):
                print(f'Did you mean Coffee?')
                self.command = 'Coffee'
                break
        #Check the occurrence of the word coffee:
        # for alphabet, occur in self.alphabets_dict.items():
        #     if alphabet == 'K' and occur > self.alphabets_occur_threshold:
        #         if 'A' in self.alphabets_dict and self.alphabets_dict['O'] > self.alphabets_occur_threshold:
        #             if 'F' in self.alphabets_dict and self.alphabets_dict['F'] > self.alphabets_occur_threshold:
        #                 if 'E' in self.alphabets_dict and self.alphabets_dict['E'] > self.alphabets_occur_threshold:
        #                     print('Did you mean Coffee? True')
        #                     self.command = 'Coffee'
        
        return
        #Print the corrected word:
        # print(f'Did you mean Coffee? {word == "COFFEE"}')


    def run(self):
        if self.is_camera:
            self.run_camera()
        else:
            self.run_image()

    def run_camera(self):

        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture('asl_video.mp4')
        total_fps = 0 # To get the final frames per second.
        frame_count = 0 # To count total frames.
        while True:
            #Waiting for frames from camera
            ret, frame = cap.read()

            #Getting the color frame from the camera
            color_frame = frame

            #converting the color frame into numpy array
            color_image = np.asanyarray(color_frame)

            #Performing inference using our model
            # Get the start time.
            start_time = time.time()
            with torch.no_grad():
                signs = self.model(color_image)
            # Get the end time.
            end_time = time.time()
            # Calculate the time it took to run the model.
            elapsed_time = end_time - start_time
            # Print the time it took to run the model.
            fps = 1 / (elapsed_time)
            total_fps += fps
            # Increment frame count.
            frame_count += 1
            #Add FPS to the frame
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #Get the alphabet from the detections:
            prediction_alphabet = self.get_prediction_alphabet(signs)
            #Save the alphabet to a dictionary with maximum number of 10 alphabets:
            self.save_alphabet(prediction_alphabet)
            # save_alphabet(prediction_alphabet)
            # print(prediction_alphabet)
            # print(signs.xyxyn)
            # print(signs.display)
            #Get the confidence of the highest probability class:
            cv2.imshow('Sign Language Detection', np.squeeze(signs.render()))    
            # cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
        self.print_alphabet()
        self.concatenate_alphabets()

    def run_image(self):
        #Inference on Image
        img = cv2.imread("/home/mrblack/Projects_DL/SignLanguage_YOLOv7/ASL/test/images/A22_jpg.rf.beab2ab0db6a2c06a0725da776ff9d74.jpg")
        signs = self.model(img)
        cv2.imshow('Sign Language Detection', np.squeeze(signs.render()))
        cv2.waitKey(0)


if __name__ == "__main__":
    sign_language_detection = SignLanguageDetection(is_camera=True)
    sign_language_detection.run()
    print("Command Passed: ", sign_language_detection.command)
