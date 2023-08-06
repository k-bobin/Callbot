# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np
import pandas as pd


# Build app and layout 
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout

#Import Anti-Spoofing
import sys
sys.path.insert(0, 'C:/Users/piai/Desktop/faceid/Silent-Face-Anti-Spoofing')
from test import test

class CamApp(App):

    def build(self):
        # Main layout components 
        self.web_cam = Image(size_hint=(1, .9))
        self.button = Button(text="인증 시작",on_press= self.verify, size_hint=(1, .1))
        self.verification_label = Label(text="대기중", size_hint=(1, .1))

        # Add items to layout
        layout = BoxLayout(orientation='horizontal')

        # Left side layout
        left_layout = BoxLayout(orientation='vertical')
        left_layout.add_widget(self.web_cam)
        left_layout.add_widget(self.button)
        left_layout.add_widget(self.verification_label)

        layout.add_widget(left_layout)

        # Right side layout
        self.right_layout = GridLayout(rows=7, size_hint=(0.7, 1))
        self.labels = ["이름:", "주민번호:", "전화번호:", "가입 상품1:", "가입 상품2:", "가입 상품3:", "가입 상품4:"]
        self.label_widgets = []  # List to store the label widgets

        for label_text in self.labels:
            label = Label(text=label_text)
            self.right_layout.add_widget(label)
            self.label_widgets.append(label)  # Add the label widget to the list

        layout.add_widget(self.right_layout)


        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={'L1Dist':L1Dist})

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout

    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        # frame = frame[120:120+250, 200:200+250, :]

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from file and conver to 100x100px
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image 
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100,100))
        # Scale image to be between 0 and 1 
        img = img / 255.0
        
        # Return image
        return img

    # Verification function to verify person
    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.9
        verification_threshold = 0.8

        # Capture input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Get the largest face detected
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            
            # Crop the image to the region of interest (face)
            face = frame[y:y+h, x:x+w]
            cv2.imwrite(SAVE_PATH, face)


            label = test(
                image=frame,
                model_dir='C:/Users/piai/Desktop/faceid/Silent-Face-Anti-Spoofing/resources/anti_spoof_models',
                device_id=0
                )


            # 1 is real image
            if label == 1:
                # Build results array
                            # Build results array
                results = []
                for image in os.listdir(os.path.join('application_data', 'verification_images')):
                    input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
                    validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            
                    
                    # Make Predictions 
                    result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
                    results.append(result)
                
                # Detection Threshold: Metric above which a prediciton is considered positive 
                detection = np.sum(np.array(results) > detection_threshold)
                
                # Verification Threshold: Proportion of positive predictions / total positive samples 
                verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
                verified = verification > verification_threshold

                # Set verification text 
                self.verification_label.text = '인증에 성공했습니다.' if verified == True else '인증에 실패했습니다.'

                
                if verified:
                    # Load customer_db
                    customer_db = pd.read_csv('C:/Users/piai/Desktop/faceid/app/customer_db.csv')

                    # Assign values to Label1 to Label7
                    for i in range(1, 8):
                        label = customer_db.loc[i-1, 'Label']
                        self.right_layout.children[i-1].text = label

                else: 
                    for i in range(7):
                        label_widget = self.label_widgets[i]
                        label_widget.text = self.labels[i]
                    
                # Log out details
                Logger.info(results)
                Logger.info(detection)
                Logger.info(verification)
                Logger.info(verified)
                
                return results, verified

            # 0 is spoofer/fake
            else:
                self.verification_label.text = '사진 또는 영상을 사용할 수 없습니다.'

        else:
            # If no face is detected, save the entire frame as the input image
            self.verification_label.text = '얼굴이 탐지되지 않습니다.'  # 얼굴이 탐지되지 않았을 때 메시지 출력
            # Reset the label texts
            for i in range(7):
                label_widget = self.label_widgets[i]
                label_widget.text = self.labels[i]

if __name__ == '__main__':
    CamApp().run()

