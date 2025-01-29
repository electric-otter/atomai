import os
import subprocess

# Install necessary packages
os.system("pip install numpy")
os.system("pip install tensorflow")
os.system("pip install opencv-python")
os.system("pip install mediapipe")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import cv2
import mediapipe as mp

# Create and save a simple emotion detection model
def create_emotion_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(7, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.save('emotion_model.h5')

# Run the main application
def run_application():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    emotion_model = tf.keras.models.load_model('emotion_model.h5')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    cap = cv2.VideoCapture(0)

    def get_emotion_color(emotion):
        colors = {
            'Angry': (0, 0, 255),
            'Disgust': (0, 255, 0),
            'Fear': (255, 255, 0),
            'Happy': (0, 255, 255),
            'Sad': (255, 0, 0),
            'Surprise': (255, 0, 255),
            'Neutral': (255, 255, 255)
        }
        return colors.get(emotion, (255, 255, 255))

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
        
        emotion = 'Neutral'
        for (x, y, w, h) in faces:
            face = gray_image[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype('float32') / 255
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)
            
            emotion_prediction = emotion_model.predict(face)
            emotion = emotion_labels[np.argmax(emotion_prediction)]
            
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        color = get_emotion_color(emotion)
        image[:] = color
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('Atom AI', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    create_emotion_model()
    run_application()

if __name__ == "__main__":
    main()
