import torch                         # Core PyTorch
import numpy as np                   # Array manipulation
import cv2                           # OpenCV for image loading and preprocessing

import matplotlib 
matplotlib.use('TkAgg')              # For WSL arch linux GUI support 
import matplotlib.pyplot as plt

import mediapipe as mp 

# load pretrained model
model = torch.jit.load("Pytorch-MobileFaceNet/save_model/mobilefacenet.pth", map_location="cpu")
model.eval()

# init images
face0 = cv2.imread("images/face0.jpg")
face1 = cv2.imread("images/face1.jpg")
group0 = cv2.imread("images/group0.jpg")
group1 = cv2.imread("images/group1.jpg")

# init mediapipe face detection 
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# convert BGR to RGB for mediapipe
rgb_image = cv2.cvtColor(group1, cv2.COLOR_BGR2RGB)

# create a face detector 
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(rgb_image)

    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(group1, detection)  # Draw on original BGR image for OpenCV display

# Show the result using OpenCV
cv2.imshow("Faces", group1)
cv2.waitKey(0)

