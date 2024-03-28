import cv2
from deepface import DeepFace


analysis = DeepFace.analyze(img_path = "Face.jpg", actions = ["age", "gender", "emotion", "race"]) 

print(analysis)