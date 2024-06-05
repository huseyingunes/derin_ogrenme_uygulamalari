# https://ai.google.dev/edge/mediapipe/solutions/vision/image_classifier?hl=tr

from typing import Tuple, Union
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np


# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options)

vid = cv2.VideoCapture(0)
while(True):
  ret, frame = vid.read()
  image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
  classification_result = classifier.classify(image)
  print(classification_result.classifications[0].categories[0])
  cv2.imshow("a", frame)
  cv2.waitKey(50)