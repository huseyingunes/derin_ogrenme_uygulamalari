# https://ai.google.dev/edge/mediapipe/solutions/vision/image_classifier?hl=tr

#https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/image_classification/python/image_classifier.ipynb?hl=tr#scrollTo=Yl_Oiye4mUuo

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("kedi.jpg")

classification_result = classifier.classify(image)

print(classification_result.classifications[0].categories[0])