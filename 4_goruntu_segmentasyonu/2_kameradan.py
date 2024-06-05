# https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter/web_js?hl=tr

#https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/image_segmentation/python/image_segmentation.ipynb?hl=tr


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math
import numpy as np

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow("a", img)
  cv2.waitKey(50)

BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white


# Create the options that will be used for ImageSegmenter
base_options = python.BaseOptions(model_asset_path='deeplab_v3.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)

vid = cv2.VideoCapture(0)
while(True):
  ret, frame = vid.read()
  image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
  with vision.ImageSegmenter.create_from_options(options) as segmenter:
      segmentation_result = segmenter.segment(image)
      category_mask = segmentation_result.category_mask

      # Generate solid color images for showing the output segmentation mask.
      image_data = image.numpy_view()
      fg_image = np.zeros(image_data.shape, dtype=np.uint8)
      fg_image[:] = MASK_COLOR
      bg_image = np.zeros(image_data.shape, dtype=np.uint8)
      bg_image[:] = BG_COLOR

      condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
      output_image = np.where(condition, fg_image, bg_image)
      resize_and_show(output_image)



