#https://ai.google.dev/edge/mediapipe/solutions/vision/interactive_segmenter?hl=tr

#https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/interactive_segmentation/python/interactive_segmenter.ipynb?hl=tr

from typing import Tuple, Union
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math
import numpy as np
from mediapipe.tasks.python.components import containers

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow("a", img)
  cv2.waitKey(0)

BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white


RegionOfInterest = vision.InteractiveSegmenterRegionOfInterest
NormalizedKeypoint = containers.keypoint.NormalizedKeypoint

# Create the options that will be used for InteractiveSegmenter
base_options = python.BaseOptions(model_asset_path='magic_touch.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)



with vision.InteractiveSegmenter.create_from_options(options) as segmenter:

    image = mp.Image.create_from_file("kedi.jpg")
    roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT,
                           keypoint=NormalizedKeypoint(200, 200))
    segmentation_result = segmenter.segment(image, roi)
    category_mask = segmentation_result.category_mask

    # Generate solid color images for showing the output segmentation mask.
    image_data = image.numpy_view()
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
    output_image = np.where(condition, fg_image, bg_image)

    # Draw a white dot with black border to denote the point of interest
    thickness, radius = 6, -1
    keypoint_px = _normalized_to_pixel_coordinates(200, 200, image.width, image.height)
    cv2.circle(output_image, keypoint_px, thickness + 5, (0, 0, 0), radius)
    cv2.circle(output_image, keypoint_px, thickness, (255, 255, 255), radius)
    resize_and_show(output_image)



