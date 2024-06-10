#https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python?hl=tr

#https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb?hl=tr#scrollTo=H4aPO-hvbw3r
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2

import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the image mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.IMAGE)
with GestureRecognizer.create_from_options(options) as recognizer:
    mp_image = mp.Image.create_from_file('1.jpg')
    gesture_recognition_result = recognizer.recognize(mp_image)
    top_gesture = gesture_recognition_result.gestures[0][0]
    hand_landmarks = gesture_recognition_result.hand_landmarks
    new_image = cv2.imread("1.jpg", cv2.IMREAD_UNCHANGED)

    gestures = gesture_recognition_result.gestures[0][0]
    title = f"{gestures.category_name} ({gestures.score:.2f})"
    print(title)

    for hand_landmarks_ in hand_landmarks:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks_
        ])

        mp_drawing.draw_landmarks(
            new_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


    cv2.imshow("a", new_image)
    cv2.waitKey()
