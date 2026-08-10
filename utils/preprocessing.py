import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------------------------------------------------------
# PATH TO MEDIAPIPE HAND LANDMARKER MODEL
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hand_landmarker.task"
)


# ---------------------------------------------------------
# CREATE MEDIAPIPE HAND LANDMARKER
# ---------------------------------------------------------

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)

image_landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ---------------------------------------------------------
# EXTRACT 63 LANDMARK FEATURES FROM IMAGE
# ---------------------------------------------------------

def extract_landmarks_image(image):

    if image is None:
        return None

    # OpenCV BGR → RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    # Detect hand
    result = image_landmarker.detect(mp_image)

    # No hand detected
    if not result.hand_landmarks:
        return None

    # Take first detected hand
    hand_landmarks = result.hand_landmarks[0]

    features = []

    # 21 landmarks × x,y,z = 63 features
    for landmark in hand_landmarks:

        features.append(landmark.x)
        features.append(landmark.y)
        features.append(landmark.z)

    return np.array(features, dtype=np.float32)