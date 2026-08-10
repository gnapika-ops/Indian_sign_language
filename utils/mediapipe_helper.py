
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# MODEL PATH
# --------------------------------------------------

MODEL_PATH = r"C:\Users\Bharath\Desktop\indianlang\models\hand_landmarker.task"


# --------------------------------------------------
# MEDIAPIPE HAND LANDMARKER SETUP
# --------------------------------------------------

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode


# --------------------------------------------------
# IMAGE MODE
# --------------------------------------------------

image_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

image_landmarker = HandLandmarker.create_from_options(
    image_options
)


# --------------------------------------------------
# CONVERT HAND LANDMARKS TO FEATURES
# --------------------------------------------------

def _landmarks_to_features(result):

    if len(result.hand_landmarks) == 0:
        return None

    hand = result.hand_landmarks[0]

    features = []

    for landmark in hand:

        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return np.array(
        features,
        dtype=np.float32
    )


# --------------------------------------------------
# EXTRACT LANDMARKS FROM IMAGE
# --------------------------------------------------

def extract_landmarks_image(image):

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    result = image_landmarker.detect(
        mp_image
    )

    return _landmarks_to_features(result)

