import streamlit as st
import cv2
import numpy as np
import joblib
import av

from PIL import Image

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Indian Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)


# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = r"C:\Users\Bharath\Desktop\indianlang\models\Ann_model.pkl"

ENCODER_PATH = r"C:\Users\Bharath\Desktop\indianlang\models\Label_encoder.pkl"

HAND_MODEL_PATH = r"C:\Users\Bharath\Desktop\indianlang\models\hand_landmarker .task"


# ==========================================================
# LOAD ANN MODEL
# ==========================================================

model = joblib.load(MODEL_PATH)

label_encoder = joblib.load(ENCODER_PATH)


# ==========================================================
# MEDIAPIPE HAND LANDMARKER
# ==========================================================

base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

hand_landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ==========================================================
# EXTRACT HAND LANDMARKS
# ==========================================================

def extract_landmarks_image(image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = hand_landmarker.detect(
        mp_image
    )

    if not result.hand_landmarks:
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


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🤟 Indian Sign Language")

st.sidebar.success(
    "Artificial Neural Network"
)

st.sidebar.info(
    "MediaPipe Hand Landmarks"
)

st.sidebar.markdown("---")


# ==========================================================
# PROJECT OBJECTIVE
# ==========================================================

st.sidebar.write("### 🎯 Project Objective")

st.sidebar.write(
    """
    The objective of this project is to recognize
    Indian Sign Language alphabet gestures using
    MediaPipe hand landmarks and an Artificial
    Neural Network (ANN).

    The system extracts 21 hand landmarks from
    an input image. Each landmark contains
    X, Y and Z coordinates, producing 63 numerical
    features.

    These features are given to the trained ANN
    model to predict the corresponding alphabet
    from A to Z.
    """
)

st.sidebar.markdown("---")


# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.sidebar.write("### 🧠 Model Information")

st.sidebar.write(
    "Input Features: 63"
)

st.sidebar.write(
    "Classes: A - Z"
)

st.sidebar.write(
    "Algorithm: MLPClassifier (ANN)"
)

st.sidebar.markdown("---")


# ==========================================================
# SELECT MODE
# ==========================================================

mode = st.sidebar.radio(
    "Select Mode",
    [
        "Upload Image",
        "Live Webcam"
    ]
)


# ==========================================================
# MAIN TITLE
# ==========================================================

st.title(
    "🤟 Indian Sign Language Alphabet Recognition"
)

st.write(
    "MediaPipe Hand Landmarks + Artificial Neural Network"
)

st.markdown("---")


# ==========================================================
# UPLOAD IMAGE MODE
# ==========================================================

if mode == "Upload Image":

    st.subheader(
        "📷 Upload Sign Language Image"
    )

    uploaded_file = st.file_uploader(
        "Upload an image of a hand sign",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        # Open uploaded image
        image = Image.open(
            uploaded_file
        )

        image = np.array(image)

        # RGB → BGR
        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        # Extract 63 features
        landmarks = extract_landmarks_image(
            image_bgr
        )

        if landmarks is None:

            st.error(
                "❌ No hand detected. Please upload a clear image showing one hand."
            )

        else:

            # Check number of features
            if len(landmarks) != 63:

                st.error(
                    f"Expected 63 features, but received {len(landmarks)}."
                )

            else:

                # Reshape for ANN
                landmarks_input = landmarks.reshape(
                    1,
                    -1
                )

                # Prediction
                prediction = model.predict(
                    landmarks_input
                )

                # Convert encoded prediction to letter
                predicted_label = label_encoder.inverse_transform(
                    prediction
                )[0]

                # Get probabilities
                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        landmarks_input
                    )

                    confidence = float(
                        np.max(probabilities)
                    )

                else:

                    confidence = 0.0


                # ==========================================
                # DISPLAY
                # ==========================================

                col1, col2 = st.columns(
                    [2, 1]
                )

                with col1:

                    st.image(
                        image,
                        caption="Uploaded Sign",
                        width=600
                    )

                with col2:

                    st.success(
                        "Prediction"
                    )

                    st.metric(
                        "Predicted Sign",
                        predicted_label
                    )

                    if confidence > 0:

                        st.metric(
                            "Confidence",
                            f"{confidence * 100:.2f}%"
                        )

                        st.progress(
                            confidence
                        )

                    st.write(
                        "### Prediction Status"
                    )

                    if confidence >= 0.80:

                        st.success(
                            "Very High Confidence"
                        )

                    elif confidence >= 0.60:

                        st.info(
                            "Good Confidence"
                        )

                    elif confidence >= 0.40:

                        st.warning(
                            "Moderate Confidence"
                        )

                    else:

                        st.error(
                            "Low Confidence"
                        )


# ==========================================================
# LIVE WEBCAM
# ==========================================================

if mode == "Live Webcam":

    st.subheader(
        "📹 Live Sign Language Recognition"
    )

    st.info(
        "Click START and allow camera permission."
    )


    class SignLanguageProcessor(
        VideoProcessorBase
    ):

        def recv(self, frame):

            image = frame.to_ndarray(
                format="bgr24"
            )

            output = image.copy()

            landmarks = extract_landmarks_image(
                image
            )

            if landmarks is not None:

                landmarks_input = landmarks.reshape(
                    1,
                    -1
                )

                prediction = model.predict(
                    landmarks_input
                )

                predicted_label = label_encoder.inverse_transform(
                    prediction
                )[0]

                # Confidence
                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        landmarks_input
                    )

                    confidence = float(
                        np.max(probabilities)
                    )

                else:

                    confidence = 0.0


                # ==========================================
                # TEXT ON VIDEO
                # ==========================================

                text = (
                    f"{predicted_label} "
                    f"{confidence * 100:.1f}%"
                )

                cv2.rectangle(
                    output,
                    (10, 10),
                    (250, 75),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    output,
                    text,
                    (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

            else:

                cv2.rectangle(
                    output,
                    (10, 10),
                    (350, 65),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    output,
                    "No hand detected",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )


            return av.VideoFrame.from_ndarray(
                output,
                format="bgr24"
            )


    # Start webcam

    webrtc_streamer(
        key="sign-language-recognition",
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        Indian Sign Language Alphabet Recognition<br>
        <b>MediaPipe Hand Landmarks + Artificial Neural Network</b>
    </div>
    """,
    unsafe_allow_html=True
)