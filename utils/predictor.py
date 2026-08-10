import numpy as np
import joblib


# -----------------------------------
# Load Saved Files
# -----------------------------------

MODEL_PATH = r"C:\Users\Bharath\Desktop\indianlang\models\Ann_model.pkl"

ENCODER_PATH = r"C:\Users\Bharath\Desktop\indianlang\models\Label_encoder.pkl"


model = joblib.load(MODEL_PATH)

encoder = joblib.load(ENCODER_PATH)


# -----------------------------------
# Prediction Function
# -----------------------------------

def predict_sign(landmarks):

    # Convert to numpy array
    landmarks = np.array(
        landmarks,
        dtype=np.float32
    )

    # Make sure we have 63 features
    if landmarks.size != 63:
        raise ValueError(
            f"Expected 63 features, but received {landmarks.size}"
        )

    # Reshape for MLPClassifier
    landmarks = landmarks.reshape(1, -1)

    # Prediction
    prediction = model.predict(
        landmarks
    )

    # Convert encoded number back to letter
    sign = encoder.inverse_transform(
        prediction
    )[0]

    # Confidence
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            landmarks
        )

        confidence = float(
            np.max(probabilities)
        )

    else:

        confidence = 0.0

    return sign, confidence