import tensorflow as tf
from pathlib import Path
from PIL import Image
import numpy as np


# ============================================================
# CNN CONFIGURATION
# ============================================================

IMG_SIZE = (128, 128)

# Project root:
# SkinLesionResearchPlatform/
# ├── models/
# │   └── my_5layer_cnn.h5
# └── modules/
#     └── cnn.py

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "my_5layer_cnn.h5"


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_cnn_model():
    """
    Load the previously trained 5-layer CNN model.

    The model is loaded with compile=False because this
    application only performs inference. This also avoids
    unnecessary deserialization of optimizer/loss configuration.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CNN model not found:\n{MODEL_PATH}"
        )

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

    except Exception as e:
        raise RuntimeError(
            "Unable to load the CNN model.\n\n"
            f"Model: {MODEL_PATH}\n"
            f"TensorFlow version: {tf.__version__}\n\n"
            "The saved H5 model may have been created with a "
            "different Keras/TensorFlow version.\n\n"
            f"Original error: {e}"
        ) from e

    return model


# ============================================================
# CNN PREDICTION
# ============================================================

def predict_image(model, image):
    """
    Predict whether an uploaded skin lesion image
    is benign or malignant.

    Parameters
    ----------
    model : TensorFlow model
        Loaded CNN model.

    image : PIL.Image
        Uploaded image.

    Returns
    -------
    probability : float
        Malignant probability.

    prediction : str
        Benign or Malignant.
    """

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to CNN input size
    image = image.resize(IMG_SIZE)

    # Convert image to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize exactly as during training
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    probability = float(
        model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    # Classification threshold
    if probability >= 0.5:
        prediction = "Malignant"
    else:
        prediction = "Benign"

    return probability, prediction


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info(model):
    """
    Return basic information about the CNN model.
    """

    return {
        "Input Size": "128 × 128 × 3",
        "Model Type": "5-Layer CNN",
        "Output": "Binary Classification",
        "Activation": "Sigmoid",
        "Loss": "Binary Crossentropy",
        "Optimizer": "Adam",
        "Model File": str(MODEL_PATH)
    }


# ============================================================
# OPTIONAL: DISPLAY MODEL SUMMARY
# ============================================================

def print_model_summary(model):
    """
    Print the CNN architecture in the terminal.
    """

    model.summary()

