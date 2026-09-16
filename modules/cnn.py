import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
from PIL import Image
import numpy as np
import h5py


IMG_SIZE = (128, 128)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "my_5layer_cnn.h5"


def build_cnn_model():
    """
    Reconstruct the original 5-layer CNN architecture.

    The architecture was recovered from the saved H5 model:
        Input: 128 x 128 x 3
        Conv2D: 32 filters
        MaxPooling2D
        Conv2D: 64 filters
        MaxPooling2D
        Conv2D: 128 filters
        MaxPooling2D
        Flatten
        Dense: 128
        Dropout: 0.5
        Dense: 1 sigmoid
    """

    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),

        layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        layers.Flatten(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.5),

        layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    return model


def _read_dataset(h5_file, dataset_path):
    """
    Read a dataset from the legacy Keras H5 weight structure.
    """
    return np.array(h5_file[dataset_path])


def load_cnn_model():
    """
    Load the CNN without using Keras H5 model deserialization.

    The original H5 was saved using Keras 3.15.1, while the
    Streamlit environment uses TensorFlow 2.15.1.

    Therefore, we reconstruct the architecture and load the
    trained weights directly from the H5 file.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CNN model not found:\n{MODEL_PATH}"
        )

    try:
        # ---------------------------------------------------------
        # Build the exact original architecture
        # ---------------------------------------------------------
        model = build_cnn_model()

        # ---------------------------------------------------------
        # Read trained weights directly from H5
        # ---------------------------------------------------------
        with h5py.File(MODEL_PATH, "r") as f:

            required_weights = [
                "conv2d/sequential/conv2d/kernel",
                "conv2d/sequential/conv2d/bias",

                "conv2d_1/sequential/conv2d_1/kernel",
                "conv2d_1/sequential/conv2d_1/bias",

                "conv2d_2/sequential/conv2d_2/kernel",
                "conv2d_2/sequential/conv2d_2/bias",

                "dense/sequential/dense/kernel",
                "dense/sequential/dense/bias",

                "dense_1/sequential/dense_1/kernel",
                "dense_1/sequential/dense_1/bias",
            ]

            # Check that all required weights exist
            missing = [
                path for path in required_weights
                if path not in f
            ]

            if missing:
                raise RuntimeError(
                    "The following CNN weights are missing from "
                    "the H5 file:\n"
                    + "\n".join(missing)
                )

            # -----------------------------------------------------
            # Load weights layer by layer
            # -----------------------------------------------------

            model.layers[0].set_weights([
                _read_dataset(
                    f,
                    "conv2d/sequential/conv2d/kernel"
                ),
                _read_dataset(
                    f,
                    "conv2d/sequential/conv2d/bias"
                ),
            ])

            model.layers[2].set_weights([
                _read_dataset(
                    f,
                    "conv2d_1/sequential/conv2d_1/kernel"
                ),
                _read_dataset(
                    f,
                    "conv2d_1/sequential/conv2d_1/bias"
                ),
            ])

            model.layers[4].set_weights([
                _read_dataset(
                    f,
                    "conv2d_2/sequential/conv2d_2/kernel"
                ),
                _read_dataset(
                    f,
                    "conv2d_2/sequential/conv2d_2/bias"
                ),
            ])

            model.layers[7].set_weights([
                _read_dataset(
                    f,
                    "dense/sequential/dense/kernel"
                ),
                _read_dataset(
                    f,
                    "dense/sequential/dense/bias"
                ),
            ])

            model.layers[9].set_weights([
                _read_dataset(
                    f,
                    "dense_1/sequential/dense_1/kernel"
                ),
                _read_dataset(
                    f,
                    "dense_1/sequential/dense_1/bias"
                ),
            ])

        # ---------------------------------------------------------
        # Compile only for compatibility.
        # Prediction does not require training.
        # ---------------------------------------------------------
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return model

    except Exception as e:
        raise RuntimeError(
            "Unable to load the CNN model.\n\n"
            f"Model: {MODEL_PATH}\n"
            f"TensorFlow version: {tf.__version__}\n\n"
            f"Original error: {str(e)}"
        ) from e


def predict_image(model, image):
    """
    Predict whether an uploaded skin lesion image
    is benign or malignant.

    Returns:
        probability: malignant probability
        prediction: Benign or Malignant
    """

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize(IMG_SIZE)

    # Convert to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize exactly as used by the original model
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


def get_model_info(model):
    """
    Return information displayed by the application.
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


def print_model_summary(model):
    """
    Print CNN architecture.
    """

    model.summary()
