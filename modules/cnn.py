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
    Reconstruct the CNN architecture from the saved H5 model.

    Input:
        128 x 128 x 3

    Architecture:
        Conv2D 32
        MaxPooling2D
        Conv2D 64
        MaxPooling2D
        Conv2D 128
        MaxPooling2D
        Flatten
        Dense 128
        Dropout 0.5
        Dense 1 sigmoid
    """

    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),

        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            name="conv2d"
        ),

        layers.MaxPooling2D(
            pool_size=(2, 2),
            name="max_pooling2d"
        ),

        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            name="conv2d_1"
        ),

        layers.MaxPooling2D(
            pool_size=(2, 2),
            name="max_pooling2d_1"
        ),

        layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            name="conv2d_2"
        ),

        layers.MaxPooling2D(
            pool_size=(2, 2),
            name="max_pooling2d_2"
        ),

        layers.Flatten(
            name="flatten"
        ),

        layers.Dense(
            128,
            activation="relu",
            name="dense"
        ),

        layers.Dropout(
            0.5,
            name="dropout"
        ),

        layers.Dense(
            1,
            activation="sigmoid",
            name="dense_1"
        )
    ])

    return model


def _read_weight(h5_file, path):
    """Read one weight dataset from the H5 file."""
    return np.array(h5_file[path])


def load_cnn_model():
    """
    Load the trained CNN.

    The H5 model was saved with Keras 3.15.1.
    Streamlit currently uses TensorFlow 2.15.1.

    Therefore, we avoid Keras H5 model deserialization and
    reconstruct the architecture before loading the trained
    weights directly from the H5 file.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CNN model not found:\n{MODEL_PATH}"
        )

    try:
        # ---------------------------------------------------------
        # Build model
        # ---------------------------------------------------------
        model = build_cnn_model()

        # ---------------------------------------------------------
        # Load trained weights directly from H5
        # ---------------------------------------------------------
        with h5py.File(MODEL_PATH, "r") as f:

            # The actual H5 structure contains:
            #
            # model_weights/
            #     conv2d/
            #     conv2d_1/
            #     conv2d_2/
            #     dense/
            #     dense_1/

            weight_paths = {
                "conv2d_kernel":
                    "model_weights/conv2d/sequential/conv2d/kernel",

                "conv2d_bias":
                    "model_weights/conv2d/sequential/conv2d/bias",

                "conv2d_1_kernel":
                    "model_weights/conv2d_1/sequential/conv2d_1/kernel",

                "conv2d_1_bias":
                    "model_weights/conv2d_1/sequential/conv2d_1/bias",

                "conv2d_2_kernel":
                    "model_weights/conv2d_2/sequential/conv2d_2/kernel",

                "conv2d_2_bias":
                    "model_weights/conv2d_2/sequential/conv2d_2/bias",

                "dense_kernel":
                    "model_weights/dense/sequential/dense/kernel",

                "dense_bias":
                    "model_weights/dense/sequential/dense/bias",

                "dense_1_kernel":
                    "model_weights/dense_1/sequential/dense_1/kernel",

                "dense_1_bias":
                    "model_weights/dense_1/sequential/dense_1/bias",
            }

            # -----------------------------------------------------
            # Verify that every weight exists
            # -----------------------------------------------------
            missing = []

            for name, path in weight_paths.items():
                if path not in f:
                    missing.append(path)

            if missing:
                raise RuntimeError(
                    "The following CNN weights are missing "
                    "from the H5 file:\n"
                    + "\n".join(missing)
                )

            # -----------------------------------------------------
            # Conv2D 1
            # -----------------------------------------------------
            model.get_layer("conv2d").set_weights([
                _read_weight(
                    f,
                    weight_paths["conv2d_kernel"]
                ),
                _read_weight(
                    f,
                    weight_paths["conv2d_bias"]
                ),
            ])

            # -----------------------------------------------------
            # Conv2D 2
            # -----------------------------------------------------
            model.get_layer("conv2d_1").set_weights([
                _read_weight(
                    f,
                    weight_paths["conv2d_1_kernel"]
                ),
                _read_weight(
                    f,
                    weight_paths["conv2d_1_bias"]
                ),
            ])

            # -----------------------------------------------------
            # Conv2D 3
            # -----------------------------------------------------
            model.get_layer("conv2d_2").set_weights([
                _read_weight(
                    f,
                    weight_paths["conv2d_2_kernel"]
                ),
                _read_weight(
                    f,
                    weight_paths["conv2d_2_bias"]
                ),
            ])

            # -----------------------------------------------------
            # Dense 128
            # -----------------------------------------------------
            model.get_layer("dense").set_weights([
                _read_weight(
                    f,
                    weight_paths["dense_kernel"]
                ),
                _read_weight(
                    f,
                    weight_paths["dense_bias"]
                ),
            ])

            # -----------------------------------------------------
            # Output Dense
            # -----------------------------------------------------
            model.get_layer("dense_1").set_weights([
                _read_weight(
                    f,
                    weight_paths["dense_1_kernel"]
                ),
                _read_weight(
                    f,
                    weight_paths["dense_1_bias"]
                ),
            ])

        # ---------------------------------------------------------
        # Compile for compatibility
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

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to CNN input size
    image = image.resize(IMG_SIZE)

    # Convert to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize to 0-1
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Run prediction
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
    Return model information for the Streamlit interface.
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
    """Print the CNN architecture."""
    model.summary()
