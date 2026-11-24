"""Prototype: Cassava leaf disease classifier (Gradio).

Upload a photo of a cassava leaf and get the predicted class with confidence scores.
Works in a Kaggle notebook (temporary public link) and on a Hugging Face Space (permanent link).
"""
import os
import numpy as np
import tensorflow as tf
import gradio as gr

# Class order = label numbers 0..4 of the Kaggle dataset
CLASS_NAMES = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mottle (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy",
]

# Find the saved model (same folder, or the Kaggle working folder)
CANDIDATES = [
    os.environ.get("MODEL_PATH", ""),
    "cassava_mobilenetv2.keras",
    "/kaggle/working/cassava_mobilenetv2.keras",
]
MODEL_PATH = next((p for p in CANDIDATES if p and os.path.exists(p)), None)
if MODEL_PATH is None:
    raise FileNotFoundError("Model file not found. Put cassava_mobilenetv2.keras next to app.py.")

model = tf.keras.models.load_model(MODEL_PATH)
HEIGHT, WIDTH = model.input_shape[1:3]   # read the input size from the model itself


def predict(img):
    if img is None:
        return None
    arr = np.array(img.convert("RGB"), dtype="float32")
    arr = tf.image.resize(arr, (HEIGHT, WIDTH)).numpy()   # same resize as in training
    probs = model.predict(arr[None], verbose=0)[0]        # the model normalises its input itself
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Photo of a cassava leaf"),
    outputs=gr.Label(num_top_classes=5, label="Prediction"),
    title="Cassava Leaf Disease Classifier (prototype)",
    description=(
        "Upload a photo of a cassava leaf. The model (MobileNetV2, transfer learning) predicts one of "
        "5 classes: 4 diseases or healthy. Prototype from a student project: about 62% accuracy on the "
        "test set, so always confirm a diagnosis with an agricultural expert."
    ),
)

if __name__ == "__main__":
    # In Kaggle, share=True gives a temporary public link (needs Internet on).
    # debug=True keeps the cell running (the link stays alive) and prints errors in the cell output.
    ON_KAGGLE = os.path.exists("/kaggle")
    demo.launch(share=ON_KAGGLE, debug=ON_KAGGLE)
