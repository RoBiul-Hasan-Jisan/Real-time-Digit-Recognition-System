import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

from utils import decode_base64_image, preprocess_image_for_model, make_gradcam_heatmap, heatmap_to_base64

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

MODEL_PATH = "../model/saved_model/digit_cnn.keras"
CONF_MATRIX_PATH = "../model/saved_model/confusion_matrix.npy"
CLASS_ACC_PATH = "../model/saved_model/class_accuracy.json"
HISTORY_PATH = "../model/saved_model/training_history.json"

# ✅ Load model (simple and stable)
model = load_model(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode image
        pil_image = decode_base64_image(data["image"])

        # Preprocess
        img_array, _ = preprocess_image_for_model(pil_image)

        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        # Top 3 predictions
        top_3_indices = predictions.argsort()[-3:][::-1]
        top_3 = [
            {"digit": int(i), "confidence": float(predictions[i])}
            for i in top_3_indices
        ]

        # 🔥 Heatmap (safe mode)
        heatmap_base64 = None
        try:
            heatmap = make_gradcam_heatmap(img_array, model)
            heatmap_base64 = heatmap_to_base64(heatmap)
        except Exception as e:
            print("Grad-CAM error:", e)

        return jsonify({
            "predicted_digit": predicted_digit,
            "confidence": round(confidence * 100, 2),
            "top_3_predictions": [
                {
                    "digit": item["digit"],
                    "confidence": round(item["confidence"] * 100, 2)
                }
                for item in top_3
            ],
            "heatmap": heatmap_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analytics", methods=["GET"])
def analytics():
    try:
        confusion_matrix = np.load(CONF_MATRIX_PATH).tolist()

        with open(CLASS_ACC_PATH, "r") as f:
            class_acc_data = json.load(f)

        with open(HISTORY_PATH, "r") as f:
            history_data = json.load(f)

        return jsonify({
            "confusion_matrix": confusion_matrix,
            "overall_accuracy": class_acc_data["overall_accuracy"],
            "overall_loss": class_acc_data["overall_loss"],
            "class_accuracy": class_acc_data["class_accuracy"],
            "training_history": history_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)