from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
import numpy as np
import tempfile
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Model and config ---
img_height, img_width = 150, 150
class_names = ['Full Water Level', 'Half Water Level', 'Overflowing']

# Load your trained model once at startup
model = keras.models.load_model("models/classifier_v1.keras")

# --- Your prediction function ---
def predict_bottle_fill(model, img_path, class_names):
    """
    Takes an image path and returns the predicted fill level.
    """
    img = keras.utils.load_img(img_path, target_size=(img_height, img_width))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(keras.activations.softmax(predictions[0])))

    return class_names[predicted_index], confidence

@app.route('/')
def home():
    return "API de procesamiento de imágenes con Flask y TensorFlow"

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve multiple images from the request
    files = request.files.getlist('images')

    if not files or len(files) == 0:
        return jsonify({'error': 'No images uploaded'}), 400

    results = []

    try:
        for file in files:
            if file.filename == '':
                continue

            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            # Run prediction
            predicted_class, confidence = predict_bottle_fill(model, tmp_path, class_names)

            # Store result
            results.append({
                'file_name': file.filename,
                'predicted_class': predicted_class,
                'confidence': confidence
            })

            # Delete temp file
            os.remove(tmp_path)

        return jsonify({'predictions': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
