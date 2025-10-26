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
class_names = ['full', 'medium', 'empty']

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

    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(keras.activations.softmax(predictions[0])))

    return class_names[predicted_index], confidence

def predict_bottle_fill_batch(model, img_paths, class_names):
    """
    Takes a list of image paths and returns predictions for all images in a single batch.
    This is much faster than predicting images one at a time.
    """
    img_arrays = []
    for img_path in img_paths:
        img = keras.utils.load_img(img_path, target_size=(img_height, img_width))
        img_array = keras.utils.img_to_array(img)
        img_arrays.append(img_array)
    
    # Stack all images into a single batch
    batch = np.stack(img_arrays, axis=0)
    
    # Single prediction call for all images
    predictions = model.predict(batch, verbose=0)
    
    results = []
    for pred in predictions:
        predicted_index = np.argmax(pred)
        confidence = float(np.max(keras.activations.softmax(pred)))
        results.append((class_names[predicted_index], confidence))
    
    return results

@app.route('/')
def home():
    return "API de procesamiento de imágenes con Flask y TensorFlow"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'service': 'bottle-classification'
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve multiple images from the request
    files = request.files.getlist('images')

    if not files or len(files) == 0:
        return jsonify({'error': 'No images uploaded'}), 400

    tmp_paths = []
    file_names = []

    try:
        # Save all files first
        for file in files:
            if file.filename == '':
                continue

            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                file.save(tmp.name)
                tmp_paths.append(tmp.name)
                file_names.append(file.filename)

        # Batch predict all images at once (much faster)
        predictions = predict_bottle_fill_batch(model, tmp_paths, class_names)

        # Format results
        results = [
            {
                'file_name': file_names[i],
                'predicted_class': pred[0],
                'confidence': pred[1]
            }
            for i, pred in enumerate(predictions)
        ]

        return jsonify({'predictions': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temp files
        for tmp_path in tmp_paths:
            try:
                os.remove(tmp_path)
            except:
                pass

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=3001)
