import os
from flask import Flask, request, jsonify

from flask_cors import CORS
from tensorflow import keras
from bottle_prediction import predict_bottle_fill_batch
from sticker_prediction import get_sticker_predictions, get_counts_as_json
import tempfile
import os
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Model and config ---
img_height, img_width = 150, 150
class_names = ['full', 'medium', 'empty']

# Load your trained model once at startup
model = keras.models.load_model("models/classifier_v1.keras")

# --- Your prediction function ---
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

@app.route('/stickers', methods=['POST'])
def stickers():
    # Get single image from request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    tmp_path = None
    try:
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Detect stickers in the image
        results = get_sticker_predictions()
        
        # Get counts in JSON format
        counts_json = get_counts_as_json(results)
        
        # Parse the JSON string back to a dictionary
        response_data = json.loads(counts_json)
                
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temp file
        if tmp_path:
            try:
                os.remove(tmp_path)
            except:
                pass

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=3001)
