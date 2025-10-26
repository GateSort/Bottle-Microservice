import numpy as np
from tensorflow import keras

img_height, img_width = 150, 150

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
