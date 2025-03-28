import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import io
from PIL import Image
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the absolute path to the model file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rice_leaf_deficiency_model.h5')

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names based on your labels
class_names = ['Nitrogen', 'Phosphorus', 'Potassium']

def preprocess_image(image_data):
    # Convert base64 to image
    try:
        # Remove the data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode base64 string
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Resize image
        image = image.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

@app.route('/')
def home():
    return jsonify({'message': 'Server is running', 'status': 'ok'})

@app.route('/test')
def test():
    return jsonify({'message': 'Test endpoint working', 'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request")  # Debug log
    
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        if not data or 'image' not in data:
            print("No image data in request")  # Debug log
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get the base64 encoded image
        image_data = data['image']
        
        print("Processing image data...")  # Debug log
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        start_time = time.time()
        predictions = model.predict(processed_image, verbose=0)
        end_time = time.time()
        
        # Get prediction details
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        inference_time = end_time - start_time
        
        return jsonify({
            'class': int(predicted_class_idx),  # 0, 1, or 2
            'class_name': class_names[predicted_class_idx],
            'confidence': confidence,
            'inference_time': inference_time
        })
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")  # Debug log
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model exists: {os.path.exists(MODEL_PATH)}")
    app.run(host='0.0.0.0', debug=True, port=5000)
