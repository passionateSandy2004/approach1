import google.generativeai as genai
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configure the Gemini API with your key
genai.configure(api_key="AIzaSyC2Rb7z7HTqOjmPWCN7ZmyVW3HQ1TOtPqQ")

def analyze_images_base64(images, prompt):
    try:
        # Initialize the model - using gemini-1.5-pro for better vision capabilities
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        # Prepare contents with prompt and base64 images
        contents = [prompt]
        for img in images:
            contents.append({
                "inline_data": {
                    "mime_type": img.get("mime_type", "image/jpeg"),
                    "data": img.get("data")
                }
            })
        
        # Generate content with the images in base64 format
        response = model.generate_content(contents)
        return response.text
    except Exception as e:
        return f"Error generating content: {e}"

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'images' not in data or 'prompt' not in data:
        return jsonify({"error": "Missing 'images' or 'prompt' in request body"}), 400
    
    images = data['images']
    prompt = data['prompt']
    
    if not isinstance(images, list):
        return jsonify({"error": "'images' must be a list"}), 400
    
    result = analyze_images_base64(images, prompt)
    return jsonify({"analysis": result})

if __name__ == '__main__':
    app.run(debug=True) 
