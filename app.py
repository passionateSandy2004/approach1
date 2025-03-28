import google.generativeai as genai
import base64
from flask import Flask, request, jsonify
import os
import tempfile
from pdf2image import convert_from_path
from docx2pdf import convert
from PIL import Image
import logging
from typing import Optional, List, Dict
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure the Gemini API with your key
genai.configure(api_key="AIzaSyC2Rb7z7HTqOjmPWCN7ZmyVW3HQ1TOtPqQ")

# Supported formats
SUPPORTED_DOC_FORMATS = ["pdf", "doc", "docx"]
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "webp", "heic"]

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Default analysis prompt
DEFAULT_PROMPT = '''
You are an expert resume analyzer. Your task is to evaluate resumes against the provided job description and rank candidates based on their suitability.

For each provided resume image, evaluate the candidate based on the following criteria:

1. **Tenure at Previous Companies**: Assess the duration of employment at each company, highlighting instances of long-term commitments (e.g., 5-10 years) and noting any patterns of short-term engagements.

2. **Job Description Alignment**: Determine how closely the candidate's experience, skills, and qualifications match the specified job description. Identify areas of strong alignment and any gaps.

3. **Skill Relevance and Proficiency**: Analyze the listed skills, their relevance to the job role, and the proficiency level indicated by the candidate's experience.

4. **Career Progression**: Examine the trajectory of the candidate's career, noting evidence of growth such as promotions, increased responsibilities, or diversification of skills.

5. **Educational Background**: Review the candidate's educational qualifications, considering the relevance of degrees, certifications, and continuous learning efforts to the job role.

6. **Stability and Commitment**: Beyond tenure, assess overall stability by evaluating the frequency of job changes and any patterns that may indicate a lack of commitment.

7. **Cultural Fit Indicators**: Identify any information suggesting the candidate's alignment with the company's values and culture, such as participation in relevant projects, initiatives, or organizations.

**Job Description:**
{job_description}

**Output Format:**
For each candidate, provide the following structured response:

[
    {{
        "Candidate ID": "001",
        "ranking": "rank of the candidate for the description given and the resume content",
        "JD Match Percentage": "85%"
    }}
]

Important:
1. Rank candidates based on their overall match with the job description
2. Calculate the match percentage based on required skills and experience
3. Keep the response concise and focused on the key metrics
GIVE THE RESPONSE ONLY IN JSON FORMAT.
Repeat this structured response for each candidate, incrementing the Candidate ID for each resume.
'''

def validate_file_size(file_data: bytes) -> bool:
    """Validate file size is within limits"""
    return len(file_data) <= MAX_FILE_SIZE

def convert_doc_to_pdf(doc_data: bytes, doc_type: str) -> Optional[bytes]:
    """Convert DOC/DOCX to PDF"""
    try:
        # Validate file size
        if not validate_file_size(doc_data):
            logger.error("Document file size exceeds limit")
            return None

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=f".{doc_type}", delete=False) as doc_file:
            doc_file.write(doc_data)
            doc_path = doc_file.name

        pdf_path = doc_path.replace(f".{doc_type}", ".pdf")
        convert(doc_path, pdf_path)

        # Read the PDF file
        with open(pdf_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()

        # Clean up temporary files
        os.remove(doc_path)
        os.remove(pdf_path)

        return pdf_data
    except Exception as e:
        logger.error(f"Error converting document to PDF: {e}")
        return None

def convert_pdf_to_image(pdf_data: bytes) -> Optional[bytes]:
    """Convert PDF to image"""
    try:
        # Validate file size
        if not validate_file_size(pdf_data):
            logger.error("PDF file size exceeds limit")
            return None

        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf_file:
            pdf_file.write(pdf_data)
            pdf_path = pdf_file.name

        # Convert PDF to image
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        if images:
            # Save the first page as PNG
            img_path = pdf_path.replace(".pdf", ".png")
            images[0].save(img_path, "PNG")

            # Read the image file
            with open(img_path, "rb") as img_file:
                img_data = img_file.read()

            # Clean up temporary files
            os.remove(pdf_path)
            os.remove(img_path)

            return img_data
    except Exception as e:
        logger.error(f"Error converting PDF to image: {e}")
        return None

def process_document(file_data: bytes, file_type: str) -> Optional[bytes]:
    """Process a document file and return image data"""
    try:
        if file_type in ["doc", "docx"]:
            # Convert DOC/DOCX to PDF
            pdf_data = convert_doc_to_pdf(file_data, file_type)
            if not pdf_data:
                return None
        elif file_type == "pdf":
            pdf_data = file_data
        else:
            return None

        # Convert PDF to image
        img_data = convert_pdf_to_image(pdf_data)
        return img_data

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return None

def analyze_images_base64(images: List[Dict], job_description: str, job_post_id: str) -> str:
    try:
        # Initialize the model - using gemini-1.5-pro for better vision capabilities
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        # Format the prompt with job description and job post ID
        analysis_prompt = DEFAULT_PROMPT.format(job_description=job_description, job_post_id=job_post_id)
        
        # Prepare contents with prompt and base64 images
        contents = [analysis_prompt]
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
        logger.error(f"Error in analyze_images_base64: {e}")
        return f"Error generating content: {e}"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Check for required fields
        if 'files' not in data or 'job_description' not in data or 'job_post_id' not in data:
            return jsonify({"error": "Missing 'files', 'job_description', or 'job_post_id' in request body"}), 400

        files = data['files']
        job_description = data['job_description']
        job_post_id = data['job_post_id']

        if not isinstance(files, list):
            return jsonify({"error": "'files' must be a list"}), 400

        if not isinstance(job_description, str) or not job_description.strip():
            return jsonify({"error": "Job description must be a non-empty string"}), 400

        if not isinstance(job_post_id, str) or not job_post_id.strip():
            return jsonify({"error": "Job post ID must be a non-empty string"}), 400

        # Process all files
        processed_images = []
        for file_data in files:
            if not isinstance(file_data, dict):
                continue

            file_type = file_data.get("type", "").lower()
            base64_data = file_data.get("data")

            if not base64_data:
                continue

            try:
                # Decode base64 data
                binary_data = base64.b64decode(base64_data)

                # Validate file size
                if not validate_file_size(binary_data):
                    logger.error(f"File size exceeds limit for type: {file_type}")
                    continue

                if file_type in SUPPORTED_IMAGE_FORMATS:
                    # Directly use image data
                    processed_images.append({
                        "mime_type": f"image/{file_type}",
                        "data": base64_data
                    })
                elif file_type in SUPPORTED_DOC_FORMATS:
                    # Convert document to image
                    img_data = process_document(binary_data, file_type)
                    if img_data:
                        processed_images.append({
                            "mime_type": "image/png",
                            "data": base64.b64encode(img_data).decode('utf-8')
                        })
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                continue

        if not processed_images:
            return jsonify({"error": "No valid files processed"}), 400

        # Analyze the processed images
        result = analyze_images_base64(processed_images, job_description, job_post_id)
        
        # Return the raw text response
        return jsonify({"analysis": result})

    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 
