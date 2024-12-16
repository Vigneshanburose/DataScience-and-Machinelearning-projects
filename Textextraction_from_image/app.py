from flask import Flask, render_template, request, jsonify  
from flask_cors import CORS
import pytesseract
from PIL import Image
import cv2
import numpy as np
import os
import platform

# Initialize Flask app
app = Flask(__name__)
CORS(app)  

if platform.system() == "Windows":
    # Path for local Windows development
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    # Path for Render/Linux deployment
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract-text', methods=['POST'])
def extract_text():
    try:
        # Check if an image is in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        image = Image.open(file.stream)

        # Convert image to OpenCV format
        open_cv_image = np.array(image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding (optional)
        _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

        # Extract text using Tesseract
        text = pytesseract.image_to_string(thresh_image)

        # Return extracted text
        return jsonify({'text': text}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
