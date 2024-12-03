🖼️ Image-to-Text Extraction Tool :

A Flask-powered web application that extracts text from uploaded images using Tesseract OCR. The app leverages image preprocessing techniques to improve recognition accuracy and provides a user-friendly interface for text extraction.

🚀 Features
1. Text Extraction: Upload an image and extract text using Tesseract OCR.
2. Image Preprocessing: Converts images to grayscale and applies thresholding for better OCR accuracy.
3. Interactive Frontend: Easy-to-use interface for uploading images and viewing extracted text.
4. Cross-Origin Support: Flask-CORS enabled for smooth API communication.
5. Cloud Deployment: Ready for deployment on platforms like Render

🛠️ Tech Stack

1.Backend: Flask, Python

2.Frontend: HTML, CSS, JavaScript

3.OCR: Tesseract OCR

4. Image Processing: OpenCV, PIL (Pillow)

5. Deployment: Render


🖥️ How It Works

1. Upload an Image: Select an image from your device.
2. Preprocessing: The image is processed (grayscale and thresholding).
3. OCR: Text is extracted from the processed image using Tesseract.
4. Result: Extracted text is displayed on the web interface.

🛠️ Setup Instructions :
Prerequisites:
1. Python 3.x
2. Tesseract OCR installed on your system (Download Tesseract)

Installation :-
1. Clone the repository  

2. Install dependencies:
pip install -r requirements.txt 

3.Update the pytesseract path in app.py:
pytesseract.pytesseract.tesseract_cmd = r'path_to_tesseract' 

4.Run the app:
python app.py 

5.Open the app in your browser at http://127.0.0.1:5000/
