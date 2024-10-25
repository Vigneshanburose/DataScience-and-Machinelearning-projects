# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model_training import SentimentAnalyzer

app = Flask(__name__)
CORS(app)

# Initialize the analyzer
analyzer = SentimentAnalyzer()

# Load the trained model
print("Loading model...")
analyzer.load_model()
print("Model loaded successfully!")

# Root route
@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
<html>
    <head>
        <title>Amazon Review Sentiment Analysis</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    </head>
    <body class="flex flex-col items-center justify-center min-h-screen bg-gray-100">
        <div class="bg-white shadow-xl rounded-lg p-8 max-w-2xl w-full m-4">
            <h1 class="text-center text-2xl font-bold mb-8 text-gray-800">Sentiment Analysis on Amazon Product Review</h1>
            
            <div class="space-y-6">
                <div class="space-y-2">
                    <label for="inputText" class="block text-gray-700 font-semibold">Enter Product Review</label>
                    <textarea 
                        id="inputText" 
                        class="w-full p-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:ring focus:ring-blue-200 transition"
                        rows="4"
                        placeholder="Type your review here..."></textarea>
                </div>
                
                <div class="flex justify-center">
                    <button 
                        onclick="analyzeSentiment()" 
                        class="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-6 rounded-lg transition transform hover:scale-105">
                        Analyze Sentiment
                    </button>
                </div>
                
                <div id="loading" class="hidden text-center">
                    <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
                    <p class="mt-2 text-gray-600">Analyzing sentiment...</p>
                </div>
                
                <div class="flex flex-col items-center space-y-2">
                    <div id="result" class="hidden bg-gray-500 text-white font-bold py-2 px-4 rounded-lg flex items-center">
                        <i id="resultIcon" class="fas fa-meh mr-2"></i>
                        NEUTRAL
                    </div>
                    <span id="confidence" class="text-gray-600 font-medium"></span>
                </div>
            </div>
        </div>
        <footer class="bg-gray-100 text-center py-4 mt-8">
            <p>Created by Vignesh Anburose | <a href="https://github.com/vigneshanburose" class="text-blue-500 hover:underline" target="_blank">GitHub</a></p>
        </footer>
        <script>
            async function analyzeSentiment() {
                const inputText = document.getElementById('inputText').value;
                const resultDiv = document.getElementById('result');
                const resultIcon = document.getElementById('resultIcon');
                const confidenceSpan = document.getElementById('confidence');
                const loadingDiv = document.getElementById('loading');
                
                if (!inputText.trim()) {
                    alert('Please enter some text to analyze');
                    return;
                }
                
                // Show loading state
                loadingDiv.classList.remove('hidden');
                resultDiv.classList.add('hidden');
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text: inputText })
                    });
                    
                    const data = await response.json();
                    
                    // Update result
                    resultDiv.classList.remove('hidden');
                    loadingDiv.classList.add('hidden');
                    
                    const sentiment = data.sentiment;
                    const confidence = data.confidence;
                    
                    resultDiv.textContent = sentiment;
                    confidenceSpan.textContent = `Confidence: ${confidence}%`;
                    
                    if (sentiment === 'POSITIVE') {
                        resultDiv.className = 'bg-green-500 text-white font-bold py-2 px-4 rounded-lg flex items-center';
                        resultIcon.className = 'fas fa-smile mr-2';
                    } else if (sentiment === 'NEGATIVE') {
                        resultDiv.className = 'bg-red-500 text-white font-bold py-2 px-4 rounded-lg flex items-center';
                        resultIcon.className = 'fas fa-frown mr-2';
                    } else {
                        resultDiv.className = 'bg-gray-500 text-white font-bold py-2 px-4 rounded-lg flex items-center';
                        resultIcon.className = 'fas fa-meh mr-2';
                    }
                } catch (error) {
                    console.error('Error:', error);
                    alert('Error analyzing sentiment. Please try again.');
                    loadingDiv.classList.add('hidden');
                }
            }
        </script>
    </body>
    </html>
    '''

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                'error': 'No text provided',
                'message': 'Please provide text to analyze'
            }), 400
        
        # Predict sentiment
        sentiment, confidence = analyzer.predict_sentiment(text)
        
        return jsonify({
            'sentiment': sentiment.upper(),
            'confidence': round(confidence, 2)
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error analyzing sentiment'
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested URL was not found on the server.'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An internal server error occurred.'
    }), 500

if __name__ == '__main__':
    app.run(debug=True)