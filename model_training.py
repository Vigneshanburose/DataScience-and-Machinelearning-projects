# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """Preprocess the text data"""
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

    def prepare_data(self, filepath):
        """Load and prepare the dataset"""
        print("Loading and preparing data...")
        
        # Read the dataset
        df = pd.read_csv("C:\\Users\\Vignesh Anburose\\Downloads\\archive (1)\\Amazon_Unlocked_Mobile.csv")
        
        # Convert ratings to sentiment
        df['sentiment'] = df['Rating'].apply(
            lambda x: 'positive' if x > 3 else 'negative' if x < 3 else 'neutral'
        )
        
        # Preprocess the reviews
        print("Preprocessing reviews...")
        df['processed_review'] = df['Reviews'].apply(self.preprocess_text)
        
        # Remove any rows with empty processed reviews
        df = df[df['processed_review'].str.strip().str.len() > 0]
        
        return df

    def train_models(self, df):
        """Train and evaluate multiple models"""
        print("Training models...")
        
        # Prepare features and target
        X = df['processed_review']
        y = df['sentiment']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Naive Bayes': MultinomialNB()
        }
        
        # Train and evaluate each model
        best_accuracy = 0
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Save the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.model = model
                print(f"New best model: {name}")
        
        return X_test, y_test

    def save_model(self):
        """Save the trained model and vectorizer"""
        print("\nSaving model and vectorizer...")
        with open('sentiment_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print("Model and vectorizer saved successfully!")

    def load_model(self):
        """Load the trained model and vectorizer"""
        with open('sentiment_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)

    def predict_sentiment(self, text):
        """Predict sentiment for new text"""
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Vectorize the text
        text_vectorized = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(text_vectorized)[0]
        
        # Get prediction probability
        proba = self.model.predict_proba(text_vectorized)[0]
        confidence = max(proba) * 100
        
        return prediction, confidence

def main():
    # Initialize the analyzer
    analyzer = SentimentAnalyzer()
    
    try:
        # Try to load existing model
        print("Attempting to load existing model...")
        analyzer.load_model()
        print("Existing model loaded successfully!")
    except FileNotFoundError:
        print("No existing model found. Training new model...")
        # Load and prepare data
        df = analyzer.prepare_data("C:\\Users\\Vignesh Anburose\\Downloads\\archive (1)\\Amazon_Unlocked_Mobile.csv")
        
        # Train models
        analyzer.train_models(df)
        
        # Save the best model
        analyzer.save_model()
    
    # Test the model with a sample review
    sample_review = "This product exceeded my expectations. Great quality and fast shipping!"
    sentiment, confidence = analyzer.predict_sentiment(sample_review)
    print(f"\nTest prediction:")
    print(f"Review: {sample_review}")
    print(f"Predicted sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()