from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import plotly.express as px
import json
import os
file_path = os.path.join(os.path.dirname(__file__), 'BankChurners.csv')

app = Flask(__name__)
def preprocess_data(df):
    """Preprocess the data consistently for both training and prediction."""
    # Drop unnecessary columns
    columns_to_drop = ['CLIENTNUM', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                      'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']
    
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Convert target variable
    if 'Attrition_Flag' in df.columns:
        df['Attrition_Flag'] = (df['Attrition_Flag'] == 'Attrited Customer').astype(int)
    
    # Handle categorical variables
    categorical_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    
    # Create dummy variables
    for col in categorical_columns:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=[col])
    
    # Feature engineering
    df['Credit_Utilization'] = df['Total_Trans_Amt'] / df['Credit_Limit']
    df['Transaction_Frequency'] = df['Total_Trans_Ct'] / df['Months_on_book']
    
    return df

def train_model():
    """Load data, preprocess, and train the model."""
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Separate features and target
    X = df_processed.drop('Attrition_Flag', axis=1)
    y = df_processed['Attrition_Flag']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = xgb.XGBClassifier(
        learning_rate=0.01,
        n_estimators=200,
        max_depth=4,
        min_child_weight=6,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X.columns

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get customer data from form
        customer_data = request.get_json()
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([customer_data])
        
        # Preprocess input data
        processed_input = preprocess_data(input_df)
        
        # Load and train model (in production, you'd want to save/load the model instead of training each time)
        model, scaler, feature_names = train_model()
        
        # Ensure input features match training features
        missing_cols = set(feature_names) - set(processed_input.columns)
        for col in missing_cols:
            processed_input[col] = 0
            
        # Reorder columns to match training data
        processed_input = processed_input[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(processed_input)
        
        # Make prediction
        churn_prob = model.predict_proba(input_scaled)[0][1]
        
        # Generate prevention strategies
        strategies = []
        if 'Total_Trans_Amt' in customer_data and customer_data['Total_Trans_Amt'] < 5000:
            strategies.append("Encourage more card usage through targeted rewards programs")
        if 'Contacts_Count_12_mon' in customer_data and customer_data['Contacts_Count_12_mon'] > 3:
            strategies.append("Proactively reach out to address potential concerns")
        
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
