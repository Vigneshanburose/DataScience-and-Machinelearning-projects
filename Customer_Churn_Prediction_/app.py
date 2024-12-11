from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os
from flask_cors import CORS

file_path = os.path.join(os.path.dirname(__file__), 'BankChurners.csv')
app = Flask(__name__)
CORS(app)

def preprocess_data(df):
    """Preprocess the data consistently for both training and prediction."""
    columns_to_drop = ['CLIENTNUM', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                      'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    if 'Attrition_Flag' in df.columns:
        df['Attrition_Flag'] = (df['Attrition_Flag'] == 'Attrited Customer').astype(int)

    categorical_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    for col in categorical_columns:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=[col])

    df['Credit_Utilization'] = df['Total_Trans_Amt'] / df['Credit_Limit']
    df['Transaction_Frequency'] = df['Total_Trans_Ct'] / df['Months_on_book']

    return df
# Global model and scaler variables
model = None
scaler = None
feature_names = None

# Function to load the model
def load_model():
    global model, scaler, feature_names
    if model is None:
        model = joblib.load('xgb_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')

load_model()  # Load the model once when the app starts
@app.route('/')
def home():
    """Render the index.html file."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Use the loaded model and scaler
        customer_data = request.get_json()
        input_df = pd.DataFrame([customer_data])

        processed_input = preprocess_data(input_df)
        missing_cols = set(feature_names) - set(processed_input.columns)
        for col in missing_cols:
            processed_input[col] = 0

        processed_input = processed_input[feature_names]
        input_scaled = scaler.transform(processed_input)

        # Ensure churn_prob is a native Python float, not numpy.float32
        churn_prob = float(model.predict_proba(input_scaled)[0][1])

        strategies = []
        if customer_data.get('Total_Trans_Amt', 0) < 5000:
            strategies.append("Encourage more card usage through targeted rewards programs")
        if customer_data.get('Contacts_Count_12_mon', 0) > 3:
            strategies.append("Proactively reach out to address potential concerns")

        return jsonify({
            'churn_probability': churn_prob,
            'prevention_strategies': strategies
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def train_model():
    df = pd.read_csv(file_path)
    df_processed = preprocess_data(df)
    X = df_processed.drop('Attrition_Flag', axis=1)
    y = df_processed['Attrition_Flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

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

    joblib.dump(model, 'xgb_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(X.columns.tolist(), 'feature_names.pkl')

if __name__ == '__main__':
    if not os.path.exists('xgb_model.pkl'):
        train_model()
    app.run(debug=True)
