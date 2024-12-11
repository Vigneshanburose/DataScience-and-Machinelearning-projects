@app.route('/')
def home():
    """Render the index.html file."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = joblib.load('xgb_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')

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
    except FileNotFoundError:
        return jsonify({'error': 'Required model or scaler files are missing.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
