import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # 🔥 Fix CORS issue

# Load trained pipeline
model = joblib.load("xgboost_salary_pipeline.pkl")

@app.route('/')
def home():
    return "🚀 XGBoost Salary Predictor API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Predict
        prediction = model.predict(df)[0]
        
        return jsonify({
            "prediction": float(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)