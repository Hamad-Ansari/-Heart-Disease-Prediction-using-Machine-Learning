from flask import Flask, request, jsonify
from prediction import HeartDiseasePredictor
import os

app = Flask(__name__)
predictor = HeartDiseasePredictor()

@app.route('/')
def home():
    return jsonify({
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "status": "active"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        result = predictor.predict(**data)
        
        return jsonify({
            "success": True,
            "prediction": result
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)