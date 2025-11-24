"""
Example usage of Heart Disease Prediction System
"""
from prediction import HeartDiseasePredictor
import joblib


def __init__(self, model_path: str = None):
    if model_path is None:
        model_path = 'models/heart_disease_predictor.pkl'  # updated path
    try:
        self.prediction_system = joblib.load('models/heart_disease_predictor.pkl'  )
        self.model = self.prediction_system['model']
        self.feature_names = self.prediction_system['feature_names']
        self.encoders = self.prediction_system['encoders']
        print("✅ Heart Disease Predictor loaded successfully!")
    except FileNotFoundError:
        print(f"❌ Model file not found at {'models/heart_disease_predictor.pkl'  }. Please train the model first.")
        raise

def main():
    print("🫀 Heart Disease Prediction System - Example Usage")
    print("=" * 50)
    
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Test cases
    test_cases = [
        {
            "name": "High Risk Patient",
            "params": {
                'age': 65, 'sex': 'male', 'chol': 300, 
                'trestbps': 180, 'thalch': 100, 'oldpeak': 4.2
            }
        },
        {
            "name": "Low Risk Patient", 
            "params": {
                'age': 45, 'sex': 'female', 'chol': 180,
                'trestbps': 110, 'thalch': 160, 'oldpeak': 0.5
            }
        }
    ]
    
    for case in test_cases:
        print(f"\n📋 Patient: {case['name']}")
        print("-" * 30)
        
        result = predictor.quick_predict(**case['params'])
        
        print(f"🔍 Prediction: {'❤️ HEART DISEASE DETECTED' if result['prediction'] else '✅ NO HEART DISEASE'}")
        print(f"📊 Probability: {result['probability']:.3f} ({result['probability']*100:.1f}%)")
        print(f"⚠️  Risk Level: {result['risk_level']}")
        print(f"🎯 Confidence: {result['confidence']}")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"   {i}. {rec}")

if __name__ == "__main__":
    main()