"""
Heart Disease Prediction System
Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Union
import warnings
warnings.filterwarnings('ignore')

class HeartDiseasePredictor:
    """
    A machine learning system for predicting heart disease risk
    based on clinical parameters.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved model file. If None, uses default.
        """
        if model_path is None:
            model_path = '../models/heart_disease_predictor.pkl'
        
        try:
            self.prediction_system = joblib.load('../models/heart_disease_predictor.pkl')
            self.model = self.prediction_system['model']
            self.feature_names = self.prediction_system['feature_names']
            self.encoders = self.prediction_system['encoders']
            print("✅ Heart Disease Predictor loaded successfully!")
        except FileNotFoundError:
            print("❌ Model file not found. Please train the model first.")
            raise
    
    def predict(self, **patient_params) -> Dict:
        """
        Predict heart disease risk for a patient.
        
        Args:
            **patient_params: Patient clinical parameters
            
        Returns:
            Dictionary with prediction results
        """
        # Process input parameters
        processed_data = self._process_input(patient_params)
        
        # Make prediction
        prediction = self.model.predict(processed_data)[0]
        probability = self.model.predict_proba(processed_data)[0, 1]
        
        # Generate comprehensive results
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': self._get_risk_level(probability),
            'confidence': self._get_confidence(probability),
            'feature_contributions': self._get_feature_contributions(processed_data),
            'recommendations': self._generate_recommendations(prediction, probability)
        }
        
        return result
    
    def quick_predict(self, age: int, sex: str, cholesterol: int, 
                    blood_pressure: int, max_heart_rate: int, 
                    st_depression: float = 0) -> Dict:
        """
        Quick prediction using essential parameters.
        
        Args:
            age: Patient age in years
            sex: 'male' or 'female'
            cholesterol: Serum cholesterol in mg/dl
            blood_pressure: Resting blood pressure
            max_heart_rate: Maximum heart rate achieved
            st_depression: ST depression induced by exercise
            
        Returns:
            Simplified prediction results
        """
        patient_data = {
            'age': age,
            'sex': 1 if sex.lower() in ['male', 'm'] else 0,
            'chol': cholesterol,
            'trestbps': blood_pressure,
            'thalch': max_heart_rate,
            'oldpeak': st_depression,
            # Default values for other parameters
            'cp': 0,  # typical angina
            'fbs': 0,
            'restecg': 0,
            'exang': 0,
            'slope': 1,
            'ca': 0,
            'thal': 1,
            'dataset': 0
        }
        
        return self.predict(**patient_data)
    
    def _process_input(self, patient_params: Dict) -> pd.DataFrame:
        """Process input parameters to match model expectations."""
        processed_data = {}
        
        for feature in self.feature_names:
            if feature in patient_params:
                processed_data[feature] = patient_params[feature]
            else:
                processed_data[feature] = self._get_default_value(feature)
        
        return pd.DataFrame([processed_data], columns=self.feature_names)
    
    def _get_default_value(self, feature: str):
        """Get sensible default values for missing features."""
        defaults = {
            'age': 50, 'trestbps': 120, 'chol': 200, 'thalch': 150,
            'oldpeak': 0, 'sex': 1, 'fbs': 0, 'exang': 0, 'cp': 0,
            'restecg': 0, 'slope': 1, 'ca': 0, 'thal': 1, 'dataset': 0
        }
        return defaults.get(feature, 0)
    
    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level."""
        if probability >= 0.7:
            return "High Risk"
        elif probability >= 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    def _get_confidence(self, probability: float) -> str:
        """Calculate prediction confidence."""
        confidence_score = 2 * abs(probability - 0.5)
        if confidence_score > 0.8:
            return "Very High"
        elif confidence_score > 0.6:
            return "High"
        elif confidence_score > 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _get_feature_contributions(self, processed_data: pd.DataFrame, top_n: int = 5) -> List:
        """Get top contributing features."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_imp = list(zip(self.feature_names, importances))
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            return feature_imp[:top_n]
        return []
    
    def _generate_recommendations(self, prediction: int, probability: float) -> List[str]:
        """Generate personalized recommendations."""
        recommendations = []
        
        if prediction == 1 or probability > 0.3:
            recommendations.extend([
                "Consult a cardiologist for comprehensive evaluation",
                "Consider lifestyle modifications: balanced diet and regular exercise",
                "Monitor blood pressure and cholesterol levels regularly",
                "Schedule follow-up appointments every 6 months"
            ])
        else:
            recommendations.extend([
                "Maintain regular health checkups annually",
                "Continue healthy lifestyle habits",
                "Monitor cardiovascular health indicators"
            ])
        
        return recommendations

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Example prediction
    result = predictor.quick_predict(
        age=55,
        sex='male',
        cholesterol=240,
        blood_pressure=140,
        max_heart_rate=130,
        st_depression=1.5
    )
    
    print("🔍 Prediction Results:")
    print(f"   Heart Disease: {'YES' if result['prediction'] else 'NO'}")
    print(f"   Probability: {result['probability']:.3f}")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Confidence: {result['confidence']}")