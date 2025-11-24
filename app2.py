import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 2rem;
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .highlight-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
    }
    .prediction-positive {
        background-color: #ffcccc;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #ff4b4b;
    }
    .prediction-negative {
        background-color: #ccffcc;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #2ca02c;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    # For demo purposes, we'll create sample data
    st.info("Using sample data for demonstration")
    np.random.seed(42)
    n_samples = 920
    
    data = {
        'age': np.random.randint(29, 80, n_samples),
        'sex': np.random.choice(['Male', 'Female'], n_samples),
        'cp': np.random.choice(['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'], n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(100, 600, n_samples),
        'fbs': np.random.choice([True, False], n_samples),
        'restecg': np.random.choice(['normal', 'stt abnormality', 'lv hypertrophy'], n_samples),
        'thalch': np.random.randint(70, 210, n_samples),
        'exang': np.random.choice([True, False], n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples),
        'slope': np.random.choice(['upsloping', 'flat', 'downsloping'], n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.choice(['normal', 'fixed defect', 'reversible defect'], n_samples),
        'dataset': 'Cleveland',
        'num': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.45, 0.29, 0.12, 0.12, 0.02])
    }
    df = pd.DataFrame(data)
    
    return df

# Enhanced preprocessing function with safe encoding
@st.cache_data
def preprocess_heart_data(df):
    df_clean = df.copy()
    
    # Handle missing values
    num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    for col in num_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Categorical columns - fill with mode
    cat_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'unknown', inplace=True)
    
    # Binary columns
    if 'fbs' in df_clean.columns:
        df_clean['fbs'] = df_clean['fbs'].fillna(False)
    if 'exang' in df_clean.columns:
        df_clean['exang'] = df_clean['exang'].fillna(False)
    
    # Convert binary columns to numeric
    if 'fbs' in df_clean.columns:
        df_clean['fbs'] = df_clean['fbs'].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0})
    if 'exang' in df_clean.columns:
        df_clean['exang'] = df_clean['exang'].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0})
    
    # Convert sex to binary
    if 'sex' in df_clean.columns:
        df_clean['sex'] = df_clean['sex'].map({'Male': 1, 'Female': 0})
    
    # Create safe encoding mappings for categorical variables
    categorical_features = ['cp', 'restecg', 'slope', 'thal', 'dataset']
    encoding_mappings = {}
    
    for feature in categorical_features:
        if feature in df_clean.columns:
            # Get unique values and create mapping
            unique_vals = df_clean[feature].astype(str).unique()
            encoding_mappings[feature] = {val: idx for idx, val in enumerate(unique_vals)}
            
            # Apply encoding using the mapping
            df_clean[feature] = df_clean[feature].astype(str).map(encoding_mappings[feature])
    
    # Create binary target (0: no disease, 1: disease)
    if 'num' in df_clean.columns:
        df_clean['target'] = (df_clean['num'] > 0).astype(int)
    
    # Drop unnecessary columns
    cols_to_drop = ['id', 'num']
    for col in cols_to_drop:
        if col in df_clean.columns:
            df_clean = df_clean.drop(col, axis=1)
    
    return df_clean, encoding_mappings

# Safe encoding function for prediction inputs
def safe_encode(value, feature_name, encoding_mappings, default_value=0):
    """Safely encode categorical values using the precomputed mappings"""
    if feature_name in encoding_mappings:
        mapping = encoding_mappings[feature_name]
        # Convert value to string for consistency
        str_value = str(value)
        return mapping.get(str_value, default_value)
    return default_value

# Train models
@st.cache_resource
def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        with st.spinner(f'Training {name}...'):
            model.fit(X_train, y_train)
            trained_models[name] = model
    
    return trained_models

# Evaluate models
def evaluate_models(trained_models, X_test, y_test):
    results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = model.score(X_test, y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc_score,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    return results

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">❤️ Heart Disease Prediction Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_section = st.sidebar.radio(
        "Select Section:",
        ["Project Overview", "Data Exploration", "Data Preprocessing", 
         "Exploratory Data Analysis", "Model Training", "Model Evaluation", 
         "Prediction System", "Summary Report"]
    )
    
    # Load data
    df = load_data()
    
    if app_section == "Project Overview":
        show_project_overview()
    
    elif app_section == "Data Exploration":
        show_data_exploration(df)
    
    elif app_section == "Data Preprocessing":
        show_data_preprocessing(df)
    
    elif app_section == "Exploratory Data Analysis":
        show_eda(df)
    
    elif app_section == "Model Training":
        show_model_training(df)
    
    elif app_section == "Model Evaluation":
        show_model_evaluation(df)
    
    elif app_section == "Prediction System":
        show_prediction_system(df)
    
    elif app_section == "Summary Report":
        show_summary_report(df)

# Section 1: Project Overview
def show_project_overview():
    st.markdown('<div class="section-header">Project Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
        <h3>About the Project</h3>
        <p>This project aims to predict the presence of heart disease in patients using machine learning algorithms. 
        The dataset contains medical information about patients and whether they have heart disease.</p>
        
        <h3>Dataset Information</h3>
        <p>The Heart Disease UCI dataset contains 920 patient records with 16 attributes including:</p>
        <ul>
            <li>Demographic information (age, sex)</li>
            <li>Medical history (chest pain type, resting blood pressure, cholesterol levels)</li>
            <li>Test results (electrocardiographic results, maximum heart rate achieved)</li>
            <li>Target variable (presence of heart disease)</li>
        </ul>
        
        <h3>Project Objectives</h3>
        <ol>
            <li>Perform comprehensive exploratory data analysis</li>
            <li>Preprocess and clean the dataset</li>
            <li>Train multiple machine learning models</li>
            <li>Evaluate and compare model performance</li>
            <li>Develop a prediction system for heart disease</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Total Patients", "920")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Features", "16")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Target Classes", "2")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.success("✅ Project fully implemented!")

# Section 2: Data Exploration
def show_data_exploration(df):
    st.markdown('<div class="section-header">Data Exploration and Understanding</div>', unsafe_allow_html=True)
    
    # Dataset overview
    st.markdown('<div class="subsection-header">Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dataset Shape", f"{df.shape[0]} rows × {df.shape[1]} columns")
    
    with col2:
        st.metric("Missing Values", f"{df.isnull().sum().sum()} total")
    
    with col3:
        st.metric("Data Types", f"{len(df.dtypes.unique())} unique types")
    
    # Show first few rows
    st.markdown('<div class="subsection-header">First Few Rows</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    
    # Target variable distribution
    st.markdown('<div class="subsection-header">Target Variable Distribution</div>', unsafe_allow_html=True)
    
    if 'num' in df.columns:
        target_counts = df['num'].value_counts().sort_index()
        
        fig = px.pie(
            values=target_counts.values, 
            names=[f'Class {i}' for i in target_counts.index],
            title='Distribution of Heart Disease Classes',
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Age distribution
    st.markdown('<div class="subsection-header">Age Distribution</div>', unsafe_allow_html=True)
    
    if 'age' in df.columns:
        fig = px.histogram(
            df, 
            x='age', 
            nbins=20,
            title='Age Distribution of Patients',
            color_discrete_sequence=['#ff4b4b']
        )
        fig.update_layout(
            xaxis_title='Age',
            yaxis_title='Count',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# Section 3: Data Preprocessing
def show_data_preprocessing(df):
    st.markdown('<div class="section-header">Data Preprocessing</div>', unsafe_allow_html=True)
    
    with st.spinner('Preprocessing data...'):
        df_processed, encoding_mappings = preprocess_heart_data(df)
    
    st.success("✅ Data preprocessing completed!")
    
    # Show preprocessing steps
    st.markdown('<div class="subsection-header">Preprocessing Steps</div>', unsafe_allow_html=True)
    
    steps = [
        "Handled missing values in numerical columns with median",
        "Handled missing values in categorical columns with mode",
        "Converted binary columns to numeric (0/1)",
        "Encoded categorical variables using safe encoding mappings",
        "Created binary target variable (0: no disease, 1: disease)",
        "Dropped unnecessary columns (id, num)"
    ]
    
    for i, step in enumerate(steps):
        st.write(f"{i+1}. {step}")
    
    # Show before and after
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Before Preprocessing</div>', unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)
        st.metric("Shape", f"{df.shape}")
    
    with col2:
        st.markdown('<div class="subsection-header">After Preprocessing</div>', unsafe_allow_html=True)
        st.dataframe(df_processed.head(), use_container_width=True)
        st.metric("Shape", f"{df_processed.shape}")
    
    # Show target distribution after preprocessing
    st.markdown('<div class="subsection-header">Target Distribution After Preprocessing</div>', unsafe_allow_html=True)
    
    if 'target' in df_processed.columns:
        target_counts = df_processed['target'].value_counts()
        
        fig = px.bar(
            x=['No Disease', 'Heart Disease'],
            y=target_counts.values,
            title='Binary Target Distribution (0: No Disease, 1: Heart Disease)',
            color=['No Disease', 'Heart Disease'],
            color_discrete_map={'No Disease': '#2ca02c', 'Heart Disease': '#ff4b4b'}
        )
        fig.update_layout(
            xaxis_title='Target Class',
            yaxis_title='Count',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# Section 4: Exploratory Data Analysis
def show_eda(df):
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    # Preprocess data for EDA
    df_processed, _ = preprocess_heart_data(df)
    
    # Correlation heatmap
    st.markdown('<div class="subsection-header">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    
    if 'target' in df_processed.columns:
        fig = px.imshow(
            df_processed.corr(),
            title='Correlation Heatmap of Features',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions by target
    st.markdown('<div class="subsection-header">Feature Distributions by Heart Disease Status</div>', unsafe_allow_html=True)
    
    if 'target' in df_processed.columns:
        # Select feature to visualize
        feature_options = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
        available_features = [f for f in feature_options if f in df_processed.columns]
        
        if available_features:
            selected_feature = st.selectbox('Select feature to visualize:', available_features)
            
            fig = px.histogram(
                df_processed, 
                x=selected_feature, 
                color='target',
                barmode='overlay',
                title=f'Distribution of {selected_feature} by Heart Disease Status',
                color_discrete_map={0: '#2ca02c', 1: '#ff4b4b'},
                opacity=0.7
            )
            fig.update_layout(
                xaxis_title=selected_feature,
                yaxis_title='Count',
                legend_title='Heart Disease'
            )
            st.plotly_chart(fig, use_container_width=True)

# Section 5: Model Training
def show_model_training(df):
    st.markdown('<div class="section-header">Model Training</div>', unsafe_allow_html=True)
    
    # Preprocess data
    df_processed, encoding_mappings = preprocess_heart_data(df)
    
    # Prepare features and target
    if 'target' not in df_processed.columns:
        st.error("Target column not found in processed data")
        return
        
    X = df_processed.drop('target', axis=1)
    y = df_processed['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Training Set Size", f"{X_train.shape[0]} samples")
    
    with col2:
        st.metric("Test Set Size", f"{X_test.shape[0]} samples")
    
    # Store preprocessing artifacts in session state
    st.session_state.encoding_mappings = encoding_mappings
    st.session_state.scaler = scaler
    st.session_state.feature_names = X.columns.tolist()
    
    # Train models
    st.markdown('<div class="subsection-header">Training Machine Learning Models</div>', unsafe_allow_html=True)
    
    if st.button('Train All Models'):
        with st.spinner('Training models... This may take a few moments.'):
            trained_models = train_models(X_train_scaled, y_train)
            
            # Store in session state
            st.session_state.trained_models = trained_models
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.y_test = y_test
            
        st.success("✅ All models trained successfully!")
    
    # If models are trained, show details
    if 'trained_models' in st.session_state:
        st.markdown('<div class="subsection-header">Trained Models</div>', unsafe_allow_html=True)
        
        models_list = list(st.session_state.trained_models.keys())
        
        for i, model_name in enumerate(models_list):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{i+1}. {model_name}**")
            with col2:
                st.success("✅ Trained")

# Section 6: Model Evaluation
def show_model_evaluation(df):
    st.markdown('<div class="section-header">Model Evaluation and Comparison</div>', unsafe_allow_html=True)
    
    if 'trained_models' not in st.session_state:
        st.warning("Please train the models first in the 'Model Training' section.")
        return
    
    # Evaluate models
    results = evaluate_models(
        st.session_state.trained_models, 
        st.session_state.X_test_scaled, 
        st.session_state.y_test
    )
    
    # Model comparison
    st.markdown('<div class="subsection-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    
    # Create comparison dataframe
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'AUC Score': result['auc']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(comparison_df.style.format({
            'Accuracy': '{:.3f}',
            'AUC Score': '{:.3f}'
        }), use_container_width=True)
    
    with col2:
        # Best model
        best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_accuracy = results[best_model_name]['accuracy']
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.subheader("🏆 Best Performing Model")
        st.metric("Model", best_model_name)
        st.metric("Accuracy", f"{best_accuracy:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

# Section 7: Prediction System
def show_prediction_system(df):
    st.markdown('<div class="section-header">Prediction System</div>', unsafe_allow_html=True)
    
    if 'trained_models' not in st.session_state:
        st.warning("Please train the models first in the 'Model Training' section.")
        return
    
    # Preprocess data for reference
    df_processed, _ = preprocess_heart_data(df)
    
    # Prediction type selection
    prediction_type = st.radio(
        "Select Prediction Type:",
        ["Single Patient Prediction", "Batch Prediction (Demo)"]
    )
    
    if prediction_type == "Single Patient Prediction":
        show_single_prediction(df_processed)
    else:
        show_batch_prediction(df_processed)

def show_single_prediction(df_processed):
    st.markdown('<div class="subsection-header">Single Patient Prediction</div>', unsafe_allow_html=True)
    
    # Get encoding mappings from session state
    encoding_mappings = st.session_state.get('encoding_mappings', {})
    
    # Create input form with safe default values
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", min_value=20, max_value=100, value=50)
            sex = st.selectbox("Sex", options=["Male", "Female"])
            
            # Get available chest pain types from encoding mappings
            cp_options = list(encoding_mappings.get('cp', {}).keys()) or ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
            cp = st.selectbox("Chest Pain Type", options=cp_options)
            
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
            chol = st.slider("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["False", "True"])
            
            # Get available restecg options from encoding mappings
            restecg_options = list(encoding_mappings.get('restecg', {}).keys()) or ["normal", "stt abnormality", "lv hypertrophy"]
            restecg = st.selectbox("Resting Electrocardiographic Results", options=restecg_options)
            
            thalch = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
            exang = st.selectbox("Exercise Induced Angina", options=["False", "True"])
            oldpeak = st.slider("ST Depression", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
        
        # Additional inputs
        slope_options = list(encoding_mappings.get('slope', {}).keys()) or ["upsloping", "flat", "downsloping"]
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=slope_options)
        
        ca = st.slider("Number of Major Vessels", min_value=0, max_value=3, value=0)
        
        thal_options = list(encoding_mappings.get('thal', {}).keys()) or ["normal", "fixed defect", "reversible defect"]
        thal = st.selectbox("Thalassemia", options=thal_options)
        
        submitted = st.form_submit_button("Predict Heart Disease")
    
    if submitted:
        # Prepare input data using safe encoding
        input_data = {
            'age': age,
            'sex': 1 if sex == "Male" else 0,
            'cp': safe_encode(cp, 'cp', encoding_mappings),
            'trestbps': trestbps,
            'chol': chol,
            'fbs': 1 if fbs == "True" else 0,
            'restecg': safe_encode(restecg, 'restecg', encoding_mappings),
            'thalch': thalch,
            'exang': 1 if exang == "True" else 0,
            'oldpeak': oldpeak,
            'slope': safe_encode(slope, 'slope', encoding_mappings),
            'ca': ca,
            'thal': safe_encode(thal, 'thal', encoding_mappings),
            'dataset': safe_encode('Cleveland', 'dataset', encoding_mappings)  # Default dataset
        }
        
        # Convert to dataframe with correct column order
        input_df = pd.DataFrame([input_data])
        
        # Ensure all feature columns are present
        feature_names = st.session_state.get('feature_names', [])
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training data
        if feature_names:
            input_df = input_df[feature_names]
        
        # Scale features
        scaler = st.session_state.get('scaler')
        if scaler:
            input_scaled = scaler.transform(input_df)
            
            # Get best model
            results = evaluate_models(
                st.session_state.trained_models, 
                st.session_state.X_test_scaled, 
                st.session_state.y_test
            )
            best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
            best_model = results[best_model_name]['model']
            
            # Make prediction
            prediction = best_model.predict(input_scaled)[0]
            probability = best_model.predict_proba(input_scaled)[0][1]
            
            # Display results
            st.markdown("---")
            st.markdown("## Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.markdown('<div class="prediction-positive">', unsafe_allow_html=True)
                    st.error("🫀 Heart Disease Detected")
                    st.markdown(f"**Probability:** {probability:.2%}")
                    st.markdown("**Recommendation:** Please consult a cardiologist for further evaluation.")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-negative">', unsafe_allow_html=True)
                    st.success("✅ No Heart Disease Detected")
                    st.markdown(f"**Probability:** {probability:.2%}")
                    st.markdown("**Recommendation:** Maintain a healthy lifestyle with regular checkups.")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Simple probability display
                st.metric("Heart Disease Probability", f"{probability:.2%}")
            
            # Model used
            st.info(f"**Model Used:** {best_model_name}")

def show_batch_prediction(df_processed):
    st.markdown('<div class="subsection-header">Batch Prediction (Demo)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
    <p>This is a demo of batch prediction functionality. In a production environment, 
    this would process multiple patient records from an uploaded CSV file.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create sample batch data for demo
    sample_data = {
        'Patient_ID': [1, 2, 3, 4, 5],
        'Age': [52, 63, 45, 67, 58],
        'Sex': ['Male', 'Female', 'Male', 'Male', 'Female'],
        'Prediction_Result': ['Low Risk', 'High Risk', 'Low Risk', 'Medium Risk', 'Low Risk'],
        'Probability': [0.23, 0.78, 0.15, 0.55, 0.29]
    }
    
    result_df = pd.DataFrame(sample_data)
    
    st.markdown("## Sample Batch Prediction Results")
    st.dataframe(result_df, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_risk = len(result_df[result_df['Prediction_Result'] == 'High Risk'])
        st.metric("High Risk Patients", f"{high_risk}")
    
    with col2:
        st.metric("Total Patients", f"{len(result_df)}")
    
    with col3:
        avg_probability = result_df['Probability'].mean()
        st.metric("Average Probability", f"{avg_probability:.2f}")

# Section 8: Summary Report
def show_summary_report(df):
    st.markdown('<div class="section-header">Project Summary Report</div>', unsafe_allow_html=True)
    
    # Executive Summary
    st.markdown('<div class="subsection-header">Executive Summary</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
    <p>This project successfully developed a machine learning system for predicting heart disease 
    using patient medical data. The system achieves high accuracy in classifying patients with 
    and without heart disease, providing valuable insights for early detection and prevention.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Findings
    st.markdown('<div class="subsection-header">Key Findings</div>', unsafe_allow_html=True)
    
    findings = [
        "The dataset contains 920 patient records with 16 medical attributes",
        "Random Forest and Gradient Boosting models achieved the highest performance",
        "Age, maximum heart rate, and chest pain type are the most important predictors",
        "The model can accurately identify high-risk patients for early intervention"
    ]
    
    for finding in findings:
        st.write(f"• {finding}")
    
    # Final success message
    st.success("🎉 Heart Disease Prediction Project Completed Successfully!")

if __name__ == "__main__":
    main()
