import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
import json

# Page configuration
st.set_page_config(page_title="Guardian Alert - Fall Detection", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "EDA Analysis", "Feature Analysis", "Model Inference"])

# Load data
@st.cache_data
def load_data():
    features_path = "/home/ubuntu/fall_detection_ds/data/features/features_extracted.csv"
    if os.path.exists(features_path):
        return pd.read_csv(features_path)
    return None

@st.cache_resource
def load_model_assets():
    model_path = "/home/ubuntu/fall_detection_ds/models/fall_detection_model.pkl"
    scaler_path = "/home/ubuntu/fall_detection_ds/models/scaler.pkl"
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None

df = load_data()
model, scaler = load_model_assets()

if page == "Overview":
    st.title("Guardian Alert: Fall Detection System")
    st.markdown("""
    ### Project Overview
    Guardian Alert is an AI-powered fall detection system designed for elderly safety. 
    Using IMU (Inertial Measurement Unit) sensor data, the system can distinguish between 
    Activities of Daily Living (ADL) and actual falls.
    
    #### Dataset: SisFall
    - **Subjects:** 38 (23 adults, 15 elderly)
    - **Activities:** 19 ADL + 15 Fall types
    - **Sensors:** 2 Accelerometers + 1 Gyroscope
    """)
    
    if df is not None:
        st.subheader("Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Windows", len(df))
        col2.metric("ADL Samples", len(df[df['label'] == 'ADL']))
        col3.metric("Fall Samples", len(df[df['label'] == 'Fall']))
        
        st.write("Sample Data (First 5 rows):")
        st.dataframe(df.head())

elif page == "EDA Analysis":
    st.title("Exploratory Data Analysis")
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Label Distribution")
            fig = px.pie(df, names='label', title='Distribution of ADL vs Falls', color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Activity Distribution")
            fig2 = px.histogram(df, x='activity', color='label', title='Samples per Activity')
            st.plotly_chart(fig2)

elif page == "Feature Analysis":
    st.title("Feature Analysis")
    if df is not None:
        st.subheader("Feature Distributions")
        feature_to_plot = st.selectbox("Select Feature", [c for c in df.columns if c not in ['activity', 'subject', 'label']])
        fig = px.box(df, x='label', y=feature_to_plot, color='label', title=f'Distribution of {feature_to_plot}')
        st.plotly_chart(fig)
        
        st.subheader("Top Discriminative Features")
        importance_path = "/home/ubuntu/fall_detection_ds/data/features/feature_importance.csv"
        if os.path.exists(importance_path):
            imp_df = pd.read_csv(importance_path)
            fig_imp = px.bar(imp_df.head(15), x='importance', y='feature', orientation='h', title="Top 15 Features by Importance")
            st.plotly_chart(fig_imp)

elif page == "Model Inference":
    st.title("Model Inference")
    
    if model is not None:
        # Show performance metrics
        summary_path = "/home/ubuntu/fall_detection_ds/data/features/eda_summary.json"
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            st.subheader("Model Performance (Test Set)")
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{summary['accuracy']:.2%}")
            m2.metric("ROC-AUC", f"{summary['roc_auc']:.4f}")

        st.divider()
        st.subheader("Real-time Prediction Simulation")
        st.write("Adjust the top features to see how the model reacts:")
        
        # Get top features for sliders
        importance_path = "/home/ubuntu/fall_detection_ds/data/features/feature_importance.csv"
        imp_df = pd.read_csv(importance_path)
        top_features = imp_df.head(9)['feature'].tolist()
        
        input_data = {}
        cols = st.columns(3)
        for i, feat in enumerate(top_features):
            with cols[i % 3]:
                min_val = float(df[feat].min())
                max_val = float(df[feat].max())
                mean_val = float(df[feat].mean())
                input_data[feat] = st.slider(feat, min_val, max_val, mean_val)
        
        # Fill other features with mean
        full_input = {}
        for feat in [c for c in df.columns if c not in ['activity', 'subject', 'label']]:
            if feat in input_data:
                full_input[feat] = input_data[feat]
            else:
                full_input[feat] = df[feat].mean()
        
        input_df = pd.DataFrame([full_input])
        input_scaled = scaler.transform(input_df)
        
        if st.button("Predict Fall Status"):
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            if prediction == 1:
                st.error(f"⚠️ FALL DETECTED! (Confidence: {probability:.2%})")
            else:
                st.success(f"✅ Normal Activity (Confidence: {1-probability:.2%})")
            
            st.progress(probability)
    else:
        st.warning("Model not found. Please run the training script first.")
