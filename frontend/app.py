import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Function to create a gauge chart
def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    return fig

# Set page config
st.set_page_config(
    page_title="Lung Cancer Risk Predictor",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
h1, h2, h3 {
    color: #0066cc;
}
.prediction-box {
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
}
.high-risk {
    background-color: #ffcccc;
    border: 2px solid #ff0000;
}
.medium-risk {
    background-color: #fff2cc;
    border: 2px solid #ffcc00;
}
.low-risk {
    background-color: #ccffcc;
    border: 2px solid #00cc00;
}
.stButton>button {
    background-color: #0066cc;
    color: white;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 5px;
    border: none;
}
.stButton>button:hover {
    background-color: #004d99;
}
</style>
""", unsafe_allow_html=True)

# App title and introduction
st.title("Lung Cancer Risk Assessment Tool")
st.markdown("""
This application uses machine learning to predict the risk of lung cancer based on various health factors and lifestyle choices.
Complete the form below for a personalized risk assessment.
""")

# Sidebar for info
with st.sidebar:
    st.header("About")
    st.info("""
    This tool uses an advanced machine learning model trained on survey data to predict lung cancer risk.
    
    **Important**: This is an educational tool and should not replace professional medical advice.
    
    If you have concerns about lung cancer or other health issues, please consult with a healthcare professional.
    """)
    
    st.header("Risk Factors")
    st.markdown("""
    Key risk factors for lung cancer include:
    
    - Smoking
    - Exposure to secondhand smoke
    - Family history
    - Exposure to radon gas
    - Exposure to asbestos and other carcinogens
    - Air pollution
    - Previous radiation therapy
    """)
    
    st.header("Symptoms")
    st.markdown("""
    Common symptoms of lung cancer:
    
    - Persistent cough
    - Chest pain
    - Shortness of breath
    - Wheezing
    - Hoarseness
    - Weight loss
    - Fatigue
    - Recurring infections
    """)

# Create a form for user input
st.header("Patient Information")

# Use columns to organize the form
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.radio("Gender", ["M", "F"])
    age = st.number_input("Age", min_value=1, max_value=100, value=50)
    smoking = st.radio("Smoking", [1, 2], 
                       format_func=lambda x: "Yes" if x == 2 else "No", 
                       horizontal=True)
    yellow_fingers = st.radio("Yellow Fingers", [1, 2], 
                              format_func=lambda x: "Yes" if x == 2 else "No", 
                              horizontal=True)
    anxiety = st.radio("Anxiety", [1, 2], 
                       format_func=lambda x: "Yes" if x == 2 else "No", 
                       horizontal=True)

with col2:
    peer_pressure = st.radio("Peer Pressure", [1, 2], 
                             format_func=lambda x: "Yes" if x == 2 else "No", 
                             horizontal=True)
    chronic_disease = st.radio("Chronic Disease", [1, 2], 
                               format_func=lambda x: "Yes" if x == 2 else "No", 
                               horizontal=True)
    fatigue = st.radio("Fatigue", [1, 2], 
                       format_func=lambda x: "Yes" if x == 2 else "No", 
                       horizontal=True)
    allergy = st.radio("Allergy", [1, 2], 
                       format_func=lambda x: "Yes" if x == 2 else "No", 
                       horizontal=True)
    wheezing = st.radio("Wheezing", [1, 2], 
                        format_func=lambda x: "Yes" if x == 2 else "No", 
                        horizontal=True)

with col3:
    alcohol_consuming = st.radio("Alcohol Consuming", [1, 2], 
                                 format_func=lambda x: "Yes" if x == 2 else "No", 
                                 horizontal=True)
    coughing = st.radio("Coughing", [1, 2], 
                        format_func=lambda x: "Yes" if x == 2 else "No", 
                        horizontal=True)
    shortness_of_breath = st.radio("Shortness of Breath", [1, 2], 
                                  format_func=lambda x: "Yes" if x == 2 else "No", 
                                  horizontal=True)
    swallowing_difficulty = st.radio("Swallowing Difficulty", [1, 2], 
                                    format_func=lambda x: "Yes" if x == 2 else "No", 
                                    horizontal=True)
    chest_pain = st.radio("Chest Pain", [1, 2], 
                         format_func=lambda x: "Yes" if x == 2 else "No", 
                         horizontal=True)

# Add a submit button
if st.button("Predict Lung Cancer Risk"):
    try:
        # Create a dictionary with user inputs
        input_data = {
            "GENDER": gender,
            "AGE": age,
            "SMOKING": smoking,
            "YELLOW_FINGERS": yellow_fingers,
            "ANXIETY": anxiety,
            "PEER_PRESSURE": peer_pressure,
            "CHRONIC_DISEASE": chronic_disease,
            "FATIGUE": fatigue,
            "ALLERGY": allergy,
            "WHEEZING": wheezing,
            "ALCOHOL_CONSUMING": alcohol_consuming,
            "COUGHING": coughing,
            "SHORTNESS_OF_BREATH": shortness_of_breath,
            "SWALLOWING_DIFFICULTY": swallowing_difficulty,
            "CHEST_PAIN": chest_pain
        }
        
        # Make API request to FastAPI backend
        api_url = "http://localhost:8000/predict"  # Adjust if needed
        
        # For development, you can use a mock response
        # Remove this block and uncomment the requests block for real API calls
        """
        response = requests.post(api_url, json=input_data)
        if response.status_code == 200:
            prediction_result = response.json()
        else:
            st.error(f"Error: {response.text}")
            st.stop()
        """
        
        # Mock response for development (remove in production)
        # Simulate different predictions based on smoking and age
        if smoking == 2 and age > 60:
            mock_probability = 0.85
            mock_prediction = "YES"
            mock_risk_level = "High"
        elif smoking == 2 or age > 60:
            mock_probability = 0.65
            mock_prediction = "YES"
            mock_risk_level = "Medium"
        else:
            mock_probability = 0.25
            mock_prediction = "NO"
            mock_risk_level = "Low"
            
        prediction_result = {
            "prediction": mock_prediction,
            "probability": mock_probability,
            "risk_level": mock_risk_level
        }
        
        # Display prediction result
        st.header("Prediction Result")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display appropriate box based on risk level
            if prediction_result["risk_level"] == "High":
                st.markdown(f"""
                <div class='prediction-box high-risk'>
                    <h2>High Risk</h2>
                    <p>The model predicts <strong>{prediction_result["prediction"]}</strong> for lung cancer risk with a probability of <strong>{prediction_result["probability"]:.2%}</strong>.</p>
                    <p>Please consider consulting with a healthcare professional as soon as possible.</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction_result["risk_level"] == "Medium":
                st.markdown(f"""
                <div class='prediction-box medium-risk'>
                    <h2>Medium Risk</h2>
                    <p>The model predicts <strong>{prediction_result["prediction"]}</strong> for lung cancer risk with a probability of <strong>{prediction_result["probability"]:.2%}</strong>.</p>
                    <p>Consider discussing these results with a healthcare provider during your next visit.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='prediction-box low-risk'>
                    <h2>Low Risk</h2>
                    <p>The model predicts <strong>{prediction_result["prediction"]}</strong> for lung cancer risk with a probability of <strong>{prediction_result["probability"]:.2%}</strong>.</p>
                    <p>Continue maintaining a healthy lifestyle.</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Add recommendations based on risk factors
            st.subheader("Recommendations")
            recommendations = []
            
            if smoking == 2:
                recommendations.append("- Consider smoking cessation programs - smoking is a major risk factor for lung cancer.")
            
            if alcohol_consuming == 2:
                recommendations.append("- Reduce alcohol consumption to improve overall health.")
                
            if fatigue == 2 and shortness_of_breath == 2:
                recommendations.append("- The combination of fatigue and shortness of breath could indicate respiratory issues. Consider consultation.")
                
            if coughing == 2 and chest_pain == 2:
                recommendations.append("- Persistent cough with chest pain should be evaluated by a healthcare professional.")
                
            if len(recommendations) > 0:
                for rec in recommendations:
                    st.markdown(rec)
            else:
                st.markdown("- Maintain a healthy lifestyle with regular exercise and a balanced diet.")
                st.markdown("- Consider regular check-ups with your healthcare provider.")
        
        with col2:
            # Display gauge chart
            st.plotly_chart(create_gauge_chart(prediction_result["probability"]), use_container_width=True)
            
            # Add a warning about model limitations
            st.info("""
            **Note:** This prediction is based on a machine learning model and should be used for informational purposes only. 
            It does not replace a professional medical diagnosis.
            """)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add more information at the bottom
st.header("Understanding Your Results")
st.markdown("""
The prediction is based on statistical patterns in data and should be interpreted with care. Here's what to consider:

- A **High Risk** result suggests strong indicators associated with lung cancer and warrants prompt medical attention.
- A **Medium Risk** result indicates potential concerns that should be discussed with a healthcare provider.
- A **Low Risk** result suggests fewer risk factors, but does not guarantee absence of disease.

Remember that early detection is key to better outcomes in cancer treatment.
""")

# Instructions on how to run the application
st.header("How to Use This Application")
st.markdown("""
1. Make sure the FastAPI backend is running on localhost:8000
2. Fill in all the fields in the form with your information
3. Click the "Predict Lung Cancer Risk" button to get your results
4. Review the prediction and recommendations
5. Consult with healthcare professionals for any concerns
""")

# Footer
st.markdown("---")
st.markdown("© 2025 Lung Cancer Risk Assessment Tool | Developed for educational purposes only")
