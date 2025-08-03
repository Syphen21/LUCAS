from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load the model and scaler
model = joblib.load('lung_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

app = FastAPI(
    title="Lung Cancer Prediction API",
    description="API for predicting lung cancer based on survey data",
    version="1.0.0"
)

# Add CORS middleware to allow requests from the Streamlit app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class PatientData(BaseModel):
    GENDER: str = Field(..., description="Gender of the patient (M/F)")
    AGE: int = Field(..., description="Age of the patient", ge=1, le=100)
    SMOKING: int = Field(..., description="Smoking status (1-2)", ge=1, le=2)
    YELLOW_FINGERS: int = Field(..., description="Yellow fingers (1-2)", ge=1, le=2)
    ANXIETY: int = Field(..., description="Anxiety (1-2)", ge=1, le=2)
    PEER_PRESSURE: int = Field(..., description="Peer pressure (1-2)", ge=1, le=2)
    CHRONIC_DISEASE: int = Field(..., description="Chronic disease (1-2)", ge=1, le=2)
    FATIGUE: int = Field(..., description="Fatigue (1-2)", ge=1, le=2)
    ALLERGY: int = Field(..., description="Allergy (1-2)", ge=1, le=2)
    WHEEZING: int = Field(..., description="Wheezing (1-2)", ge=1, le=2)
    ALCOHOL_CONSUMING: int = Field(..., description="Alcohol consuming (1-2)", ge=1, le=2)
    COUGHING: int = Field(..., description="Coughing (1-2)", ge=1, le=2)
    SHORTNESS_OF_BREATH: int = Field(..., description="Shortness of breath (1-2)", ge=1, le=2)
    SWALLOWING_DIFFICULTY: int = Field(..., description="Swallowing difficulty (1-2)", ge=1, le=2)
    CHEST_PAIN: int = Field(..., description="Chest pain (1-2)", ge=1, le=2)

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Lung Cancer Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_lung_cancer(patient_data: PatientData):
    try:
        # Convert the input data to a format the model expects
        input_data = {
            'GENDER': 1 if patient_data.GENDER == 'F' else 0,  # Convert to numeric
            'AGE': patient_data.AGE,
            'SMOKING': patient_data.SMOKING,
            'YELLOW_FINGERS': patient_data.YELLOW_FINGERS,
            'ANXIETY': patient_data.ANXIETY,
            'PEER_PRESSURE': patient_data.PEER_PRESSURE,
            'CHRONIC DISEASE': patient_data.CHRONIC_DISEASE,
            'FATIGUE': patient_data.FATIGUE,
            'ALLERGY': patient_data.ALLERGY,
            'WHEEZING': patient_data.WHEEZING,
            'ALCOHOL CONSUMING': patient_data.ALCOHOL_CONSUMING,
            'COUGHING': patient_data.COUGHING,
            'SHORTNESS OF BREATH': patient_data.SHORTNESS_OF_BREATH,
            'SWALLOWING DIFFICULTY': patient_data.SWALLOWING_DIFFICULTY,
            'CHEST PAIN': patient_data.CHEST_PAIN
        }
        
        # Convert to DataFrame to ensure consistent column order
        df = pd.DataFrame([input_data])
        
        # Scale the features
        df_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][prediction]
        
        # Determine risk level based on probability
        if probability > 0.8:
            risk_level = "High"
        elif probability > 0.5:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            "prediction": "YES" if prediction == 1 else "NO",
            "probability": float(probability),
            "risk_level": risk_level
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
