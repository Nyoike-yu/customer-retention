from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

try:
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("model_columns.pkl")
except Exception as e:
    raise RuntimeError("Missing .pkl files. Did you run train_model.py first?") from e

app = FastAPI(
    title="Customer Retention ML API", 
    description="Predicts churn probability using optimal feature selection."
)

# Defines exact JSON structure expected from frontend based on our processed DataFrame
class CustomerData(BaseModel):
    Tenure: int
    MonthlyCharge: float
    TotalCharges: float
    InternetService_Fiber_optic: int 
    InternetService_No: int
    Contract_One_year: int
    Contract_Two_year: int
    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

@app.post("/predict_churn")
async def predict_churn(customer: CustomerData):
    try:
        input_df = pd.DataFrame([customer.model_dump()])
        
        # Ensures column alignment
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)
        
        # Scale and Predict
        scaled_features = scaler.transform(input_df)
        prediction_binary = model.predict(scaled_features)[0]
        prediction_probability = model.predict_proba(scaled_features)[0][1]
        
        return {
            "churn_risk_percentage": round(prediction_probability * 100, 2),
            "will_churn_prediction": bool(prediction_binary),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))