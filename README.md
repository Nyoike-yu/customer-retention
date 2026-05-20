Predictive Customer Retention Engine (ML API)
- The project encompasses an end-to-end Machine Learning pipeline and RESTful API designed to predict customer churn.
- It bridges the gap between Data Science and Software Engineering by combining raw data wrangling, feature selection, mathematical model training (Random Forest), and deployment via a high performance FastAPI backend.

Technology stack: Python, Pandas, Scikit-Learn, Random Forest Classifier, FastAPI, Uvicorn, Pydantic
Model Serialization: Joblib

Key Learnings & Technical Challenges Overcome
- Building this pipeline from scratch exposed several critical cases that separate theoretical machine learning from production grade engineering. Here are the core concepts mastered during this project:

1. Feature Selection & Defeating "Probability Compression"
Initially, the model was fed the entire dataset (over 20 features) while the API only collected 6. This discrepancy forced the API to automatically fill the missing 14 data points with 0s for every request. As a result, the model suffered from Probability Compression; it saw a profile entirely filled with zeros, got confused, and safely guessed the dataset's average churn rate (~26%) every single time.

The Fix: Conducted a mathematical Feature Importance analysis to isolate the top drivers of churn (TotalCharges, MonthlyCharge, Tenure, InternetService, Contract, PaymentMethod). By stripping away statistical noise (like Gender or Dependents) and training only on high-impact features, the model's decisiveness skyrocketed, accurately flagging flight risks at >70% and loyal customers at <10%.

2. The Blank Space Trap (Data Cleaning)
- Real-world data is messy. In the raw dataset, brand new customers with a tenure of 0 had blank spaces (" ") in their TotalCharges column instead of actual zeros. Pandas interpreted these spaces as text strings. When one-hot encoding was applied, Pandas created over 6,000 unique binary columns for every individual dollar amount, completely crashing the API's matrix alignment.

The Fix: Implemented robust preprocessing to force coercion of string data to numerics (pd.to_numeric(errors='coerce')), gracefully converting hidden blank spaces to NaNs, and securely filling them with 0.

3. The Pandas vs Pydantic Space Constraint
- Pandas dynamically generated feature names with spaces (e.g., InternetService_Fiber optic), but Python variables and Pydantic schema validation require underscores (InternetService_Fiber_optic). This caused the API safety net to reject critical input data right before prediction, resulting in silent failures.

The Fix: Engineered a universal formatting step before model serialization (df.columns = df.columns.str.replace(' ', '_')) ensuring absolute strict 1-to-1 parity between the Pandas training matrix and the Pydantic API payload.

How to Run the Project
1. Install Dependencies
Ensure you have Python installed, then install the required libraries: pip install fastapi uvicorn scikit-learn pandas pydantic joblib

2. Train the Model 
Before starting the server, you must train the machine learning model. This script cleans the CSV data, isolates the core features, trains the Random Forest, and exports the .pkl files to your hard drive.

Bash
python train_model.py
(You will see a Classification Report printed in the terminal showing the Precision, Recall, and F1-Score of the mathematical model)

3. Start the API Server
Boot up the FastAPI microservice to begin listening for JSON prediction requests.

Bash
uvicorn main:app --reload

API Endpoints & Usage
Go to your browser and input this: http://localhost:8000/docs
It accepts a structured JSON payload of customer data and returns the mathematical probability of that customer churning.

Test Payload A: The "Flight Risk" (High Churn Probability)
New customer, month-to-month, expensive fiber optic, paying by electronic check.

JSON
{
  "Tenure": 1,
  "MonthlyCharge": 110.50,
  "TotalCharges": 110.50,
  "InternetService_Fiber_optic": 1,
  "InternetService_No": 0,
  "Contract_One_year": 0,
  "Contract_Two_year": 0,
  "PaymentMethod_Credit_card_automatic": 0,
  "PaymentMethod_Electronic_check": 1,
  "PaymentMethod_Mailed_check": 0
}

Test Payload B: The "Loyal Customer" (Low Churn Probability)
5-year customer, cheap DSL internet, locked into a 2-year contract.

JSON
{
  "Tenure": 60,
  "MonthlyCharge": 35.00,
  "TotalCharges": 2100.00,
  "InternetService_Fiber_optic": 0,
  "InternetService_No": 0,
  "Contract_One_year": 0,
  "Contract_Two_year": 1,
  "PaymentMethod_Credit_card_automatic": 1,
  "PaymentMethod_Electronic_check": 0,
  "PaymentMethod_Mailed_check": 0
}
Example API Response:

JSON
{
  "churn_risk_percentage": 78.4,
  "will_churn_prediction": true,
  "status": "success"
}

Author
Nyoro Fadhili Wanyoike 
Computer Science Student 
@ University of Nairobi

Backend Development | Systems Architecture | Applied Machine Learning
