import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def main():
    print("1. Loading Data...")
    df = pd.read_csv('customer_data.csv') 

    print("2. Cleaning and Engineering Features...")
    # Forces TotalCharges to be numbers and fill blanks with 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # keeps only the top statistical predictors of churn to avoid probability compression
    columns_to_keep = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 
        'InternetService', 'Contract', 'PaymentMethod', 'Churn'
    ]
    df = df[columns_to_keep]
    
    # Rename for standard API conventions
    df.rename(columns={'tenure': 'Tenure', 'MonthlyCharges': 'MonthlyCharge'}, inplace=True)

    # Converts Yes/No churn to 1/0, and one-hot encode the text data
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, drop_first=True)


    # Strips spaces and parentheses so Pydantic doesn't crash
    df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    print("3. Splitting and Scaling Data...")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("4. Training the Random Forest Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    print("5. Evaluating Model Performance...")
    predictions = model.predict(X_test_scaled)
    print(classification_report(y_test, predictions))

    print("6. Exporting Assets...")
    joblib.dump(model, 'churn_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(list(X.columns), 'model_columns.pkl') 
    print("Success! .pkl files saved to disk.")

if __name__ == "__main__":
    main()