import pandas as pd
import argparse
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Standard feature names the model expects
STANDARD_FEATURES = [
    'Contract', 'LatePayments', 'Plan',
    'TotalCharges', 'MonthlyCharges', 'CreditScore', 'Age'
]

# Mapping of standard feature names to possible column name variations in input files
COLUMN_ALIASES = {
    'Contract': ['Contract', 'ContractLength', 'Tenure', 'tenure'],
    'LatePayments': ['LatePayments', 'IsActiveMember', 'MissedPayments', 'credit_card', 'HasCrCard', 'active'],
    'Plan': ['Plan', 'NumOfProducts', 'SubscriptionType', 'products', 'n_products'],
    'TotalCharges': ['TotalCharges', 'Balance', 'balance', 'Total'],
    'MonthlyCharges': ['MonthlyCharges', 'EstimatedSalary', 'monthly_fee', 'billing_amount', 'monthly_payment', 'salary'],
    'CreditScore': ['CreditScore', 'RiskScore', 'creditscore', 'score', 'customer_score', 'cs'],
    'Age': ['Age', 'age', 'Customer_Age', 'customerage', 'years', 'Age (Yrs)', 'client_age']
}

def resolve_csv_path(csv_path):
    """
    Resolve the CSV file path by checking multiple possible locations.
    
    Args:
        csv_path (str): Input path to CSV file
        
    Returns:
        str: Valid path to existing CSV file
        
    Raises:
        FileNotFoundError: If file cannot be found in either location
    """
    if os.path.isfile(csv_path):
        return csv_path
    uploads_path = os.path.join("uploads", csv_path)
    if os.path.isfile(uploads_path):
        print(f" CSV resolved to: {uploads_path}")
        return uploads_path
    raise FileNotFoundError(f"CSV file not found: {csv_path} or {uploads_path}")

def safe_map_columns(df):
    """
    Map various column names to standard feature names used by the model.
    
    Args:
        df (DataFrame): Input dataframe with raw columns
        
    Returns:
        DataFrame: Processed dataframe with standardized columns
    """
    # Clean column names by stripping whitespace and standardizing format
    df.columns = [col.strip().replace(" ", "").lower() for col in df.columns]
    mapped_df = pd.DataFrame()

    # Handle target column (different possible names)
    if 'attrition' in df.columns:
        mapped_df['target'] = df['attrition'].replace({'Yes': 1, 'No': 0})
        print(" Found target column: 'Attrition'")
    elif 'exited' in df.columns:
        mapped_df['target'] = df['exited']
        print(" Found target column: 'Exited'")
    else:
        print(" No valid target column found in the CSV.")

    # Map features using aliases dictionary
    for std_feature, aliases in COLUMN_ALIASES.items():
        # Find first matching alias in the input data
        match = next((alias for alias in aliases if alias.lower() in df.columns), None)
        if match:
            mapped_df[std_feature] = df[match.lower()]
            print(f" Mapped '{std_feature}' from column '{match}'")
        else:
            mapped_df[std_feature] = 0
            print(f" '{std_feature}' not found — defaulted to 0")

    # Special case: Calculate monthly charges from annual salary if available
    if 'estimatedsalary' in df.columns and 'MonthlyCharges' not in mapped_df.columns:
        mapped_df['MonthlyCharges'] = pd.to_numeric(df['estimatedsalary'], errors='coerce') / 12
        print(" Derived MonthlyCharges from EstimatedSalary")

    return mapped_df

def preprocess_numerics(df):
    """
    Clean and convert numeric columns to proper types.
    
    Args:
        df (DataFrame): Input dataframe
        
    Returns:
        DataFrame: Processed dataframe with clean numeric columns
    """
    for col in ['TotalCharges', 'MonthlyCharges', 'CreditScore', 'Age']:
        # Convert to numeric, coerce errors to NaN, then fill NaN with 0
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)
    return df

def encode_categoricals(df):
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df (DataFrame): Input dataframe
        
    Returns:
        tuple: (processed DataFrame, dictionary of label encoders)
    """
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f" Encoded categorical: {col}")
    return df, label_encoders

def train_model(csv_path):
    """
    Main training function that handles the full pipeline:
    1. Data loading
    2. Feature mapping
    3. Preprocessing
    4. Model training
    5. Evaluation
    6. Saving artifacts
    
    Args:
        csv_path (str): Path to training data CSV file
    """
    # Resolve file path and load data
    path = resolve_csv_path(csv_path)
    print("\n===  Starting Model Training ===")
    print(f" Reading CSV from: {path}\n")

    try:
        df = pd.read_csv(path)
        print(" CSV Loaded. Columns:", df.columns.tolist())
    except Exception as e:
        raise Exception(f" CSV read error: {e}")

    # Standardize columns and features
    df = safe_map_columns(df)

    # Ensure all standard features are present (fill missing with 0)
    for col in STANDARD_FEATURES:
        if col not in df.columns:
            df[col] = 0
            print(f" Feature '{col}' not in final DataFrame — defaulted to 0")

    # Clean numeric columns and encode categoricals
    df = preprocess_numerics(df)
    df, label_encoders = encode_categoricals(df)

    # Validate we have target variable
    if 'target' not in df.columns:
        raise Exception(" No target column found — expected 'Attrition' or 'Exited'")

    # Prepare features and target
    X = df[STANDARD_FEATURES]
    y = df['target']

    print("\n Final Features Used:", X.columns.tolist())
    print(f" X shape: {X.shape} | y shape: {y.shape}")

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )

    # Initialize and train model
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )
    model.fit(X_train, y_train)

    # Print evaluation metrics
    print("\n Model Training Complete!")
    print(f" Train Accuracy: {model.score(X_train, y_train):.2f}")
    print(f" Test Accuracy: {model.score(X_test, y_test):.2f}")
    print(classification_report(y_test, model.predict(X_test)))

    # Save model artifacts
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/churn_model.pkl")
    joblib.dump(label_encoders, "model/label_encoders.pkl")

    print("\n Model saved to: model/churn_model.pkl")
    print(" Encoders saved to: model/label_encoders.pkl")
    print(" All done!")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Train churn prediction model using uploaded CSV"
    )
    parser.add_argument(
        "--csv", 
        required=True, 
        help="Path to the CSV file containing customer data with 'Attrition' or 'Exited' column"
    )
    args = parser.parse_args()
    
    # Run training pipeline
    train_model(args.csv)
