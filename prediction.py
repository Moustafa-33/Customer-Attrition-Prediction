import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from flask import request, flash, redirect, url_for, render_template
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask application
app = Flask(__name__) # type: ignore
app.secret_key = 'your_secret_key'  # replace this with a secure key in production

# === 1. Load Model and Encoders ===
# Load pre-trained machine learning model for churn prediction
model = joblib.load("model/churn_model.pkl")
# Load label encoders used during model training for categorical features
label_encoders = joblib.load("model/label_encoders.pkl")

# === 2. Feature Expectations ===
# Standard feature names expected by the model
STANDARD_FEATURES = [
    'Contract', 'LatePayments', 'Plan',
    'TotalCharges', 'MonthlyCharges', 'CreditScore', 'Age'
]

# Mapping of standard feature names to possible column name variations in input files
COLUMN_ALIASES = {
    'Contract': ['ContractLength', 'Tenure', 'Contract'],
    'LatePayments': ['IsActiveMember', 'LatePayments', 'MissedPayments'],
    'Plan': ['NumOfProducts', 'Plan', 'SubscriptionType'],
    'TotalCharges': ['TotalCharges', 'Balance', 'EstimatedSalary'],
    'MonthlyCharges': ['MonthlyCharges', 'Balance', 'EstimatedSalary'],
    'CreditScore': ['CreditScore', 'RiskScore'],
    'Age': ['Age', 'Customer_Age']
}

# === 3. Helper Functions ===
def safe_map_columns(df):
    """Map various column names to standard feature names used by the model."""
    # Clean column names by stripping whitespace
    df.columns = [col.strip() for col in df.columns]
    mapped_df = pd.DataFrame()

    # For each standard feature, find matching column in input data
    for standard, aliases in COLUMN_ALIASES.items():
        match = next((alias for alias in aliases if alias in df.columns), None)
        if match:
            mapped_df[standard] = df[match]  # Use matched column
        else:
            mapped_df[standard] = 0  # Default to 0 if column not found

    # Special case: Calculate monthly charges from balance and tenure if available
    if 'Balance' in df.columns and 'Tenure' in df.columns:
        mapped_df['MonthlyCharges'] = df['Balance'] / df['Tenure'].replace(0, 1)

    return mapped_df

def preprocess_numerics(df):
    """Convert numeric columns to proper type and handle missing values."""
    for col in ['TotalCharges', 'MonthlyCharges', 'CreditScore', 'Age']:
        # Convert to numeric, coerce errors to NaN, then fill NaN with 0
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)
    return df

def encode_categoricals(df):
    """Encode categorical features using pre-trained label encoders."""
    for col in df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            # Convert to string and handle unseen categories
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            # Transform categories to numerical values
            df[col] = le.transform(df[col])
    return df

# === 4. Prediction Route ===
@app.route('/predict-attrition', methods=['POST'])
@login_required # type: ignore
def predict_attrition():
    """Handle file upload, process data, make predictions, and return results."""
    # Check if file was included in request
    if 'file' not in request.files:
        flash("No file uploaded", "error")
        return redirect(url_for('upload'))

    file = request.files['file']
    # Check if file was selected
    if file.filename == '':
        flash("No selected file", "error")
        return redirect(url_for('upload'))

    try:
        # === File Handling ===
        # Save uploaded file securely
        filename = secure_filename(file.filename) # type: ignore
        filepath = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)  # Create uploads directory if needed
        file.save(filepath)

        # === Data Processing ===
        # Load CSV data
        df = pd.read_csv(filepath)
        # Standardize column names and structure
        df = safe_map_columns(df)
        # Clean and convert numeric columns
        df = preprocess_numerics(df)
        # Encode categorical features
        df = encode_categoricals(df)
        # Keep only features the model expects
        df = df[STANDARD_FEATURES]

        # === Make Predictions ===
        # Get binary predictions (0=No churn, 1=Churn)
        predictions = model.predict(df)
        # Get probability scores for churn
        probs = model.predict_proba(df)[:, 1]

        # === Prepare Results ===
        # Add human-readable risk labels
        df['AttritionRisk'] = ['Yes' if p == 1 else 'No' for p in predictions]
        # Convert probabilities to percentages with 1 decimal place
        df['Confidence'] = [round(p * 100, 1) for p in probs]
        # Add customer IDs (1-based index)
        df['CustomerID'] = range(1, len(df)+1)

        # Calculate overall churn rate percentage
        churn_rate = round(df['AttritionRisk'].value_counts(normalize=True).get('Yes', 0) * 100, 2)

        # === Save Outputs ===
        # Save prediction results to CSV
        result_path = os.path.join("static", "results.csv")
        df.to_csv(result_path, index=False)

        # === Create Visualization ===
        # Generate pie chart of churn distribution
        plt.figure(figsize=(4,4))
        plt.pie(df['AttritionRisk'].value_counts(), 
                labels=['Staying', 'Leaving'], 
                autopct='%1.1f%%', 
                colors=['#4CAF50', '#F44336'])  # Green for staying, red for leaving
        chart_path = os.path.join("static", "churn_pie.png")
        plt.savefig(chart_path)
        plt.close()

        # === Render Results ===
        return render_template("upload.html",
            churn_percentage=churn_rate,  # Overall churn rate
            chart_url=url_for('static', filename='churn_pie.png'),  # Pie chart image
            table=df.to_html(classes='table table-hover',  # Results table
                            index=False, 
                            columns=['CustomerID', 'AttritionRisk', 'Confidence']),
            download_url=url_for('static', filename='results.csv'),  # CSV download link
            total_customers=len(df),  # Total customers analyzed
            high_risk_count=sum(predictions),  # Number of high-risk customers
            low_risk_count=len(predictions) - sum(predictions)  # Number of low-risk customers
        )

    except Exception as e:
        # Handle any errors during processing
        flash(f"Prediction failed: {str(e)}", "error")
        return redirect(url_for('upload'))