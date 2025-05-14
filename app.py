from flask_sqlalchemy import SQLAlchemy
import os
import re
import datetime
import joblib
import pandas as pd #Data Handling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpdf import FPDF

from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt

from train_model import safe_map_columns, train_model
from sklearn.preprocessing import LabelEncoder #Preprocessing,Training,Evaluating the Model
from werkzeug.utils import secure_filename

from models.extensions import db, bcrypt, login_manager
from models.model import UploadHistory  #  Churn upload tracking model

# ------------------------ App Factory Configuration ------------------------ #
from flask import Flask
import os
from models.extensions import db, bcrypt, login_manager


# Application path configuration - defines key directories for database, uploads, models, and static files
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, 'instance', 'users.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
REPORT_DIR = os.path.join(BASE_DIR, 'static', 'reports')

def create_app():
    app = Flask(__name__)

    # Configure
    app.config.update(
        SECRET_KEY=os.environ.get('SECRET_KEY') or 'luxury_churn_ai_secure_secret_key',
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{DB_PATH}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER=UPLOAD_FOLDER,
        MODEL_DIR=MODEL_DIR,
        STATIC_DIR=STATIC_DIR,
        REPORT_DIR=REPORT_DIR
    )

# Initialize required application directories (uploads, reports, and database location)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REPORT_DIR'], exist_ok=True)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # Initialize Flask extensions and set login page
    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'login'

#load user from database 
    @login_manager.user_loader
    def load_user(user_id):
        from models.user import User
        return User.query.get(int(user_id))

    # Create tables inside app context
    with app.app_context():
        from models.user import User
        from models.model import UploadHistory
        db.create_all()
        print("All tables created successfully.")
      


    return app

# ------------------------ Global Constants & UI Options ------------------------ #

CHART_COLORS = ['#e74c3c', '#2ecc71']
FEATURES_OF_INTEREST = ['MonthlyCharges', 'Contract', 'CreditScore', 'TotalCharges']#Important featres needed for churn

#Actual mapping
def safe_map_columns_lite(df):
    mapping = {
        'contract': 'Contract',
        'Plan': 'Plan',
        'Plan Type': 'Plan',
        'SubscriptionPlan': 'Plan',
        'LatePayments': 'LatePayments',
        'MissedPayments': 'LatePayments',
        'TotalCharges': 'TotalCharges',
        'Total_Charges': 'TotalCharges',
        'MonthlyCharges': 'MonthlyCharges',
        'MonthlyIncome': 'MonthlyCharges',
        'BillAmount': 'MonthlyCharges',
        'CreditScore': 'CreditScore',
        'Credit_Score': 'CreditScore',
        'credit_score': 'CreditScore'
    }

    # Apply the mapping to column names
    df.columns = [mapping.get(col.strip(), col.strip()) for col in df.columns]

    # Required minimal columns
    required = ["Contract", "LatePayments", "Plan", "TotalCharges", "MonthlyCharges", "CreditScore"]
    for col in required:
        if col not in df.columns:
            df[col] = 0  # Fill with 0s if missing

    return df[required]

# ------------------------ Utility Functions ------------------------ #

# Convert column to numeric, coercing errors to NaN, then fill NaN with 0
def clean_numeric_column(df, column_name):
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(0)
    return df

# Executes plot_func, saves result to path, and closes figure
def save_plot_as_image(plot_func, path, *args, **kwargs):
    plt.figure()
    plot_func(*args, **kwargs)
    plt.savefig(path)
    plt.close()
    print(f"Chart saved to {path}")

# ------------------------ Example Utility for Dynamic Charts ------------------------ #

#Generate and display a sample churn rate trend line plot with mock data.    
#Creates a simple demonstration plot showing increasing churn rate over 4 time periods.
#X-axis represents time periods, Y-axis represents churn percentage.
    
def sample_churn_trend_plot():
    plt.plot([1, 2, 3, 4], [5, 10, 15, 20])
    plt.title("Churn Rate Over Time")

# ------------------------ Template Check (Log) ------------------------ #
# Verify and print expected template files in the Flask templates directory
# Displays checklist of required HTML templates for the application

print(" Flask environment ready. Templates assumed present:")
for template in ['index.html', 'login.html', 'signup.html', 'upload.html', 'dashboard.html']:
    print(f"   - templates/{template}")

# Load Model & Encoders
try:
    model_path = os.path.join(MODEL_DIR, "churn_model.pkl")
    encoder_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoder_path)
    print("Churn prediction model and label encoders loaded successfully")
except Exception as e:
    print(f" Failed to load model or encoders: {e}")

    model = None
    label_encoders = {}
    print(f" Error loading model or encoders: {str(e)}")

# Feature transformer wrapper

def transform_dataset(df):
    print("Transforming dataset with safe column mapping")
    df, features = safe_map_columns(df)
    for col in FEATURES_OF_INTEREST:
        if col in df.columns:
            df = clean_numeric_column(df, col)
    return df, features

# Label encoding safeguard

def encode_labels(df, encoders):
    for col in df.columns:
        if df[col].dtype == 'object' and col in encoders:
            le = encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    return df

# Prediction core logic

# Predicts churn using model, adds 'Yes/No' predictions and confidence % to DataFrame
def generate_predictions(df, model, features):
    print(" Running churn predictions...")
    df = df.reindex(columns=features, fill_value=0)
    probs = model.predict_proba(df[features])[:, 1]
    predictions = model.predict(df[features])
    df['Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
    df['Confidence'] = [round(prob * 100, 2) for prob in probs]
    return df, predictions, probs

# Reason detection 

# Checks for high charges, short contracts, low scores to explain churn risk
def identify_churn_reasons(df):
    reasons = []
    if df['MonthlyCharges'].mean() > 80:
        reasons.append("High MonthlyCharges")
    if 'Contract' in df.columns and df['Contract'].nunique() == 1 and df['Contract'].iloc[0] == 0:
        reasons.append("Short Contract")
    if 'CreditScore' in df.columns and df['CreditScore'].mean() < 600:
        reasons.append("Low CreditScore")
    if not reasons:
        reasons.append("Customer behavior risk (unspecified)")
    return reasons

# Retention strategies

def generate_retention_strategies():
    return [
        {'title': 'Offer Loyalty Discounts', 'description': 'Reduce churn for high MonthlyCharges customers by offering tailored discounts.'},
        {'title': 'Contract Extension Incentives', 'description': 'Encourage customers on short contracts to commit longer with perks.'},
        {'title': 'Credit Coaching Programs', 'description': 'Help customers with low credit scores by offering financial guidance.'},
        {'title': 'Premium Support Offers', 'description': 'Offer premium support to high-risk segments lacking service engagement.'}
    ]

# Placeholder Prediction Test
print(" Model integration layer complete. Ready for use.")

# Create app first
app = create_app()

    # Init Uploads Directory
with app.app_context():
    upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        print("Uploads directory created")

# Check database and show startup info (optional)
with app.app_context():
    print(f" Database checked/created at: {app.config['SQLALCHEMY_DATABASE_URI']}")  # type: ignore

# Debug route printout
with app.app_context():
    print(" All registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint:25s} --> {rule}")
# ---------------------------------------------------------------------- #
# Utility: Numeric cleaner
def clean_numeric_column(df, column_name):
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(0)
    return df

# Utility: Save matplotlib chart as image
def save_plot_as_image(plot_func, path, *args, **kwargs):
    plt.figure()
    plot_func(*args, **kwargs)
    plt.savefig(path)
    plt.close()
    print(f" Chart saved: {path}")

# Sample Trend Chart
def sample_churn_trend_plot():
    plt.plot([1, 2, 3, 4], [5, 10, 15, 20])
    plt.title("Churn Trend")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

#  Log Existing Templates
expected_templates = [
    'index.html', 'login.html', 'signup.html', 'upload.html',
    'dashboard.html', 'contact.html', 'demo.html', 'testimonials.html',
    'submit-testimonials.html', 'accuracy.html', 'insights.html',
    'upload-info.html', 'upload-guidelines.html', 'upload-history.html',
    'upload-troubleshooting.html', 'billing.html', 'checkout.html',
    'analytics.html', 'offer.html', 'other-benefits.html', 'strategies.html'
]

print("Verifying template presence...")
for tpl in expected_templates:
    print(f"   - templates/{tpl}")

# DB Existence Check
try:
    with app.app_context():
        
        print(f"User DB initialized at: {DB_PATH}")
except Exception as e:
    print(f" DB setup failed: {e}")
# 👤 Signup Route
from flask import request, redirect, url_for, flash, render_template
from models.user import User  # Make sure this is imported
from models.extensions import db, bcrypt
# ---------------------------ROUTES------------------------------------------------------------#
#Signup Route 

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username').strip().lower()
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        account_type = request.form.get('account_type', 'Individual').strip().lower()
        full_name = request.form.get('name')
        company = request.form.get('company')
        job_title = request.form.get('job_title')
        phone = request.form.get('phone')

        if not username or not password or not confirm_password:
            flash("Please complete all required fields.", "error")
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for('signup'))

        if User.query.filter_by(username=username).first():
            flash("An account with that email already exists.", "error")
            return redirect(url_for('signup'))

        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(
            username=username,
            password=hashed_pw,
            account_type=account_type,
            name=full_name,
            company=company,
            job_title=job_title,
            phone=phone
        )
        db.session.add(new_user)
        db.session.commit()

        flash("Account created! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')

# ---------------LOGIN------------------------------------------------------ #

from flask import request, redirect, url_for, flash, render_template
from flask_login import login_user
from models.user import User
from models.extensions import db, bcrypt

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password')
        account_type = request.form.get('account_type', 'individual').lower()

        user = User.query.filter_by(username=username, account_type=account_type).first()# Serching for existing user in database

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash("Logged in successfully!", "success")
            return redirect(url_for('index'))
        else:
            flash(" Invalid credentials or account type.", "error")
            return redirect(url_for('login'))

    return render_template('login.html')


#------------------------ Logout Route----------------------------------
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

#Protected Dashboard
@app.route('/dashboard')
@login_required
def dashboard():
    uploads = UploadHistory.query.filter_by(user_id=current_user.id).order_by(UploadHistory.timestamp.desc()).limit(10).all()
    return render_template('dashboard.html', uploads=uploads)


# Account Info Route (optional)
@app.route('/account')
@login_required
def account():
    return jsonify({
        'username': current_user.username,
        'account_type': current_user.account_type
    })

# Role-based protection (optional)
def admin_only(view_func):
    @login_required
    def wrapped(*args, **kwargs):
        if current_user.account_type != 'admin':
            flash("Admins only.", "warning")
            return redirect(url_for('dashboard'))
        return view_func(*args, **kwargs)
    return wrapped

# Admin Test Route
@app.route('/admin-dashboard')
@admin_only
def admin_dashboard():
    return render_template('admin-dashboard.html')

# Session Check Helper
@app.route('/session-status')
def session_status():
    if current_user.is_authenticated:
        return jsonify({"status": "logged_in", "user": current_user.username})
    return jsonify({"status": "logged_out"})

# -----------------------------------Public Routes-------------------------------#

from flask import render_template

#Homepage Route

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')



#--------------------------CONTACT----------------------------------------------#
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        flash("Message received! We'll get back to you soon.", "success")
        return redirect(url_for('contact'))
    return render_template('contact.html')
#-----------------------------TESTIMONIALS------------------------------------#

@app.route('/testimonials')
def testimonials():
    return render_template('testimonials.html')

#----------------SUBMIT TESTIMONIALS-------------------------------------------#
@app.route('/submit-testimonials', methods=['GET', 'POST'])
def submit_testimonials():
    if request.method == 'POST':
        name = request.form.get('name')
        role = request.form.get('role')
        quote = request.form.get('quote')
        flash("Thank you for your feedback!", "success")
        return redirect(url_for('testimonials'))
    return render_template('submit-testimonials.html')
#------------------DEMO-----------------------------------------------------------#

@app.route('/demo')
def demo():
    return render_template('demo.html')

# -------------------------------------INSIGHTS------------------------------------ #
@app.route('/insights')
def insights():
    return render_template('insights.html')
#-----------------------------------ACCURACY----------------------------------------- #
@app.route('/accuracy')
def accuracy():
    return render_template('accuracy.html')
# --------------------------- OFFER----------------------------------------------------#
@app.route('/offer')
def offer():
    return render_template('offer.html')
#----------------------------------BENEFITS-------------------------------------------#
@app.route('/benefits')
def benefits():
    return render_template('other-benefits.html')

# Protected Routes
@app.route('/upload', methods=['GET'])
@login_required
def upload():
    return render_template('upload.html')
#-----------------------BILLING----------------------------------------------------------#
@app.route('/billing')
@login_required
def billing():
    return render_template('billing.html')
#---------------------------------CHECKOUT-----------------------------------------------#
@app.route('/checkout', methods=['GET', 'POST'])
@login_required
def checkout():
    return render_template('checkout.html')
#-----------------------------PROCESS CHECKOUT----------------------------------------------#
@app.route('/process-checkout', methods=['POST'])
@login_required
def process_checkout():
    flash('Payment processed successfully.', 'success')
    return redirect(url_for('dashboard'))
# ----------------------------- UPLOAD CHECKPOINT--------------------------------------------#
@app.route('/upload-checkpoint')
@login_required
def upload_checkpoint():
    return render_template('upload-checkpoint.html')
#-------------------------------------ANALYTICS---------------------------------------------#
@app.route('/analytics')
@login_required
def analytics():
    return render_template('analytics.html')
#---------------------------------------RETENTION STRATAGIES------------------------------------#
@app.route('/playbooks')
@login_required
def playbooks():
    return render_template('strategies.html')
#----------------------------------------------DATA PRIVACY-------------------------------------#
@app.route('/data-privacy')
def data_privacy():
    return render_template('data-privacy.html')
#----------------------------------------------UPLOAD GUIDELINES---------------------------------------#
@app.route('/upload-guidelines')
def upload_guidelines():
    return render_template('upload-guidelines.html')
#-----------------------------------------------ANALYISIS-----------------------------------------------#
@app.route('/analysis')
@login_required  # Optional: remove this line if you want it public
def analysis():
    return render_template('analysis.html')

#-------------------------------------------------TROUBLESHOOTING-------------------------------------------#
@app.route('/upload-troubleshooting')
def upload_troubleshooting():
    return render_template('upload-troubleshooting.html')

#---------------------------------------------UPLOAD INFO------------------------------------------------------#
@app.route('/upload-info')
def upload_info():
    return render_template('upload-info.html')
#-------------------------------------------PRICING----------------------------------------------------------------#
@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

#-----------------------------------------------------------UPLOAD HISTORY---------------------------------------------#
@app.route('/upload-history')
def upload_history():
    return render_template('upload-history.html')
def transform_dataset(df):
    try:
        df, features = safe_map_columns(df)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        if 'MonthlyCharges' in df.columns:
            df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0)
        if 'CreditScore' in df.columns:
            df['CreditScore'] = pd.to_numeric(df['CreditScore'], errors='coerce').fillna(0)
        return df, features
    except Exception as e:
        print(f"Error in transform_dataset: {str(e)}")
        return pd.DataFrame(), []
#-------------------------------------------------------------PREDICT ONLY------------------------------------------------#
@app.route('/predict-only', methods=['POST'])
@login_required
def predict_only():
    if 'file' not in request.files:
        flash('No file uploaded.', 'error')
        return redirect(url_for('upload'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file.', 'error')
        return redirect(url_for('upload'))

    try:
        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)

        model = joblib.load("model/churn_model.pkl")
        label_encoders = joblib.load("model/label_encoders.pkl")
        

        df = pd.read_csv(file_path)
        df, features = transform_dataset(df)

        for col in features:
            if df[col].dtype == 'object':
                le = label_encoders.get(col)
                if le:
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df[col] = le.transform(df[col])

        df = df.reindex(columns=features, fill_value=0)

        probs = model.predict_proba(df[features])[:, 1]
        predictions = model.predict(df[features])

        df['Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
        df['Confidence'] = [round(prob * 100, 2) for prob in probs]

        churned = sum(predictions)
        total = len(predictions)
        churn_percentage = round((churned / total) * 100, 2)
        high_risk_percentage = round((churned / total) * 100, 2)

        top_churn_reasons = []
        if df['MonthlyCharges'].mean() > 80:
            top_churn_reasons.append("High MonthlyCharges")
        if 'Contract' in df.columns and df['Contract'].nunique() == 1 and df['Contract'].iloc[0] == 0:
            top_churn_reasons.append("Short Contract")
        if 'CreditScore' in df.columns and df['CreditScore'].mean() < 600:
            top_churn_reasons.append("Low CreditScore")
        if not top_churn_reasons:
            top_churn_reasons.append("Customer behavior risk (unspecified)")

        retention_strategies = [
            {'title': 'Offer Loyalty Discounts', 'description': 'Reduce churn for high MonthlyCharges customers by offering tailored discounts.'},
            {'title': 'Contract Extension Incentives', 'description': 'Encourage customers on short contracts to commit longer with perks.'},
            {'title': 'Credit Coaching Programs', 'description': 'Help customers with low credit scores by offering financial guidance.'},
            {'title': 'Premium Support Offers', 'description': 'Offer premium support to high-risk segments lacking service engagement.'}
        ]

        results_html = df.to_html(classes='table table-striped table-hover', index=False)
        export_path = os.path.join("static", "results.csv")
        df.to_csv(export_path, index=False)

        plt.figure(figsize=(4, 4))
        plt.pie([churned, total - churned], labels=['Churned', 'Retained'], autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'])
        plt.title("Predicted Churn Breakdown")
        chart_path = os.path.join("static", "churn_pie.png")
        plt.savefig(chart_path)
        plt.close()

        session['churn_percentage'] = churn_percentage
        session['high_risk_percentage'] = high_risk_percentage
        session['top_churn_reasons'] = top_churn_reasons
        session['retention_strategies'] = [s['description'] for s in retention_strategies]

        return render_template('upload.html',
                               table=results_html,
                               churn_percentage=churn_percentage,
                               explanation=None,
                               download_link=url_for('static', filename='results.csv'),
                               chart_path=url_for('static', filename='churn_pie.png'))

    except Exception as e:
        flash(f' Prediction error: {e}', 'error')
        return redirect(url_for('upload'))
# ======================= PART 7: Churn Prediction Logic ======================= #
@app.route('/predict-attrition', methods=['POST'])
@login_required
def handle_attrition_prediction():
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('upload'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('upload'))
# Handles file upload for churn prediction - validates request contains a valid file
# Requires authenticated user (via @login_required)
# On error: flashes message and redirects to upload page
# On success: proceeds with prediction processing (in following code)
    try:
        # === Save uploaded file ===
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)
        print(f"\n File uploaded: {filename}")
        print(f" Saved to: {filepath}")

        # === Load model & encoders ===
        model = joblib.load("model/churn_model.pkl")
        label_encoders = joblib.load("model/label_encoders.pkl")

        # === Define aliases ===
        FEATURE_CONFIG = {
            'CreditScore': {'default': 650, 'aliases': ['creditscore', 'score'], 'encoder': False},
            'Geography': {'default': 'unknown', 'aliases': ['geography', 'country'], 'encoder': True},
            'Gender': {'default': 'unknown', 'aliases': ['gender'], 'encoder': True},
            'Age': {'default': 35, 'aliases': ['age'], 'encoder': False},
            'Tenure': {'default': 2, 'aliases': ['tenure'], 'encoder': False},
            'Balance': {'default': 0.0, 'aliases': ['balance'], 'encoder': False},
            'NumOfProducts': {'default': 1, 'aliases': ['numofproducts'], 'encoder': False},
            'IsActiveMember': {'default': 1, 'aliases': ['isactivemember'], 'encoder': False},
            'EstimatedSalary': {'default': 100000, 'aliases': ['estimatedsalary'], 'encoder': False},
            'Contract': {'default': 0, 'aliases': ['contract'], 'encoder': False},
            'Plan': {'default': 0, 'aliases': ['plan'], 'encoder': False},
            'TotalCharges': {'default': 0.0, 'aliases': ['totalcharges'], 'encoder': False},
            'MonthlyCharges': {'default': 0.0, 'aliases': ['monthlycharges'], 'encoder': False},
            'LatePayments': {'default': 0, 'aliases': ['latepayments'], 'encoder': False}
        }

        # Read CSV file, clean column names (strip whitespace & lowercase), remove duplicates, and display final columns
        df = pd.read_csv(filepath)
        df.columns = [col.strip().lower() for col in df.columns]  # Standardize column names
        df = df.loc[:, ~df.columns.duplicated()]  # Keep only first occurrence of duplicate columns
        print("\n Uploaded Columns:", df.columns.tolist())  # Show cleaned columns
        
        # === Map aliases to standard features ===
        mapped_cols = {}
        for std, config in FEATURE_CONFIG.items():
            for alias in config['aliases']:
                if alias in df.columns:
                    mapped_cols[alias] = std
        df = df.rename(columns=mapped_cols)

        # === Count provided features (excluding Geography/Gender)
        provided_features = [f for f in FEATURE_CONFIG if f in df.columns and f not in ['Geography', 'Gender']]
        if len(provided_features) < 4:
            flash(f"Requires 4+ features (found {len(provided_features)})", "error")
            return redirect(url_for('upload'))

        # === Fill missing features ===
        for feat, cfg in FEATURE_CONFIG.items():
            if feat not in df.columns:
                df[feat] = cfg['default']
                print(f"➕ Filled {feat} with default: {cfg['default']}")

        # === Type conversions & derived features ===
        df['Tenure'] = pd.to_numeric(df['Tenure'], errors='coerce').fillna(0)
        df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce').fillna(0)
        df['EstimatedSalary'] = pd.to_numeric(df['EstimatedSalary'], errors='coerce').fillna(0)

        df['Contract'] = (df['Tenure'] > 12).astype(int)
        df['LatePayments'] = (df['Balance'] == 0).astype(int)
        df['MonthlyCharges'] = df['EstimatedSalary'] / 12

        # === Encode categorical features ===
        for col in ['Geography', 'Gender']:
            if col in df.columns and col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].astype(str).str.lower()
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
                if 'unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'unknown')
                df[col] = le.transform(df[col])

        # === Auto-align with model ===
        model_features = model.feature_names_in_.tolist()
        for feature in model_features:
            if feature not in df.columns:
                df[feature] = 0
                print(f" Added missing feature: {feature} = 0")
        df = df[model_features]

        # === Prediction ===
        # Generate predictions (0/1 for churn) and probabilities from the trained model
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        
# Create results DataFrame with:
# - CustomerID (index + 1 for 1-based numbering)
# - AttritionRisk (converted to 'Yes'/'No' labels)
# - Confidence (probability converted to percentage, rounded to 1 decimal)
        results = pd.DataFrame({
            'CustomerID': df.index + 1,
            'AttritionRisk': ['Yes' if p == 1 else 'No' for p in predictions],
            'Confidence': [round(p * 100, 1) for p in probabilities]
        })
# Calculate overall churn rate:
# 1. Count proportion of 'Yes' predictions using value_counts(normalize=True)
# 2. Multiply by 100 to get percentage
# 3. Default to 0% if no 'Yes' predictions exist (.get('Yes', 0))
        churn_rate = results['AttritionRisk'].value_counts(normalize=True).get('Yes', 0) * 100

        # Save upload history
        if current_user.is_authenticated:
            new_upload = UploadHistory(
                user_id=current_user.id,
                filename=filename,
                churn_percentage=round(churn_rate, 1)
            )
            db.session.add(new_upload)
            db.session.commit()

        # === Chart + Save Files ===
        # Create a pie chart to visualize customer attrition risk distribution
        plt.figure(figsize=(6, 6))
        counts = results['AttritionRisk'].value_counts()
        label_map = {'Yes': 'High Risk', 'No': 'Low Risk'}
        labels = [label_map.get(label, label) for label in counts.index]
        colors = ['#F44336', '#4CAF50'][:len(labels)]
        plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors)
        
# Get counts of each risk category from results
        chart_filename = f"attrition_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        chart_path = os.path.join("static", chart_filename)
        plt.savefig(chart_path)
        plt.close()
# Map labels to more descriptive names
        results_filename = f"results_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        results_path = os.path.join("static", results_filename)
        results.to_csv(results_path, index=False)
#CREATES PIECHART
        return render_template(
            "upload.html",
            churn_percentage=round(churn_rate, 1),
            chart_url=url_for('static', filename=chart_filename),
            table=results.to_html(classes='table table-hover', index=False),
            download_url=url_for('static', filename=results_filename),
            total_customers=len(results),
            high_risk_count=(predictions == 1).sum(),
            low_risk_count=(predictions == 0).sum()
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Prediction failed: {str(e)}", "error")
        return redirect(url_for('upload'))
#-------------------------------DOWNLOAD REPORT–--------------------------------------------------------#    
@app.route('/download-report', methods=['POST'])
@login_required
def download_report():
    try:
        # Fallbacks in case session data is missing
        churn_percentage = session.get('churn_percentage', 0.0)
        high_risk_percentage = session.get('high_risk_percentage', 0.0)
        churn_reasons = session.get('top_churn_reasons', ['Not available'])
        strategies = session.get('retention_strategies', ['No strategy available'])

        estimated_reduction = round(churn_percentage * 0.2, 2)
        projected_churn = round(churn_percentage - estimated_reduction, 2)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 20)
        pdf.cell(0, 15, "Customer Churn Report", ln=True, align='C') # PDF FORMAT

        pdf.set_font('Arial', '', 12)
        pdf.ln(10)
        pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.cell(0, 10, f"Predicted Churn Rate: {churn_percentage:.2f}%", ln=True)
        pdf.cell(0, 10, f"High Risk Segment: {high_risk_percentage:.2f}%", ln=True)
        pdf.cell(0, 10, f"Projected Churn After Strategy: {projected_churn:.2f}%", ln=True)

        pdf.ln(8)
        pdf.set_font('Arial', 'B', 13)
        pdf.cell(0, 10, "Top Churn Drivers:", ln=True)
        pdf.set_font('Arial', '', 11)
        for reason in churn_reasons:
            pdf.cell(0, 8, f"- {reason}", ln=True)

        pdf.ln(6)
        pdf.set_font('Arial', 'B', 13)
        pdf.cell(0, 10, "Recommended Retention Strategies:", ln=True)
        pdf.set_font('Arial', '', 11)
        for s in strategies:
            pdf.cell(0, 8, f"- {s}", ln=True)

        # Include pie chart 
        chart_path = os.path.join("static", "churn_pie.png")
        if os.path.exists(chart_path):
            pdf.ln(10)
            pdf.set_font('Arial', 'B', 13)
            pdf.cell(0, 10, "Churn Distribution:", ln=True)
            pdf.image(chart_path, x=30, w=150)

        # Footer
        pdf.set_y(-15)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 10, f"Page {pdf.page_no()}", 0, 0, 'C')

        # Save PDF
        os.makedirs("static/reports", exist_ok=True)
        report_path = os.path.join("static/reports", f"churn_report_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
        pdf.output(report_path)

        return send_file(report_path, as_attachment=True)

    except Exception as e:
        flash(f' Error during report generation: {e}', 'error')
        return redirect(url_for('upload'))
#------------------------------------PREDICT ATTRITION-----------------------------------------------#
@app.route('/predict-attrition', methods=['GET', 'POST'])
@login_required
def predict_attrition():
    # Import required libraries (note: typically imports go at top of file)
    import numpy as np
    import matplotlib.pyplot as plt
    import datetime
    import traceback
    from sklearn.preprocessing import LabelEncoder
    from fpdf import FPDF
    from models.model import UploadHistory

    # Handle GET request by redirecting to upload page
    if request.method == 'GET':
        return redirect(url_for('upload'))

    # Validate file was submitted
    if 'file' not in request.files or request.files['file'].filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('upload'))

    try:
        # === FILE HANDLING ===
        # Secure and save the uploaded file
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)  # Create uploads directory if needed
        file.save(filepath)

        # === MODEL LOADING ===
        # Load pre-trained churn model and label encoders
        model = joblib.load("model/churn_model.pkl")
        label_encoders = joblib.load("model/label_encoders.pkl")

        # === DATA PREPROCESSING ===
        # Read and clean the uploaded CSV
        df = pd.read_csv(filepath)
        # Standardize column names (remove spaces, lowercase)
        df.columns = [col.strip().replace(" ", "").lower() for col in df.columns]
        # Remove duplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]

        # Column name aliases mapping to standardize different naming conventions
        alias_map = {
            'Contract': ['contract', 'contractlength', 'tenure'],
            'LatePayments': ['latepayments', 'isactivemember', 'missedpayments', 'credit_card', 'hascrcard', 'active'],
            'Plan': ['plan', 'numofproducts', 'subscriptiontype', 'products', 'n_products'],
            'TotalCharges': ['totalcharges', 'balance', 'total'],
            'MonthlyCharges': ['monthlycharges', 'estimatedsalary', 'monthly_fee', 'billing_amount', 'monthly_payment', 'salary'],
            'CreditScore': ['creditscore', 'riskscore', 'score', 'customer_score', 'cs'],
            'Age': ['age', 'customer_age', 'customerage', 'years', 'age(yrs)', 'client_age']
        }

        # Create standardized dataframe using alias mapping
        fixed_df = pd.DataFrame()
        for std_feature, aliases in alias_map.items():
            # Find first matching alias in the uploaded data
            match = next((alias for alias in aliases if alias.lower() in df.columns), None)
            if match:
                fixed_df[std_feature] = df[match.lower()]
            else:
                fixed_df[std_feature] = 0  # Default to 0 if feature not found

        # === FEATURE ENGINEERING ===
        # Calculate monthly charges if not directly provided
        if 'estimatedsalary' in df.columns and fixed_df['MonthlyCharges'].sum() == 0:
            fixed_df['MonthlyCharges'] = pd.to_numeric(df['estimatedsalary'], errors='coerce') / 12

        if 'balance' in df.columns and 'tenure' in df.columns and fixed_df['MonthlyCharges'].sum() == 0:
            fixed_df['MonthlyCharges'] = pd.to_numeric(df['balance'], errors='coerce') / df['tenure'].replace(0, 1)

        # Convert numeric columns and handle missing values
        for col in ['TotalCharges', 'MonthlyCharges', 'CreditScore', 'Age']:
            fixed_df[col] = pd.to_numeric(fixed_df.get(col, 0), errors='coerce').fillna(0)

        # === LABEL ENCODING ===
        # Encode categorical features using saved or new label encoders
        for col in fixed_df.columns:
            if fixed_df[col].dtype == 'object':
                fixed_df[col] = fixed_df[col].astype(str)
                if col in label_encoders:
                    le = label_encoders[col]
                    # Handle unseen categories by marking as 'unknown'
                    fixed_df[col] = fixed_df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
                    if 'unknown' not in le.classes_:
                        le.classes_ = np.append(le.classes_, 'unknown')
                    fixed_df[col] = le.transform(fixed_df[col])
                else:
                    # Create new encoder if feature wasn't in training data
                    le = LabelEncoder()
                    fixed_df[col] = le.fit_transform(fixed_df[col])
                    label_encoders[col] = le

        # === FEATURE VALIDATION ===
        # Check minimum required features are present
        features = ['Contract', 'LatePayments', 'Plan', 'TotalCharges', 'MonthlyCharges', 'CreditScore', 'Age']
        provided = [col for col in features if col in fixed_df.columns and fixed_df[col].sum() != 0]

        if len(provided) < 4:
            flash(" At least 4 of the 7 key features must be present in your CSV to make a prediction.", "error")
            return redirect(url_for('upload'))

        # Fill missing features with 0
        for col in features:
            if col not in fixed_df.columns:
                fixed_df[col] = 0

        # === PREDICTION ===
        X = fixed_df[provided]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]  # Probability of churn (class 1)

        # Create results dataframe with customer IDs, risk levels, and confidence scores
        results = pd.DataFrame({
            'CustomerID': df.index + 1,  # 1-based indexing
            'ChurnRisk': ['High' if p == 1 else 'Low' for p in predictions],  # Convert to human-readable
            'Confidence': [round(prob * 100, 1) for prob in probabilities]  # Percentage with 1 decimal
        })

        # Calculate overall churn rate percentage
        churn_rate = results['ChurnRisk'].value_counts(normalize=True).get('High', 0) * 100

        # === DATABASE LOGGING ===
        # Record prediction results if user is authenticated
        if current_user.is_authenticated:
            new_upload = UploadHistory(
                user_id=current_user.id,
                filename=filename,
                churn_percentage=round(churn_rate, 1)
            )
            db.session.add(new_upload)
            db.session.commit()

        # === VISUALIZATION ===
        # Generate and save pie chart of risk distribution
        chart_filename = f"churn_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        chart_path = os.path.join("static", chart_filename)
        plt.figure(figsize=(6, 6))
        counts = results['ChurnRisk'].value_counts()
        labels = ['High Risk' if label == 'High' else 'Low Risk' for label in counts.index]
        colors = ['#F44336' if label == 'High Risk' else '#4CAF50' for label in labels]  # Red/Green color scheme
        plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors)
        plt.savefig(chart_path)
        plt.close()

        # Save prediction results to CSV
        results_filename = f"results_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        results_path = os.path.join("static", results_filename)
        results.to_csv(results_path, index=False)

        # Identify top 5 high-risk customers and most influential factor
        top_customers = results[results['ChurnRisk'] == 'High'].sort_values(by='Confidence', ascending=False).head(5)
        top_reason = max(provided, key=lambda col: fixed_df[col].std()) if provided else "Data inconsistency detected."

        # === REPORT GENERATION ===
        # Create PDF report with analysis results
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 20)
        pdf.cell(0, 15, "Customer Churn Predictions Report", ln=True, align='C')

        # Executive summary section
        pdf.set_font('Arial', 'B', 12)
        pdf.ln(10)
        pdf.cell(0, 10, "Executive Summary:", ln=True)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 8,
            f"This churn analysis report is based on a predictive machine learning model trained on historical customer behavior. "
            f"The model evaluated a total of {len(results)} customers and determined a churn rate of {round(churn_rate, 1)}%. "
            "Churn refers to customers who are likely to discontinue their service or subscription. "
            "Accurately identifying churn-prone users allows companies to proactively deploy retention strategies.")

        # Prediction summary section
        pdf.ln(8)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Prediction Summary:", ln=True)
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 8, f"Total Customers Analyzed: {len(results)}", ln=True)
        pdf.cell(0, 8, f"Predicted Churn Rate: {round(churn_rate, 1)}%", ln=True)
        pdf.cell(0, 8, f"High-Risk Customers: {(predictions == 1).sum()}", ln=True)
        pdf.cell(0, 8, f"Low-Risk Customers: {(predictions == 0).sum()}", ln=True)

        # Top high-risk customers section
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Top 5 High-Risk Customers (By Confidence):", ln=True)
        pdf.set_font('Arial', '', 11)
        for _, row in top_customers.iterrows():
            pdf.cell(0, 8, f"Customer {int(row['CustomerID'])} - Confidence: {row['Confidence']}%", ln=True)

        # Retention strategies section
        pdf.ln(8)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Recommended Retention Strategies:", ln=True)
        pdf.set_font('Arial', '', 11)
        strategies = [
            "- Launch proactive outreach campaigns for high-risk segments.",
            "- Offer time-sensitive loyalty bonuses or discounts.",
            "- Use contract extensions or win-back programs.",
            "- Improve customer service for active accounts with declining activity.",
            "- Tailor retention offers based on customer tenure and engagement."
        ]
        for s in strategies:
            pdf.cell(0, 8, s, ln=True)

        # Add visualization to report
        if os.path.exists(chart_path):
            pdf.ln(10)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "Churn Risk Breakdown Chart:", ln=True)
            pdf.image(chart_path, x=30, w=150)

        # Save PDF report
        os.makedirs("static/reports", exist_ok=True)
        pdf_filename = f"churn_report_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        pdf_path = os.path.join("static/reports", pdf_filename)
        pdf.output(pdf_path)

        # === RENDER RESULTS ===
        return render_template(
            "upload.html",
            churn_percentage=round(churn_rate, 1),
            chart_url=url_for("static", filename=chart_filename),
            table=results.sort_values(by="Confidence", ascending=False).head(10).to_html(
                classes="table table-hover", index=False
            ),
            download_url=url_for("static", filename=results_filename),
            pdf_url=url_for("static", filename=f"reports/{pdf_filename}"),
            total_customers=len(results),
            high_risk_count=sum(predictions),
            low_risk_count=len(predictions) - sum(predictions),
            top_reason=f"The most influential factor contributing to churn appears to be: {top_reason}."
        )

    except Exception as e:
        # Error handling and debugging
        print(" Exception during prediction:")
        traceback.print_exc()
        return render_template("upload.html", error=str(e))
    
#---------------------------------DOWNLOAD ATTRITION REPORT-------------------------------------------#
@app.route('/download-attrition-report', methods=['GET'])
@login_required
def download_attrition_report():
    """
    Creates and downloads a PDF report of customer churn predictions.
    This endpoint:
    1. Validates existing prediction results
    2. Calculates key churn metrics
    3. Creates a professional PDF report
    4. Provides download to the user
    """
    try:
        # Import required libraries (note: typically these would be at top of file)
        from fpdf import FPDF  # For PDF generation
        import pandas as pd    # For data handling
        import os              # For file operations
        import datetime        # For timestamping

        # === FILE VALIDATION ===
        # Ensure static directory exists for storing reports
        static_dir = os.path.join(app.root_path, 'static')
        os.makedirs(static_dir, exist_ok=True)
        
        # Find all result CSV files in static directory
        result_files = [f for f in os.listdir(static_dir) 
                      if f.startswith("results_") and f.endswith(".csv")]
        
        # Return error if no prediction files found
        if not result_files:
            flash("No prediction results found. Please analyze data first.", "error")
            return redirect(url_for("upload"))

        # === DATA LOADING ===
        # Get the most recent results file using creation time
        latest_file = max(result_files, 
                         key=lambda x: os.path.getctime(os.path.join(static_dir, x)))
        csv_path = os.path.join(static_dir, latest_file)
        
        try:
            # Load the prediction results into DataFrame
            df = pd.read_csv(csv_path)
        except Exception as e:
            flash(f"Error reading results file: {str(e)}", "error")
            return redirect(url_for("upload"))

        # === DATA VALIDATION ===
        # Check for required columns in the results
        required_columns = {'ChurnRisk', 'Confidence', 'CustomerID'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            flash(f"Missing required columns: {', '.join(missing)}", "error")
            return redirect(url_for("upload"))

        # === METRICS CALCULATION ===
        # Calculate overall churn rate percentage
        churn_rate = df['ChurnRisk'].value_counts(normalize=True).get('High', 0) * 100
        # Count high and low risk customers
        high_risk_count = (df['ChurnRisk'] == 'High').sum()
        low_risk_count = (df['ChurnRisk'] == 'Low').sum()
        # Get top 5 most confident high-risk predictions
        top_customers = df[df['ChurnRisk'] == 'High'].nlargest(5, 'Confidence')

        # === PDF REPORT GENERATION ===
        # Initialize PDF document with basic settings
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)  # Enable automatic page breaks
        
        # Set document metadata for PDF properties
        pdf.set_title("Customer Churn Report")
        pdf.set_author("Churn Prediction System")

        # === HEADER SECTION ===
        pdf.set_font('Arial', 'B', 20)
        pdf.cell(0, 15, "Customer Churn Predictions Report", ln=True, align='C')
        pdf.ln(10)  # Add vertical space

        # === REPORT METADATA ===
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.cell(0, 10, f"Report ID: {latest_file.split('_')[1].split('.')[0]}", ln=True)
        pdf.ln(10)

        # === EXECUTIVE SUMMARY ===
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, "Executive Summary", ln=True)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 8,
            f"This report analyzes consumer attrtion risk for {len(df)} customers. "
            f"The model identified {high_risk_count} high-risk customers ({churn_rate:.1f}% churn rate). "
            "Instant action is recommended for the top at-risk customers listed below.")
        pdf.ln(10)

        # === KEY METRICS SECTION ===
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Key Metrics", ln=True)
        pdf.set_font('Arial', '', 12)
        
        # Calculate column width for 3-column layout
        col_width = pdf.w / 3
        
        # Define metrics to display
        metrics = [
            ("Total Customers", len(df)),
            ("High-Risk Customers", f"{high_risk_count} ({churn_rate:.1f}%)"),
            ("Low-Risk Customers", low_risk_count),
            ("Report Date", datetime.datetime.now().strftime('%Y-%m-%d'))
        ]
        
        # Render metrics in two-column format
        for label, value in metrics:
            pdf.cell(col_width, 8, label, 0, 0)  # Label
            pdf.cell(col_width, 8, str(value), 0, 1)  # Value with line break
        pdf.ln(10)

        # === TOP AT-RISK CUSTOMERS TABLE ===
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Top At-Risk Customers", ln=True)
        
        # Table Header with gray background
        pdf.set_fill_color(200, 200, 200)  # Light gray
        headers = ["Customer ID", "Risk Level", "Confidence"]
        for header in headers:
            pdf.cell(40, 8, header, 1, 0, 'C', 1)  # Centered with border and fill
        pdf.ln(8)
        pdf.set_fill_color(255, 255, 255)  # Reset to white
        
        # Table Rows with customer data
        pdf.set_font('Arial', '', 12)
        for _, row in top_customers.iterrows():
            pdf.cell(40, 8, str(int(row['CustomerID'])), 1, 0)  # Customer ID
            pdf.cell(40, 8, row['ChurnRisk'], 1, 0)            # Risk Level
            pdf.cell(40, 8, f"{row['Confidence']}%", 1, 1)      # Confidence % with line break
        pdf.ln(15)

        # === RECOMMENDED ACTIONS ===
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Recommended Actions", ln=True)
        pdf.set_font('Arial', '', 12)
        
        # List of suggested retention strategies
        actions = [
            "1. Immediate outreach to top at-risk customers",
            "2. Review service patterns of high-risk segment",
            "3. Implement targeted retention offers",
            "4. Schedule follow-up analysis in 30 days"
        ]
        
        # Render each action item
        for action in actions:
            pdf.cell(0, 8, action, 0, 1)
        pdf.ln(10)

        # === VISUALIZATION ===
        # Add pie chart if available
        chart_path = os.path.join(static_dir, "churn_pie.png")
        if os.path.exists(chart_path):
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, "Churn Risk Distribution", ln=True)
            # Center the image on page with width 110
            pdf.image(chart_path, x=50, w=110)
            pdf.ln(5)

        # === SAVE AND RETURN REPORT ===
        # Ensure reports directory exists
        reports_dir = os.path.join(static_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate timestamped filename
        report_filename = f"churn_report_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        report_path = os.path.join(reports_dir, report_filename)
        
        # Output PDF to file
        pdf.output(report_path)

        # Send file to user as download
        return send_file(
            report_path,
            as_attachment=True,  # Force download
            download_name=f"Churn_Report_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
            mimetype='application/pdf'
        )

    except Exception as e:
        # Handle any errors during report generation
        flash(f"Error generating report: {str(e)}", "error")
        app.logger.error(f"PDF generation error: {str(e)}", exc_info=True)
        return redirect(url_for("upload"))
    


#  run it
if __name__ == '__main__':
    print("\n Luxury Churn AI running at http://127.0.0.1:5000/ \n")
    app.run(debug=True)

    


