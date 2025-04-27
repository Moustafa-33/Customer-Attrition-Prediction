from flask import Flask, render_template, request, redirect, url_for, flash, session # type: ignore
from flask_sqlalchemy import SQLAlchemy # type: ignore
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user # type: ignore
from flask_bcrypt import Bcrypt  # type: ignore # Secure password hashing
import os

# Absolute Path for Database
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "instance", "users.db")

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'luxury_churn_ai'  # Security key

# Configure Database
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{DB_PATH}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize Flask Extensions
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)  # Stores hashed password
    account_type = db.Column(db.String(20), nullable=False)  # "Individual" or "Business"

# Create Database Tables (Ensures Database is Created)
with app.app_context():
    db.create_all()
    print(f"✅ Database created successfully at: {DB_PATH}")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        account_type = request.form['account_type']

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose another.', 'error')
            return redirect(url_for('signup'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        new_user = User(username=username, password=hashed_password, account_type=account_type)
        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')

# Logout Route (LOGOUT WITHOUT DELETING USER ACCOUNT)
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

# Dashboard Route (Requires Login)
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username, account_type=current_user.account_type)

# Upload Page (Requires Login)
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded.', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file.', 'error')
            return redirect(request.url)

        filename = file.filename
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        flash('File uploaded successfully!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('upload.html')

# Pricing Page
@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

# Testimonials Page
@app.route('/testimonials')
def testimonials():
    return render_template('testimonials.html')

# Contact Page
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        flash('Message sent successfully!', 'success')
    return render_template('contact.html')

# Run Flask App
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Ensures database is created on startup
        print(f"✅ Database checked and created if missing at: {DB_PATH}")
    
    print("\n🚀 Flask is running at: http://127.0.0.1:5000/ 🚀\n")
    app.run(debug=True)
