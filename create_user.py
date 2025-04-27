# Import required application components from the app package
from app import app, db, User, bcrypt

# Create an application context to work with the database
with app.app_context():
    # Define user credentials and account type to be created
    username = 'Moustafa123@business.com'
    password = 'pass123'
    account_type = 'Business'

    # Check if a user with this username already exists in the database
    existing_user = User.query.filter_by(username=username).first()
    
    if existing_user:
        # If user exists, print warning message (don't create duplicate)
        print(f"  User '{username}' already exists.")
    else:
        # If user doesn't exist:
        # 1. Hash the plaintext password for secure storage
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # 2. Create new User object with provided credentials
        user = User(username=username, password=hashed_pw, account_type=account_type)
        
        # 3. Add new user to database session
        db.session.add(user)
        
        # 4. Commit the session to save to database
        db.session.commit()
        
        # Print success message
        print(f" User '{username}' created successfully!")
