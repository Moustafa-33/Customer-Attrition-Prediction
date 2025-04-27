# Import the Flask application factory function and database instance
from app import create_app
from models.extensions import db

# Create the Flask application instance using the factory function
app = create_app()

# Execute database operations within the application context
with app.app_context():
    # Print status message for database cleanup
    print(" Dropping all existing tables (if any)...")
    # Remove all existing database tables (clean slate)
    db.drop_all()

    # Print status message for table creation
    print(" Creating all tables...")
    # Create all database tables based on defined models
    db.create_all()

    # Print success confirmation
    print(" Database initialized successfully.")