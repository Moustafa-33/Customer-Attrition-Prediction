# models/extensions.py

from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

#  Shared instances used across the app
db = SQLAlchemy()
bcrypt = Bcrypt()
login_manager = LoginManager()
