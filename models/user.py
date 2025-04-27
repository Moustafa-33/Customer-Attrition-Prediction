from flask_login import UserMixin
from models.extensions import db

class User(UserMixin, db.Model):
    __tablename__ = 'user'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    account_type = db.Column(db.String(20), nullable=False)
    name = db.Column(db.String(120), nullable=True)
    company = db.Column(db.String(120), nullable=True)
    job_title = db.Column(db.String(120), nullable=True)
    phone = db.Column(db.String(20), nullable=True)

    uploads = db.relationship('UploadHistory', backref='user', lazy=True)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', account_type='{self.account_type}')>"
