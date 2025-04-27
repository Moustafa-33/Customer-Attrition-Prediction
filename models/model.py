from datetime import datetime
from models.extensions import db

class UploadHistory(db.Model):
    __tablename__ = 'upload_history'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    churn_percentage = db.Column(db.Float, nullable=False)
    report_file = db.Column(db.String(255), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


    def __repr__(self):
        return (f"<UploadHistory(filename='{self.filename}', churn={self.churn_percentage}%, "
                f"user_id='{self.user_id}', timestamp='{self.timestamp}')>")
