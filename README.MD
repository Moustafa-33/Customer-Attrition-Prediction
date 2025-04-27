CUSTOMER ATTRTION,PREDICTION,RETENTION STRATEGY (Churn AI)

OVERVIEW

This project was developed to predict customer attrition using machine learning. The goal is to help businesses identify customers who are most likely to leave, enabling them to take early action and improve retention rates.

I designed this system to bring together practical machine learning, web development, and project management skills into a complete, professional-grade solution.

KEY FEATURES

Accurate Attrition Prediction: Trained on real customer data to provide meaningful churn predictions.
User-Friendly CSV Upload: Allows users to easily upload datasets without complex formatting.
Insightful Outputs: Displays clear prediction results in an accessible, professional interface.
Clean, Responsive Front-End: Built with HTML and CSS, designed for a smooth user experience.
Version Control: Full GitHub repository management and project tracking.
Technologies Used

STACK
Python (Pandas, scikit-learn)
Flask (backend framework)
HTML5, CSS3 (frontend)
Git and GitHub (version control and collaboration)
How It Works

The user uploads a CSV file containing customer data.
The system processes the file and applies the trained machine learning model.
Prediction results are generated and displayed on the dashboard.
Users can view customer churn likelihoods and take action based on the insights.
Project Structure

Customer-Attrition-Prediction/
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   ├── index.html
│   ├── upload.html
│   └── dashboard.html
│
├── app.py
├── model.pkl
└── README.md

Future Improvements

Improve model performance through advanced feature engineering and hyperparameter tuning.
Add the ability to download a PDF report summarising prediction results.
Extend the platform with user accounts and upload history tracking.
Getting Started

To run the project locally:

git clone https://github.com/Moustafa-33/Customer-Attrition-Prediction.git
cd Customer-Attrition-Prediction
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
pip install -r requirements.txt
python app.py
Then open your browser and navigate to http://localhost:5000.