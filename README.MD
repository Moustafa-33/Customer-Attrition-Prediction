CUSTOMER ATTRITION, PREDICTION, RETENTION STRATEGY (Churn AI)

Overview

This project was developed to predict customer attrition using machine learning. The goal is to help businesses identify customers who are most likely to leave, enabling them to take early action and improve retention rates. I designed this system to bring together practical machine learning, web development, and project management skills into a complete, professional-grade solution.

Key Features

- Accurate Attrition Prediction: Trained on real customer data to generate meaningful churn predictions.  
- User-Friendly CSV Upload: Users can easily upload datasets without needing complex formatting.  
- Insightful Outputs: Prediction results are displayed clearly and professionally for easy interpretation.  
- Clean, Responsive Front-End: Built with HTML and CSS, ensuring a smooth and accessible user experience.  
- Version Control: Full GitHub repository management with integrated project tracking.

Technologies Used

- Python (Pandas, scikit-learn)  
- Flask (backend framework)  
- HTML5, CSS3 (frontend)  
- Git and GitHub (version control and collaboration)

How It Works

The user uploads a CSV file containing customer data. The system processes the file and applies a pre-trained machine learning model. Prediction results are then generated and displayed on a clean, interactive dashboard, helping users understand which customers are at higher risk of churn.

Project Structure

- Customer-Attrition-Prediction/  
  - static/  
    - css/  
      - style.css  
  - templates/  
    - index.html  
    - upload.html  
    - dashboard.html  
  - app.py  
  - model.pkl  
  - README.md

Future Improvements

- Enhance model performance through advanced feature engineering and hyperparameter tuning.  
- Add the ability to download a full PDF report summarising prediction results.  
- Extend the platform to support user accounts and track historical uploads for better insights.

Getting Started

To run the project locally:

1. Clone the repository:  
   
   git clone https://github.com/Moustafa-33/Customer-Attrition-Prediction.git
   

2. Navigate into the project directory:  
   
   cd Customer-Attrition-Prediction
   

3. Set up a virtual environment:  
   
   python -m venv venv
   

4. Activate the virtual environment:  
   - On macOS/Linux:  
     
     source venv/bin/activate
     
   - On Windows:  
     
     .\venv\Scripts\activate
     

5. Install the required dependencies:  
   
   pip install -r requirements.txt
   

6. Run the Flask application:  
   
   python app.py
   

7. Open your browser and navigate to:  
   
   http://localhost:5000
   

