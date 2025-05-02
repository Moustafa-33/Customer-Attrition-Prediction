CUSTOMER ATTRITION, PREDICTION, RETENTION STRATEGY (Churn AI)

Overview

The main target of this project was to initally predict customer attrition through the use of machine learning. Main objective being to aid businesses through identifing which sutomer is most at risk of leaving. Thus enabling the consumer to take iniative and overall imporve retention rates. I've managed to create this project with the main gial of being to combine both practical machine learinign, web development as well as project management skills for it to result into a high-level soltuion to the main problem of "Churn".

Key Features

- Percise Attrition Prediction: Ive initally trinaed this model through the use of real customer data in order to generate attrition predictions.  
- CSV Upload: consumers are able to easily upload datasets (CSVs) with peersonal consumer information.  
- Insightful Outputs: The results are then presented to the conusmer both visually on the wbeiste , but also able to download a PDF genrated with further attrtion techniques that can be utilised for reduciong churn.  
- A Responsive Front-End: Created with HTML and CSS, assuring a low level of cogantive overalod and accessible user experience.  
- Version Control:GitHub repository management as well as project tracking.

Technologies Used

- Python (Pandas, scikit-learn)  
- Flask (backend framework)  
- HTML5, CSS3 (frontend)  
- Git and GitHub (version control and collaboration)

How It Works

The consumer ( a business) initally uploads a CSV file that conatins  personal customer data. The webiste utlimaltley processes the file and applies a pre-trained machine learning model. Prediction results are then displayed on the website through the use of the results page, aiding consumers to understand which customers are at higher risk of attricting and thus leaving their business or service that they provide.

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

- Further improve model performance through the use of feature engineering as well as hyperparameter tuning.  
- Further fine tune the feature on my website where it allows the conusmers to download a full PDF report summarising prediction results.  
- Lastly to enhance the platform to support business accounts and to account for historical uploads for better insights.


To start this project 


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
   

