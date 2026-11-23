# Hospital Predictive Analytics – 30-Day Readmission Prediction

## Project Overview
This project uses Machine Learning to predict the likelihood of a diabetic patient being readmitted to the hospital within 30 days of discharge. It analyzes patient demographics, medical history, hospital visits, medications, and treatment information to estimate readmission risk.

## Objectives
- Analyze hospital patient data.
- Perform data cleaning and preprocessing.
- Identify important factors associated with readmission.
- Train and evaluate Machine Learning models.
- Predict readmission risk for new patient inputs.
- Develop an interactive web application using Flask.

## Dataset
**Dataset:** Diabetes 130-US Hospitals for Years 1999–2008

The dataset contains hospital encounters involving diabetic patients and includes patient demographics, diagnoses, medications, procedures, and readmission information.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Flask
- HTML
- CSS
- Joblib

## Machine Learning Models
- Logistic Regression
- Random Forest
- Balanced Logistic Regression

## Project Workflow
1. Data acquisition
2. Data exploration
3. Data cleaning
4. Missing value handling
5. Feature engineering
6. Data preprocessing
7. Model training
8. Model evaluation
9. Model comparison
10. Model saving
11. Flask web application development
12. Patient readmission prediction

## Web Application
The application allows users to enter patient details such as age, gender, hospital stay duration, laboratory procedures, medications, previous hospital visits, and diagnoses.

The application displays:
- Predicted readmission risk
- Readmission probability percentage

## Project Structure
```text
Hospital_Predictive_Analytics/
│
├── dataset/
│   └── diabetic_data.csv
├── model/
│   └── readmission_model.pkl
├── templates/
│   └── index.html
├── screenshots/
├── report/
├── presentation/
├── main.py
├── app.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the model
```bash
python main.py
```

### 3. Run the web application
```bash
python app.py
```

### 4. Open in browser
```text
http://127.0.0.1:5000
```

## Disclaimer
This project is developed for educational purposes. Predictions are experimental and should not be used as a substitute for professional medical judgment or clinical decision-making.