# Student Performance Prediction System

## About the Project

This project is an end-to-end Machine Learning web application that predicts a student’s final grade and pass/fail status based on behavioral factors such as study time, travel time, health, and absences.

The goal was to build a complete ML pipeline — from data preprocessing and model training to deployment using Flask.

----------

## What This Project Does

- Predicts the final grade (Regression)
- Predicts pass/fail outcome (Classification)
- Provides a simple web interface for real-time prediction

----------

## How It Was Built

### 1. Data Processing
- Loaded and explored the dataset
- Performed feature selection
- Removed demographic features (like parental education) to reduce bias

### 2. Model Training
- Compared multiple models:
        - Linear Regression
        - Random Forest Regressor
        - Logistic Regression
        - Random Forest Classifier
- Handled class imbalance using class weighting
- Evaluated performance using R², accuracy, precision, recall, and F1-score

### 3. Deployment
- Saved trained models using Joblib
- Built a Flask web application
- Added input validation (both frontend and backend)
- Displayed prediction results dynamically with clear UI feedback

----------

## Tech Stack

- Python
- Pandas & NumPy
- Scikit-learn
- Flask
- HTML & CSS

----------

## Project Structure

Student-Performance-ML/
│
├── data/
├── models/
├── train.py
├── app.py
├── templates/
│ └── index.html
├── requirements.txt
└── README.md

----------

## ▶ How To Run

1. Install dependencies:
        pip install -r requirements.txt

2. Train models:
        python train.py

3. Run the application:
        python app.py

4. Open in browser:
        http://127.0.0.1:5000/

----------

## Key Highlights
- End-to-end ML pipeline
- Bias-aware feature engineering
- Model comparison & evaluation
- Class imbalance handling
- Real-time web deployment

----------

## Future Improvements
- Deploy to cloud (Render / AWS)
- Add advanced feature engineering
- Improve UI/UX
- Add model monitoring

----------

✨ New Features:
- Dark/Light theme toggle
- ML confidence probability bar
- Modern dropdown UI
- Responsive design

----------

## Author

Kousik Varma Gattupalli
B.Tech CSE (AI & ML)
