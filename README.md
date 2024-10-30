# ANN-Classification-Churn
## Customer Churn Prediction Using ANN
This project uses an Artificial Neural Network (ANN) to predict customer churn, deployed on Streamlit for user interaction. The repository includes the dataset, model files, and preprocessing encoders for replicable predictions and feature engineering.

Project Structure
```bash
Copy code
.
├── Churn_Modelling.csv         # Dataset used for training and evaluation
├── LICENSE                     # License file (GPL-3.0)
├── README.md                   # Project description and usage instructions
├── app.py                      # Streamlit app for real-time predictions
├── experiments.ipynb           # Notebook for exploratory data analysis and model training
├── prediction.ipynb            # Notebook for generating predictions
├── label_encoder_gender.pkl     # Pickle file for label encoding gender feature
├── model.keras                 # Trained ANN model for churn prediction
├── onehot_encoder_geo.pkl      # Pickle file for one-hot encoding geography feature
├── scaler.pkl                  # Pickle file for feature scaling
└── requirements.txt            # List of dependencies for project setup
```
## Project Overview
This project predicts whether a customer is likely to churn based on demographic and account data. It uses an ANN model trained on the provided Churn_Modelling.csv dataset, with preprocessing steps handled through saved encoder files (label_encoder_gender.pkl, onehot_encoder_geo.pkl, scaler.pkl) to ensure consistent feature encoding and scaling.

## Streamlit Deployment
The model is deployed using Streamlit, allowing interactive churn predictions. Users input customer details and receive predictions in real-time, with additional data visualizations in app.py for better insights into model output.

Link : https://ann-classification-churn-k2taqmjxnsx7yg6wjtbgux.streamlit.app/
