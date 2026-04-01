# Third_project_taxi_fare_predict

Problem Statement for Understanding
       As a Data Analyst at an urban mobility analytics firm. Your mission is to unloack insights from real-world taxi trip data to enhance fare estimation systems and promote pricing transperancy for passengers. This project focuses on analyzing historical taxi trip records collected from a metropolitan transportation network.
The goal is to build a predictive model that accuracy estimates the total taxi fare amount based on various ride-related featues. Learners will preprocess the raw data, engineer meaningful features, handle data quality issues, train and evaluate multiple regression models, and finally deploy the best-performing model using Streamlit.
       
# Taxi Fare Prediction: End-to-End Machine Learning Project
## Overview
This project focuses on predicting the total fare amount for taxi rides in New York City using machine learning. The objective was to develop an accurate regression model from real trip data, transform and engineer useful features, and deploy a user-friendly web application for interactive predictions.

## Project Workflow
## Problem Statement
The main goal was to build a robust ML pipeline capable of estimating the total taxi fare (including all surcharges and tips) based on trip details such as pickup/dropoff locations, trip time, passenger count, and other features.

## Data Preparation
The project began with exploring and understanding the provided taxi trip dataset, which contains columns like pickup_datetime, pickup/dropoff coordinates, passenger_count, rates, extras, tips, and total fare amount. The data was cleaned by handling missing values and outliers to ensure high-quality inputs for model training.

## Feature Engineering
Additional features were engineered to enhance model performance, including calculating trip distance using the Haversine formula and extracting time-based features such as hour of day, night/weekend indicators, and log-transformed continuous values to account for skewed distributions.

## Model Building
Multiple regression algorithms were evaluated, including Random Forest, Linear Regression, and others. Hyperparameter tuning was conducted using techniques like RandomizedSearchCV to optimize model performance efficiently while managing computational resources.

## Model Selection and Saving
Models were compared on validation set metrics (e.g., RMSE, R²), and the best performing model was chosen. The final model pipeline integrates both feature preprocessing and prediction steps. This pipeline was saved using joblib for deployment purposes.

## Streamlit Web App Deployment
A Streamlit application was developed to provide an interactive user interface. The app collects user input for the most relevant trip parameters, computes required features, and predicts the total fare instantly using the trained model pipeline. Input validation and automatic feature engineering ensure accurate, user-friendly predictions.
