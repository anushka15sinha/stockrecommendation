# Stock Recommendation Studio

## Overview

Stock Recommendation Studio is an AI-powered stock analysis and recommendation system built using Streamlit. It combines deep learning-based stock price prediction with financial news sentiment analysis to generate explainable buy, hold, or sell recommendations.

The system helps users analyze market trends, forecast future stock prices, and understand sentiment-driven market signals.

---

## Features

### Stock Price Prediction

* Uses LSTM (Long Short-Term Memory) neural networks
* Predicts future stock prices based on historical data
* Supports multiple stocks including INFY.NS, RELIANCE.NS, and TCS.NS

### News Sentiment Analysis

* Fetches financial news using APIs
* Uses FinBERT model for sentiment classification
* Classifies news as positive, neutral, or negative

### Recommendation System

* Combines price prediction and sentiment analysis
* Generates buy, hold, or sell signals
* Provides explainable outputs based on model inputs

### Interactive Dashboard

* Built using Streamlit
* Displays charts, predictions, and sentiment results
* Simple and user-friendly interface

---

## Tech Stack

* Frontend: Streamlit
* Backend: Python
* Deep Learning: TensorFlow / Keras
* NLP: FinBERT
* Data Processing: Pandas, NumPy
* Visualization: Matplotlib, Plotly
* APIs: News API, Yahoo Finance

---

## Project Structure

stockrecommendor-main/
│── app.py                  Main Streamlit application
│── api.py                  API integration utilities
│── requirements.txt        Python dependencies
│── .env.example            Environment variable template
│
│── INFY.NS_model.h5        Trained LSTM model
│── RELIANCE.NS_model.h5
│── TCS.NS_model.h5
│
│── INFY.NS_scaler.pkl      Saved scalers
│── RELIANCE.NS_scaler.pkl
│── TCS.NS_scaler.pkl
│
│── .streamlit/
│   └── config.toml         Streamlit configuration
│
└── README.md

---

## Installation

### 1. Clone the repository

git clone [https://github.com/anushka15sinha/stockrecommendation.git](https://github.com/anushka15sinha/stockrecommendation.git)
cd stockrecommendation

### 2. Create virtual environment

Windows:
python -m venv venv
venv\Scripts\activate

Mac/Linux:
python3 -m venv venv
source venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Setup environment variables

Create a .env file:

NEWS_API_KEY=your_news_api_key

---

## Run the application

streamlit run app.py

---

## Deployment

This project can be deployed using Hugging Face Spaces.

Steps:

1. Create a new Space
2. Select Streamlit SDK
3. Upload the repository
4. Add environment variables in settings
5. Deploy

---


## Disclaimer

This project is for educational purposes only. It should not be considered financial advice.

---

## Public deployed link 
https://stockrecommendation-15.streamlit.app/

---
