# ⛽ Gas Price Forecasting System (Quant Research Project)

## Overview

This project is a **time-series forecasting application** that estimates and predicts gas prices using a machine learning pipeline. The system is built using **Flask** for deployment and integrates a modular ML pipeline for forecasting future prices based on historical data.

The model is trained and validated in a Jupyter notebook and deployed as a web application that allows users to estimate gas prices for a specific date.

---

## 🚀 Key Features

* Gas price forecasting using time-series modeling
* End-to-end ML pipeline (training → prediction → deployment)
* Flask-based web application
* Future price generation using recursive forecasting
* Dynamic prediction for user-selected dates
* Modular and production-style project structure

---

## 🧠 Problem Statement

Gas prices fluctuate over time due to economic and market factors. The goal of this project is to:

* Predict future gas prices based on historical trends
* Provide a simple interface for users to estimate prices for any given date
* Build a scalable forecasting pipeline that can be extended to other financial time-series problems

---

## 🚀 How to Use the App

### 1. Clone the Repository

```bash
git clone https://github.com/OlaSuaifan93/Quant_Research.git
cd Quant_Research
```

---

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

**Mac/Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Ensure Model Artifacts Exist

Make sure the following folder exists after training:

```text
artifacts/
    ├── model.pkl
```

This file is required for predictions.

---

### 5. Run the Flask Application

```bash
python app.py
```

The app will start at:

```
http://0.0.0.0:8000/
```

or open in your browser:

```
http://localhost:8000/
```

---

### 6. Make a Prediction

1. Open the web app in your browser
2. Navigate to the prediction page
3. Enter a **date** in the input field
4. Click **Predict**

The system will:

* Load historical gas price data
* Generate future predictions using the ML pipeline
* Return the estimated gas price for the selected date

---

## 📌 Expected Workflow

User enters date → Flask app receives input →
CustomData pipeline generates future forecast →
Model predicts prices →
Result is displayed on UI

---

## 📊 Output Example

```
Predicted Gas Price for 2025-01-01: $3.42
```

---

## ⚠️ Notes

* Ensure `artifacts/model.pkl` exists before running the app
* Model must be trained using the corresponding notebook
* Date input must be valid and within forecastable range
* The system relies on historical data preprocessing inside `DataTransformation`

---

## 🔮 Future Improvements

* Deploy on cloud (AWS / Azure / Render)
* Add confidence intervals to predictions
* Replace Flask with FastAPI for better performance
* Add interactive charts (Plotly)
* Extend model to multivariate forecasting
* Add real-time gas price API comparison

---



