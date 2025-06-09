# 🌫️ AQI Prediction Using Random Forest Regression

[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Libraries](https://img.shields.io/badge/libs-numpy%20|%20pandas%20|%20matplotlib-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 Project Overview

This project implements an **Air Quality Index (AQI) prediction model** using a **Random Forest Regression algorithm built entirely from scratch** — without using any machine learning libraries like scikit-learn.  
The implementation uses only **NumPy**, **Pandas**, and **Matplotlib** to demonstrate fundamental machine learning concepts and build a functional model for predicting AQI based on sensor data.

---

## 📊 Dataset

The dataset consists of hourly sensor readings related to air pollutants and weather conditions:

| Feature | Description                        |
|---------|----------------------------------|
| Date    | Date of measurement              |
| Time    | Time of measurement              |
| CO      | Carbon Monoxide concentration    |
| PT08S1  | Tin Oxide sensor response (CO)   |
| NMHC    | Non-methane hydrocarbons         |
| C6H6    | Benzene concentration            |
| PT08S2  | Sensor response (NMHC)            |
| NOx     | Nitrogen Oxides concentration    |
| PT08S3  | Sensor response (NOx)             |
| NO2     | Nitrogen Dioxide concentration   |
| PT08S4  | Sensor response (NO2)             |
| PT08S5  | Sensor response (O3)              |
| T       | Temperature (°C)                  |
| RH      | Relative Humidity (%)             |
| AH      | Absolute Humidity                 |

> **Note:** The target variable, AQI, is derived from these pollutant measurements.

---

## 🧠 Algorithm: Random Forest Regression (From Scratch)

### How It Works:

- **Decision Trees:**
  - Built by recursively splitting data based on the best feature and threshold.
  - Splits minimize the **Mean Squared Error (MSE)** to reduce prediction error.
  
- **Random Forest:**
  - Ensemble method combining multiple decision trees.
  - Each tree trained on a **bootstrapped sample** of the data (sampling with replacement).
  - Predictions from all trees are averaged to improve accuracy and reduce overfitting.

---

## 📈 Features Implemented

- Data preprocessing using **Pandas** (handling missing values, feature selection)
- Decision tree training with:
  - Recursive binary splits
  - MSE-based split criterion
- Bootstrapped sampling for tree diversity
- Prediction aggregation across multiple trees
- Model evaluation with **MSE** and **RMSE**
- Visualization of actual vs predicted AQI values using **Matplotlib**

---

## 💻 How to Use

### 1. Clone the repository

```bash
git clone https://github.com/your-username/aqi-prediction.git
cd aqi-prediction
