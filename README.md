# 💎 Diamond Total Sales Price Prediction App

This project is a **Deep Learning–powered Flask web app** that predicts the **total sales price of a diamond** using a trained neural network model.

The model is trained on **all diamond attributes**, while the app requires only **minimal user input** for a clean user experience.

---

## 🚀 Features

- Deep Learning model (TensorFlow / Keras)
- Advanced preprocessing (scaling + encoding)
- Log-transformed target for better accuracy
- Minimal user inputs (UX-friendly)
- Production-ready Flask backend
- No data leakage
- Kaggle & local compatible

---

## 🧠 Model Details

- **Target**: `total_sales_price`
- **Model Type**: Regression (Neural Network)
- **Loss**: Mean Squared Error (MSE)
- **Metric**: MAE + R²
- **R² Score**: ~0.79 (Good real-world performance)

### Preprocessing
- Numerical → RobustScaler
- Ordinal Categorical → OrdinalEncoder
- Nominal Categorical → OneHotEncoder
- Missing values → Imputed
- Target → `log1p(price)`

---

## 🧾 User Inputs (Frontend)

The app only asks for:

- Carat Weight
- Cut (Shape)
- Color
- Clarity

All other features are **auto-filled in the backend** using dataset statistics.

---

## 📁 Project Structure

