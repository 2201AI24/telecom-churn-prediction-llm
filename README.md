# Telecom Customer Churn Predictor with LLM Explanation

This project is an interactive web application that predicts whether a telecom customer will churn and provides a plain-language explanation for the prediction using a Large Language Model (LLM).

---

## 📌 Overview

The application takes customer information as input, uses a trained XGBoost model to predict churn, and then uses an LLM (via OpenRouter API) to explain the prediction in simple terms.

---

## 🚀 Features

* Predict customer churn using a machine learning model (XGBoost)
* Collect user input through a friendly Streamlit interface
* Provide human-readable explanations using DeepSeek LLM (Qwen3-8B)
* Visualize processed input data and prediction confidence
* Secure API key handling via `.env` file

---

## 🛠️ Technologies Used

* Python
* Streamlit
* pandas & NumPy
* scikit-learn
* XGBoost
* OpenRouter API (LLM: DeepSeek Qwen3-8B)
* joblib

---

## 🧠 Model Details

The XGBoost model is trained on telecom customer data with features such as:

* Demographics (Gender, SeniorCitizen, Partner, Dependents)
* Account details (Tenure, Contract, PaperlessBilling, PaymentMethod)
* Services (PhoneService, InternetService, StreamingTV, etc.)
* Charges (MonthlyCharges, TotalCharges)

---

## 💡 Explanation Engine

The app generates a detailed, easy-to-understand explanation using the DeepSeek LLM hosted on OpenRouter. It interprets key factors like tenure, contract type, service bundles, and payment method to justify the prediction.

---

## 📷 Sample Output

* **Prediction**: Will the customer churn? No
* **Confidence**: 0.45
* **LLM Explanation**: Highlights customer's strong service engagement, non-senior status, regular payment, and bundled features as reasons for non-churn.

---

## 🔐 Environment Variables

Create a `.env` file in your project root:

```
OPENROUTER_API_KEY=your-api-key-here
```

---

## 📦 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/telecom-churn-prediction-llm.git
   cd telecom-churn-prediction-llm
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Add your `.env` file with the OpenRouter API key.
4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## 📁 Project Structure

```
├── app.py
├── model_training.ipynb  # Colab notebook
├── xgb_churn_model.pkl
├── feature_columns.pkl
├── label_encoders.pkl
├── .env
├── requirements.txt
└── README.md
```

---

## 👨‍💻 Author

M. Umesh Chandra
BTech Artificial Intelligence and Data Science (Batch 2022)
Project: Telecom Churn Prediction + LLM Explanation

---

## 📄 License

This project is for educational use only.
