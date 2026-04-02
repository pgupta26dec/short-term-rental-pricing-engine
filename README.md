# 🏙️ Dynamic Pricing & Market Strategy Engine

An end-to-end Machine Learning application built for the PropTech/Short-Term Rental industry. This engine leverages **XGBoost** to predict optimal nightly rates and layers custom business logic to generate actionable pricing strategies. 

Unlike standard black-box models, this application prioritizes **algorithmic transparency** (via SHAP) and **production-grade data validation** (via Pydantic) to ensure reliable, explainable outputs for business stakeholders.

## ✨ Key Features

* **Algorithmic Pricing:** Utilizes an `XGBoost` Regressor trained on market saturation, booking lead times, property specs, and historical review scores.
* **Explainable AI (XAI):** Integrates `SHAP` (SHapley Additive exPlanations) to generate dynamic waterfall charts, visually explaining exactly how each feature impacts the final predicted price.
* **Strategic Business Logic:** Translates raw ML predictions into actionable market strategies (e.g., "Last-Minute Liquidation" vs. "Premium Hold") based on time-to-booking and competitor density constraints.
* **Strict Data Intake Validation:** Employs `Pydantic` schemas to catch logical anomalies (e.g., occupancy limit violations or invalid review scores) before they reach the inference layer, preventing data drift and model failure.

## 🛠️ Tech Stack

* **Framework:** Streamlit
* **Machine Learning:** XGBoost, Scikit-Learn
* **Explainability:** SHAP, Matplotlib
* **Data Processing:** Pandas, NumPy
* **Data Validation:** Pydantic

## 🚀 Installation & Usage

To run this application locally, follow these steps:

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/proptech-dynamic-pricing.git](https://github.com/your-username/proptech-dynamic-pricing.git)
cd proptech-dynamic-pricing
```

### 2. Install dependencies
It is recommended to use a virtual environment. Install the required packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. Run the application
Launch the Streamlit server:
```bash
streamlit run app.py
```
The application will automatically open in your default web browser at `http://localhost:8501`.

## 📂 Project Structure

```text
proptech-dynamic-pricing/
│
├── app.py                # Main Streamlit application and ML pipeline
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```


## 👨‍💻 Author
**Prashansa Gupta** *Data Scientist & ML Engineer* <br>
[LinkedIn](https://www.linkedin.com/in/prashansa-gupta-india) | [Portfolio](https://pgupta26dec.github.io/)
