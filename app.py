import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pydantic import BaseModel, ValidationError, field_validator

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Dynamic Pricing Engine", page_icon="🏙️", layout="wide")
st.title("🏙️ Short-Term Rental Dynamic Pricing & Strategy Engine")
st.markdown("""
**Domain:** PropTech / Real Estate  
**Algorithm:** XGBoost Regressor with SHAP Explainability  
This engine doesn't just predict the optimal nightly rate for a property—it validates business inputs, prevents data drift, and recommends an actionable market strategy based on current booking windows and competitor density.
""")

# --- 2. PRODUCTION GUARDRAILS (PYDANTIC VALIDATION) ---
class PropertyInput(BaseModel):
    bedrooms: int
    bathrooms: float
    guests_included: int
    cleaning_fee: float
    review_scores_rating: float
    days_to_booking: int
    competitor_density: float

    @field_validator('guests_included')
    def check_capacity(cls, v, info):
        bedrooms = info.data.get('bedrooms')
        if bedrooms and v > (bedrooms * 4):
            raise ValueError(f"Fire code violation: Cannot fit {v} guests in {bedrooms} bedrooms.")
        return v
    
    @field_validator('review_scores_rating')
    def check_rating(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Review score must be between 0 and 100.")
        return v

# --- 3. DATA & MODEL PIPELINE ---
@st.cache_resource
def load_data_and_train_model():
    # Simulating a realistic PropTech dataset for immediate runnability
    np.random.seed(42)
    n_samples = 5000
    
    df = pd.DataFrame({
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.uniform(1, 4, n_samples).round(1),
        'guests_included': np.random.randint(1, 10, n_samples),
        'cleaning_fee': np.random.uniform(20, 150, n_samples),
        'review_scores_rating': np.random.uniform(80, 100, n_samples),
        'days_to_booking': np.random.randint(1, 120, n_samples),
        'competitor_density': np.random.uniform(0.1, 1.0, n_samples) # 1.0 = highly saturated market
    })
    
    # Target Variable: Nightly Rate (Engineered to have logical relationships)
    df['NightlyRate'] = (
        50 + 
        (df['bedrooms'] * 40) + 
        (df['bathrooms'] * 20) + 
        (df['guests_included'] * 10) - 
        (df['competitor_density'] * 30) + 
        (df['review_scores_rating'] * 0.5)
    )
    # Add non-linear noise
    df['NightlyRate'] += np.where(df['days_to_booking'] < 7, -20, 15)
    df['NightlyRate'] += np.random.normal(0, 15, n_samples)

    X = df.drop(columns=['NightlyRate'])
    y = df['NightlyRate']

    # Train Model
    model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    
    # Initialize SHAP
    explainer = shap.Explainer(model, X)
    
    return X, model, explainer

X, model, explainer = load_data_and_train_model()

# --- 4. UI / USER INPUT ---
st.sidebar.header("🏡 Property & Market Inputs")
with st.sidebar.form("input_form"):
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
    bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=1.5, step=0.5)
    guests_included = st.number_input("Max Guests", min_value=1, max_value=20, value=4)
    cleaning_fee = st.number_input("Cleaning Fee ($)", min_value=0.0, max_value=300.0, value=50.0)
    review_scores_rating = st.slider("Average Review Score (0-100)", 0.0, 100.0, 95.0)
    days_to_booking = st.slider("Days Until Check-In", 0, 180, 14)
    competitor_density = st.slider("Market Saturation (0=Empty, 1=Saturated)", 0.0, 1.0, 0.7)
    
    submitted = st.form_submit_button("Generate Pricing Strategy")

# --- 5. EXECUTION & BUSINESS LOGIC ---
if submitted:
    try:
        # Validate inputs via Pydantic
        validated_data = PropertyInput(
            bedrooms=bedrooms, bathrooms=bathrooms, guests_included=guests_included,
            cleaning_fee=cleaning_fee, review_scores_rating=review_scores_rating,
            days_to_booking=days_to_booking, competitor_density=competitor_density
        )
        
        # Convert to DataFrame
        input_df = pd.DataFrame([validated_data.model_dump()])
        
        # Predict
        predicted_rate = model.predict(input_df)[0]
        
        # --- BUSINESS STRATEGY LAYER ---
        st.subheader("📊 Market Intelligence Output")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Algorithm Recommended Nightly Rate", value=f"${predicted_rate:,.2f}")
        
        with col2:
            if days_to_booking < 7 and competitor_density > 0.7:
                st.error("**Strategy:** Last-Minute Liquidation. \nMarket is highly saturated and check-in is imminent. Drop price by 10% to secure occupancy.")
                st.metric(label="Adjusted Strategic Rate", value=f"${predicted_rate * 0.9:,.2f}")
            elif days_to_booking > 60:
                st.success("**Strategy:** Premium Hold. \nCheck-in is far out. Increase price by 8% to capture early, less price-sensitive bookers.")
                st.metric(label="Adjusted Strategic Rate", value=f"${predicted_rate * 1.08:,.2f}")
            else:
                st.info("**Strategy:** Market Match. \nHold at algorithm recommended rate.")
                st.metric(label="Adjusted Strategic Rate", value=f"${predicted_rate:,.2f}")

        st.divider()

        # --- SHAP EXPLAINABILITY ---
        st.subheader("🧠 Algorithmic Transparency (SHAP)")
        st.markdown("Understanding the *why* behind the algorithm's baseline recommendation:")
        
        shap_values = explainer(input_df)
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    except ValidationError as e:
        # Catch and display business logic errors beautifully
        st.error("🛑 Business Logic Validation Failed")
        for error in e.errors():
            st.warning(error['msg'])