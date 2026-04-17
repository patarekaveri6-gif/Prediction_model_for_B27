import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from streamlit_lottie import st_lottie

# --- PAGE CONFIG ---
st.set_page_config(page_title="Salary Predictor Pro", page_icon="💰", layout="centered")

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
    }
    .main-card {
        padding: 30px;
        border-radius: 20px;
        background-color: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- SAFE ANIMATION LOADER ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Using a high-quality finance animation
lottie_finance = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ai9m89ub.json")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        with open('model (3).pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file 'model (3).pkl' not found. Please ensure it is in the same directory.")
        return None

model = load_model()

# --- MAIN UI ---
with st.container():
    col1, col2 = st.columns([3, 2])
    with col1:
        st.title("Salary Estimator")
        st.markdown("### Predict your earning potential based on industry experience.")
    with col2:
        if lottie_finance:
            st_lottie(lottie_finance, height=180, key="money")
        else:
            st.title("💸")

st.write("---")

# --- PREDICTION SECTION ---
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    st.subheader("Professional Details")
    years_exp = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0, step=0.5)
    
    if st.button("Calculate Expected Salary"):
        if model:
            # Prepare input as a 2D array for the Linear Regression model
            features = np.array([[years_exp]])
            prediction = model.predict(features)
            
            # Formatting the result
            salary = prediction[0]
            
            st.write("---")
            st.balloons()
            st.success(f"### Estimated Annual Salary: ${salary:,.2f}")
            
            # Progress bar for visual flair
            st.progress(min(int(years_exp * 2), 100))
        else:
            st.error("Model failed to load.")
            
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.sidebar.markdown("### About this Model")
st.sidebar.info("""
This application uses a **Linear Regression** model to calculate salary estimates based on historical data trends.
""")
