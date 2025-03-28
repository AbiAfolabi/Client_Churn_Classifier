import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import gspread
from gspread_dataframe import get_as_dataframe
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Set page configuration
st.set_page_config(
    layout="wide",
    page_title="IFSSA Return Predictor"
)

# Load and Display Logos
col1, col2, _ = st.columns([0.15, 0.15, 0.7])
with col1:
    st.image("logo1.jpeg", width=120)
with col2:
    st.image("logo2.png", width=120)

# Header
st.markdown(
    """
    <h1 style='text-align: center; color: #ff5733; padding: 20px;' >
    IFSSA Client Return Prediction
    </h1>
    <p style='text-align: center; font-size: 1.1rem;' >
    Predict which clients will return within 3 months using statistically validated features
    </p>
    """,
    unsafe_allow_html=True
)

# ================== Google Sheets Connection (Using OAuth2 Authentication) ==================
# Fetch Data from Google Sheets using OAuth2 Authentication
@st.cache_data
def load_google_sheet():
    # Define the scope and credentials path
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    credentials = None

    # The file token.json stores the user's access and refresh tokens, and is created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        credentials = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            credentials = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(credentials.to_json())

    # Connect to Google Sheets using OAuth2 credentials
    gc = gspread.authorize(credentials)

    # Open the Google Sheet by URL (replace with your actual sheet URL)
    sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQwjh9k0hk536tHDO3cgmCb6xvu6GMAcLUUW1aVqKI-bBw-3mb5mz1PTRZ9XSfeLnlmrYs1eTJH3bvJ/pubhtml"
    spreadsheet = gc.open_by_url(sheet_url)
    worksheet = spreadsheet.get_worksheet(0)

    # Fetch the data as a DataFrame
    sheet_data = get_as_dataframe(worksheet)
    return sheet_data

# Load Google Sheets data
try:
    google_sheet_data = load_google_sheet()
    st.write(google_sheet_data)
except Exception as e:
    st.error(f"Failed to load data from Google Sheets: {e}")

# ================== Navigation ==================
page = st.sidebar.radio(
    "Navigation",
    ["About", "Feature Analysis", "Make Prediction"],
    index=2
)

# ================== About Page ==================
if page == "About":
    st.markdown(""" 
    ## About This Tool
    
    This application helps IFSSA predict which clients are likely to return for services 
    within the next 3 months using machine learning.
    
    ### How It Works
    1. Staff enter client visit information
    2. The system analyzes patterns from historical data
    3. Predictions guide outreach efforts
    
    ### Key Benefits
    - Enhance Response Accuracy
    - Improve Operational Efficiency
    - Streamline Stakeholder Communication
    - Facilitate Informed Decision Making
    - Ensure Scalability and Flexibility
    """)
    
# ================== Feature Analysis ==================
elif page == "Feature Analysis":
    st.markdown("## Statistically Validated Predictors")
    
    chi2_results = {
        'monthly_visits': 0.000000e+00,
        'weekly_visits': 0.000000e+00,
        'total_dependents_3_months': 0.000000e+00,
        'pickup_count_last_90_days': 0.000000e+00,
        'pickup_count_last_30_days': 0.000000e+00,
        'pickup_count_last_14_days': 0.000000e+00,
        'pickup_count_last_7_days': 0.000000e+00,
        'Holidays': 8.394089e-90,
        'pickup_week': 1.064300e-69,
        'postal_code': 2.397603e-16,
        'time_since_first_visit': 7.845354e-04
    }
    
    chi_df = pd.DataFrame.from_dict(chi2_results, orient='index', columns=['p-value'])
    chi_df['-log10(p)'] = -np.log10(chi_df['p-value'].replace(0, 1e-300))
    chi_df = chi_df.sort_values('-log10(p)', ascending=False)
    
    st.markdown("### Feature Significance (-log10 p-values)")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='-log10(p)', y=chi_df.index, data=chi_df, palette="viridis")
    plt.axvline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
    plt.xlabel("Statistical Significance (-log10 p-value)")
    plt.ylabel("Features")
    plt.title("Chi-Square Test Results for Feature Selection")
    st.pyplot(plt)

    st.markdown(""" **Key Insights**:
    - All shown features are statistically significant (p < 0.05)
    - Visit frequency metrics are strongest predictors (p ≈ 0)
    - Holiday effects are 10^90 times more significant than chance
    - Postal code explains location-based patterns (p=2.4e-16)
    """)

# ================== Make Prediction Page ==================
elif page == "Make Prediction":
    st.markdown("<h2 style='color: #33aaff;'>Client Return Prediction</h2>", unsafe_allow_html=True)
    
    # Load Model Function
    def load_model():
        model_path = "RF_model.pkl"
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None

    # Load Model
    model = load_model()
    if model is None:
        st.error("⚠️ No trained model found. Please upload a trained model to 'RF_model.pkl'.")
        st.stop()

    # Input Features Section
    st.markdown("<h3 style='color: #ff5733;'>Client Information</h3>", unsafe_allow_html=True)

    # Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        # Recent Pickup Information
        pickup_count_last_14_days = st.number_input("Pickups in last 14 days:", min_value=0, value=0)
        pickup_count_last_30_days = st.number_input("Pickups in last 30_
