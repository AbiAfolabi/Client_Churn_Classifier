import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re  # Added this import for postal code validation

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
    <h1 style='text-align: center; color: #ff5733; padding: 20px;'>
    IFSSA Client Return Prediction
    </h1>
    <p style='text-align: center; font-size: 1.1rem;'>
    Predict which clients will return within 3 months using statistically validated features
    </p>
    """,
    unsafe_allow_html=True
)

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
    
    # Chi-square test results (from your data)
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
    
    # Convert to dataframe
    chi_df = pd.DataFrame.from_dict(chi2_results, orient='index', columns=['p-value'])
    chi_df['-log10(p)'] = -np.log10(chi_df['p-value'].replace(0, 1e-300))  # Handle zero p-values
    chi_df = chi_df.sort_values('-log10(p)', ascending=False)
    
    # Visualization
    st.markdown("### Feature Significance (-log10 p-values)")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='-log10(p)', y=chi_df.index, data=chi_df, palette="viridis")
    plt.axvline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
    plt.xlabel("Statistical Significance (-log10 p-value)")
    plt.ylabel("Features")
    plt.title("Chi-Square Test Results for Feature Selection")
    st.pyplot(plt)
    
    # Interpretation
    st.markdown("""
    **Key Insights**:
    - All shown features are statistically significant (p < 0.05)
    - Visit frequency metrics are strongest predictors (p ≈ 0)
    - Holiday effects are 10^90 times more significant than chance
    - Postal code explains location-based patterns (p=2.4e-16)
    """)
    
    # Feature correlations (placeholder - replace with your actual data)
    st.markdown("### Feature Relationships")
    try:
        # Generate sample correlation data if real data isn't available
        corr_data = pd.DataFrame(np.random.rand(6, 6), 
                               columns=chi_df.index[:6], 
                               index=chi_df.index[:6])
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0)
        st.pyplot(plt)
    except:
        st.warning("Correlation data not available. Displaying sample visualization.")

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
        pickup_count_last_30_days = st.number_input("Pickups in last 30 days:", min_value=0, value=0)
        weekly_visits = st.number_input("Weekly Visits:", min_value=0, value=3)

    with col2:
        total_dependents_3_months = st.number_input("Total Dependents in Last 3 Months:", min_value=0, value=2)
        time_since_first_visit = st.number_input("Time Since First Visit (days):", min_value=1, max_value=366, value=30)
        pickup_week = st.number_input("Pickup Week (1-52):", min_value=1, max_value=52, value=1)

    # Additional Features
    st.markdown("<h3 style='color: #ff5733;'>Additional Information</h3>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        Holidays = 1 if st.radio("Is this pick-up during a holiday?", ["No", "Yes"]) == "Yes" else 0
    
    with col4:
        # Canadian Postal Code Input with validation
        def validate_postal_code(postal_code):
            if not postal_code:
                return False
            # Remove spaces and convert to uppercase
            clean_code = postal_code.replace(" ", "").upper()
            # Check pattern (A1A1A1)
            if len(clean_code) != 6:
                return False
            pattern = re.compile(r'^[A-Z]\d[A-Z]\d[A-Z]\d$')
            return bool(pattern.match(clean_code))
        
        postal_code = st.text_input("Postal Code (A1A 1A1 format):", 
                                  max_chars=7,
                                  help="Enter Canadian postal code (e.g., M5V 3L9)")
        
        # Validate format
        if postal_code and not validate_postal_code(postal_code):
            st.warning("Please enter a valid Canadian postal code (e.g., A1A 1A1)")

    # Prepare input data
    input_data = pd.DataFrame([[
        weekly_visits,
        total_dependents_3_months,
        pickup_count_last_30_days,
        pickup_count_last_14_days,
        Holidays,
        pickup_week,
        postal_code.replace(" ", "").upper()[:6] if postal_code else "",
        time_since_first_visit
    ]], columns=[
        'weekly_visits',
        'total_dependents_3_months',
        'pickup_count_last_30_days',
        'pickup_count_last_14_days',
        'Holidays',
        'pickup_week',
        'postal_code',
        'time_since_first_visit'
    ])

    # Prediction Button
    if st.button("Predict Return Probability"):
        if not postal_code:
            st.error("Please enter a postal code")
        elif not validate_postal_code(postal_code):
            st.error("Please enter a valid Canadian postal code (format: A1A 1A1)")
        elif model is None:
            st.error("❌ No trained model found.")
        else:
            try:
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[:, 1][0]
                
                st.markdown("<h3 style='color: #ff33aa;'>Prediction Result</h3>", unsafe_allow_html=True)
                st.write(f"🎯 **Predicted Outcome:** {'Will Return' if prediction[0] == 1 else 'Will Not Return'}")
                st.write(f"📊 **Probability of Returning:** {probability:.1%}")
                
                if prediction[0] == 1:
                    st.success("✅ This client is likely to return within 3 months")
                else:
                    st.warning("⚠️ This client is unlikely to return within 3 months")
                    
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")

!pip install streamlit pandas gspread gspread-dataframe

import streamlit as st
import pandas as pd
import gspread
from gspread_dataframe import get_as_dataframe

# Authenticate and connect to Google Sheets (if you're using OAuth)
# gc = gspread.service_account(filename='path_to_your_credentials.json')

gc = gspread.authorize(credentials=None)  # This would use an unauthenticated public access to the sheet

# Alternatively, for a public sheet, you can use the URL or sheet key directly:
sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSXUCJRkYyqkfNFbyjRkB5NyP4pL4Khh00bmHegBZOpFf9BparWuCsxx7-C7m-Uy6DNBn7fSBs21NKi/pubhtml"

# Open the sheet using the URL or the sheet ID
worksheet = gc.open_by_url(sheet_url).sheet1
