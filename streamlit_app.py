import streamlit as st
import pandas as pd
import numpy as np
from model import preprocess_and_train

# Page configuration
st.set_page_config(
    page_title="Cinema Ticket Predictor",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Cinema Ticket Sales Predictor")
st.write("This application uses a LightGBM ML model to predict the number of tickets likely to be sold for a movie screening.")

# Cache the model training so it doesn't run every time a user changes an input slider
@st.cache_resource
def get_trained_model():
    with st.spinner("Downloading Kaggle dataset and training the AI model... Please wait..."):
        model, feature_names, metrics = preprocess_and_train()
    return model, feature_names, metrics

# Initialize model pipeline
try:
    model, feature_names, metrics = get_trained_model()
    
except Exception as e:
    st.error(f"Failed to load dataset/model. Error: {e}")
    st.stop()

# Layout: Split into sidebar metrics and main prediction engine
col1, col2 = st.columns([1, 2])



with col2:
    st.header("🔮 Predict New Screening Sales")
    st.write("Adjust the features of your upcoming screening below to get a real-time sales forecast:")
    
    # We build input widgets dynamically based on the dataset features
    # Standard columns in this dataset typically include: ticket_price, capacity, cinema_code, month, day, etc.
    input_data = {}
    
    # Create columns inside the prediction panel for a clean layout
    ui_cols = st.columns(2)
    
    for i, feature in enumerate(feature_names):
        # Alternate inputs across the 2 UI columns
        with ui_cols[i % 2]:
            # Customize controls based on common dataset features
            if 'price' in feature.lower():
                input_data[feature] = st.slider(f"Ticket Price (Normalized/Value)", 0.0, 100.0, 15.0)
            elif 'capacity' in feature.lower():
                input_data[feature] = st.slider(f"Theater Capacity", 10, 500, 150)
            elif 'month' in feature.lower():
                input_data[feature] = st.selectbox(f"Month Number", list(range(1, 13)), index=5)
            elif 'day' in feature.lower():
                input_data[feature] = st.selectbox(f"Day of Month", list(range(1, 32)), index=14)
            else:
                # Fallback slider for remaining numerical/encoded ID columns (like cinema_code, film_code)
                input_data[feature] = st.number_input(f"{feature}", value=1.0, step=1.0)

    # Trigger Prediction
    if st.button("Calculate Expected Ticket Sales", type="primary"):
        # Convert user inputs into a DataFrame format matching the model's structure
        input_df = pd.DataFrame([input_data])
        
        # Ensure columns are in the exact order the model expects
        input_df = input_df[feature_names]
        
        # Generate prediction
        prediction = model.predict(input_df)[0]
        
        # Guard against edge-case negative numbers from regression models
        predicted_tickets = max(0, int(np.round(prediction)))
        
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        st.metric(label="Predicted Tickets Sold", value=f"{predicted_tickets} Tickets")
        
        # Contextual logic based on theater capacity input
        capacity_key = [c for c in feature_names if 'capacity' in c.lower()]
        if capacity_key:
            cap = input_data[capacity_key[0]]
            fill_rate = (predicted_tickets / cap) * 100
            st.info(f"Estimated Theater Occupancy Rate: **{min(fill_rate, 100.0):.1f}%**")