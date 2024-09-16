import streamlit as st
import pickle
import numpy as np

# Load the saved Random Forest model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the saved MinMaxScaler
with open('minmaxscaler.pkl', 'rb') as file1:
    scaler = pickle.load(file1)

# Streamlit UI
st.title("Crop Recommendation System 🌱")

# Input fields for user input
N = st.number_input("Nitrogen Content (N)", min_value=0, max_value=100)
P = st.number_input("Phosphorus Content (P)", min_value=0, max_value=100)
K = st.number_input("Potassium Content (K)", min_value=0, max_value=100)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0)
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100)
ph = st.number_input("pH Value", min_value=0.0, max_value=14.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=300)

# Button to make a prediction
if st.button("Get Recommendation"):
    # Prepare the input data
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Apply MinMaxScaler to the input
    scaled_features = scaler.transform(features)
    
    # Make a prediction using the loaded model
    prediction = model.predict(scaled_features)
    
    # Dictionary of crop predictions
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    # Display the recommendation
    recommended_crop = crop_dict.get(prediction[0], "Unknown Crop")
    st.success(f"The recommended crop is: {recommended_crop}")
