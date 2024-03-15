import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load the pickled model
with open('model_pickle', 'rb') as f:
    model = pickle.load(f)

#Load the final dataset
df_filename = 'house_prices.csv'
df = pd.read_csv(df_filename)

# Streamlit App
st.title('House Price Prediction')
    
# Sidebar inputs
st.sidebar.header('Input Parameters')
latitude = st.sidebar.number_input('Latitude', format='%f')
longitude = st.sidebar.number_input('Longitude', format='%f')
bedrooms = st.sidebar.number_input('Bedrooms', step=1, format='%d')
bathrooms = st.sidebar.number_input('Bathrooms', step=1, format='%d')
parkingSpaces = st.sidebar.number_input('Parking Spaces', step=1, format='%d')
    
    
propertyTypes = df.columns[6:]
propertyType = st.sidebar.selectbox('Property Type', propertyTypes)        
    
# Make prediction
if st.sidebar.button('Predict'):
    # Convert propertyType to one-hot encoding
    propertyType_encoded = np.zeros(len(propertyTypes))
    propertyType_index = np.where(propertyTypes == propertyType)[0][0]
    propertyType_encoded[propertyType_index] = 1
    
    features = np.array([[latitude, longitude, bedrooms, bathrooms, parkingSpaces, *propertyType_encoded]])
    prediction = model.predict(features)
    st.write('Predicted Price:', f"<span style='font-size:24px;'>{prediction[0]}</span>", unsafe_allow_html=True)

