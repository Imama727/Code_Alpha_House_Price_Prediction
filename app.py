# Streamlit app for house price prediction

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Set page configuration
st.set_page_config(page_title="House Price Prediction", layout="centered")

# Load the trained model
model = joblib.load('house_price_model.pkl')

# Define a function to make predictions
def predict_house_price(area, bedrooms, bathrooms, stories, parking):
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, parking]], 
                              columns=['area', 'bedrooms', 'bathrooms', 'stories', 'parking'])
    # Make the prediction
    return model.predict(input_data)[0]

# Streamlit UI
st.title("House Price Prediction")
st.write("This app predicts house prices based on various features like area, bedrooms, bathrooms, stories, and parking.")

# Input fields for the features
area = st.slider('Area (in square feet)', 500, 10000, 500)
bedrooms = st.selectbox('Number of Bedrooms', [1, 2, 3, 4, 5])
bathrooms = st.selectbox('Number of Bathrooms', [1, 2, 3])
stories = st.selectbox('Number of Stories', [1, 2, 3, 4])
parking = st.selectbox('Number of Parking Spaces', [0, 1, 2, 3, 4])

# Button to make predictions
if st.button('Predict House Price'):
    # Make prediction based on the user inputs
    predicted_price = predict_house_price(area, bedrooms, bathrooms, stories, parking)
    
    # Display the predicted price
    st.success(f"Predicted House Price: ${predicted_price:,.2f}")

# Display performance metrics (if any)
#st.subheader("Model Performance")
#mse = 25134500  # Example value for Mean Squared Error, replace with actual value
#r2 = 0.85       # Example value for R^2 score, replace with actual value

#st.write(f"Mean Squared Error (MSE): {mse}")
#st.write(f"R^2 Score: {r2}")

# Show a brief description of the model
#st.write("This model was trained using Linear Regression on a dataset of housing features like area, bedrooms, and more.")

# Optional: Add a plot or chart for better visualization (e.g., actual vs predicted prices)
# Example of how to visualize actual vs predicted prices using matplotlib
#import matplotlib.pyplot as plt

# Load the dataset again for visualization
#df = pd.read_csv('Housing.csv')

# Preprocessing: Select features and target variable
#X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
#y = df['price']

# Predict on the full dataset for visualization
#y_pred = model.predict(X)

# Create a scatter plot for actual vs predicted
#fig, ax = plt.subplots()
#ax.scatter(y, y_pred, edgecolors=(0, 0, 0))
#ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
#ax.set_xlabel('Actual Price')
#ax.set_ylabel('Predicted Price')
#ax.set_title('Actual vs Predicted House Prices')

# Display the plot in Streamlit
#st.pyplot(fig)
