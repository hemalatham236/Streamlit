import pandas as pd
import streamlit as st
import datetime
import pickle
import numpy as np

cars_df = pd.read_excel(r"C:\Users\DELL\Desktop\MLOPs\Streamlit\hemalatha\cars24-car-price.xlsx")

st.write(
    """
     # Cars24 Used Car Price Prediction
    """
)
st.dataframe(cars_df.head())

encode_dict = {
    "fuel_type": {'Diesel': 1, 'Petrol': 2, 'CNG': 3, 'LPG': 4, 'Electric': 5},
    "seller_type": {'Dealer': 1, 'Individual': 2, 'Trustmark Dealer': 3},
    "transmission_type": {'Manual': 1, 'Automatic': 2}
}

def model_pred(year, seller_type, fuel_type, transmission_type, engine, seats):
    ## loading the model
    with open(r"C:\Users\DELL\Desktop\MLOPs\Streamlit\hemalatha/car_pred", 'rb') as file:
        reg_model = pickle.load(file)

        input_features = np.array([year, seller_type, 4000, fuel_type, transmission_type, 19.70, engine, 86.30, seats])
        input_features = input_features.reshape(1, -1)
        return reg_model.predict(input_features)

## Formatting and adding dropdowns and sliders
col1, col2 = st.columns(2)

year = col1.selectbox("Select the Year", list(range(1990, 2025)))

seller_type = col2.selectbox("Select the Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])

fuel_type = col1.selectbox("Select the fuel type",
                           ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
engine = col1.slider("Set the Engine Power",
                     500, 5000, step=100)

transmission_type = col2.selectbox("Select the transmission type",
                                   ["Manual", "Automatic"])

seats = col2.selectbox("Enter the number of seats",
                       [4,5,7,9,11])


if (st.button("Predict Price")):
    fuel_type = encode_dict['fuel_type'][fuel_type]
    transmission_type = encode_dict['transmission_type'][transmission_type]
    seller_type = encode_dict['seller_type'][seller_type]

    price = model_pred(year, seller_type, fuel_type, transmission_type, engine, seats)
    st.text("Predicted Price of the car is: " + str(price))

