import streamlit as st  # Import the Streamlit library for creating web apps
import pandas as pd  # Import the pandas library for data manipulation
import joblib  # Import the joblib library for loading Python objects

st.title("E-Commerce Churn Prediction")  # Set the title of the Streamlit app

# Load the model and scaler
@st.cache_data  # Cache the function to speed up the app
def load_model():
    model = joblib.load('model.pkl')  # Load the pre-trained model from a file
    scaler = joblib.load('scaler.pkl')  # Load the scaler used during training
    return model, scaler  # Return the model and scaler

# Load the label encoders used during training
@st.cache_data  # Cache the function to speed up the app
def load_label_encoders():
    # label encoders are saved during training
    label_encoders = {}
    label_encoders['PreferredLoginDevice'] = joblib.load('le_PreferredLoginDevice.pkl')  # Load the label encoder for 'PreferredLoginDevice'
    label_encoders['PreferredPaymentMode'] = joblib.load('le_PreferredPaymentMode.pkl')  # Load the label encoder for 'PreferredPaymentMode'
    label_encoders['Gender'] = joblib.load('le_Gender.pkl')  # Load the label encoder for 'Gender'
    label_encoders['PreferedOrderCat'] = joblib.load('le_PreferedOrderCat.pkl')  # Load the label encoder for 'PreferedOrderCat'
    label_encoders['MaritalStatus'] = joblib.load('le_MaritalStatus.pkl')  # Load the label encoder for 'MaritalStatus'

    return label_encoders  # Return the dictionary of label encoders

# Streamlit app
def main():
    model, scaler = load_model()  # Load the model and scaler
    label_encoders = load_label_encoders()  # Load the label encoders

    # Input features in columns
    st.header("Input Features")  # Set the header for the input features section

    col1, col2, col3, col4 = st.columns(4)  # Create four columns for organizing input fields

    with col1:
        features = {}
        features['Tenure'] = st.number_input('Tenure', min_value=0, max_value=100, value=0)  # Input field for 'Tenure'
        features['PreferredLoginDevice'] = st.selectbox('PreferredLoginDevice', ['Mobile Phone', 'Computer'])  # Select box for 'PreferredLoginDevice'
        features['CityTier'] = st.number_input('CityTier', min_value=1, max_value=3, value=None)  # Input field for 'CityTier'
        features['WarehouseToHome'] = st.number_input('WarehouseToHome', min_value=0, max_value=100, value=0)  # Input field for 'WarehouseToHome'

    with col2:
        features['PreferredPaymentMode'] = st.selectbox('PreferredPaymentMode',
                                                        ['Debit Card', 'UPI', 'Cash on Delivery', 'E wallet',
                                                         'Credit Card'])  # Select box for 'PreferredPaymentMode'
        features['Gender'] = st.selectbox('Gender', ['Female', 'Male'])  # Select box for 'Gender'
        features['HourSpendOnApp'] = st.number_input('HourSpendOnApp', min_value=0, max_value=24, value=0)  # Input field for 'HourSpendOnApp'
        features['NumberOfDeviceRegistered'] = st.number_input('NumberOfDeviceRegistered', min_value=0, max_value=10,
                                                               value=1)  # Input field for 'NumberOfDeviceRegistered'
        features['PreferedOrderCat'] = st.selectbox('PreferedOrderCat',
                                                    ['Laptop & Accessory', 'Mobile Phone', 'Others', 'Fashion',
                                                     'Grocery'])  # Select box for 'PreferedOrderCat'

    with col3:
        features['SatisfactionScore'] = st.selectbox('SatisfactionScore', [1, 2, 3, 4, 5])  # Select box for 'SatisfactionScore'
        features['MaritalStatus'] = st.selectbox('MaritalStatus', ['Single', 'Divorced', 'Married'])  # Select box for 'MaritalStatus'
        features['NumberOfAddress'] = st.number_input('NumberOfAddress', min_value=0, max_value=100, value=1)  # Input field for 'NumberOfAddress'
        features['Complain'] = st.selectbox('Complain', [1, 0])  # Select box for 'Complain'
        features['OrderAmountHikeFromlastYear'] = st.number_input('OrderHikeLastYear', min_value=0.0, max_value=100.0, value=1.0)  # Input field for 'OrderAmountHikeFromlastYear'

    with col4:
        features['CouponUsed'] = st.number_input('CouponUsed', min_value=0, max_value=10, value=0)  # Input field for 'CouponUsed'
        features['OrderCount'] = st.number_input('OrderCount', min_value=0, max_value=100, value=0)  # Input field for 'OrderCount'
        features['DaySinceLastOrder'] = st.number_input('DaySinceLastOrder', min_value=0, max_value=365, value=0)  # Input field for 'DaySinceLastOrder'
        features['CashbackAmount'] = st.number_input('CashbackAmount', min_value=0, max_value=10000, value=0)  # Input field for 'CashbackAmount'
    st.write('______________________________')  # Add a horizontal line for visual separation

    # Predict button
    col1, col2 = st.columns(2)  # Create two columns for organizing the predict button and result
    with col1:
        if st.button("Predict"):  # Button to trigger the prediction
                # Convert features dictionary to a DataFrame
                input_data = pd.DataFrame([features])
                for col in input_data.columns:
                    if input_data[col].dtype == 'object':
                        input_data[col] = label_encoders[col].transform(input_data[col])  # Transform categorical features using the corresponding label encoder
                input_data_scaled = scaler.transform(input_data)  # Scale the input data using the scaler
                # Make prediction
                prediction = model.predict(input_data_scaled)  # Predict using the model
    with col2:
        if 'prediction' in locals():  # Check if the prediction variable exists
            # Display the prediction result in the same line
            st.write(f"Prediction Result:  {'Churn' if prediction[0] == 1 else 'No Churn'}")  # Display the prediction result
            st.write(prediction[0])  # Display the raw prediction value


if __name__ == "__main__":
    main()  # Run the main function