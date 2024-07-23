import pandas as pd  # Import the pandas library for data manipulation and analysis
import numpy as np  # Import the numpy library for numerical operations
from sklearn.impute import SimpleImputer, KNNImputer  # Import imputation methods for handling missing data
from sklearn.preprocessing import LabelEncoder, \
    MinMaxScaler  # Import tools for encoding categorical data and scaling features
from sklearn.model_selection import \
    train_test_split  # Import function for splitting the dataset into training and testing sets
from xgboost import XGBClassifier  # Import the XGBoost classifier for machine learning
from imblearn.combine import SMOTETomek  # Import method for handling imbalanced data
import joblib  # Import joblib for saving and loading Python objects


# Load the dataset
def load_data():
    """
    Load the dataset from an Excel file.

    Returns:
        DataFrame: The loaded dataset.
    """
    df = pd.read_excel('/home/yaser/practice/kaggle_projects/E Commerce Dataset.xlsx',
                       sheet_name='E Comm')  # Read the Excel file into a DataFrame
    return df  # Return the DataFrame


# Preprocess the data
def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, encoding categorical variables, and removing outliers.

    Args:
        df (DataFrame): The input dataset.

    Returns:
        DataFrame: The preprocessed dataset.
    """
    # Impute missing values
    df['Tenure'] = df['Tenure'].fillna(
        method='bfill')  # Fill missing values in 'Tenure' column with the next valid observation
    s_imp = SimpleImputer(missing_values=np.nan,
                          strategy='most_frequent')  # Initialize SimpleImputer to fill missing values with the most frequent value
    df['WarehouseToHome'] = s_imp.fit_transform(
        pd.DataFrame(df['WarehouseToHome']))  # Apply SimpleImputer to 'WarehouseToHome' column
    fill_list = df['HourSpendOnApp'].dropna()  # Create a list of non-missing values in 'HourSpendOnApp' column
    df['HourSpendOnApp'] = df['HourSpendOnApp'].fillna(pd.Series(np.random.choice(fill_list, size=len(
        df['HourSpendOnApp'].index))))  # Fill missing values in 'HourSpendOnApp' with random values from fill_list
    df['OrderAmountHikeFromlastYear'] = df['OrderAmountHikeFromlastYear'].fillna(
        method='ffill')  # Fill missing values in 'OrderAmountHikeFromlastYear' with the previous valid observation
    imputer = KNNImputer(n_neighbors=2)  # Initialize KNNImputer to fill missing values based on the nearest neighbors
    df['CouponUsed'] = imputer.fit_transform(df[['CouponUsed']])  # Apply KNNImputer to 'CouponUsed' column
    imputer_2 = KNNImputer(n_neighbors=2)  # Initialize another KNNImputer
    df['OrderCount'] = imputer_2.fit_transform(df[['OrderCount']])  # Apply KNNImputer to 'OrderCount' column
    df['DaySinceLastOrder'] = df[
        'DaySinceLastOrder'].bfill()  # Fill missing values in 'DaySinceLastOrder' with the next valid observation
    df.drop('CustomerID', axis=1, inplace=True)  # Drop the 'CustomerID' column as it is not needed for modeling

    # Encode categorical variables
    for i in df.columns:  # Iterate over each column in the DataFrame
        if df[i].dtype == 'object':  # Check if the column data type is 'object' (categorical)
            le = LabelEncoder()  # Initialize LabelEncoder
            df[i] = le.fit_transform(df[i])  # Transform categorical values to numerical using LabelEncoder
            joblib.dump(le, f'le_{i}.pkl')  # Save the LabelEncoder for future use

    # Handle outliers
    def handle_outliers(df, column_name):
        """
        Remove outliers from a specified column using the IQR method.

        Args:
            df (DataFrame): The input dataset.
            column_name (str): The name of the column to handle outliers for.

        Returns:
            DataFrame: The dataset with outliers removed.
        """
        Q1 = df[column_name].quantile(0.25)  # Calculate the first quartile (Q1) of the column
        Q3 = df[column_name].quantile(0.75)  # Calculate the third quartile (Q3) of the column
        IQR = Q3 - Q1  # Calculate the interquartile range (IQR)
        Upper = Q3 + IQR * 1.5  # Define the upper bound for outliers
        lower = Q1 - IQR * 1.5  # Define the lower bound for outliers
        new_df = df[(df[column_name] > lower) & (
                    df[column_name] < Upper)]  # Filter out rows where the column value is outside the bounds
        return new_df  # Return the filtered DataFrame

    cols_outliers = ['Tenure', 'WarehouseToHome', 'NumberOfAddress', 'DaySinceLastOrder', 'HourSpendOnApp',
                     'NumberOfDeviceRegistered']  # List of columns to handle outliers for
    for col in cols_outliers:  # Iterate over each column in the list
        df = handle_outliers(df, col)  # Apply the handle_outliers function to the column

    return df  # Return the preprocessed DataFrame


# Train the model
def train_model(df):
    """
    Train an XGBoost classifier on the preprocessed dataset.

    Args:
        df (DataFrame): The preprocessed dataset.

    Returns:
        tuple: The trained model and the MinMaxScaler used for scaling.
    """
    X = df.drop('Churn', axis=1)  # Separate the features (X) from the target ('Churn')
    Y = df['Churn']  # Separate the target (Y)
    smt = SMOTETomek(random_state=42)  # Initialize SMOTETomek for handling imbalanced data
    x_over, y_over = smt.fit_resample(X, Y)  # Apply SMOTETomek to balance the dataset
    x_train, x_test, y_train, y_test = train_test_split(x_over, y_over, test_size=0.30,
                                                        random_state=42)  # Split the balanced dataset into training and testing sets
    MN = MinMaxScaler()  # Initialize MinMaxScaler for scaling features
    x_train_scaled = MN.fit_transform(x_train)  # Scale the training features
    x_test_scaled = MN.transform(x_test)  # Scale the testing features
    model = XGBClassifier()  # Initialize the XGBoost classifier
    model.fit(x_train_scaled, y_train)  # Train the model on the scaled training data
    return model, MN  # Return the trained model and the scaler


# Save the model and scaler
def save_model(model, scaler):
    """
    Save the trained model and scaler to disk.

    Args:
        model: The trained model.
        scaler: The MinMaxScaler used for scaling.
    """
    joblib.dump(model, 'model.pkl')  # Save the trained model to a file
    joblib.dump(scaler, 'scaler.pkl')  # Save the scaler to a file


if __name__ == "__main__":
    df = load_data()  # Load the dataset
    df = preprocess_data(df)  # Preprocess the dataset
    model, scaler = train_model(df)  # Train the model
    save_model(model, scaler)  # Save the trained model and scaler