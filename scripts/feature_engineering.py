import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

def create_aggregate_features(df):
    try:
        # Group transactions by customer
        grouped = df.groupby('CustomerId')

        # Calculate aggregate features
        aggregate_features = grouped['Amount'].agg(['sum', 'mean', 'count', 'std']).reset_index()
        aggregate_features.columns = ['CustomerId', 'TotalTransactionAmount', 'AverageTransactionAmount', 'TransactionCount', 'StdTransactionAmount']

        # Merge aggregate features with original dataframe
        df = pd.merge(df, aggregate_features, on='CustomerId', how='left')

        return df

    except Exception as e:
        print("An error occurred:", e)



def extract_time_features(df):
    try:
        # Convert TransactionStartTime to datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

        # Extract time-related features
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year

        return df

    except Exception as e:
        print("An error occurred:", e)


def encode_categorical_variables(df):
    """
    Encodes categorical variables in the input dataframe using Label Encoding.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing the data.

    Returns:
    pandas.DataFrame: The dataframe with the categorical variables encoded and converted to numerical type.
    """
    try:
        # Copy the dataframe to avoid modifying the original
        data = df.copy()

        # Label Encoding
        label_encoder = LabelEncoder()
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = label_encoder.fit_transform(data[col])

        # Convert data types to numerical
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].astype('int64')

        return data
    except Exception as e:
        print(f"Error occurred during encoding categorical variables: {e}")
        return None


def handle_missing_values(df):
    """
    Handles missing values in the input dataframe using imputation or removal.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing the data.

    Returns:
    pandas.DataFrame: The dataframe with missing values handled.
    """
    try:
        # Copy the dataframe to avoid modifying the original
        data = df.copy()

        # Impute missing values with mean
        imputer = SimpleImputer(strategy='mean')
        numerical_cols = data.select_dtypes(include=['int32', 'int64', 'float64']).columns
        data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

        # Remove rows with missing values if they are few
        if data.isnull().sum().sum() / len(data) < 0.05:
            data = data.dropna()

        return data
    except Exception as e:
        print(f"Error occurred during handling missing values: {e}")
        return None

def normalize_and_standardize_numerical_features(df):
    """
    Normalizes and standardizes the numerical features in the input dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing the data.

    Returns:
    pandas.DataFrame: The dataframe with the numerical features normalized and standardized.
    """
    try:
        # Copy the dataframe to avoid modifying the original
        data = df.copy()

        # Normalize numerical features
        numerical_cols = data.select_dtypes(include=['int32', 'int64', 'float64']).columns
        min_max_scaler = MinMaxScaler()
        data[numerical_cols] = min_max_scaler.fit_transform(data[numerical_cols])

        # Standardize numerical features
        standard_scaler = StandardScaler()
        data[numerical_cols] = standard_scaler.fit_transform(data[numerical_cols])

        return data
    except Exception as e:
        print(f"Error occurred during normalization and standardization of numerical features: {e}")
        return None
