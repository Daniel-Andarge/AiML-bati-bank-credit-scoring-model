import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from category_encoders.woe import WOEEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder

def calculate_rfms_score(df):
    """
    Calculates the Recency, Frequency, Monetary, and Standard Deviation (RFMS) features
    for a given Pandas DataFrame 'df' and assigns a 'Good', 'Average', or 'Bad' label based on the RFMS_Score.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the necessary columns.
    
    Returns:
    pandas.DataFrame: The input DataFrame with the calculated RFMS features, the RFMS_Score, the 'RFMS_Segment' column, and the 'Assessment_Binary' column.
    """

    df = df.copy()
    
    # Calculate Recency, Frequency, Monetary, and Standard Deviation
    df['Recency'] = (df['TransactionYear'] * 365 + df['TransactionMonth'] * 30 + df['TransactionDay']) - (df.groupby('AccountId')[['TransactionYear', 'TransactionMonth', 'TransactionDay']].transform('max')).sum(axis=1)
    df['Frequency'] = df.groupby('AccountId')['TransactionId'].transform('count')
    df['Monetary'] = df.groupby('AccountId')['Amount'].transform('sum')
    df['StdDev'] = df.groupby('AccountId')['Amount'].transform('std')
    
    # Standardize the RFMS features
    scaler = StandardScaler()
    df[['Recency', 'Frequency', 'Monetary', 'StdDev']] = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary', 'StdDev']])
    

    
    
    # Calculate RFMS score
    df['RFMS_Score'] = df['Recency'] + df['Frequency'] + df['Monetary'] + df['StdDev']
    
    # Bin the RFMS_Score values
    df['RFMS_bin'] = pd.qcut(df['RFMS_Score'], q=3)

    # Calculate the central tendency and variability
    mean_rfms = df['RFMS_Score'].mean()
    median_rfms = df['RFMS_Score'].median()
    std_rfms = df['RFMS_Score'].std()

    print(f"Mean RFMS Score: {mean_rfms:.2f}")
    print(f"Median RFMS Score: {median_rfms:.2f}")
    print(f"Standard Deviation of RFMS Scores: {std_rfms:.2f}")

    # Determine the "Good" and "Bad" thresholds
    good_rfms_threshold = median_rfms
    bad_rfms_threshold = median_rfms - std_rfms

    # Create the "Assessment" column
    df['Assessment'] = np.where(df['RFMS_Score'] >= good_rfms_threshold, 'Good', 'Bad')
    
    # Create the "Assessment_Binary" column
    df['Assessment_Binary'] = np.where(df['Assessment'] == 'Good', 1, 0)
    
    return df





def preprocess_data(df):
    """
    Preprocess the input DataFrame by encoding categorical features and scaling numerical features.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the necessary columns.
    
    Returns:
    pandas.DataFrame: The preprocessed DataFrame.
    """
    # Remove the specified fields
    df = df[['RFMS_Score', 'RFMS_bin', 'Amount', 'Value', 'FraudResult', 'TotalTransactionAmount',
             'AverageTransactionAmount', 'TransactionCount', 'StdTransactionAmount', 'TransactionHour', 'TransactionDay',
             'TransactionMonth', 'TransactionYear', 'Recency', 'Frequency', 'Monetary', 'StdDev', 'Assessment_Binary']]
    
    # Encode categorical features
    le = LabelEncoder()
    df['RFMS_bin'] = le.fit_transform(df['RFMS_bin'])
    
    # Scale numerical features
    scaler = StandardScaler()
    num_cols = ['RFMS_Score', 'Recency', 'Frequency', 'Monetary', 'StdDev']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Add the WoE feature
    woe_encoder = WOEEncoder(cols=['RFMS_bin'])
    df['RFMS_bin_woe'] = woe_encoder.fit_transform(df['RFMS_bin'], df['Assessment_Binary'])
    
    return df

