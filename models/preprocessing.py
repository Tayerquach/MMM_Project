import argparse
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def detect_outlier(col):
    # Using Interquartile range (IQR)
    sorted(col)
    lower_quartile, upper_quartile = np.percentile(col, [25, 75])
    IQR = upper_quartile - lower_quartile
    lower_range = lower_quartile - (1.5 * IQR)
    upper_range = upper_quartile + (1.5 * IQR)
    return lower_range, upper_range

def replace_missing_values(df):
    # Replace missing values with the average of the preceding and following row
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:  # Apply only to numeric columns
            for index in df[col][df[col].isna()].index:  # Get indices of NaN values
                prev_value = df[col].iloc[index - 1] if index > 0 else np.nan
                next_value = df[col].iloc[index + 1] if index < len(df) - 1 else np.nan
                
                # Compute average only if both values exist
                if not np.isnan(prev_value) and not np.isnan(next_value):
                    df.at[index, col] = (prev_value + next_value) / 2
                elif not np.isnan(prev_value):  # If only previous exists
                    df.at[index, col] = prev_value
                elif not np.isnan(next_value):  # If only next exists
                    df.at[index, col] = next_value
    return df

def clean_data(df):
    # dataframe dimensions
    print(f"This dataframe has {df.shape[0]} rows and {df.shape[1]} columns.")

    # Remove duplicates
    print("Remove duplicates")
    df = df.drop_duplicates(subset=['date'], keep='first').reset_index(drop=True)
    print(f"The dataframe now has {df.shape[0]} rows and {df.shape[1]} columns.")

    # Preprocessing missing values
    print("Replace missing values")
    df = replace_missing_values(df)
    print(f"The dataframe now has {df.shape[0]} rows and {df.shape[1]} columns.")

    # Remove outliers
    for col in df.select_dtypes(include=['number']).columns: # [float or int64]
        lower_range, upper_range = detect_outlier(df[col])
        df[col] = np.where(df[col] > upper_range, upper_range, df[col])
        df[col] = np.where(df[col] < lower_range, lower_range, df[col])

    # Sorted by date and reset index
    print("Sorted")
    df['date'] = pd.to_datetime(df['date'])  # Convert to datetime format if not already
    df = df.sort_values(by='date').reset_index(drop=True)  # Sort and reset index
    df = df.set_index('date')
    print(f"The dataframe now has {df.shape[0]} rows and {df.shape[1]} columns.")

    return df


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Simulate MMM data')
    parser.add_argument('-freq', type=str, help='Dayily or Weekly data')

    # Add the arguments to the parser
    args = parser.parse_args()
    freq = args.freq
    
    # Read csv file into a dataframe
    file_path = f"data/{freq}_raw_three_year_data.csv"
    df = pd.read_csv(file_path)
    # Preprocessing data
    preprocessed_df = clean_data(df)
    # Save data
    preprocessed_df.to_csv(f"data/{freq}_preprocessed_data.csv")