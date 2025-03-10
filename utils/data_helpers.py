import pandas as pd 

def train_test_split_time_series(df, test_size=0.3):
    """
    Splits a time series DataFrame into train and test sets while preserving time order.

    Parameters:
        df (pd.DataFrame): Time series DataFrame with a datetime index.
        test_size (float): Proportion of the dataset to use as the test set (e.g., 0.3 for 30%).

    Returns:
        pd.DataFrame, pd.DataFrame: Train and test DataFrames.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    split_index = int(len(df) * (1 - test_size))  # Determine the split index
    train = df.iloc[:split_index]  # Take the first (1 - test_size)% for training
    test = df.iloc[split_index:]   # Take the last test_size% for testing
    start_predicted_index = split_index
    end_predicted_index = df.shape[0]

    return train, test, start_predicted_index, end_predicted_index

