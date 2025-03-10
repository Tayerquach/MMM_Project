import argparse
import pandas as pd
import numpy as np
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
from config import data_generator_params
from sklearn.preprocessing import MaxAbsScaler

def data_generator(start_date, periods, channels, spend_scalar, adstock_alphas, saturation_lamdas, betas, freq="W", random_state=42):
    '''
    Generates a synthetic dataset for a MMM with trend, seasonality, and channel-specific contributions.

    Args:
        start_date (str or pd.Timestamp): The start date for the generated time series data.
        periods (int): The number of time periods (e.g., days, weeks) to generate data for.
        channels (list of str): A list of channel names for which the model will generate spend and conversion data.
        spend_scalar (list of float): Scalars that adjust the raw spend for each channel to a desired scale.
        adstock_alphas (list of float): The adstock decay factors for each channel, determining how much past spend influences the current period.
        saturation_lamdas (list of float): Lambda values for the logistic saturation function, controlling the saturation effect on each channel.
        betas (list of float): The coefficients for each channel, representing the contribution of each channel's impact on conversion.

    Returns:
        pd.DataFrame: A DataFrame containing the generated time series data, including organic, conversion, and channel-specific metrics.
    '''
    np.random.seed(random_state)
    # 0. Create time dimension
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
    df = pd.DataFrame({'date': date_range})
    
    # 1. Add trend component with some growth
    df["trend"]= (np.linspace(start=0.0, stop=20, num=periods) + 5) ** (1 / 8) - 1
    
    # 2. Add seasonal component with oscillation around 0
    if freq=="W":
        df["seasonality"] = 0.1 * np.sin(2 * np.pi * df.index / 52)
    elif freq=="D":
        df["seasonality"] = 0.1 * np.sin(2 * np.pi * df.index / 365)
    
    # 3. Multiply trend and seasonality to create overall organic with noise
    df["organic"] = df["trend"] * (1 + df["seasonality"]) + np.random.normal(loc=0, scale=0.10, size=periods)
    df["organic"] = df["organic"] * 1000
    
    # 4. Create proxy for organic, which is able to follow organic but has some noise added
    df["organic_proxy"] = np.abs(df["organic"]* np.random.normal(loc=1, scale=0.10, size=periods))
    
    # 5. Initialize sales based on organic
    df["conversion"] = df["organic"]
    
    # 6. Loop through each channel and add channel-specific contribution
    for i, channel in enumerate(channels):
        
        # Create raw channel spend, following organic with some random noise added
        df[f"{channel}_spend_raw"] = df["organic"] * spend_scalar[i]
        df[f"{channel}_spend_raw"] = np.abs(df[f"{channel}_spend_raw"] * np.random.normal(loc=1, scale=0.30, size=periods))
               
        # Scale channel spend
        channel_transformer = MaxAbsScaler().fit(df[f"{channel}_spend_raw"].values.reshape(-1, 1))
        df[f"{channel}_spend"] = channel_transformer.transform(df[f"{channel}_spend_raw"].values.reshape(-1, 1))
        
        # Apply adstock transformation
        df[f"{channel}_adstock"] = geometric_adstock(
            x=df[f"{channel}_spend"].to_numpy(),
            alpha=adstock_alphas[i],
            l_max=8, normalize=True
        ).eval().flatten()
        
        # Apply saturation transformation
        df[f"{channel}_saturated"] = logistic_saturation(
            x=df[f"{channel}_adstock"].to_numpy(),
            lam=saturation_lamdas[i]
        ).eval()
        
        # Calculate contribution to conversion
        df[f"{channel}_conversion"] = df[f"{channel}_saturated"] * betas[i]
        
        # Add the channel-specific contribution to conversion
        df["conversion"] += df[f"{channel}_conversion"]

    channels_spend_columns = [channel + "_spend_raw" for channel in channels]
    db = df[["date", "organic", "organic_proxy"] + channels_spend_columns + ["conversion"]]

    # Introduce duplicate rows randomly
    duplicate_rows = db.sample(frac=0.05, random_state=42)  # 5% duplicate rows
    db = pd.concat([db, duplicate_rows]).reset_index(drop=True)

    # Introduce null values in spend of some channels
    rows_to_null = db.sample(frac=0.05, random_state=42).index  # 5% missing rows

    for row in rows_to_null:
        col_to_null = np.random.choice(channels_spend_columns)  # Randomly pick one column
        db.loc[row, col_to_null] = np.nan  # Set only one column to NaN per row
    
    return db

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Simulate MMM data')
    parser.add_argument('-start_date', type=str, help='The start date for the generated time series data (e.g., 2021-01-01).')
    parser.add_argument('-periods', type=int, help='The number of time periods (e.g., days, weeks) to generate data for.')
    parser.add_argument('-freq', type=str, help='Dayily or Weekly data')

    # Add the arguments to the parser
    args = parser.parse_args()

    start_date = args.start_date
    periods = args.periods
    freq = args.freq
    channels = data_generator_params["channels"]
    adstock_alphas = data_generator_params["adstock_alphas"]
    saturation_lamdas = data_generator_params["saturation_lamdas"]
    betas = data_generator_params["betas"]
    spend_scalars = data_generator_params["spend_scalars"]

    # Generate data
    df = data_generator(start_date, periods, channels, spend_scalars, adstock_alphas, saturation_lamdas, betas, freq=freq)
    # Convert int for conversion and organic
    df['organic'] = df['organic'].astype(int)
    df['conversion'] = df['conversion'].astype(int)
    # Save data
    df.to_csv(f"data/{freq}_raw_three_year_data.csv", index=False)