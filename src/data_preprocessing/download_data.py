
import sqlite3
import requests
import zipfile
import io
import pandas as pd
import numpy as np

def download_and_process_data():
    """
    Downloads and preprocesses the household power consumption dataset.
    """
    # URL of the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"

    # Download the zip file
    print("Downloading data...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # Extract and read the data
    print("Extracting and reading data...")
    with z.open("household_power_consumption.txt") as f:
        df = pd.read_csv(f, sep=';', low_memory=False)

    # --- Data Preprocessing ---
    print("Preprocessing data...")

    # 1. Combine Date and Time into a single datetime column
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df = df.set_index('datetime')
    df = df.drop(['Date', 'Time'], axis=1)

    # 2. Handle missing values (represented as '?')
    df = df.replace('?', np.nan)

    # 3. Convert columns to numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])

    # 4. Resample to hourly data to make the dataset more manageable
    # We will use the mean for resampling.
    df_resampled = df.resample('h').mean()

    # 5. Handle missing values in resampled data (e.g., using forward fill)
    df_resampled = df_resampled.ffill()

    # --- Save the data to SQLite ---
    db_path = "data/energy_data.db"
    conn = sqlite3.connect(db_path)

    print(f"Saving processed data to SQLite database: {db_path}...")
    df_resampled.to_sql('energy_data', conn, if_exists='replace', index=True, index_label='datetime')

    conn.close()
    print("Data download and processing complete.")

if __name__ == "__main__":
    download_and_process_data()
