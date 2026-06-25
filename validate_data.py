#!/usr/bin/env python3
"""
Data validation script for Energy Demand Forecasting project.
"""

import pandas as pd
import numpy as np
import os

def validate_data():
    """Validate the processed data for quality and consistency."""
    print("Validating Energy Demand Forecasting Data...")
    print("=" * 50)
    
    # Check if processed data exists
    data_path = "data/processed/household_power_consumption_processed.csv"
    if not os.path.exists(data_path):
        print("❌ Processed data file not found!")
        print("   Run: python src/data_preprocessing/download_data.py")
        return False
    
    try:
        # Load data
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        print(f"✅ Data loaded successfully - Shape: {df.shape}")
        
        # Check data types
        print(f"✅ Data types: {df.dtypes.to_dict()}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() == 0:
            print("✅ No missing values found")
        else:
            print(f"❌ Missing values found:")
            for col, count in missing_values[missing_values > 0].items():
                print(f"   {col}: {count}")
        
        # Check for infinite values
        inf_values = np.isinf(df.select_dtypes(include=[np.number])).sum()
        if inf_values.sum() == 0:
            print("✅ No infinite values found")
        else:
            print(f"❌ Infinite values found:")
            for col, count in inf_values[inf_values > 0].items():
                print(f"   {col}: {count}")
        
        # Check target column
        target_col = 'Global_active_power'
        if target_col in df.columns:
            print(f"✅ Target column '{target_col}' found")
            
            # Check for negative values (unusual for power consumption)
            negative_count = (df[target_col] < 0).sum()
            if negative_count == 0:
                print("✅ No negative values in target column")
            else:
                print(f"⚠️  {negative_count} negative values found in target column")
            
            # Check for zero values
            zero_count = (df[target_col] == 0).sum()
            print(f"ℹ️  {zero_count} zero values in target column")
            
            # Basic statistics
            print(f"✅ Target column statistics:")
            print(f"   Min: {df[target_col].min():.4f}")
            print(f"   Max: {df[target_col].max():.4f}")
            print(f"   Mean: {df[target_col].mean():.4f}")
            print(f"   Std: {df[target_col].std():.4f}")
        else:
            print(f"❌ Target column '{target_col}' not found!")
            return False
        
        # Check datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            print("✅ Datetime index is valid")
            print(f"   Date range: {df.index.min()} to {df.index.max()}")
            
            # Check for duplicate timestamps
            duplicate_times = df.index.duplicated().sum()
            if duplicate_times == 0:
                print("✅ No duplicate timestamps")
            else:
                print(f"❌ {duplicate_times} duplicate timestamps found")
        else:
            print("❌ Index is not a datetime index!")
            return False
        
        # Check data frequency
        time_diff = df.index.to_series().diff().dropna()
        most_common_diff = time_diff.mode().iloc[0]
        print(f"✅ Most common time difference: {most_common_diff}")
        
        print("=" * 50)
        print("✅ Data validation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error validating data: {e}")
        return False

if __name__ == "__main__":
    validate_data()