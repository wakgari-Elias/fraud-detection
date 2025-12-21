# src/data_processing.py
import pandas as pd
import numpy as np
import logging


# Initialize logger for this module
logger = logging.getLogger(__name__)
def map_ips_to_countries(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps IP addresses in the fraud dataset to countries using the IP lookup table.
    Handles integer overflow, sorting, and out-of-range IPs.

    Args:
        fraud_df (pd.DataFrame): The dataframe containing 'ip_address'.
        ip_df (pd.DataFrame): The dataframe mapping IP ranges to countries.

    Returns:
        pd.DataFrame: fraud_df with a new 'country' column.
    """
    # 1. Type Casting to prevent Integer Overflow (Negative numbers issue)
    # We use .copy() to ensure we don't modify the original dataframes outside the function unintentionally
    fraud_df = fraud_df.copy()
    ip_df = ip_df.copy()

    logger.info("Converting IP addresses to int64...")
    fraud_df['ip_address'] = fraud_df['ip_address'].astype('float').astype('int64')
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype('float').astype('int64')
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype('float').astype('int64')

    # 2. Sort (Required for merge_asof)
    ip_df = ip_df.sort_values('lower_bound_ip_address')
    fraud_df = fraud_df.sort_values('ip_address')

    # 3. Perform the range merge
    print("Merging IP data (this may take a moment)...")
    fraud_df = pd.merge_asof(
        fraud_df,
        ip_df,
        left_on='ip_address',
        right_on='lower_bound_ip_address',
        direction='backward'
    )

    # 4. Cleanup Logic
    # Case A: Gap or Above Max (IP > matched upper bound)
    is_above_upper_bound = fraud_df['ip_address'] > fraud_df['upper_bound_ip_address']
    
    # Case B: Below Min (No match found, resulting in NaN)
    is_country_nan = fraud_df['country'].isnull()

    # Assign 'Unknown'
    n_unknowns = (is_above_upper_bound | is_country_nan).sum()
    fraud_df.loc[is_above_upper_bound | is_country_nan, 'country'] = 'Unknown'

    # Drop the extra bound columns
    fraud_df.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address','ip_address'], inplace=True)

    logger.info(f"✅ Mapping complete. Found {n_unknowns:,} IPs with unknown countries.")
    
    return fraud_df

