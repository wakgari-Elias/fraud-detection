import pandas as pd
import logging


# Initialize logger for this module
logger = logging.getLogger(__name__)
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from a DataFrame and prints a summary report.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        pd.DataFrame: The DataFrame with duplicates removed.
    """
    # Calculate initial size
    initial_rows = df.shape[0]
    num_duplicates = df.duplicated().sum()

    # Drop duplicates (returning a new dataframe is safer than inplace=True in functions)
    df_cleaned = df.drop_duplicates()

    # Calculate final size
    final_rows = df_cleaned.shape[0]

    # Print formatted summary
    # Print formatted summary
    logger.info(f"Initial Row Count: {initial_rows:,}")
    logger.info(f"Duplicate Rows Found: {num_duplicates:,}")
    logger.info(f"Rows After Cleaning: {final_rows:,}")

    if initial_rows != final_rows:
        logger.info(f"✅ Removed {initial_rows - final_rows} duplicate rows.")
    else:
        logger.info("✅ No duplicates found.")
        
    return df_cleaned




def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows with missing values from the DataFrame and prints a summary.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with missing values removed.
    """
    # Calculate initial size
    initial_rows = df.shape[0]
    
    # Drop missing values (We assign to a new variable to avoid side effects)
    df_cleaned = df.dropna()
    
    # Calculate final size
    remaining_rows = df_cleaned.shape[0]
    
    # Check if any nulls remain (Verification)
    total_nulls = df_cleaned.isnull().sum().sum()

    # Print Summary
    # Print Summary
    if total_nulls == 0:
        logger.info(f"✅ No missing values detected. Dataset is clean.")
        if initial_rows != remaining_rows:
            logger.info(f"   (Removed {initial_rows - remaining_rows} rows containing null values)")
    else:
        # This theoretically shouldn't happen after dropna(), but good for sanity checking
        logger.warning(f"⚠️ Warning: {total_nulls} missing values remain.")
        
    return df_cleaned