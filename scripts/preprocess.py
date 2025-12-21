import sys
import os
import logging
import pandas as pd
from pathlib import Path

# --- 1. Setup Logging ---
# This configures the logger to show time, level (INFO/ERROR), and the message.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Print to console
        # You could add a FileHandler here to write to a log file
    ]
)
logger = logging.getLogger(__name__)

# --- 2. Path Setup ---
# Robust way to find the project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import custom modules after setting path
try:
    from src.data_cleaning import remove_duplicates, remove_missing_values
    from src.data_processing import map_ips_to_countries
    from src.data_preprocessing import engineer_features
except ImportError as e:
    logger.error(f"Failed to import src modules: {e}")
    sys.exit(1)

def validate_schema(df: pd.DataFrame, required_columns: list, dataset_name: str):
    """
    Checks if the dataframe contains the required columns.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        error_msg = f"Missing columns in {dataset_name}: {missing}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def load_and_clean_fraud_data(
    fraud_path: Path,
    ip_path: Path
) -> pd.DataFrame:
    """
    Full cleaning + feature engineering for Fraud_Data.csv
    """
    logger.info("Starting Fraud_Data preprocessing pipeline...")

    # 1. Load Data with Error Handling
    try:
        if not fraud_path.exists():
            raise FileNotFoundError(f"Fraud data not found at {fraud_path}")
        if not ip_path.exists():
            raise FileNotFoundError(f"IP data not found at {ip_path}")

        df = pd.read_csv(fraud_path)
        ip_df = pd.read_csv(ip_path)
        logger.info(f"Loaded Fraud_Data: {df.shape}, IP_Data: {ip_df.shape}")

    except Exception as e:
        logger.error(f"Error loading raw fraud data: {e}")
        raise

    # 2. Schema Validation
    # Basic check to ensure we have critical columns before processing
    expected_cols = ['user_id', 'signup_time', 'purchase_time', 'ip_address']
    validate_schema(df, expected_cols, "Fraud_Data")

    # 3. Processing Steps
    try:
        df = remove_duplicates(df)
        df = remove_missing_values(df)
        
        logger.info("Mapping IP addresses to countries...")
        df = map_ips_to_countries(df, ip_df)
        
        logger.info("Engineering features...")
        df = engineer_features(df)
        
    except Exception as e:
        logger.error(f"Error during Fraud_Data transformation: {e}")
        raise

    logger.info(f"✅ Fraud_Data preprocessing complete. Final Shape: {df.shape}")
    return df


def load_and_clean_creditcard_data(path: Path) -> pd.DataFrame:
    """
    Minimal cleaning for creditcard.csv
    """
    logger.info("Starting creditcard.csv preprocessing pipeline...")

    try:
        if not path.exists():
            raise FileNotFoundError(f"Credit card data not found at {path}")

        df = pd.read_csv(path)
        logger.info(f"Loaded creditcard.csv: {df.shape}")
        
        # Schema check
        validate_schema(df, ['Time', 'Amount', 'Class'], "creditcard.csv")

        # Deduplication
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - df.shape[0]
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows.")

        # Missing Value Check
        if df.isnull().sum().sum() > 0:
            logger.warning("Missing values found in creditcard data. Dropping rows...")
            df = df.dropna()
        
    except Exception as e:
        logger.error(f"Error processing creditcard data: {e}")
        raise

    logger.info(f"✅ creditcard.csv preprocessing complete. Final Shape: {df.shape}")
    return df


def main():
    """
    Main execution entry point.
    """
    # Define paths using pathlib
    data_raw = project_root / 'data' / 'raw'
    data_processed = project_root / 'data' / 'processed'
    
    # Ensure output directory exists
    data_processed.mkdir(parents=True, exist_ok=True)

    try:
        # Process Fraud Data
        fraud_df = load_and_clean_fraud_data(
            fraud_path=data_raw / 'Fraud_Data.csv',
            ip_path=data_raw / 'IpAddress_to_Country.csv'
        )
        fraud_output = data_processed / 'fraud_data_engineered.csv'
        fraud_df.to_csv(fraud_output, index=False)
        logger.info(f"Saved to {fraud_output}")

        # Process Credit Card Data
        cc_df = load_and_clean_creditcard_data(
            path=data_raw / 'creditcard.csv'
        )
        cc_output = data_processed / 'creditcard_processed.csv'
        cc_df.to_csv(cc_output, index=False)
        logger.info(f"Saved to {cc_output}")
        
        logger.info("🚀 All datasets processed successfully.")

    except Exception as e:
        logger.critical(f"Preprocessing pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()