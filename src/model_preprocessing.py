# src/model_preprocessing.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import logging


# Initialize logger for this module
logger = logging.getLogger(__name__)
def build_preprocessor(X: pd.DataFrame):
    """
    Builds a ColumnTransformer for numeric scaling and categorical one-hot encoding.
    
    Args:
        X (pd.DataFrame): Feature DataFrame (no target column).
    
    Returns:
        ColumnTransformer: Unfitted preprocessor.
    """
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    logger.info(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )

    logger.info("✅ Preprocessing pipeline (scaling + encoding) built.")
    return preprocessor




def prepare_data_for_modeling(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "Dataset",
    imbalance_technique: str = "smote",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Complete preprocessing + imbalance handling pipeline.
    """
    logger.info(f"\n=== Preparing {dataset_name} for Modeling ===")

    # 1. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # 2. Build preprocessor (Fit on Train ONLY)
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    logger.info("Before Handling imbalance:")
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train) # Fit and Transform Train
    X_test_processed = preprocessor.transform(X_test)       # Transform Test

    # Convert to DataFrame (Optional, keeps column names)
    feature_names = preprocessor.get_feature_names_out()
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)

    # 3. Imbalance handling (On Processed Train Data)
    logger.info(f"Applying {imbalance_technique.upper()}...")
    
    if imbalance_technique == "smote":
        balancer = SMOTE(random_state=random_state)
    elif imbalance_technique == "undersample":
        balancer = RandomUnderSampler(random_state=random_state)
    elif imbalance_technique == "smotetomek":
        balancer = SMOTETomek(random_state=random_state)
    else:
        # If 'none', just return the processed data
        return X_train_processed, y_train, X_test_processed, y_test, preprocessor

    # Apply the selected balancer
    X_train_bal, y_train_bal = balancer.fit_resample(X_train_processed, y_train)
    
    logger.info("Class distribution BEFORE balancing:")
    logger.info(pd.Series(y_train).value_counts(normalize=True).round(4).to_dict())
    logger.info("Class distribution AFTER balancing:")
    logger.info(pd.Series(y_train_bal).value_counts(normalize=True).round(4).to_dict())

    logger.info(f"✅ Ready for modeling! Train Shape: {X_train_bal.shape}")

    return X_train_bal, y_train_bal, X_test_processed, y_test, preprocessor