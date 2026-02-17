import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_fraud_full

if __name__ == "__main__":
    df = preprocess_fraud_full(save=True)
    print(df.head())
    print("\nColumns:", df.columns.tolist())