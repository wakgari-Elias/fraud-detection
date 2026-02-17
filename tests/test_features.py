import pandas as pd
from src.features import calculate_time_since_signup

def test_time_since_signup():
    df = pd.DataFrame({
        "signup_time": ["2024-01-01"],
        "purchase_time": ["2024-01-02"]
    })
    df = calculate_time_since_signup(df)
    assert df["time_since_signup"].iloc[0] > 0
