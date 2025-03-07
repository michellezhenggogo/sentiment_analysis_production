import os
import pandas as pd
from src.utils.config_loader import CONFIG


def load_preprocessed_data():
    """
    Load preprocessed sentiment analysis dataset.

    Returns:
        X (pd.Series): Processed text data.
        y (pd.Series): Encoded sentiment labels.
    """
    processed_file = CONFIG["data"]["processed"]

    # Check if the file exists
    if not os.path.exists(processed_file):
        print(f"Processed data not found at: {processed_file}")
        return None, None

    df = pd.read_csv(processed_file)
    X = df['processed_text'].fillna('').astype(str).tolist()
    y = df['Encoded_sentiment'].values

    return X, y
