import os
import pandas as pd
from src.preprocessing import Preprocessor
from src.models.train_ml import train_svm_model
from src.models.train_transformer import train_bert_model

# Define file paths
RAW_DATA_PATH = "data/raw/sentiment_analysis.csv"
PROCESSED_DATA_PATH = "data/processed/sentiment_analysis_cleaned.csv"

def main():

    Preprocessor.preprocess_and_save(RAW_DATA_PATH, PROCESSED_DATA_PATH)

    train_svm_model(PROCESSED_DATA_PATH)

    train_bert_model(PROCESSED_DATA_PATH)

    print("Pipeline completed!")

if __name__ == "__main__":
    main()