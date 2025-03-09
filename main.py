import os
import pandas as pd
from src.preprocessing import Preprocessor
from src.models.train_ml import MLTrainer
from src.models.train_transformer import TransformerTrainer
from src.evaluate import Evaluator
from src.utils.config_loader import CONFIG

# Define file paths
RAW_DATA_PATH = CONFIG['data']['raw']
PROCESSED_DATA_PATH = CONFIG['data']['processed']


def main():
    # Preprocess raw data and save
    preprocessor = Preprocessor()
    preprocessor.preprocess_and_save(RAW_DATA_PATH, PROCESSED_DATA_PATH)

    # Train ML models
    ml_trainer = MLTrainer()
    ml_trainer.train_ml()

    # Train Transformer model
    transformer_trainer = TransformerTrainer()
    transformer_trainer.train_transformer()

    # Evaluate models
    evaluator = Evaluator()
    evaluator.evaluate_ml_models()
    evaluator.evaluate_transformer()


if __name__ == "__main__":
    main()