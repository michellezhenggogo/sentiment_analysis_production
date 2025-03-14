import os
import joblib
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from src.utils.config_loader import CONFIG
import logging


# Configure Logging
log_dir = CONFIG["logs"]["evaluate"]
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "evaluation.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class Evaluator:
    """
    Class for evaluating both traditional ML models and Transformer models.
    """

    def __init__(self):
        self.processed_ml_file = CONFIG['data']['ml_split']
        self.processed_tf_file = CONFIG['data']['tf_split']
        self.vectorizer_path = os.path.join(self.processed_ml_file, "tfidf_vectorizer.pkl")
        self.model_ml_path = CONFIG['models']['model_path']
        self.model_tf_path = CONFIG['models']['model_tf_path']
        self.path_tf_test = CONFIG['data']['tf_test_dataset']

        self.model_name = CONFIG['training']['transformers']['model']
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.max_length = CONFIG['training']['transformers']['max_length']
        self.truncation = CONFIG['training']['tokenizer']['truncation']
        self.padding = CONFIG['training']['tokenizer']['padding']
        self.return_tensors = CONFIG['training']['tokenizer']['return_tensors']
        self.batch_size = CONFIG['training']['transformers']['batch_size']

    def evaluate_ml_models(self):
        """Evaluate traditional ML models."""
        print("\nEvaluating Traditional ML Models...")

        # Load test data
        X_test, y_test = self.load_ml_test_data()

        # Load the TF-IDF vectorizer
        if not os.path.exists(self.vectorizer_path):
            print("TF-IDF vectorizer not found.")
            return

        # Evaluate each ML model
        for model_name in CONFIG["training"]["models"].keys():
            model_path = os.path.join(self.model_ml_path, f"{model_name}_model.pkl")

            if not os.path.exists(model_path):
                print(f"Skipping {model_name}: Model not found.")
                continue

            model = joblib.load(model_path)

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds)

            print(f"{model_name} Accuracy: {acc:.2f}")
            print(report)

            # Log results
            logging.info(f"Model: {model_name}")
            logging.info(f"Accuracy: {acc:.2f}")
            logging.info(f"Classification Report:\n{report}\n")

    def evaluate_transformer(self):

        """Evaluate Transformer-based model."""
        print("\nEvaluating Transformer Model...")

        # Load test datasets
        X_test, y_test, test_masks = self.load_transformer_test_data()
        test_dataset = tf.data.Dataset.from_tensor_slices(
            ({"input_ids": X_test, "attention_mask": test_masks}, y_test)).batch(self.batch_size, drop_remainder=True)  # ensure dataset does not pad extra batches.

        # Load model
        model = TFBertForSequenceClassification.from_pretrained(self.model_tf_path)

        # Predict labels
        predicted_labels = model.predict(test_dataset).logits
        predicted_labels = tf.argmax(tf.nn.softmax(predicted_labels, axis=1), axis=1).numpy()
        y_test = y_test[:len(predicted_labels)]

        # Compute accuracy
        acc = accuracy_score(y_test, predicted_labels)
        report = classification_report(y_test, predicted_labels)

        # Print evaluation metrics
        print(f"\nTransformer Model Accuracy: {acc:.2f}")
        print(report)

        # Log results
        logging.info(f"Transformer Accuracy: {acc:.2f}")
        logging.info(f"Transformer Classification Report:\n{report}\n")

    def load_ml_test_data(self):
        try:
            X_test = joblib.load(os.path.join(self.processed_ml_file, "X_test.pkl"))
            y_test = joblib.load(os.path.join(self.processed_ml_file, "y_test.pkl"))
            print(f"Loaded ml test data from {self.processed_ml_file}")
            return X_test, y_test
        except FileNotFoundError:
            print("ML test data not found.")
            return None, None

    def load_transformer_test_data(self):
        try:
            X_test = joblib.load(os.path.join(self.processed_tf_file, "X_test.pkl"))
            y_test = joblib.load(os.path.join(self.processed_tf_file, "y_test.pkl"))
            test_masks = joblib.load(os.path.join(self.processed_tf_file, "test_masks.pkl"))

            print(f"Loaded ml test data from {self.processed_ml_file}")
            return X_test, y_test, test_masks
        except FileNotFoundError:
            print("ML test data not found.")
            return None, None, None


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate_ml_models()
    evaluator.evaluate_transformer()
