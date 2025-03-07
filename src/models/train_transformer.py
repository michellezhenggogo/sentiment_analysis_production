import logging
import os
import joblib
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from src.utils.config_loader import CONFIG

# Configure Logging
log_dir = CONFIG["logs"]["train_transformer"]
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train_transformer.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class TransformerTrainer:
    """
    Train Transformer-based model (BERT) for sentiment analysis.
    """

    def __init__(self):
        self.model_name = CONFIG['training']['transformers']['model']
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = TFBertForSequenceClassification.from_pretrained(self.model_name,
                                                                     num_labels=CONFIG['training']['transformers']['num_labels'])

        # Use fixed hyperparameters from config
        self.batch_size = CONFIG['training']['transformers']['batch_size']
        self.epochs = CONFIG['training']['transformers']['epochs']
        self.learning_rate = float(CONFIG['training']['transformers']['learning_rate'])
        self.max_length = CONFIG['training']['transformers']['max_length']
        self.num_labels = CONFIG['training']['transformers']['num_labels']
        self.dropout = CONFIG['training']['transformers']['dropout_rate']
        self.optimizer_type = CONFIG['training']['transformers']['optimizer']

        self.split_path_tf = CONFIG['data']['tf_split']
        self.path_tf_train = CONFIG['data']['tf_train_dataset']
        self.path_tf_test = CONFIG['data']['tf_test_dataset']
        self.processed_tf_file = CONFIG['data']['tf_split']

        self.random_state = CONFIG['training']['random_state']

    def train_transformer(self):
        # Load TensorFlow dataset
        train_dataset, test_dataset = self.load_tf_dataset()

        # Load model with dropout
        self.model.config.hidden_dropout_prob = self.dropout

        # Select optimizer
        optimizer_mapping = {
            "adam": tf.keras.optimizers.Adam(self.learning_rate),
            "adamw": tf.keras.optimizers.AdamW(self.learning_rate),
            "sgd": tf.keras.optimizers.SGD(self.learning_rate),
        }
        optimizer = optimizer_mapping.get(self.optimizer_type, tf.keras.optimizers.AdamW(self.learning_rate))

        # Compile model
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        # Train model
        self.model.fit(train_dataset, validation_data=test_dataset, epochs=self.epochs)

        # Save the trained model
        self.save_model()

    def load_saved_tf_data(self):
        try:
            X_train = joblib.load(os.path.join(self.split_path_tf, "X_train.pkl"))
            X_test = joblib.load(os.path.join(self.split_path_tf, "X_test.pkl"))
            train_masks = joblib.load(os.path.join(self.split_path_tf, "train_masks.pkl"))
            test_masks = joblib.load(os.path.join(self.split_path_tf, "test_masks.pkl"))
            y_train = joblib.load(os.path.join(self.split_path_tf, "y_train.pkl"))
            y_test = joblib.load(os.path.join(self.split_path_tf, "y_test.pkl"))
            print(f"Loaded train-test data from {self.split_path_tf}")
            return X_train, X_test, train_masks, test_masks, y_train, y_test

        except FileNotFoundError:
            print("Train-test split not found.")
            return None, None, None, None, None, None

    def load_tf_dataset(self):
        X_train, X_test, train_masks, test_masks, y_train, y_test = self.load_saved_tf_data()

        # Convert to TensorFlow dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ({"input_ids": X_train, "attention_mask": train_masks}, y_train)).batch(self.batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices(
            ({"input_ids": X_test, "attention_mask": test_masks}, y_test)).batch(self.batch_size)

        return train_dataset, test_dataset

    def save_model(self):
        save_path = f"models/{self.model_name.replace('/', '_')}_model"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved at path: {save_path}")


if __name__ == "__main__":
    trainer = TransformerTrainer()
    trainer.train_transformer()
