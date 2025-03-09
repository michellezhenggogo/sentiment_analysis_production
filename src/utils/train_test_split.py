import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from src.utils.config_loader import CONFIG
from src.utils.data_loader import load_preprocessed_data


class TrainTestSplit:

    def __init__(self):
        self.model_name = CONFIG['training']['transformers']['model']
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.vectorizer = TfidfVectorizer(
            ngram_range=tuple(CONFIG['training']['tfidf']['ngram_range']),
            max_features=CONFIG['training']['tfidf']['max_features']
        )
        self.random_state = CONFIG['training']["random_state"]
        self.test_size = CONFIG['training']["test_size"]
        self.batch_size = CONFIG['training']['transformers']['batch_size']
        self.max_length = CONFIG['training']['transformers']['max_length']
        self.split_path_ml = CONFIG['data']['ml_split']
        self.split_path_tf = CONFIG['data']['tf_split']
        self.truncation = CONFIG['training']['tokenizer']['truncation']
        self.padding = CONFIG['training']['tokenizer']['padding']
        self.return_tensors = CONFIG['training']['tokenizer']['return_tensors']

    def split_save_ml_data(self):
        # Load split data
        X, y = load_preprocessed_data()

        # TF-IDF Feature Extraction
        X_tfidf = self.vectorizer.fit_transform(X)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=self.test_size, random_state=self.random_state
        )

        # Ensure the directory exists
        os.makedirs(self.split_path_ml, exist_ok=True)

        # Save the split data
        joblib.dump(X_train, os.path.join(self.split_path_ml, 'X_train.pkl'))
        joblib.dump(X_test, os.path.join(self.split_path_ml, 'X_test.pkl'))
        joblib.dump(y_train, os.path.join(self.split_path_ml, 'y_train.pkl'))
        joblib.dump(y_test, os.path.join(self.split_path_ml, 'y_test.pkl'))
        joblib.dump(self.vectorizer, os.path.join(self.split_path_ml, 'tfidf_vectorizer.pkl'))

        print(f"Saved train-test split and TF-IDF vectorizer in {self.split_path_ml}")

    def split_save_transformer_data(self):
        # Load split data
        X, y = load_preprocessed_data()
        input_ids = self.tokenize_texts(X)['input_ids'].numpy()
        attention_mask = self.tokenize_texts(X)['attention_mask'].numpy()

        # Train-Test Split
        X_train, X_test, train_masks, test_masks, y_train, y_test = train_test_split(
            input_ids, attention_mask, y, test_size=self.test_size, random_state=self.random_state
        )

        # Train-Validate Split
        X_train_final, X_val, train_masks_final, val_masks, y_train_final, y_val = train_test_split(
            X_train, train_masks, y_train, test_size=self.test_size, random_state=self.random_state
        )

        # Ensure the directory exists
        os.makedirs(self.split_path_tf, exist_ok=True)

        # Save the split data
        joblib.dump(X_train_final, os.path.join(self.split_path_tf, 'X_train.pkl'))
        joblib.dump(X_val, os.path.join(self.split_path_tf, 'X_val.pkl'))
        joblib.dump(X_test, os.path.join(self.split_path_tf, 'X_test.pkl'))
        joblib.dump(y_train_final, os.path.join(self.split_path_tf, 'y_train.pkl'))
        joblib.dump(y_val, os.path.join(self.split_path_tf, 'y_val.pkl'))
        joblib.dump(y_test, os.path.join(self.split_path_tf, 'y_test.pkl'))
        joblib.dump(train_masks_final, os.path.join(self.split_path_tf, 'train_masks.pkl'))
        joblib.dump(val_masks, os.path.join(self.split_path_tf, 'val_masks.pkl'))
        joblib.dump(test_masks, os.path.join(self.split_path_tf, 'test_masks.pkl'))

        print(f"Saved transformer train-validate-test split in {self.split_path_tf}")

    def tokenize_texts(self, texts):
        return self.tokenizer(
            texts,
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors
        )


if __name__ == "__main__":
    split = TrainTestSplit()
    split.split_save_ml_data()
    split.split_save_transformer_data()
