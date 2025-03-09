import logging
import os
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, GridSearchCV
from src.utils.config_loader import CONFIG
from src.preprocessing import Preprocessor
import importlib

# Configure Logging
log_dir = CONFIG['logs']['train_ml']
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train_ml.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class MLTrainer:
    """
    Train and tune traditional ML models for sentiment analysis.
    Models:
        - Naïve Bayes (MultinomialNB)
        - Support Vector Machine (SVM) with hyperparameter tuning
        - Random Forest with hyperparameter tuning
        - XGBoost with hyperparameter tuning
    """

    def __init__(self):
        self.processed_file = CONFIG["data"]["processed"]
        self.param_grids = CONFIG['training']['param_grids']
        self.preprocessor = Preprocessor()
        self.models = self.load_models_from_config()
        self.split_path = CONFIG['data']['ml_split']
        self.cv = CONFIG['training']['hyper_param']['cv']
        self.scoring = CONFIG['training']['hyper_param']['scoring']

    def train_ml(self):

        # load training data
        X_train, y_train, vectorizer = self.load_saved_ml_data()

        # Train and Save Models
        results = {}
        for name, model in self.models.items():
            # Train and tune ml models
            print(f"Training {name}...")
            model = self.tune_hyperparameters(model, name, X_train, y_train)
            model.fit(X_train, y_train)

            # Calculate Training Accuracy
            train_preds = model.predict(X_train)
            train_acc = accuracy_score(y_train, train_preds)
            print(f"{name} Training Accuracy: {train_acc:.4f}")

            # Save results
            results[name] = {"train_accuracy": train_acc}

            # Save trained model
            self.save_model(model, f"models/{name}_model.pkl")

            # Log results
            logging.info(f"Model: {name}")
            logging.info(f"Training Accuracy: {train_acc:.4f}")

    def tune_hyperparameters(self, model, model_name, X_train, y_train):
        if model_name in self.param_grids:
            param_grid = self.param_grids[model_name]

            print(f"Hyperparameter tuning for {model_name} using {self.cv}-fold CV...")

            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1,
                verbose=2
            )

            grid_search.fit(X_train, y_train)

            # Best Model Selection
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            print(f"Best Params for {model_name}: {best_params}, CV Accuracy: {best_score:.4f}")

            logging.info(f"Best Params for {model_name}: {best_params}")
            logging.info(f"Cross-Validation Accuracy: {best_score:.4f}")

            return best_model

        return model

    def load_saved_ml_data(self):
        try:
            X_train = joblib.load(os.path.join(self.split_path, "X_train.pkl"))
            y_train = joblib.load(os.path.join(self.split_path, "y_train.pkl"))
            vectorizer = joblib.load(os.path.join(self.split_path, "tfidf_vectorizer.pkl"))

            print(f"Loaded train-test data from {self.split_path}")
            return X_train, y_train, vectorizer
        except FileNotFoundError:
            print("Train-test split not found.")
            return None, None, None

    def load_models_from_config(self):
        models = {}
        for name, details in CONFIG['training']['models'].items():
            module_name, class_name = details["class"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            models[name] = model_class()
        return models

    def save_model(self, model, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        print(f" Model saved at path: {path}")


if __name__ == "__main__":
    trainer = MLTrainer()
    trainer.train_ml()
