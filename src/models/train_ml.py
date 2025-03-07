import os
import joblib
from sklearn.model_selection import GridSearchCV
from src.utils.config_loader import CONFIG
from src.preprocessing import Preprocessor
import importlib

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

    def train_ml(self):

        # load ml split data
        X_train, X_test, y_train, y_test, vectorizer = self.load_saved_ml_data()

        # Train and Save Models
        for name, model in self.models.items():

            print(f"Training {name}...")
            model = self.tune_hyperparameters(model, name, X_train, y_train)

            model.fit(X_train, y_train)

            self.save_model(model, f"models/{name}_model.pkl")

    def load_saved_ml_data(self):
        try:
            X_train = joblib.load(os.path.join(self.split_path, "X_train.pkl"))
            X_test = joblib.load(os.path.join(self.split_path, "X_test.pkl"))
            y_train = joblib.load(os.path.join(self.split_path, "y_train.pkl"))
            y_test = joblib.load(os.path.join(self.split_path, "y_test.pkl"))
            vectorizer = joblib.load(os.path.join(self.split_path, "tfidf_vectorizer.pkl"))

            print(f"Loaded train-test data from {self.split_path}")
            return X_train, X_test, y_train, y_test, vectorizer
        except FileNotFoundError:
            print("Train-test split not found.")
            return None, None, None, None, None

    def load_models_from_config(self):
        models = {}
        for name, details in CONFIG['training']['models'].items():
            module_name, class_name = details["class"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            models[name] = model_class()
        return models

    def tune_hyperparameters(self, model, model_name, X_train, y_train):
        if model_name in self.param_grids:
            grid_search = GridSearchCV(model,
                                       self.param_grids[model_name],
                                       cv=CONFIG['training']['hyper_param']['cv'],
                                       scoring=CONFIG['training']['hyper_param']['scoring'],
                                       n_jobs=-1)
            grid_search.fit(X_train, y_train)
            print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
            return grid_search.best_estimator_
        return model

    def save_model(self, model, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        print(f" Model saved at path: {path}")


if __name__ == "__main__":
    trainer = MLTrainer()
    trainer.train_ml()
