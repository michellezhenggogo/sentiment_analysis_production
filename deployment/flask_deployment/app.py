from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np

app = Flask(__name__)

# Load ML Model & Vectorizer
ml_model = joblib.load("models/SVM_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Load Transformer Model & Tokenizer
transformer_model = TFBertForSequenceClassification.from_pretrained("models/bert-base-uncased_model")
tokenizer = BertTokenizer.from_pretrained("models/bert-base-uncased_model")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # ML Model Prediction
    text_tfidf = vectorizer.transform([text])
    ml_prediction = ml_model.predict(text_tfidf)[0]

    # Transformer Model Prediction
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    outputs = transformer_model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    tf_prediction = tf.nn.softmax(outputs.logits, axis=1).numpy()
    sentiment_label = np.argmax(tf_prediction, axis=1)[0]

    response = {
        "text": text,
        "ml_prediction": int(ml_prediction),  # ML model sentiment output
        "transformer_prediction": int(sentiment_label)  # Transformer model sentiment output
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
