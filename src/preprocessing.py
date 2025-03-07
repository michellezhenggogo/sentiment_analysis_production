import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
import emoji
from sklearn.preprocessing import LabelEncoder
from src.utils.config_loader import CONFIG

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words(CONFIG['preprocessing']['stopwords_language']))
        self.lemmatizer = WordNetLemmatizer()
        self.label_encoder = LabelEncoder()

        # Replace common emoticons with textual labels
        self.emoticon_dict = CONFIG['preprocessing']['emoticons']

    def text_preprocessing(self, text):

        text = emoji.demojize(text)

        for pattern, replacement in self.emoticon_dict.items():
            text = re.sub(pattern, replacement, text)

        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        text = contractions.fix(text)
        text = re.sub(r'\[.*?\]', '', text)  # Remove square brackets
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
        text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags (if not useful for sentiment)
        text = re.sub(r'\n', ' ', text)  # Replace new lines with space
        text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
        text = re.sub(r'[^a-zA-Z!?]', ' ', text)  # Keep special sentiment-related symbols

        tokens = text.split()  # Transform text into tokens
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return ' '.join(tokens)


    def preprocess_and_save(self, input_file, output_file):

        # Load dataset
        df = pd.read_csv(input_file)

        # Apply text preprocessing
        df['processed_text'] = df['text'].apply(self.text_preprocessing)

        # Encode sentiment labels
        df['Encoded_sentiment'] = self.label_encoder.fit_transform(df["sentiment"])

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save processed data
        df.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")


if __name__ == "__main__":
    input_path = CONFIG['data']['raw']
    output_path = CONFIG['data']['processed']

    preprocessor = Preprocessor()
    preprocessor.preprocess_and_save(input_path, output_path)
