import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
import nltk
from googletrans import Translator

def process_text(sentence):
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    def word_token(sentence):
        return word_tokenize(sentence)
    
    def remove_stop_words(sentence):
        tokens = word_token(sentence)
        return [token.lower() for token in tokens if token.lower() not in stop_words and not token.isdigit()]
    
    return ' '.join(remove_stop_words(sentence))

def load_saved_model():
    model_path = 'language_model.h5'
    pipeline_path = 'pipeline.pkl'
    label_encoder_path = 'label_encoder.pkl'
    
    # Load the model, pipeline, and label encoder
    loaded_model = tf.keras.models.load_model(model_path)
    loaded_pipeline = joblib.load(pipeline_path)
    loaded_label_encoder = joblib.load(label_encoder_path)
    
    def predict(sentence):
        sentence_processed = loaded_pipeline.transform([process_text(sentence)])
        sentence_processed = pd.DataFrame.sparse.from_spmatrix(sentence_processed)
        
        prediction = loaded_model.predict(sentence_processed)
        predicted_label = loaded_label_encoder.inverse_transform([np.argmax(prediction)])
        
        return predicted_label[0]
    
    return predict

def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text

if __name__ == "__main__":
    predictor = load_saved_model()
    
    while True:
        user_input = input("Enter a sentence to predict its language (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        prediction = predictor(user_input)
        print(f"Predicted language: {prediction}")
        translated_text = translate_to_english(user_input)
        print(f"Translated to English: {translated_text}")
