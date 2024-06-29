import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
import tkinter as tk
from tkinter import filedialog

def select_file():
    """
    Opens a file dialog for the user to select a CSV file.

    Returns:
        str: Path of the selected CSV file.
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    return file_path

def process_text_data(data):
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    
    def word_token(sentence):
        return word_tokenize(sentence)
    
    def remove_stop_words(sentence):
        tokens = word_token(sentence)
        return [token.lower() for token in tokens if token.lower() not in stop_words and not token.isdigit()]
    
    data['Text'] = data['Text'].apply(lambda x: ' '.join(remove_stop_words(x)))
    
    return data

def train_and_evaluate_model(data):
    tfidf = TfidfVectorizer()
    pipeline_model = Pipeline([('tfid', tfidf)])
    
    X_input = pipeline_model.fit_transform(data['Text'])
    X_input = pd.DataFrame.sparse.from_spmatrix(X_input)
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['Language'])
    
    X_train, X_test, y_train, y_test = train_test_split(X_input, y, test_size=0.1, random_state=33)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation=tf.nn.softmax)
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5)
    
    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")
    
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return model, pipeline_model, label_encoder

def run_complete_process():
    csv_file_path = select_file()
    if not csv_file_path:
        print("No file selected.")
        return
    
    data = pd.read_csv(csv_file_path)
    
    processed_data = process_text_data(data)
    trained_model, pipeline_model, label_encoder = train_and_evaluate_model(processed_data)
    
    # Save the model, pipeline, and label encoder
    model_path = 'language_model.h5'
    pipeline_path = 'pipeline.pkl'
    label_encoder_path = 'label_encoder.pkl'
    
    trained_model.save(model_path)
    joblib.dump(pipeline_model, pipeline_path)
    joblib.dump(label_encoder, label_encoder_path)
    
    print(f"Model saved to {model_path}")
    print(f"Pipeline saved to {pipeline_path}")
    print(f"Label encoder saved to {label_encoder_path}")

if __name__ == "__main__":
    run_complete_process()
