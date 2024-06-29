
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from googletrans import Translator
import nltk
import os

from database import SessionLocal, engine
import models

# Create the database tables
models.Base.metadata.create_all(bind=engine)

# Load the model, pipeline, and label encoder
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

def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text

class TextInput(BaseModel):
    text: str

app = FastAPI()

predictor = load_saved_model()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/predict/")
def predict_language(text_input: TextInput, db: Session = Depends(get_db)):
    try:
        prediction = predictor(text_input.text)
        translated_text = translate_to_english(text_input.text)
        
        # Save prediction to the database
        db_prediction = models.Prediction(
            original_text=text_input.text,
            predicted_language=prediction,
            translated_text=translated_text
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)

        return {
            "id": db_prediction.id,
            "original_text": text_input.text,
            "predicted_language": prediction,
            "translated_text": translated_text,
            "timestamp": db_prediction.timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static HTML page
@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("templates/index.html") as f:
        return HTMLResponse(f.read())

@app.get("/predictions/")
def get_predictions(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    predictions = db.query(models.Prediction).offset(skip).limit(limit).all()
    return predictions
