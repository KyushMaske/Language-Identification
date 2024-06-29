from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    original_text = Column(Text, nullable=False)
    predicted_language = Column(String, index=True)
    translated_text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
