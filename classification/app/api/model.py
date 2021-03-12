# Importing libraries
from pydantic import BaseModel
from typing import Optional


# Model for recieveing input
class ClassificationInput(BaseModel):
    text: str

# Model for sentiment analysis service response
class ClassificationResponse(BaseModel):
    prediction: str
    confidence: float
