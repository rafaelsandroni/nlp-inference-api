import json
from typing import List
from fastapi import APIRouter

from app.api.model import ClassificationInput, ClassificationResponse
from app.api.classificationpro import ClassificationProcessor

classification = APIRouter()
classification_process = ClassificationProcessor()

# Path for sentiment analysis service
@classification.post("/inference", response_model=ClassificationResponse)
async def classify(item: ClassificationInput):

    output_dict = dict()    
    text = item.text
    
    perdiction, confidence = classification_process.inference(input_text=text)

    output_dict["prediction"] = perdiction
    output_dict["confidence"] = confidence
    
    return output_dict
