from fastapi import FastAPI
from app.api.classification import classification
import time

# Path to provide details of the services loaded in the backend. Details are obtained from the config json file
@classification.get("/ping")
async def get_models():
    status = {"status": 'healthy'}
    return status

app = FastAPI(
    openapi_url="/classification/openapi.json", docs_url="/classification/docs"
)

app.include_router(classification, prefix="/classification", tags=["classification"])
