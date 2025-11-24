from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from contextlib import asynccontextmanager
from async_lru import alru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Global State & Lifespan Management ---
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    model_name = "lusamaki/distilbert_fine_tuned_fake_news_detection_model"
    
    # Load the pipeline
    ml_models["classifier"] = pipeline(
        "text-classification", 
        model=model_name, 
        tokenizer=model_name
    )
    logger.info("Model loaded successfully!")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan, title="Fake News Detector API")

# --- 2. Data Models ---
class NewsRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float

# --- 3. Cached Prediction Logic ---
@alru_cache(maxsize=128)
async def get_prediction_cached(text: str):
    classifier = ml_models.get("classifier")
    if not classifier:
        raise RuntimeError("Model not initialized")
    
    # REMOVED top_k=None. 
    # By default, pipeline returns the single highest scoring label.
    return classifier(text)

# --- 4. API Endpoints ---

@app.get("/health")
async def health_check():
    return {"status": "running", "model_loaded": "classifier" in ml_models}

@app.post("/predict", response_model=PredictionResponse)
async def predict_news(request: NewsRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Call the cached function
        # The result typically looks like: [{'label': 'Fake', 'score': 0.99...}]
        raw_result = await get_prediction_cached(request.text)
        
        # Extract the top result from the list
        top_result = raw_result[0]
        
        # Return simple JSON map
        return {
            "label": top_result['label'],
            "confidence": top_result['score']
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
