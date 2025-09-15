import os
import json
import logging
import yaml
import torch
import mlflow
from fastapi import FastAPI, HTTPException, Request
# --- FIX: Corrected typo from 'pantic' to 'pydantic' ---
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import Pipeline
from src.text_preprocessing import EnhancedTextPreprocessor

# --- 1. Logging and Configuration Setup ---

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuration is 100% environment-driven
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "intent-classifier-svc")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

# --- 2. Application and State Setup ---

class AppState:
    model: Pipeline = None
    preprocessor: EnhancedTextPreprocessor = None
    model_version: str = "unknown"

app = FastAPI(
    title="Intent Classification API",
    description="API to classify user intent based on text.",
    version="1.1.0"
)
app.state = AppState()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS.split(','),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Model Loading (FastAPI Startup Event) ---

@app.on_event("startup")
def load_model_and_preprocessor():
    """
    Load the model and preprocessor during application startup.
    If the primary model load fails, it lists available models and tries a local fallback.
    """
    logger.info("Application startup: Initializing preprocessor...")
    app.state.preprocessor = EnhancedTextPreprocessor()

    model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MODEL_STAGE}"
    logger.info(f"Attempting to load model from MLflow Registry: {model_uri}")

    try:
        app.state.model = mlflow.transformers.load_model(model_uri)
        logger.info("Successfully loaded model using 'mlflow.transformers' flavor.")

        client = mlflow.MlflowClient()
        version_details = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MODEL_STAGE)
        app.state.model_version = version_details.version
        logger.info(f"Model version '{app.state.model_version}' loaded for alias '{MODEL_STAGE}'.")

    except Exception as e:
        logger.warning(f"MLflow load failed for model '{MLFLOW_MODEL_NAME}' in stage '{MODEL_STAGE}'.")

        try:
            logger.info("Attempting to list available models from the registry...")
            client = mlflow.MlflowClient()
            available_models = [m.name for m in client.search_registered_models()]
            if available_models:
                logger.info(f"Available registered models are: {available_models}")
            else:
                logger.warning("No registered models found in the MLflow registry.")
        except Exception as client_error:
            logger.error(f"Could not connect to MLflow to list available models. Error: {client_error}")

        logger.warning("Falling back to local artifacts.")
        fallback_path = "models/production_model"
        if os.path.isdir(fallback_path):
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
                tokenizer = AutoTokenizer.from_pretrained(fallback_path)
                model = AutoModelForSequenceClassification.from_pretrained(fallback_path)
                app.state.model = pipeline("text-classification", model=model, tokenizer=tokenizer)
                app.state.model_version = "local_fallback"
                logger.info(f"Successfully loaded local fallback model from '{fallback_path}'.")
            except Exception as e2:
                logger.error(f"FATAL: Failed to load even the local fallback model: {e2}")
        else:
            logger.error(f"FATAL: No model available. MLflow load failed and local fallback path '{fallback_path}' not found.")


# --- 4. API Endpoints ---

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    intent: str
    confidence: float
    model_version: str

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    model_loaded = app.state.model is not None
    return {"status": "ok" if model_loaded else "error", "model_loaded": model_loaded}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Accepts text input and returns the predicted intent and confidence score."""
    if not app.state.model:
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")

    if not isinstance(request.text, str) or not request.text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' must be a non-empty string.")

    try:
        processed_text = app.state.preprocessor.preprocess(request.text)
        prediction = app.state.model(processed_text)[0]

        predicted_intent = prediction['label']
        confidence_score = prediction['score']

        logger.info({
            "event": "prediction",
            "input_length": len(request.text),
            "output_intent": predicted_intent,
            "confidence": round(confidence_score, 4),
            "model_version": app.state.model_version
        })

        return PredictionResponse(
            intent=predicted_intent,
            confidence=confidence_score,
            model_version=app.state.model_version
        )
    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction.")