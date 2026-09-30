import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import is_gemini_api_key_configured, GEMINI_MODEL
from app.data_loader import data_loader
from app.models import QueryRequest, QueryResponse, AnomalyResponse, HealthResponse
from app.llm_interpreter import parse_question_to_intent
from app.query_engine import execute_query_intent
from app.anomaly_detector import detect_anomalies

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("support_tickets_app")

app = FastAPI(
    title="Customer Support Ticket Analysis API",
    description="AI-powered support ticket analysis system using Gemini LLM intent translation and deterministic Pandas execution.",
    version="1.0.0"
)

STATIC_DIR = Path(__file__).parent / "static"

# Load dataset at startup
@app.on_event("startup")
def startup_event():
    try:
        df = data_loader.load_data()
        logger.info(f"Successfully loaded dataset with {len(df)} rows.")
    except Exception as e:
        logger.error(f"Failed to load dataset during startup: {e}")

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint confirming application status and configuration."""
    try:
        df = data_loader.get_df()
        dataset_loaded = True
        dataset_rows = len(df)
    except Exception:
        dataset_loaded = False
        dataset_rows = 0

    return HealthResponse(
        status="healthy",
        dataset_loaded=dataset_loaded,
        dataset_rows=dataset_rows,
        gemini_key_configured=is_gemini_api_key_configured(),
        gemini_model=GEMINI_MODEL
    )

@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    """Parses natural language question with Gemini LLM and executes deterministically against dataset."""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    logger.info(f"Processing query: '{request.question}'")
    
    # 1. Parse question with Gemini LLM to extract structured QueryIntent
    intent = parse_question_to_intent(request.question)
    
    # 2. Execute structured intent deterministically using Pandas
    response = execute_query_intent(request.question, intent)
    return response

@app.get("/anomalies", response_model=AnomalyResponse)
def get_anomalies():
    """Returns detected resolution time outliers (IQR method) and unresolved high-priority tickets > 24h old."""
    try:
        return detect_anomalies()
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to detect anomalies: {str(e)}")

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves the minimal web interface."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Support Ticket AI System API</h1><p>Visit /docs for API documentation.</p>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
