"""
FASTAPI BACKEND API FOR BUG CLASSIFIER
Modern REST API to serve predictions with automatic documentation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import pandas as pd
import joblib
import os
import uvicorn

# ============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE VALIDATION
# ============================================================================

class PredictionRequest(BaseModel):
    description: str = Field(..., min_length=10, description="Bug description (minimum 10 characters)")

    @validator('description')
    def validate_description(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError('Bug description too short (minimum 10 characters)')
        return v

class BatchPredictionRequest(BaseModel):
    descriptions: List[str] = Field(..., description="List of bug descriptions")

class AlternativePrediction(BaseModel):
    cause: Optional[str] = None
    team: Optional[str] = None
    confidence: float

class RootCausePrediction(BaseModel):
    primary: str
    confidence: float
    alternatives: List[AlternativePrediction]

class FixTeamPrediction(BaseModel):
    primary: str
    confidence: float
    alternatives: List[AlternativePrediction]

class ModelAccuracy(BaseModel):
    rootCause: float
    fixTeam: float

class PredictionResult(BaseModel):
    rootCause: RootCausePrediction
    fixTeam: FixTeamPrediction

class PredictionResponse(BaseModel):
    success: bool
    prediction: Optional[PredictionResult] = None
    modelAccuracy: Optional[ModelAccuracy] = None
    description: Optional[str] = None
    error: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    success: bool
    predictions: Optional[List[PredictionResponse]] = None
    count: Optional[int] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    accuracy: Dict[str, str]

class StatsResponse(BaseModel):
    modelInfo: Dict

class HomeResponse(BaseModel):
    message: str
    endpoints: Dict[str, str]

# ============================================================================
# INITIALIZE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Bug Classifier API",
    description="AI-powered bug classification and team assignment API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LOAD THE TRAINED MODEL (Once at startup)
# ============================================================================

@app.on_event("startup")
async def load_model():
    """Load the trained model on application startup"""
    global vectorizer, root_cause_model, fix_team_model
    global root_cause_encoder, fix_team_encoder, accuracy_scores

    print("Loading trained model...")

    MODEL_FILE = 'trained_bug_classifier.pkl'

    # Check if model exists
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: Model file '{MODEL_FILE}' not found!")
        print("Please train the model first using simplified_bug_classifier.py")
        raise RuntimeError(f"Model file '{MODEL_FILE}' not found!")

    # Load the model package
    model_package = joblib.load(MODEL_FILE)

    # Extract all components
    vectorizer = model_package['vectorizer']
    root_cause_model = model_package['root_cause_model']
    fix_team_model = model_package['fix_team_model']
    root_cause_encoder = model_package['root_cause_encoder']
    fix_team_encoder = model_package['fix_team_encoder']
    accuracy_scores = model_package['accuracy_scores']

    print("✅ Model loaded successfully!")
    print(f"   Root Cause Accuracy: {accuracy_scores['root_cause']:.2%}")
    print(f"   Fix Team Accuracy: {accuracy_scores['fix_team']:.2%}")

# ============================================================================
# HELPER FUNCTION FOR PREDICTIONS
# ============================================================================

def predict_bug(description: str) -> Dict:
    """
    Make prediction for a single bug description
    """
    try:
        # Clean the text (same preprocessing as training)
        clean_text = description.lower()
        clean_text = pd.Series([clean_text]).str.replace(r'[^a-z0-9\s]', ' ', regex=True)[0]
        clean_text = pd.Series([clean_text]).str.replace(r'\s+', ' ', regex=True)[0].strip()

        # Transform to features
        features = vectorizer.transform([clean_text])

        # Predict root cause
        root_pred = root_cause_model.predict(features)[0]
        root_proba = root_cause_model.predict_proba(features)[0]
        root_label = root_cause_encoder.inverse_transform([root_pred])[0]
        root_confidence = float(max(root_proba))

        # Get top 3 root causes
        top_3_root_idx = root_proba.argsort()[-3:][::-1]
        top_3_root = []
        for idx in top_3_root_idx:
            label = root_cause_encoder.inverse_transform([idx])[0]
            prob = root_proba[idx]
            top_3_root.append({
                'cause': label,
                'confidence': float(prob)
            })

        # Predict fix team
        team_pred = fix_team_model.predict(features)[0]
        team_proba = fix_team_model.predict_proba(features)[0]
        team_label = fix_team_encoder.inverse_transform([team_pred])[0]
        team_confidence = float(max(team_proba))

        # Get top 3 teams
        top_3_team_idx = team_proba.argsort()[-3:][::-1]
        top_3_team = []
        for idx in top_3_team_idx:
            label = fix_team_encoder.inverse_transform([idx])[0]
            prob = team_proba[idx]
            top_3_team.append({
                'team': label,
                'confidence': float(prob)
            })

        return {
            'success': True,
            'prediction': {
                'rootCause': {
                    'primary': root_label,
                    'confidence': root_confidence,
                    'alternatives': top_3_root
                },
                'fixTeam': {
                    'primary': team_label,
                    'confidence': team_confidence,
                    'alternatives': top_3_team
                }
            },
            'modelAccuracy': {
                'rootCause': float(accuracy_scores['root_cause']),
                'fixTeam': float(accuracy_scores['fix_team'])
            }
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/", response_model=HomeResponse, tags=["General"])
async def home():
    """Home route to check if API is running"""
    return {
        'message': 'Bug Classifier API is running!',
        'endpoints': {
            'predict': '/api/predict',
            'health': '/api/health',
            'stats': '/api/stats',
            'docs': '/docs',
            'redoc': '/redoc'
        }
    }

@app.get("/api/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'model_loaded': True,
        'accuracy': {
            'rootCause': f"{accuracy_scores['root_cause']:.2%}",
            'fixTeam': f"{accuracy_scores['fix_team']:.2%}"
        }
    }

@app.get("/api/stats", response_model=StatsResponse, tags=["General"])
async def stats():
    """Get model statistics"""
    return {
        'modelInfo': {
            'rootCauseCategories': len(root_cause_encoder.classes_),
            'fixTeams': len(fix_team_encoder.classes_),
            'rootCauseList': root_cause_encoder.classes_.tolist(),
            'fixTeamList': fix_team_encoder.classes_.tolist(),
            'modelAccuracy': {
                'rootCause': f"{accuracy_scores['root_cause']:.2%}",
                'fixTeam': f"{accuracy_scores['fix_team']:.2%}"
            }
        }
    }

@app.post("/api/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """Main prediction endpoint"""
    try:
        # Make prediction
        result = predict_bug(request.description)

        # Add the original description to response
        result['description'] = request.description

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Server error: {str(e)}')

@app.post("/api/batch-predict", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction endpoint for multiple bugs"""
    try:
        descriptions = request.descriptions

        # Process each description
        results = []
        for desc in descriptions:
            if len(desc.strip()) < 10:
                results.append({
                    'success': False,
                    'error': 'Bug description too short (minimum 10 characters)',
                    'description': desc
                })
            else:
                result = predict_bug(desc)
                result['description'] = desc
                results.append(result)

        return {
            'success': True,
            'predictions': results,
            'count': len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Server error: {str(e)}')

# ============================================================================
# RUN THE SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("STARTING BUG CLASSIFIER API SERVER (FastAPI)")
    print("=" * 60)
    print("\n📡 API will be available at: http://localhost:5000")
    print("📱 Frontend should connect to: http://localhost:5000/api/predict")
    print("📚 API Documentation: http://localhost:5000/docs")
    print("📖 Alternative Docs: http://localhost:5000/redoc")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 60)

    # Run the FastAPI app using uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
