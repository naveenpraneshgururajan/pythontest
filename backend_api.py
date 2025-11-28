"""
FLASK BACKEND API FOR BUG CLASSIFIER
Simple REST API to serve predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ============================================================================
# LOAD THE TRAINED MODEL (Once at startup)
# ============================================================================
print("Loading trained model...")

MODEL_FILE = 'trained_bug_classifier.pkl'

# Check if model exists
if not os.path.exists(MODEL_FILE):
    print(f"ERROR: Model file '{MODEL_FILE}' not found!")
    print("Please train the model first using simplified_bug_classifier.py")
    exit(1)

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
def predict_bug(description):
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

@app.route('/')
def home():
    """Home route to check if API is running"""
    return jsonify({
        'message': 'Bug Classifier API is running!',
        'endpoints': {
            'predict': '/api/predict',
            'health': '/api/health',
            'stats': '/api/stats'
        }
    })

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'accuracy': {
            'rootCause': f"{accuracy_scores['root_cause']:.2%}",
            'fixTeam': f"{accuracy_scores['fix_team']:.2%}"
        }
    })

@app.route('/api/stats')
def stats():
    """Get model statistics"""
    return jsonify({
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
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Get bug description from request
        data = request.json
        
        if not data or 'description' not in data:
            return jsonify({
                'success': False,
                'error': 'No bug description provided'
            }), 400
        
        description = data['description']
        
        # Validate description
        if not description or len(description.strip()) < 10:
            return jsonify({
                'success': False,
                'error': 'Bug description too short (minimum 10 characters)'
            }), 400
        
        # Make prediction
        result = predict_bug(description)
        
        # Add the original description to response
        result['description'] = description
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple bugs"""
    try:
        data = request.json
        
        if not data or 'descriptions' not in data:
            return jsonify({
                'success': False,
                'error': 'No bug descriptions provided'
            }), 400
        
        descriptions = data['descriptions']
        
        if not isinstance(descriptions, list):
            return jsonify({
                'success': False,
                'error': 'Descriptions must be an array'
            }), 400
        
        # Process each description
        results = []
        for desc in descriptions:
            result = predict_bug(desc)
            result['description'] = desc
            results.append(result)
        
        return jsonify({
            'success': True,
            'predictions': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

# ============================================================================
# RUN THE SERVER
# ============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("STARTING BUG CLASSIFIER API SERVER")
    print("=" * 60)
    print("\n📡 API will be available at: http://localhost:5000")
    print("📱 Frontend should connect to: http://localhost:5000/api/predict")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 60)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
