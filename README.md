# Bug Classification System

An intelligent bug classification system that uses machine learning to automatically categorize bugs by root cause and assign them to the appropriate fix teams.

## Features

- **Automated Root Cause Analysis**: Predicts the root cause of bugs based on their descriptions
- **Team Assignment**: Automatically assigns bugs to the most appropriate fix team
- **Confidence Scores**: Provides confidence levels for predictions
- **Alternative Suggestions**: Shows alternative predictions when confidence is lower
- **Web Interface**: Clean, modern React interface for easy interaction
- **REST API**: Flask-based backend API for integration with other systems

## Tech Stack

**Backend:**
- Python 3.8+
- Flask (REST API)
- scikit-learn (Machine Learning)
- pandas (Data Processing)

**Frontend:**
- React
- Modern CSS with animations
- Responsive design

## Quick Start

See the [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed setup instructions.

### Prerequisites

- Python 3.8+
- Node.js 14+
- Your Excel file with bug data for training

### Basic Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the backend API:**
   ```bash
   python backend_api.py
   ```

3. **Install and start the frontend:**
   ```bash
   npm install
   npm start
   ```

The application will be available at `http://localhost:3000`

## Project Structure

```
├── backend_api.py           # Flask REST API server
├── requirements.txt         # Python dependencies
├── package.json            # Node.js dependencies
├── App.js                  # React main component
├── App.css                 # Styling
├── index.js                # React entry point
├── SETUP_GUIDE.md          # Detailed setup instructions
└── README.md               # This file
```

## Usage

1. Enter a bug description (minimum 10 characters)
2. Click "PREDICT" to get the classification
3. View the predicted root cause and fix team with confidence scores
4. Review alternative predictions if available

### Example Bug Descriptions

- "Database connection timeout during peak hours"
- "Login button not responding on mobile app"
- "API returns 500 error when processing payment"

## API Endpoints

### POST /predict
Classifies a bug description.

**Request:**
```json
{
  "bug_description": "Your bug description here"
}
```

**Response:**
```json
{
  "root_cause": "Database Issue",
  "root_cause_confidence": 0.85,
  "fix_team": "Backend Team",
  "fix_team_confidence": 0.92,
  "alternatives": {...},
  "model_accuracy": {...}
}
```

### GET /health
Health check endpoint.

## Development

To run in development mode:

```bash
# Backend (with auto-reload)
python backend_api.py

# Frontend (with hot-reload)
npm start
```

## Production Deployment

For production deployment:

1. Use a production WSGI server (e.g., Gunicorn) for the Flask backend
2. Build the React app: `npm run build`
3. Serve the built files with a web server (e.g., nginx)
4. Configure environment variables
5. Set up HTTPS
6. Implement authentication and rate limiting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for internal use.

## Support

For issues or questions, refer to the [SETUP_GUIDE.md](SETUP_GUIDE.md) troubleshooting section.
