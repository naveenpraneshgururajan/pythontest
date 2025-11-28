# Bug Classification System - Setup Guide

## 🚀 Quick Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 14+
- Your Excel file with bug data

---

## Step 1: Train the Model First

1. **Install Python dependencies:**
```bash
pip3 install -r requirements.txt
```

**Note:** This installs FastAPI, Uvicorn, and all ML dependencies. Pydantic may be installed as a FastAPI dependency, but we don't use it directly in our code.

2. **Train your model:**
```bash
# Edit simplified_bug_classifier.py line 32 with your Excel file path
python3 simplified_bug_classifier.py
```

This will create `trained_bug_classifier.pkl` which the API needs.

---

## Step 2: Start the Backend API

1. **Run the FastAPI server:**
```bash
python3 backend_api.py
```

The API will start at `http://localhost:5000`

You should see:
```
STARTING BUG CLASSIFIER API SERVER (FastAPI)
📡 API will be available at: http://localhost:5000
📚 API Documentation: http://localhost:5000/docs
📖 Alternative Docs: http://localhost:5000/redoc
```

**✨ New Feature:** Visit `http://localhost:5000/docs` for interactive API documentation!

---

## Step 3: Start the React Frontend

1. **Open a new terminal** (keep the API running)

2. **Navigate to frontend folder:**
```bash
cd frontend
```

3. **Install React dependencies:**
```bash
npm install
```

4. **Create required React files:**

Create `frontend/src/index.js`:
```javascript
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

Create `frontend/src/index.css`:
```css
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}
```

Create `frontend/public/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="Bug Classification System" />
    <title>Bug Classifier - Lloyds Banking Group</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
```

5. **Move the App.js and App.css files to src folder:**
```bash
mv App.js src/App.js
mv App.css src/App.css
```

6. **Start the React app:**
```bash
npm start
```

The frontend will open at `http://localhost:3000`

---

## 📁 Final Folder Structure

```
bug-classifier/
│
├── simplified_bug_classifier.py    # Model training script
├── backend_api.py                  # FastAPI server
├── trained_bug_classifier.pkl      # Trained model (created after training)
├── requirements.txt                # Python dependencies (no direct Pydantic usage)
├── your_bug_data.xlsx             # Your Excel data
│
└── frontend/                       # React app folder
    ├── package.json
    ├── public/
    │   └── index.html
    └── src/
        ├── App.js
        ├── App.css
        ├── index.js
        └── index.css
```

---

## 🎯 Using the Application

1. **Enter a bug description** in the text area (minimum 10 characters)
2. **Click "PREDICT"** to get predictions
3. **View results:**
   - Primary root cause with confidence
   - Primary fix team with confidence
   - Alternative predictions
   - Model accuracy scores

### Sample Bugs to Test:
- Database connection timeout during peak hours
- Login button not responding on mobile app
- API returns 500 error when processing payment

### 📚 Exploring the API:
- Visit `http://localhost:5000/docs` for interactive Swagger UI documentation
- Visit `http://localhost:5000/redoc` for alternative ReDoc documentation
- Test API endpoints directly from the browser using the interactive docs

---

## 🔧 Troubleshooting

### Backend Issues:

**"No module named 'fastapi'" error:**
```bash
# Reinstall dependencies
pip3 install -r requirements.txt
```

**Port already in use:**
```bash
# Change port in backend_api.py last line:
uvicorn.run(app, host="0.0.0.0", port=5001, reload=True)  # Change to 5001
```

**Model file not found:**
- Make sure you run `simplified_bug_classifier.py` first
- Check that `trained_bug_classifier.pkl` exists in the same directory

**Uvicorn not starting:**
```bash
# Make sure uvicorn is installed
pip3 install uvicorn[standard]
```

### Frontend Issues:

**Cannot connect to backend:**
- Make sure backend is running on port 5000
- Check that you see "Uvicorn running on http://0.0.0.0:5000"
- Check CORS is enabled in backend_api.py

**npm install fails:**
```bash
# Try clearing cache
npm cache clean --force
npm install
```

### API Testing:

**Test the API without frontend:**
```bash
# Using curl
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "Database connection timeout during peak hours"}'

# Or visit http://localhost:5000/docs for interactive testing
```

---

## 🚀 Production Deployment

### For Production:

1. **Backend:**
   ```bash
   # Use Uvicorn with multiple workers
   uvicorn backend_api:app --host 0.0.0.0 --port 5000 --workers 4

   # Or use Gunicorn with Uvicorn workers
   gunicorn backend_api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000
   ```
   - Add authentication/API keys
   - Use environment variables for config
   - Enable access logs

2. **Frontend:**
   - Build production version: `npm run build`
   - Serve with nginx or similar
   - Update API endpoint URL

3. **Security:**
   - Input validation (already implemented)
   - Add rate limiting middleware
   - HTTPS certificates
   - Update CORS settings for specific origins

---

## 📞 Support

For issues or questions:
1. Check the console for error messages
2. Ensure all dependencies are installed
3. Verify file paths are correct
4. Check that both backend and frontend are running
