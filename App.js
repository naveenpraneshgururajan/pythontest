import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  const [recentPredictions, setRecentPredictions] = useState([]);

  // Fetch model stats on component mount
  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/stats');
      const data = await response.json();
      setStats(data.modelInfo);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!description.trim() || description.length < 10) {
      setError('Please enter a bug description (at least 10 characters)');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ description }),
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
        // Add to recent predictions
        setRecentPredictions(prev => [{
          description: description,
          rootCause: data.prediction.rootCause.primary,
          fixTeam: data.prediction.fixTeam.primary,
          timestamp: new Date().toLocaleTimeString()
        }, ...prev.slice(0, 4)]);
      } else {
        setError(data.error || 'Failed to get prediction');
      }
    } catch (err) {
      setError('Failed to connect to server. Make sure the backend is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setDescription('');
    setResult(null);
    setError(null);
  };

  // Sample bug descriptions for testing
  const sampleBugs = [
    "Database connection timeout during peak hours",
    "Login button not responding on mobile app",
    "API returns 500 error when processing payment",
    "Customer data not syncing between CRM and billing system",
    "Search functionality showing duplicate results",
    "Password reset email not being delivered to users"
  ];

  const useSampleBug = (sampleDescription) => {
    setDescription(sampleDescription);
    setError(null);
    setResult(null);
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <h1>🐛 Bug Classification System</h1>
          <p>AI-Powered Bug Triage for Lloyds Banking Group</p>
        </div>
      </header>

      {/* Main Content */}
      <div className="container">
        <div className="main-grid">
          {/* Left Column - Input Form */}
          <div className="input-section">
            <div className="card">
              <h2>Enter Bug Description</h2>
              
              <form onSubmit={handleSubmit}>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Describe the bug in detail... (minimum 10 characters)"
                  rows={6}
                  className="bug-input"
                  disabled={loading}
                />
                
                <div className="character-count">
                  {description.length} characters
                </div>

                <div className="button-group">
                  <button 
                    type="submit" 
                    className="btn btn-primary"
                    disabled={loading || !description.trim()}
                  >
                    {loading ? 'Analyzing...' : 'Predict'}
                  </button>
                  
                  <button 
                    type="button" 
                    onClick={handleClear}
                    className="btn btn-secondary"
                    disabled={loading}
                  >
                    Clear
                  </button>
                </div>
              </form>

              {/* Sample Bugs */}
              <div className="samples-section">
                <h3>Try Sample Bugs:</h3>
                <div className="sample-bugs">
                  {sampleBugs.map((bug, index) => (
                    <button
                      key={index}
                      className="sample-bug"
                      onClick={() => useSampleBug(bug)}
                      disabled={loading}
                    >
                      {bug}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Results */}
          <div className="results-section">
            {/* Error Message */}
            {error && (
              <div className="alert alert-error">
                <strong>Error:</strong> {error}
              </div>
            )}

            {/* Loading State */}
            {loading && (
              <div className="loading-container">
                <div className="loader"></div>
                <p>Analyzing bug description...</p>
              </div>
            )}

            {/* Prediction Results */}
            {result && result.success && (
              <div className="card result-card">
                <h2>Prediction Results</h2>
                
                {/* Root Cause */}
                <div className="prediction-block">
                  <h3>🔍 Root Cause</h3>
                  <div className="primary-prediction">
                    <span className="label">{result.prediction.rootCause.primary}</span>
                    <span className="confidence">
                      {(result.prediction.rootCause.confidence * 100).toFixed(1)}% confidence
                    </span>
                  </div>
                  
                  <div className="alternatives">
                    <p>Alternative causes:</p>
                    {result.prediction.rootCause.alternatives.slice(1).map((alt, index) => (
                      <div key={index} className="alternative-item">
                        <span>{alt.cause}</span>
                        <span className="alt-confidence">
                          {(alt.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Fix Team */}
                <div className="prediction-block">
                  <h3>👥 Fix Team</h3>
                  <div className="primary-prediction">
                    <span className="label">{result.prediction.fixTeam.primary}</span>
                    <span className="confidence">
                      {(result.prediction.fixTeam.confidence * 100).toFixed(1)}% confidence
                    </span>
                  </div>
                  
                  <div className="alternatives">
                    <p>Alternative teams:</p>
                    {result.prediction.fixTeam.alternatives.slice(1).map((alt, index) => (
                      <div key={index} className="alternative-item">
                        <span>{alt.team}</span>
                        <span className="alt-confidence">
                          {(alt.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Model Accuracy */}
                <div className="accuracy-info">
                  <h4>Model Performance</h4>
                  <div className="accuracy-metrics">
                    <div className="metric">
                      <span>Root Cause Accuracy:</span>
                      <span className="metric-value">
                        {(result.modelAccuracy.rootCause * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="metric">
                      <span>Fix Team Accuracy:</span>
                      <span className="metric-value">
                        {(result.modelAccuracy.fixTeam * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Model Stats */}
            {stats && !result && !loading && (
              <div className="card stats-card">
                <h2>Model Information</h2>
                <div className="stats-grid">
                  <div className="stat-item">
                    <span className="stat-value">{stats.rootCauseCategories}</span>
                    <span className="stat-label">Root Cause Categories</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">{stats.fixTeams}</span>
                    <span className="stat-label">Fix Teams</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">{stats.modelAccuracy.rootCause}</span>
                    <span className="stat-label">Root Cause Accuracy</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">{stats.modelAccuracy.fixTeam}</span>
                    <span className="stat-label">Fix Team Accuracy</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Recent Predictions History */}
        {recentPredictions.length > 0 && (
          <div className="history-section">
            <h2>Recent Predictions</h2>
            <div className="history-table">
              <table>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Bug Description</th>
                    <th>Root Cause</th>
                    <th>Fix Team</th>
                  </tr>
                </thead>
                <tbody>
                  {recentPredictions.map((pred, index) => (
                    <tr key={index}>
                      <td>{pred.timestamp}</td>
                      <td className="desc-cell">
                        {pred.description.substring(0, 50)}...
                      </td>
                      <td>{pred.rootCause}</td>
                      <td>{pred.fixTeam}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="app-footer">
        <p>Bug Classification System v1.0 | Lloyds Banking Group</p>
      </footer>
    </div>
  );
}

export default App;
