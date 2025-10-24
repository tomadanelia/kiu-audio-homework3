import React, { useState } from 'react';
import './App.css';

const API_BASE_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setResults(null);
    setError('');
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError('Please select a file first.');
      return;
    }

    setIsLoading(true);
    setError('');
    setResults(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/process-audio/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'An error occurred during processing.');
      }

      const data = await response.json();
      setResults(data);

    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const ConfidenceBadge = ({ level }) => {
    const levelClass = level ? level.toLowerCase() : '';
    return <span className={`badge ${levelClass}`}>{level}</span>;
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Audio Processing Pipeline</h1>
        <p>Upload an audio file to transcribe, redact, and summarize.</p>
      </header>
      <main>
        <form onSubmit={handleSubmit} className="upload-form">
          <input type="file" onChange={handleFileChange} accept="audio/mp3,audio/wav" />
          <button type="submit" disabled={isLoading || !file}>
            {isLoading ? 'Processing...' : 'Process Audio'}
          </button>
        </form>

        {error && <div className="error-message">{error}</div>}

        {isLoading && <div className="loader"></div>}

        {results && (
          <div className="results-container">
            <h2>Analysis Results</h2>
            
            <div className="result-card">
              <h3>Confidence Score</h3>
              <p className="confidence-score">
                {results.confidence_score} <ConfidenceBadge level={results.confidence_level} />
              </p>
            </div>
            
            <div className="result-card">
              <h3>Original Transcript</h3>
              <p className="transcript">{results.transcript}</p>
            </div>

            <div className="result-card">
              <h3>Redacted Transcript (PII Removed)</h3>
              <p className="transcript">{results.redacted_transcript}</p>
            </div>
            
            <div className="result-card">
              <h3>Generated Summary</h3>
              <p className="summary">{results.summary}</p>
              {results.summary_audio_url && (
                <div className="audio-player">
                  <h4>Listen to Summary:</h4>
                  <audio controls src={`${API_BASE_URL}${results.summary_audio_url}`}>
                    Your browser does not support the audio element.
                  </audio>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;