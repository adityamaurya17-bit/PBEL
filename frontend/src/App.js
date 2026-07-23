import React, { useState } from "react";
import "./App.css"; // external CSS file

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await fetch("https://pbel.onrender.com/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const data = await response.json();
      console.log("Backend response:", data);

      setResult(data);
      setHistory([{ input: text, ...data }, ...history]);
    } catch (error) {
      console.error("Error:", error);
      setResult({ error: "Server not reachable" });
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText("");
    setResult(null);
  };

  return (
    <div className="app-container">
      <header className="app-header">📰 Fake News Detection</header>

      <textarea
        className="input-box"
        rows="4"
        placeholder="Enter news text here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <div className="button-group">
        <button onClick={handlePredict} disabled={loading} className="btn-primary">
          {loading ? "Checking..." : "Check News"}
        </button>
        <button onClick={handleClear} className="btn-secondary">Clear</button>
      </div>

      {result && (
        <div className={`result-card ${result.error ? "error" : result.prediction === "Fake News" ? "fake" : "real"}`}>
          {result.error ? (
            <p>{result.error}</p>
          ) : (
            <>
              <p><strong>Prediction:</strong> {result.prediction}</p>
              <p><strong>Confidence:</strong> {result.confidence}%</p>
              <div className="confidence-bar">
                <div style={{ width: `${result.confidence}%` }}></div>
              </div>
            </>
          )}
        </div>
      )}

      {history.length > 0 && (
        <div className="history-section">
          <h2>📜 History</h2>
          <ul>
            {history.map((item, index) => (
              <li key={index}>
                <strong>Input:</strong> {item.input} <br />
                <strong>Prediction:</strong> {item.prediction} <br />
                <strong>Confidence:</strong> {item.confidence}%
              </li>
            ))}
          </ul>
        </div>
      )}

      <footer className="app-footer">Made by Aditya Maurya | CSE-DS Project</footer>
    </div>
  );
}

export default App;
