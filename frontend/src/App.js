import React, { useState } from "react";
import "./App.css";

const API_URL = process.env.REACT_APP_API_URL || "https://pbel2.akashprojects.dev";

function verdictClass(verdict) {
  if (verdict === "SUPPORTED") return "supported";
  if (verdict === "REFUTED") return "refuted";
  return "uncertain";
}

function verdictIcon(verdict) {
  if (verdict === "SUPPORTED") return "✓";
  if (verdict === "REFUTED") return "✕";
  return "?";
}

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleVerify = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/verify`, {
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
    <div className="app-shell">
      <div className="glow" aria-hidden="true" />

      <div className="app-container">
        <header className="app-header">
          <span className="app-title">News Fact Checker</span>
          <span className="app-subtitle">
            Retrieval + NLI fact-checking against a local Wikipedia index
          </span>
        </header>

        <div className="knowledge-notice">
          Knowledge base is a static Wikipedia snapshot from June 2017 — claims
          about events after that date can't be verified.
        </div>

        <div className="panel input-panel">
          <textarea
            className="input-box"
            rows="4"
            placeholder="Enter a claim to verify against Wikipedia..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />

          <div className="button-group">
            <button
              onClick={handleVerify}
              disabled={loading || !text.trim()}
              className="btn-primary"
            >
              {loading && <span className="spinner" aria-hidden="true" />}
              {loading ? "Verifying..." : "Verify Claim"}
            </button>
            <button onClick={handleClear} className="btn-secondary">
              Clear
            </button>
          </div>
        </div>

        {result && (
          <div
            className={`panel result-card ${
              result.error ? "error" : verdictClass(result.verdict)
            }`}
          >
            {result.error ? (
              <p>{result.error}</p>
            ) : (
              <>
                <div className="result-head">
                  <span className={`verdict-badge ${verdictClass(result.verdict)}`}>
                    <span className="verdict-icon">{verdictIcon(result.verdict)}</span>
                    {result.verdict}
                  </span>
                  <span className="confidence-value">{result.confidence}% confidence</span>
                </div>
                <div className="confidence-bar">
                  <div
                    className={`confidence-fill ${verdictClass(result.verdict)}`}
                    style={{ width: `${result.confidence}%` }}
                  />
                </div>
                {result.cited_sentence && (
                  <blockquote className="cited-sentence">
                    "{result.cited_sentence}"
                  </blockquote>
                )}
                {result.evidence && result.evidence.length > 0 && (
                  <div className="evidence-list">
                    <span className="evidence-label">Evidence</span>
                    <ul>
                      {result.evidence.map((e, i) => (
                        <li key={i} className="evidence-item">
                          <span className="evidence-sentence">{e.sentence}</span>
                          <a
                            className="evidence-source"
                            href={e.url}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            {e.source_title} ↗
                          </a>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {history.length > 0 && (
          <div className="panel history-section">
            <h2>History</h2>
            <ul>
              {history.map((item, index) => (
                <li key={index} className="history-item">
                  <div className="history-input">{item.input}</div>
                  <div className="history-meta">
                    <span className={`verdict-badge small ${verdictClass(item.verdict)}`}>
                      <span className="verdict-icon">{verdictIcon(item.verdict)}</span>
                      {item.verdict || "Error"}
                    </span>
                    {item.confidence != null && (
                      <span className="history-confidence">{item.confidence}%</span>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}

        <footer className="app-footer">Made by Aditya Maurya | IBM-PBEL Virtual Internship Project</footer>
      </div>
    </div>
  );
}

export default App;
