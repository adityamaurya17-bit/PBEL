import React, { useState } from "react";

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
      setHistory([{ input: text, ...data }, ...history]); // save prediction to history
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
    <div style={{ maxWidth: "600px", margin: "0 auto", padding: "20px", fontFamily: "Arial" }}>
      <h1 style={{ textAlign: "center", color: "#333" }}>📰 Fake News Detection</h1>

      <textarea
        rows="4"
        cols="50"
        placeholder="Enter news text here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        style={{ width: "100%", padding: "10px", marginBottom: "10px" }}
      />
      <div style={{ display: "flex", gap: "10px" }}>
        <button
          onClick={handlePredict}
          disabled={loading}
          style={{ flex: 1, padding: "10px", background: "#007bff", color: "white", border: "none", cursor: "pointer" }}
        >
          {loading ? "Checking..." : "Check News"}
        </button>
        <button
          onClick={handleClear}
          style={{ flex: 1, padding: "10px", background: "#6c757d", color: "white", border: "none", cursor: "pointer" }}
        >
          Clear
        </button>
      </div>

      {result && (
        <div
          style={{
            marginTop: "20px",
            padding: "15px",
            borderRadius: "8px",
            background: result.error
              ? "#f8d7da"
              : result.prediction === "Fake News"
              ? "#f5c6cb"
              : "#d4edda",
          }}
        >
          {result.error ? (
            <p style={{ color: "red" }}>{result.error}</p>
          ) : (
            <>
              <p><strong>Prediction:</strong> {result.prediction}</p>
              <p><strong>Confidence:</strong> {result.confidence}%</p>
            </>
          )}
        </div>
      )}

      {history.length > 0 && (
        <div style={{ marginTop: "30px" }}>
          <h2>📜 History</h2>
          <ul>
            {history.map((item, index) => (
              <li key={index} style={{ marginBottom: "10px" }}>
                <strong>Input:</strong> {item.input} <br />
                <strong>Prediction:</strong> {item.prediction} <br />
                <strong>Confidence:</strong> {item.confidence}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
