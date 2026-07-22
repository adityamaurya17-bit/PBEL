import React, { useState } from "react";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    try {
      const response = await fetch("https://pbel.onrender.com/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const data = await response.json();

      console.log("Backend response:", data);

      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      setResult({ error: "Server not reachable" });
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial" }}>
      <h1>Fake News Detection</h1>
      <textarea
        rows="4"
        cols="50"
        placeholder="Enter news text here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <br />
      <button onClick={handlePredict}>Check News</button>

      {result && (
        <div style={{ marginTop: "20px" }}>
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
    </div>
  );
}

export default App;
