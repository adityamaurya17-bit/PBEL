import React, { useState } from "react";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState("");
  const [history, setHistory] = useState([]);

  const checkNews = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      const output = `${data.prediction} (${data.confidence}% confidence)`;
      setResult(output);
      setHistory([...history, { text, prediction: output }]);
    } catch (error) {
      console.error("Error fetching:", error);
      setResult("Error connecting to backend");
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial", textAlign: "center" }}>
      <h1 style={{ color: "#2c3e50" }}>📰 Fake News Detection</h1>
      <textarea
        rows="6"
        cols="60"
        placeholder="Enter news text here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        style={{ padding: "10px", borderRadius: "5px" }}
      />
      <br /><br />
      <button
        onClick={checkNews}
        style={{
          padding: "10px 20px",
          backgroundColor: "#3498db",
          color: "white",
          border: "none",
          borderRadius: "5px",
        }}
      >
        Check
      </button>
      <h2 style={{ marginTop: "20px" }}>Result: {result}</h2>

      <h3>History</h3>
      <ul style={{ listStyle: "none", padding: 0 }}>
        {history.map((item, index) => (
          <li key={index}>
            <b>{item.text}</b> → {item.prediction}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;
