import React, { useState } from "react";

const embeddingModels = [
  { label: "BAAI/bge-small-en-v1.5", value: "BAAI/bge-small-en-v1.5" },
];
const answerGenLLMModels = [
  { label: "llama-3.1-8b-instant", value: "llama-3.1-8b-instant" },
  { label: "openai/gpt-oss-120b", value: "openai/gpt-oss-120b" },
];

function RAGSettings({ title, selectedModel, onModelChange, 
  topN, onTopNChange, semanticWeight, onSemanticWeightChange,
  agLLM, onAGLLMChange,
 }) {
  return (
    <div style={{
      flex: 1,
      border: "1px solid #ccc",
      borderRadius: "8px",
      padding: "24px",
      margin: "12px",
      background: "#fafbfc",
      boxSizing: "border-box",
      height: "100%",
      minWidth: 0
    }}>
      <h2>{title}</h2>
      <div style={{ marginBottom: "16px" }}>
        <label htmlFor={`${title}-embedding-model`} style={{ fontWeight: "bold" }}>
          Embedding Model:
        </label>
        <br />
        <select
          id={`${title}-embedding-model`}
          value={selectedModel}
          onChange={e => onModelChange(e.target.value)}
          style={{ marginTop: "8px", padding: "6px", width: "100%" }}
        >
          {embeddingModels.map(model => (
            <option key={model.value} value={model.value}>
              {model.label}
            </option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: "16px" }}>
          <label style={{ fontWeight: "bold" }}>Top N Retrieved Content:</label>
          <br />
          <select value={topN} onChange={e => onTopNChange(Number(e.target.value))} style={{ marginTop: "8px", padding: "6px", width: "100%" }}>
              {[1,2,3,4,5].map(n => <option key={n} value={n}>Top {n}</option>)}
          </select>
      </div>
      
      <div style={{ marginBottom: "16px" }}>
                <label style={{ fontWeight: "bold" }}>Semantic Retrieval Weight (0 ~ 1):</label>
                <br />
                <input
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={semanticWeight}
                    onChange={e => {
                        let v = Number(e.target.value);
                        if (Number.isNaN(v)) v = 0;
                        if (v < 0) v = 0;
                        if (v > 1) v = 1;
                        onSemanticWeightChange(v);
                    }}
                    style={{ marginTop: "8px", padding: "6px", width: "100%" }}
                />
      </div>

      <div style={{ marginBottom: "8px", color: "#333" }}>
          <strong>Key Word Retrieval Weight:</strong>
          <div style={{ marginTop: "6px" }}>{(1 - (Number(semanticWeight) || 0)).toFixed(2)}</div>
      </div>

      <div style={{ marginBottom: "16px" }}>
        <label htmlFor={`${title}-answer-gen-llm`} style={{ fontWeight: "bold" }}>
          Answer Generation LLM:
        </label>
        <br />
        <select
          id={`${title}-answer-gen-llm`}
          value={agLLM}
          onChange={e => onAGLLMChange(e.target.value)}
          style={{ marginTop: "8px", padding: "6px", width: "100%" }}
        >
          {answerGenLLMModels.map(model => (
            <option key={model.value} value={model.value}>
              {model.label}
            </option>
          ))}
        </select>
      </div>

    </div>
  );
}

function App() {
  const [rag1Model, setRag1Model] = useState(embeddingModels[0].value);
  const [rag2Model, setRag2Model] = useState(embeddingModels[0].value);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [datasetClicked, setDatasetClicked] = useState(false);
  const [rag1TopN, setRag1TopN] = useState(1);
  const [rag2TopN, setRag2TopN] = useState(1);
  const [rag1SemanticWeight, setRag1SemanticWeight] = useState(0.5);
  const [rag2SemanticWeight, setRag2SemanticWeight] = useState(0.5);
  const [rag1AGLLM, setRag1AGLLM] = useState(answerGenLLMModels[0].value);
  const [rag2AGLLM, setRag2AGLLM] = useState(answerGenLLMModels[0].value);
  const [preprocessingStatus, setPreprocessingStatus] = useState("idle");  // "idle" | "running" | "done" | "error"
  const [preprocessingMessage, setPreprocessingMessage] = useState("");

  const BACKEND_URL = "hanhanchatbot-production.up.railway.app";

  const handleRunRAGs = async () => {
      try {
        const response = await fetch(`https://${BACKEND_URL}/run-rags`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ dataset: selectedDataset }),
        });
        const data = await response.json();
        console.log("Response:", data);
      } catch (error) {
        console.error("Error running RAGs:", error);
      }
  };

  const handleFIQASelect = async () => {
    setSelectedDataset("FIQA Data");
    setDatasetClicked(true);
    setPreprocessingStatus("running");
    setPreprocessingMessage("preprocessing the data ...");
    try {
      await fetch(`https://${BACKEND_URL}/load-fiqa`, { method: "POST" });
      const poll = setInterval(async () => {
        const res = await fetch(`https://${BACKEND_URL}/preprocessing-status`);
        const data = await res.json();
        setPreprocessingMessage(data.message);
        if (data.status === "done" || data.status === "error") {
          setPreprocessingStatus(data.status);
          clearInterval(poll);
        }
      }, 2000);
    } catch (err) {
      setPreprocessingStatus("error");
      setPreprocessingMessage("Error starting preprocessing.");
    }
  };
  
  return (
    <div style={{
      display: "flex",
      flexDirection: "row",
      alignItems: "stretch",
      justifyContent: "center",
      width: "100vw",
      height: "100vh",
      background: "#f5f6fa",
      boxSizing: "border-box"
    }}>
      <RAGSettings
        title="RAG1 Settings"
        selectedModel={rag1Model}
        onModelChange={setRag1Model}
        topN={rag1TopN}
        onTopNChange={setRag1TopN}
        semanticWeight={rag1SemanticWeight}
        onSemanticWeightChange={setRag1SemanticWeight}
        agLLM={rag1AGLLM}
        onAGLLMChange={setRag1AGLLM}
        style={{ flex: 1 }}
      />
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "flex-start",
          flex: 2.2,
          boxSizing: "border-box",
          height: "100%",
          paddingTop: "40px"
        }}
      >
        <div style={{ textAlign: "center", marginBottom: "16px" }}>
          <h1 style={{ margin: 0, fontSize: "3.6rem", fontWeight: 800, color: "#800000" }}>RAG Doctor</h1>
          <div style={{ marginTop: "12px", color: "#a52a2a", fontFamily: "monospace" }}>
            ----------------------- Made by super Hanhan! -----------------------
          </div>
        </div>
          <table style={{ width: "80%", borderCollapse: "collapse", marginBottom: "64px", border: "2px solid #888" }}></table>
        <table style={{ width: "80%", borderCollapse: "collapse", marginBottom: "64px", border: "2px solid #888" }}>
          <thead>
            <tr>
              <th style={{
                textAlign: "center",
                fontWeight: "bold",
                fontSize: "1.1rem",
                border: "1px solid #888",
                padding: "12px",
                background: "#f3f3f3"
              }}>Dataset</th>
              <th style={{
                textAlign: "center",
                fontWeight: "bold",
                fontSize: "1.1rem",
                border: "1px solid #888",
                padding: "12px",
                background: "#f3f3f3"
              }}>Description</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={{
                textAlign: "center",
                padding: "12px",
                border: "1px solid #888",
                background: "#fff"
              }}>
                <button
                  style={{
                    background: datasetClicked
                      ? (selectedDataset === "FIQA Data" ? "#F0620A" : "#549d07")
                      : "#549d07",
                    color: "#fff",
                    border: "none",
                    borderRadius: "6px",
                    padding: "10px 24px",
                    fontSize: "1rem",
                    fontWeight: "bold",
                    cursor: "pointer",
                    transition: "background 0.2s"
                  }}
                  onClick={() => {
                    if (datasetClicked && selectedDataset === "FIQA Data") {
                      setSelectedDataset("");
                      setDatasetClicked(false);
                      setPreprocessingStatus("idle");
                      setPreprocessingMessage("");
                    } else {
                      handleFIQASelect();
                    }
                  }}
                >
                  FIQA Data
                </button>
                {selectedDataset === "FIQA Data" && preprocessingStatus !== "idle" && (
                   <div style={{ marginTop: "10px", display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", fontSize: "0.85rem", color: "#333" }}>
                     {preprocessingStatus === "running" && (
                       <div style={{
                         width: "14px", height: "14px",
                         border: "2px solid #ccc",
                         borderTop: "2px solid #F0620A",
                         borderRadius: "50%",
                         animation: "spin 0.8s linear infinite",
                         flexShrink: 0
                       }} />
                     )}
                     <span>{preprocessingMessage}</span>
                   </div>
                 )}
              </td>
              <td style={{
                textAlign: "left",
                padding: "12px",
                border: "1px solid #888",
                background: "#fff"
              }}>
                30 records of finance Q&amp;A.<br /> See details{" "}
                <a href="https://huggingface.co/datasets/vibrantlabsai/fiqa" target="_blank" rel="noopener noreferrer">
                  here &gt;&gt;
                </a>
              </td>
            </tr>
          </tbody>
        </table>
        <div
          style={{
            width: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "1.3rem",
            fontWeight: "500",
            color: "#555"
          }}
        >
          {datasetClicked && selectedDataset
            ? `Selected Dataset: ${selectedDataset}`
            : "Please config the settings of each RAG"}
        </div>
          {datasetClicked && selectedDataset && (
            <button
              onClick={handleRunRAGs}
              style={{
                width: "80%",
                background: "#000",
                color: "#fff",
                border: "none",
                borderRadius: "6px",
                padding: "12px 24px",
                fontSize: "1rem",
                fontWeight: "bold",
                cursor: "pointer",
                marginTop: "24px"
              }}
            >
              Confirmed all the selections, run RAGs now!
            </button>
          )}
      </div>
      <RAGSettings
        title="RAG2 Settings"
        selectedModel={rag2Model}
        onModelChange={setRag2Model}
        topN={rag2TopN}
        onTopNChange={setRag2TopN}
        semanticWeight={rag2SemanticWeight}
        onSemanticWeightChange={setRag2SemanticWeight}
        agLLM={rag2AGLLM}
        onAGLLMChange={setRag2AGLLM}
        style={{ flex: 1 }}
      />
    </div>
  );
}

export default App;