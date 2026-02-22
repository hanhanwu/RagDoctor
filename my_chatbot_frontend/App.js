import React, { useState } from "react";

const embeddingModels = [
  { label: "BAAI/bge-small-en-v1.5", value: "BAAI/bge-small-en-v1.5" },
  // Add more models here if needed
];

function RAGSettings({ title, selectedModel, onModelChange }) {
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
      {/* Add more settings here if needed */}
    </div>
  );
}

function App() {
  const [rag1Model, setRag1Model] = useState(embeddingModels[0].value);
  const [rag2Model, setRag2Model] = useState(embeddingModels[0].value);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [datasetClicked, setDatasetClicked] = useState(false);

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
        style={{ flex: 1 }}
      />
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "flex-start",
          flex: 2,
          boxSizing: "border-box",
          height: "100%",
          paddingTop: "40px"
        }}
      >
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
                    background: datasetClicked ? "#F0620A" : "#4caf50",
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
                    setSelectedDataset("FIQA Data");
                    setDatasetClicked(true);
                  }}
                  disabled={datasetClicked}
                >
                  FIQA Data
                </button>
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
          {datasetClicked
            ? `Selected Dataset: ${selectedDataset}`
            : "Please config the settings of each RAG"}
        </div>
      </div>
      <RAGSettings
        title="RAG2 Settings"
        selectedModel={rag2Model}
        onModelChange={setRag2Model}
        style={{ flex: 1 }}
      />
    </div>
  );
}

export default App;