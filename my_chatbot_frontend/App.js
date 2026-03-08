import React, { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";

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
      overflowY: "auto",
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

const SCORE_COLORS = {
   "-1": "#e74c3c",
   "0":  "#e67e22",
   "1":  "#f1c40f",
   "2":  "#2ecc71",
   "3":  "#27ae60",
   "4":  "#1a6937",
};

function EvalStackedBarChart({ title, rag1Counts, rag2Counts }) {
  if (!rag1Counts && !rag2Counts) return null;
  const allScores = Array.from(
    new Set([
      ...Object.keys(rag1Counts || {}),
      ...Object.keys(rag2Counts || {}),
    ])
  ).sort((a, b) => Number(a) - Number(b));

  const data = [
    { name: "RAG1", ...(rag1Counts || {}) },
    { name: "RAG2", ...(rag2Counts || {}) },
  ];

  const totals = data.map(d =>
    allScores.reduce((sum, score) => sum + (Number(d[score]) || 0), 0)
  );

  const makeRenderLabel = (score) => ({ x, y, width, height, index }) => {
   const actualValue = Number(data[index][score]) || 0;
   if (!actualValue) return null;
   const pct = Math.round((actualValue / (totals[index] || 1)) * 100);
   return (
     <text x={x + width + 4} y={y + height / 2}
       dominantBaseline="middle" fontSize={11} fill="#333">
       {`${actualValue} (${pct}%)`}
     </text>
   );
 };

  return (
    <div style={{ flex: 1, minWidth: "280px" }}>
      <h3 style={{ textAlign: "center", marginBottom: "8px" }}>{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ right: 80 }} barCategoryGap="20%">
          <XAxis dataKey="name" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Legend />
          {allScores.map(score => (
            <Bar key={score} dataKey={score} stackId="a"
              fill={SCORE_COLORS[score] || "#8884d8"} name={`Score ${score}`}
              label={makeRenderLabel(score)} />
          ))}
        </BarChart>
      </ResponsiveContainer>
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
  const [preprocessingStatus, setPreprocessingStatus] = useState("idle"); // "idle" | "running" | "done" | "error"
  const [preprocessingMessage, setPreprocessingMessage] = useState("");
  const [ragStatus, setRagStatus] = useState("idle"); // "idle" | "running" | "done"
  const [evalResults, setEvalResults] = useState({ rag1: null, rag2: null });

  const BACKEND_URL = "hanhanchatbot-production.up.railway.app";

  useEffect(() => {
     if (ragStatus !== "busy") return;
     const poll = setInterval(async () => {
         try {
             const res = await fetch(`https://${BACKEND_URL}/rag-lock-status`);
             const data = await res.json();
             if (!data.locked) {
                 setRagStatus("idle");
                 clearInterval(poll);
             }
         } catch {
             clearInterval(poll);
         }
     }, 5000);
     return () => clearInterval(poll);
 }, [ragStatus]);

  const handleRunRAGs = async () => {
    setRagStatus("running");
    try {
        const response = await fetch(`https://${BACKEND_URL}/run-rags`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
           dataset: selectedDataset,
           rag1: {
             embedding_model: rag1Model,
             top_n: rag1TopN,
             semantic_weight: rag1SemanticWeight,
             keyword_weight: parseFloat((1 - rag1SemanticWeight).toFixed(2)),
             answer_gen_llm: rag1AGLLM,
           },
           rag2: {
             embedding_model: rag2Model,
             top_n: rag2TopN,
             semantic_weight: rag2SemanticWeight,
             keyword_weight: parseFloat((1 - rag2SemanticWeight).toFixed(2)),
             answer_gen_llm: rag2AGLLM,
           },
         }),
        });
        const data = await response.json();
        if (data.status === "busy") {
            setRagStatus("busy"); 
            return;
        }
        console.log("Response:", data);
        setEvalResults({ rag1: data.rag1, rag2: data.rag2 });
        setRagStatus("done");
      } catch (error) {
        console.error("Error running RAGs:", error);
        setRagStatus("error");
      }
  };

  const handleFIQASelect = async () => {
    setSelectedDataset("FIQA Data");
    setDatasetClicked(true);
    setPreprocessingStatus("running");
    setPreprocessingMessage("Preprocessing the data ...");
    try {
      await fetch(`https://${BACKEND_URL}/load-fiqa`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_name: "FIQA Data" }),
      });
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
      fontFamily: "Calibri, sans-serif",
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
          paddingTop: "16px",
          overflowY: "auto",
        }}
      >
        <div style={{ textAlign: "center", marginBottom: "16px" }}>
          <h1 style={{ margin: 0, fontSize: "3.6rem", fontWeight: 800, color: "#800000" }}>RAG Doctor</h1>
          <div style={{ marginTop: "12px", color: "#a52a2a", fontFamily: "monospace" }}>
            ----------------------- Made by super Hanhan! -----------------------
          </div>
        </div>
          <table style={{ width: "80%", borderCollapse: "collapse", marginBottom: "12px", border: "2px solid #888" }}></table>
          <table style={{ width: "80%", borderCollapse: "collapse", marginBottom: "12px", border: "2px solid #888" }}>
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
                   <div style={{ marginTop: "10px", display: "flex", alignItems: "center",
                    justifyContent: "center", gap: "8px", fontSize: "0.85rem"}}>
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
                     <span style={{ color: preprocessingStatus === "running" ? "#0000ff" : "#333" }}>{preprocessingMessage}</span>
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
          {datasetClicked && selectedDataset && (
            ragStatus === "running" ? (
             <div style={{ marginTop: "24px", fontSize: "2rem", fontWeight: "bold", color: "#0000ff" }}>
               Running RAG Pipelines...
             </div>
            ) : ragStatus === "busy" ? (
            <div style={{ marginTop: "24px", fontSize: "1.2rem", fontWeight: "bold", color: "#e67e22" }}>
                Another user is running. Please wait 3~ min and try again.
            </div>
            ) : (
             <>
               {preprocessingStatus === "done" && <button
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
                   marginTop: "8px"
                 }}
               >
                 Confirmed all the selections, run RAGs now!
               </button>}
               {preprocessingStatus === "done" && ragStatus === "done" && (
                <>
                  <div style={{ marginTop: "12px", fontSize: "2rem", color: "#9932cc", fontWeight: "bold" }}>
                    RAG Performance Results Are Ready!
                  </div>
                  <div style={{ display: "flex", gap: "16px", marginTop: "20px", 
                    width: "100%", boxSizing: "border-box", padding: "0 16px",
                    flexWrap: "wrap" }}>
                   <EvalStackedBarChart
                     title="Retrieval Quality Score"
                     rag1Counts={evalResults.rag1?.retrieval_quality_counts}
                     rag2Counts={evalResults.rag2?.retrieval_quality_counts}
                   />
                   <EvalStackedBarChart
                     title="Answer Quality Score"
                     rag1Counts={evalResults.rag1?.answer_quality_counts}
                     rag2Counts={evalResults.rag2?.answer_quality_counts}
                   />
                 </div>
                </>
               )}
             </>
           )
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