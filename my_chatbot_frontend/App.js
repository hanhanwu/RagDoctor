import { useState, useEffect, useRef } from "react";
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
   "-1": "#9932cc",
   "0":  "#e74c3c",
   "1":  "#f1c40f",
   "2":  "#2ecc71",
   "3":  "#1a6937",
};

const RETRIEVAL_SCORE_DEFS = [
  { score: "-1", label: "Retrieved content is more relevant than human labeled context" },
  { score: "0",  label: "Completely irrelevant retrieved content" },
  { score: "1",  label: "Relevant retrieved content but low value" },
  { score: "2",  label: "Partially relevant retrieved content" },
  { score: "3",  label: "Highly relevant retrieved content" },
];

const ANSWER_SCORE_DEFS = [
  { score: "-1", label: "AI's answer is better than the 'ground truth'" },
  { score: "0",  label: "Completely irrelevant answer" },
  { score: "1",  label: "Relevant answer but low value" },
  { score: "2",  label: "Partially correct answer" },
  { score: "3",  label: "Highly accurate answer" },
];


function EvalStackedBarChart({ title, rag1Counts, rag2Counts, scoreDefinitions }) {
  if (!rag1Counts && !rag2Counts) return null;
  const [showTip, setShowTip] = useState(false);
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
      <div style={{ textAlign: "center", marginBottom: "8px", position: "relative" }}>
        <h3
          style={{ display: "inline", cursor: "help" }}
          onMouseEnter={() => setShowTip(true)}
          onMouseLeave={() => setShowTip(false)}
        >
          {title}
        </h3>
        {showTip && scoreDefinitions && (
          <div style={{
            position: "absolute",
            top: "100%",
            left: "50%",
            transform: "translateX(-50%)",
            background: "#fff",
            border: "1px solid #ccc",
            borderRadius: "6px",
            padding: "10px 14px",
            zIndex: 100,
            whiteSpace: "nowrap",
            boxShadow: "0 2px 8px rgba(0,0,0,0.18)",
            fontSize: "0.85rem",
            textAlign: "left",
            pointerEvents: "none",
          }}>
            {scoreDefinitions.map(({ score, label }) => (
              <div key={score} style={{ marginBottom: "4px" }}>
                <span style={{ fontWeight: "bold", color: SCORE_COLORS[score] || "#333" }}>Score {score}:</span> {label}
              </div>
            ))}
          </div>
        )}
      </div>
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

function RCAResultsPage({ results }) {
  if (!results) return (
    <div style={{ padding: "32px", fontFamily: "Calibri, sans-serif", color: "#0000ff", fontSize: "1.8rem",
      height: "100vh", overflowY: "auto", boxSizing: "border-box" }}>
      ⏳ Analysis is running... this page will update automatically when done.
    </div>
  );
  return (
    <div style={{ padding: "32px", fontFamily: "Calibri, sans-serif",
      height: "100vh", overflowY: "auto", boxSizing: "border-box" }}>
      <h1 style={{ color: "#800000", marginBottom: "24px" }}>Root Cause Analysis Results</h1>
      {results.rag1.map((item, i) => {
        const rag2Item = results.rag2[i];
        return (
          <div key={i} style={{ border: "1px solid #ddd", borderRadius: "8px", marginBottom: "24px", padding: "16px" }}>
            <div style={{ fontWeight: "bold", marginBottom: "12px", fontSize: "1.1rem" }}>
              Query {i + 1}: {item.query}
            </div>
            <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
              {[["RAG1", item], ["RAG2", rag2Item]].map(([label, r]) => r && (
                <div key={label} style={{ flex: 1, minWidth: "280px", background: "#fafbfc", border: "1px solid #eee", borderRadius: "6px", padding: "12px" }}>
                  <div style={{ fontWeight: "bold", color: "#800000", marginBottom: "8px" }}>{label}</div>
                  <div><strong>New Retrieval Score:</strong> {r.new_retrieval_quality_score}</div>
                  <div><strong>New Answer Score:</strong> {r.new_answer_quality_score}</div>
                  <div><strong>Query Quality:</strong> {r.query_quality}</div>
                  {r.re_eval_needed?.length > 0 && (
                    <div style={{ color: "#e74c3c", marginTop: "4px" }}>⚠ Re-evaluation needed</div>
                  )}
                  {r.root_cause_analysis?.length > 0 && (
                    <div style={{ marginTop: "8px" }}>
                      <strong>Root Causes:</strong>
                      {r.root_cause_analysis.map((rca, j) => (
                        <div key={j} style={{ marginTop: "4px", padding: "6px", background: "#fff3cd", borderRadius: "4px" }}>
                          {Object.entries(rca).map(([k, v]) => (
                            <div key={k}>
                              <em style={{ color: "#800000" }}>{k}:</em>{" "}
                              {Array.isArray(v) ? v.join("; ") : v}
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function AppMain() {
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
  const [jobId, setJobId] = useState(null);
  const [queuePosition, setQueuePosition] = useState(null);
  const pollRef = useRef(null);
  const [rcaStatus, setRcaStatus] = useState("idle"); // "idle" | "running" | "done" | "error"
  const [rcaResults, setRcaResults] = useState(null);
  const [rcaJobId, setRcaJobId] = useState(null);
  const rcaPollRef = useRef(null);
  const rcaTabRef = useRef(null);

  const BACKEND_URL = "hanhanchatbot-production.up.railway.app";

  useEffect(() => {
    if (!rcaJobId) return;
    rcaPollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`https://${BACKEND_URL}/rca-status/${rcaJobId}`);
        const data = await res.json();
        if (data.status === "done") {
          const results = { rag1: data.rca_records_1, rag2: data.rca_records_2 };
          localStorage.setItem('rcaResults', JSON.stringify(results));
          setRcaResults(results);
          setRcaStatus("done");
          clearInterval(rcaPollRef.current);
          if (rcaTabRef.current) rcaTabRef.current.location.reload();
        } else if (data.status === "error") {
          setRcaStatus("error");
          clearInterval(rcaPollRef.current);
        }
      } catch {
        clearInterval(rcaPollRef.current);
      }
    }, 5000);
    return () => clearInterval(rcaPollRef.current);
  }, [rcaJobId]);

  const handleRunRCA = async () => {
    setRcaStatus("running");
    localStorage.removeItem('rcaResults');
    rcaTabRef.current = window.open(`${window.location.pathname}?view=rca`, '_blank');
    try {
      const res = await fetch(`https://${BACKEND_URL}/run-rca/${jobId}`, { method: "POST" });
      const data = await res.json();
      setRcaJobId(data.rca_job_id);
    } catch {
      setRcaStatus("error");
    }
  };

  useEffect(() => {
     if (!jobId) return;
     pollRef.current = setInterval(async () => {
         try {
             const res = await fetch(`https://${BACKEND_URL}/job-status/${jobId}`);
             const data = await res.json();
             if (data.status === "queued") {
                 setRagStatus("queued");
                 setQueuePosition(data.position);
             } else if (data.status === "running") {
                 setRagStatus("running");
                 setQueuePosition(null);
             } else if (data.status === "done") {
                 setEvalResults({ rag1: data.rag1, rag2: data.rag2 });
                 setRagStatus("done");
                clearInterval(pollRef.current);
              } else if (data.status === "error") {
                 setRagStatus("error");
                 clearInterval(pollRef.current);
             }
         } catch {
             clearInterval(pollRef.current);
         }
     }, 5000);
     return () => clearInterval(pollRef.current);
}, [jobId]);

  const handleRunRAGs = async () => {
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
        console.log("Job queued:", data);
        setJobId(data.job_id);
        setQueuePosition(data.position);
        setRagStatus(data.position === 0 ? "running" : "queued");
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
            ragStatus === "queued" ? (
             <div style={{ marginTop: "24px", fontSize: "1.2rem", fontWeight: "bold", color: "#e67e22" }}>
                 {queuePosition - 1 === 0
                    ? "You're next! Waiting for the current run to finish..."
                    : `${queuePosition - 1} user(s) waiting ahead of you...`}
             </div>
            ) : ragStatus === "running" ? (
             <div style={{ marginTop: "24px", fontSize: "2rem", fontWeight: "bold", color: "#0000ff" }}>
               Running RAG Pipelines...
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
                 {ragStatus === "done"
                  ? <>Specify <span style={{ color: "#FFFF00" }}>New</span> RAG Settings, Click Me <span style={{ color: "#FFFF00" }}>Again</span> to Run RAGs!</>
                  : "Specify RAG Settings, then Click Me to Run RAGs!"}
               </button>}
               {preprocessingStatus === "done" && ragStatus === "done" && (
                <>
                  <div style={{ marginTop: "12px", fontSize: "2rem", color: "#9932cc", fontWeight: "bold" }}>
                    RAG Performance Comparison Shown Below:
                  </div>
                  <div style={{ display: "flex", gap: "16px", marginTop: "20px", 
                    width: "100%", boxSizing: "border-box", padding: "0 16px",
                    flexWrap: "wrap" }}>
                   <EvalStackedBarChart
                     title="Retrieval Quality Score"
                     rag1Counts={evalResults.rag1?.retrieval_quality_counts}
                     rag2Counts={evalResults.rag2?.retrieval_quality_counts}
                     scoreDefinitions={RETRIEVAL_SCORE_DEFS}
                   />
                   <EvalStackedBarChart
                     title="Answer Quality Score"
                     rag1Counts={evalResults.rag1?.answer_quality_counts}
                     rag2Counts={evalResults.rag2?.answer_quality_counts}
                     scoreDefinitions={ANSWER_SCORE_DEFS}
                   />
                 </div>
                 <button
                    style={{
                      marginTop: "66px",
                      background: "#000",
                      color: "#fff",
                      border: "none",
                      borderRadius: "6px",
                      padding: "12px 24px",
                      fontSize: "1rem",
                      fontWeight: "bold",
                      cursor: rcaStatus === "running" ? "not-allowed" : "pointer",
                      opacity: rcaStatus === "running" ? 0.7 : 1,
                    }}
                    onClick={handleRunRCA}
                    disabled={rcaStatus === "running"}
                  >
                    {rcaStatus === "running" ? "Running now..." : "🔍 Root Cause Analysis"}
                  </button>
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

function App() {
   const params = new URLSearchParams(window.location.search);
   if (params.get('view') === 'rca') {
     const stored = localStorage.getItem('rcaResults');
     return <RCAResultsPage results={stored ? JSON.parse(stored) : null} />;
   }
   return <AppMain />;
 }

export default App;