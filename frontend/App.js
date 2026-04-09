import { useState, useEffect, useRef } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";

const BACKEND_URL = "hanhanchatbot-production.up.railway.app";

const embeddingModels = [
  { label: "text-embedding-3-small", value: "text-embedding-3-small" },
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
            <option key={model.value} value={model.value}>{model.label}</option>
          ))}
        </select>
      </div>
      <div style={{ marginBottom: "16px" }}>
        <label style={{ fontWeight: "bold" }}>Top N Retrieved Content:</label>
        <br />
        <select value={topN} onChange={e => onTopNChange(Number(e.target.value))} style={{ marginTop: "8px", padding: "6px", width: "100%" }}>
          {[1, 2, 3, 4, 5].map(n => <option key={n} value={n}>Top {n}</option>)}
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
            <option key={model.value} value={model.value}>{model.label}</option>
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
  const [showTip, setShowTip] = useState(false);
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
          <Legend iconSize={10} wrapperStyle={{ fontSize: "15px", paddingLeft: "37px" }} />
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

function ExpandableText({ text, maxTokens = 66 }) {
  const [expanded, setExpanded] = useState(false);
  const tokens = (text || "").split(/\s+/).filter(Boolean);
  const isLong = tokens.length > maxTokens;
  const display = (!isLong || expanded) ? text : tokens.slice(0, maxTokens).join(" ") + "…";
  return (
    <>
      {display}
      {isLong && (
        <div>
          <button onClick={() => setExpanded(e => !e)} style={{
            marginTop: "4px", background: "none", border: "none",
            color: "#800000", cursor: "pointer", fontSize: "0.75rem",
            padding: 0, fontWeight: "bold",
          }}>
            {expanded ? "▲ Less" : "▼ More"}
          </button>
        </div>
      )}
    </>
  );
}

function ExpandableCell({ text, style, maxTokens = 66 }) {
  return <td style={style}><ExpandableText text={text} maxTokens={maxTokens} /></td>;
}

function RCAResultsPage({ results }) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!results) return (
    <div style={{ padding: "32px", fontFamily: "Calibri, sans-serif", color: "#0000ff", fontSize: "1.8rem",
      height: "100vh", overflowY: "auto", boxSizing: "border-box" }}>
      ⏳ Analysis is running... this page will update automatically when done.
    </div>
  );

  const sharedReEvalRows = results.rag1
    .map((item, i) => ({ item, rag2Item: results.rag2[i], i }))
    .filter(({ item, rag2Item }) =>
      item?.needs_re_eval === 1 && rag2Item?.needs_re_eval === 1
    );

  function mergeDistinctSuggestions(rca1, rca2) {
    const seen = new Set();
    const result = [];
    for (const obj of [...(rca1 || []), ...(rca2 || [])]) {
      for (const [k, v] of Object.entries(obj)) {
        const display = Array.isArray(v) ? v.join("; ") : v;
        const key = `${k}: ${display}`;
        if (!seen.has(key)) { seen.add(key); result.push({ k, display }); }
      }
    }
    return result;
  }

  return (
    <div style={{ padding: "32px", fontFamily: "Calibri, sans-serif",
      height: "100vh", overflowY: "auto", boxSizing: "border-box" }}>
      <h1 style={{ color: "#800000", marginBottom: "24px" }}>Root Cause Analysis Results</h1>

      {sharedReEvalRows.length > 0 && (
        <div style={{ marginBottom: "40px" }}>
          <h2 style={{ color: "#e74c3c", marginBottom: "16px" }}>⚠ Suggest to re-evaluate records below:</h2>
          <div style={{ position: "relative" }}>
            <div style={{
              overflowX: "auto",
              overflowY: "hidden",
              width: "100%",
              maxHeight: isExpanded ? "none" : "120px",
            }}>
              <table style={{ borderCollapse: "collapse", minWidth: "1500px", fontSize: "0.85rem", width: "100%" }}>
                <thead>
                  <tr style={{ background: "#fdf0f0" }}>
                    {[
                      { label: "Row Number",                          minW: "40px"  },
                      { label: "Query",                               minW: "180px" },
                      { label: "Referenced Content",                  minW: "200px" },
                      { label: "Retrieved Content",                   minW: "200px" },
                      { label: "Referenced Answer",                   minW: "180px" },
                      { label: "AI's Answer\nRAG1 / RAG2",            minW: "180px" },
                      { label: "Retrieval Quality Score\nRAG1 | RAG2", minW: "160px" },
                      { label: "Answer Quality Score\nRAG1 | RAG2",   minW: "160px" },
                      { label: "Suggestions",                         minW: "260px" },
                    ].map(({ label, minW }) => (
                      <th key={label} style={{
                        border: "1px solid #ccc", padding: "8px 12px", textAlign: "center",
                        whiteSpace: "pre-line", color: "#800000", fontWeight: "bold",
                        minWidth: minW, background: "#fdf0f0",
                      }}>
                        {label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sharedReEvalRows.map(({ item, rag2Item, i }) => {
                    const suggestions = mergeDistinctSuggestions(
                      item.root_cause_analysis, rag2Item.root_cause_analysis
                    );
                    const cellStyle = {
                      border: "1px solid #ccc", padding: "8px 12px",
                      verticalAlign: "top", wordBreak: "break-word",
                      background: i % 2 === 0 ? "#fff" : "#fafafa",
                    };
                    return (
                      <tr key={i}>
                        <td style={cellStyle}>{i + 1}</td>
                        <td style={cellStyle}>{item.query}</td>
                        <ExpandableCell text={item.context} style={cellStyle} />
                        <ExpandableCell text={item.retrieved_content} style={cellStyle} />
                        <ExpandableCell text={item.referenced_answer ?? item.expected_answer} style={cellStyle} />
                        <td style={{ ...cellStyle, padding: 0 }}>
                          <div style={{ padding: "8px 12px", borderBottom: "1px solid #e0e0e0" }}>
                            <span style={{ fontSize: "0.75rem", color: "#888", fontWeight: "bold" }}>RAG1</span><br />
                            <ExpandableText text={item.ai_answer} />
                          </div>
                          <div style={{ padding: "8px 6px" }}>
                            <span style={{ fontSize: "0.75rem", color: "#888", fontWeight: "bold" }}>RAG2</span><br />
                            <ExpandableText text={rag2Item.ai_answer} />
                          </div>
                        </td>
                        <td style={{ ...cellStyle, textAlign: "center" }}>
                          <span style={{ color: SCORE_COLORS[String(item.new_retrieval_quality_score)] || "#333" }}>
                            {item.new_retrieval_quality_score}
                          </span>
                          {" | "}
                          <span style={{ color: SCORE_COLORS[String(rag2Item.new_retrieval_quality_score)] || "#333" }}>
                            {rag2Item.new_retrieval_quality_score}
                          </span>
                        </td>
                        <td style={{ ...cellStyle, textAlign: "center" }}>
                          <span style={{ color: SCORE_COLORS[String(item.new_answer_quality_score)] || "#333" }}>
                            {item.new_answer_quality_score}
                          </span>
                          {" | "}
                          <span style={{ color: SCORE_COLORS[String(rag2Item.new_answer_quality_score)] || "#333" }}>
                            {rag2Item.new_answer_quality_score}
                          </span>
                        </td>
                        <td style={cellStyle}>
                          {suggestions.map(({ k, display }, j) => (
                            <div key={j} style={{ marginBottom: "4px" }}>
                              <em style={{ color: "#800000" }}>{k}:</em> {display}
                            </div>
                          ))}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {!isExpanded && (
              <div style={{
                position: "absolute",
                bottom: 0, left: 0, right: 0,
                height: "48px",
                background: "linear-gradient(transparent, #fff)",
                pointerEvents: "none",
              }} />
            )}
          </div>
          <button
            onClick={() => setIsExpanded(e => !e)}
            style={{
              marginTop: "8px",
              background: "none",
              border: "1px solid #ccc",
              borderRadius: "4px",
              padding: "4px 16px",
              cursor: "pointer",
              fontSize: "0.85rem",
              color: "#800000",
              fontWeight: "bold",
            }}
          >
            {isExpanded ? "▲ Collapse" : "▼ Expand table"}
          </button>
        </div>
      )}
      {(results.agg_review_1 || results.agg_review_2) && (
        <div style={{ marginTop: "40px" }}>
          <h2 style={{ color: "#800000", marginBottom: "16px" }}>📊 Aggregate RAG System Review</h2>
          <div style={{ display: "flex", gap: "24px", alignItems: "flex-start" }}>
            {[
              { label: "RAG 1", review: results.agg_review_1 },
              { label: "RAG 2", review: results.agg_review_2 },
            ].map(({ label, review }) => (
              <div key={label} style={{ flex: 1, border: "1px solid #ccc", borderRadius: "8px",
                padding: "20px", background: "#fafbfc" }}>
                <h3 style={{ color: "#800000", marginBottom: "12px" }}>{label}</h3>
                {review ? (
                  <>
                    <div style={{ marginBottom: "16px" }}>
                      <strong style={{ color: "#e74c3c" }}>Root Cause Analysis:</strong>
                      <p style={{ marginTop: "8px", whiteSpace: "pre-wrap", lineHeight: "1.6" }}>
                        {review.root_cause_analysis}
                      </p>
                    </div>
                    <div>
                      <strong style={{ color: "#27ae60" }}>Improvement Suggestions:</strong>
                      <p style={{ marginTop: "8px", whiteSpace: "pre-wrap", lineHeight: "1.6" }}>
                        {review.improvement_suggestions}
                      </p>
                    </div>
                  </>
                ) : (
                  <p style={{ color: "#888", fontStyle: "italic" }}>
                    All records flagged for re-evaluation — no aggregate review available.
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Page 1: Dataset Selection ───────────────────────────────────────────────

function DatasetPage({ onDatasetReady }) {
  const [selectedDataset, setSelectedDataset] = useState("");
  const [datasetClicked, setDatasetClicked] = useState(false);
  const [preprocessingStatus, setPreprocessingStatus] = useState("idle");
  const [preprocessingMessage, setPreprocessingMessage] = useState("");

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
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "flex-start",
        width: "100%",
        minHeight: "100vh",
        background: "#f5f6fa",
        fontFamily: "Calibri, sans-serif",
        padding: "40px 24px 12px",
        boxSizing: "border-box",
      }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }
        @keyframes arrowBounce { 0% { transform: translateX(0); opacity: 1; } 50% { transform: translateX(10px); opacity: 0.95; } 100% { transform: translateX(0); opacity: 1; } }
      `}</style>

      <h1 style={{ fontSize: "5.6rem", fontWeight: 800, color: "#800000", margin: "0 0 32px 0" }}>
        RAG Doctor
      </h1>

      <div style={{ width: "620px" }}>
        <p style={{ color: "#666", fontSize: "1.1rem", marginBottom: "8px", marginTop: 0 }}>
          Select a dataset:
        </p>
      <table style={{ width: "620px", borderCollapse: "collapse", marginBottom: "12px", border: "2px solid #888" }}>
        <thead>
          <tr>
            <th style={{ textAlign: "center", fontWeight: "bold", fontSize: "1.1rem", border: "1px solid #888", padding: "12px", background: "#f3f3f3" }}>
              Dataset
            </th>
            <th style={{ textAlign: "center", fontWeight: "bold", fontSize: "1.1rem", border: "1px solid #888", padding: "12px", background: "#f3f3f3" }}>
              Description
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ textAlign: "center", padding: "12px", border: "1px solid #888", background: "#fff" }}>
              <button
                style={{
                  background: datasetClicked && selectedDataset === "FIQA Data" ? "#F0620A" : "#549d07",
                  color: "#fff",
                  border: "none",
                  borderRadius: "6px",
                  padding: "10px 24px",
                  fontSize: "1rem",
                  fontWeight: "bold",
                  cursor: "pointer",
                  transition: "background 0.2s",
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
                  justifyContent: "center", gap: "8px", fontSize: "0.85rem" }}>
                  {preprocessingStatus === "running" && (
                    <div style={{
                      width: "14px", height: "14px",
                      border: "2px solid #ccc",
                      borderTop: "2px solid #F0620A",
                      borderRadius: "50%",
                      animation: "spin 0.8s linear infinite",
                      flexShrink: 0,
                    }} />
                  )}
                  <span style={{ color: preprocessingStatus === "running" ? "#0000ff" : "#333" }}>
                    {preprocessingMessage}
                  </span>
                </div>
              )}
            </td>
            <td style={{ textAlign: "left", padding: "12px", border: "1px solid #888", background: "#fff" }}>
              30 records of finance Q&amp;A.<br /> See details{" "}
              <a href="https://huggingface.co/datasets/vibrantlabsai/fiqa" target="_blank" rel="noopener noreferrer">
                here &gt;&gt;
              </a>
            </td>
          </tr>
        </tbody>
      </table>
      </div>

      <div style={{ flex: 1, display: "flex", alignItems: "flex-start", justifyContent: "flex-start", paddingTop: "44px" }}>
        {preprocessingStatus === "done" && (
        <button
          onClick={() => onDatasetReady(selectedDataset)}
          style={{
            background: "#800000",
            color: "#fff",
            border: "none",
            borderRadius: "8px",
            padding: "16px 56px",
            fontSize: "1.3rem",
            fontWeight: "bold",
            cursor: "pointer",
            boxShadow: "0 4px 16px rgba(128,0,0,0.25)",
            transition: "background 0.2s",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 12,
          }}
        >
          <span style={{ display: "inline-flex", alignItems: "center", gap: 12 }}>
            <span>RAG A/B Test</span>
            <span style={{ color: "#fff", fontWeight: 900, display: "inline-block", fontSize: "2.2rem", lineHeight: 1, textShadow: "0 1px 0 rgba(0,0,0,0.25)", animation: "arrowBounce 1s ease-in-out infinite" }}>➜</span>
          </span>
        </button>
        )}
      </div>
    </div>
  );
}

// ─── Page 2: RAG A/B Test ─────────────────────────────────────────────────────

function ABTestPage({ selectedDataset }) {
  const [rag1Model, setRag1Model] = useState(embeddingModels[0].value);
  const [rag2Model, setRag2Model] = useState(embeddingModels[0].value);
  const [rag1TopN, setRag1TopN] = useState(1);
  const [rag2TopN, setRag2TopN] = useState(1);
  const [rag1SemanticWeight, setRag1SemanticWeight] = useState(0.5);
  const [rag2SemanticWeight, setRag2SemanticWeight] = useState(0.5);
  const [rag1AGLLM, setRag1AGLLM] = useState(answerGenLLMModels[0].value);
  const [rag2AGLLM, setRag2AGLLM] = useState(answerGenLLMModels[0].value);
  const [ragStatus, setRagStatus] = useState("idle");
  const [evalResults, setEvalResults] = useState({ rag1: null, rag2: null });
  const [jobId, setJobId] = useState(null);
  const [queuePosition, setQueuePosition] = useState(null);
  const pollRef = useRef(null);
  const [rcaStatus, setRcaStatus] = useState("idle");
  const [rcaJobId, setRcaJobId] = useState(null);
  const rcaPollRef = useRef(null);
  const rcaTabRef = useRef(null);
  const [settingsChangedAfterRCA, setSettingsChangedAfterRCA] = useState(false);

  useEffect(() => {
    if (!rcaJobId) return;
    rcaPollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`https://${BACKEND_URL}/rca-status/${rcaJobId}`);
        const data = await res.json();
        if (data.status === "done") {
          const results = {
            rag1: data.rca_records_1,
            rag2: data.rca_records_2,
            agg_review_1: data.agg_review_1,
            agg_review_2: data.agg_review_2,
          };
          localStorage.setItem('rcaResults', JSON.stringify(results));
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

  useEffect(() => {
    if (rcaStatus === "done") setRcaStatus("idle");
  }, [rag1Model, rag1TopN, rag1SemanticWeight, rag1AGLLM,
      rag2Model, rag2TopN, rag2SemanticWeight, rag2AGLLM]);

  useEffect(() => {
    if (rcaStatus === "done") setSettingsChangedAfterRCA(true);
  }, [rag1Model, rag1TopN, rag1SemanticWeight, rag1AGLLM,
      rag2Model, rag2TopN, rag2SemanticWeight, rag2AGLLM]);

  useEffect(() => {
    if (ragStatus === "done") setSettingsChangedAfterRCA(false);
  }, [ragStatus]);

  const handleRunRCA = async () => {
    setSettingsChangedAfterRCA(false);
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
        headers: { "Content-Type": "application/json" },
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
      setJobId(data.job_id);
      setQueuePosition(data.position);
      setRagStatus(data.position === 0 ? "running" : "queued");
    } catch (error) {
      console.error("Error running RAGs:", error);
      setRagStatus("error");
    }
  };

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      width: "100%",
      height: "100vh",
      background: "#f5f6fa",
      fontFamily: "Calibri, sans-serif",
      boxSizing: "border-box",
      overflow: "hidden",
    }}>
      {/* ── Top Header Bar ── */}
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "10px 24px",
        background: "#fff",
        borderBottom: "2px solid #e0e0e0",
        flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <h1 style={{ margin: 0, fontSize: "4.4rem", fontWeight: 800, color: "#800000" }}>RAG Doctor</h1>
          <span style={{
            background: "#fdf0e8",
            border: "1px solid #F0620A",
            borderRadius: "6px",
            padding: "4px 14px",
            fontWeight: "bold",
            color: "#F0620A",
            fontSize: "0.95rem",
          }}>
            📊 {selectedDataset}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <a
            href="https://github.com/hanhanwu/RagDoctor"
            target="_blank"
            rel="noopener noreferrer"
            title="Open GitHub repo"
            style={{ display: "inline-flex", alignItems: "center", textDecoration: "none", color: "inherit" }}
          >
            <svg height="28" viewBox="0 0 16 16" version="1.1" width="28" aria-hidden="true" fill="#24292f">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.5-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38C13.71 14.53 16 11.54 16 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
          </a>
          <iframe
            src="https://ghbtns.com/github-btn.html?user=hanhanwu&repo=RagDoctor&type=star&count=true"
            frameBorder="0" scrolling="0" width="100" height="20" title="GitHub Stars"
          />
        </div>
      </div>

      {/* ── Body: 3-column layout ── */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>

        {/* Left: RAG 1 Settings */}
        <div style={{ flex: 1, overflowY: "auto", minWidth: 0 }}>
          <RAGSettings
            title="RAG 1 Settings"
            selectedModel={rag1Model}
            onModelChange={setRag1Model}
            topN={rag1TopN}
            onTopNChange={setRag1TopN}
            semanticWeight={rag1SemanticWeight}
            onSemanticWeightChange={setRag1SemanticWeight}
            agLLM={rag1AGLLM}
            onAGLLMChange={setRag1AGLLM}
          />
        </div>

        {/* Center: Compare button + results */}
        <div style={{
          flex: 2.2,
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          padding: "24px 16px",
          boxSizing: "border-box",
        }}>
          {ragStatus === "queued" ? (
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
              <button
                onClick={handleRunRAGs}
                style={{
                  width: "80%",
                  background: "#800000",
                  color: "#fff",
                  border: "none",
                  borderRadius: "6px",
                  padding: "14px 24px",
                  fontSize: "1.1rem",
                  fontWeight: "bold",
                  cursor: "pointer",
                  marginTop: "8px",
                  letterSpacing: "0.04em",
                }}
              >
                {ragStatus === "done"
                  ? <>Specify <span style={{ color: "#FFFF00" }}>New</span> RAG Settings, Click <span style={{ color: "#FFFF00" }}>Compare</span> Again!</>
                  : "Compare"}
              </button>

              {ragStatus === "done" && (
                <>
                  <div style={{ marginTop: "12px", fontSize: "2rem", color: "#9932cc", fontWeight: "bold" }}>
                    RAG Performance Comparison Shown Below:
                  </div>
                  <div style={{ display: "flex", gap: "16px", marginTop: "20px",
                    width: "100%", boxSizing: "border-box", padding: "0 16px", flexWrap: "wrap" }}>
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
                  {!settingsChangedAfterRCA && (
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
                  )}
                </>
              )}
            </>
          )}
        </div>

        {/* Right: RAG 2 Settings */}
        <div style={{ flex: 1, overflowY: "auto", minWidth: 0 }}>
          <RAGSettings
            title="RAG 2 Settings"
            selectedModel={rag2Model}
            onModelChange={setRag2Model}
            topN={rag2TopN}
            onTopNChange={setRag2TopN}
            semanticWeight={rag2SemanticWeight}
            onSemanticWeightChange={setRag2SemanticWeight}
            agLLM={rag2AGLLM}
            onAGLLMChange={setRag2AGLLM}
          />
        </div>
      </div>
    </div>
  );
}

// ─── App Router ───────────────────────────────────────────────────────────────

function AppMain() {
  const [page, setPage] = useState("dataset");
  const [selectedDataset, setSelectedDataset] = useState("");

  if (page === "dataset") {
    return (
      <DatasetPage
        onDatasetReady={(ds) => {
          setSelectedDataset(ds);
          setPage("abtest");
        }}
      />
    );
  }
  return <ABTestPage selectedDataset={selectedDataset} />;
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
