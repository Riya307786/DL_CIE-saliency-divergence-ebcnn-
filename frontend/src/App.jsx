import React, { useState, useEffect } from 'react';
import { 
  Activity, Cpu, Eye, Layers, ShieldCheck, Zap, Upload, RefreshCw, 
  CheckCircle2, ArrowRight, BarChart3, Database, AlertCircle, Sparkles
} from 'lucide-react';

const CLASS_NAMES = [
  "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", 
  "Near-full", "Random", "Scratch", "none"
];

export default function App() {
  const [activeTab, setActiveTab] = useState('inference'); // 'inference' | 'stats'
  const [health, setHealth] = useState(null);
  const [samples, setSamples] = useState([]);
  const [selectedSampleId, setSelectedSampleId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [statsData, setStatsData] = useState(null);
  const [error, setError] = useState(null);

  // Fetch health check and sample wafers on startup
  useEffect(() => {
    fetchHealth();
    fetchSamples();
    fetchStats();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      }
    } catch (err) {
      console.warn("Backend offline or starting up...", err);
    }
  };

  const fetchSamples = async () => {
    try {
      const res = await fetch('/api/samples?limit=9');
      if (res.ok) {
        const data = await res.json();
        setSamples(data.samples || []);
        if (data.samples && data.samples.length > 0) {
          setSelectedSampleId(data.samples[0].sample_id);
        }
      }
    } catch (err) {
      console.warn("Failed to fetch test samples:", err);
    }
  };

  const fetchStats = async () => {
    try {
      const res = await fetch('/api/stats');
      if (res.ok) {
        const data = await res.json();
        setStatsData(data);
      }
    } catch (err) {
      console.warn("Failed to fetch stats:", err);
    }
  };

  const handleRunInference = async (sampleIdToRun = selectedSampleId) => {
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      if (sampleIdToRun !== null) {
        formData.append('sample_id', sampleIdToRun);
      }

      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Inference failed: ${res.statusText}`);
      }

      const data = await res.json();
      setPrediction(data);
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to complete real inference.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`File inference failed: ${res.statusText}`);
      }

      const data = await res.json();
      setPrediction(data);
      setSelectedSampleId(null);
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to process uploaded wafer image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '1.5rem' }}>
      {/* Header Bar */}
      <header className="glass-panel" style={{ padding: '1.25rem 2rem', marginBottom: '2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.25rem' }}>
            <Sparkles className="text-cyan-400" size={24} color="#38bdf8" />
            <h1 style={{ fontSize: '1.4rem' }}>Saliency-Divergence Branch Selection</h1>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
            Replacing Accuracy-Heuristic Ensembling with Explainability-Driven Criteria for Spatially-Scattered Defect Classification
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div className={`badge ${health?.model_loaded ? 'badge-green' : 'badge-amber'}`}>
            <Activity size={12} />
            {health?.model_loaded ? `Backend Device: ${health.device}` : 'API Disconnected'}
          </div>

          <div style={{ display: 'flex', background: 'rgba(30, 41, 59, 0.6)', padding: '0.25rem', borderRadius: '10px' }}>
            <button
              onClick={() => setActiveTab('inference')}
              className={activeTab === 'inference' ? 'btn-primary' : 'btn-secondary'}
              style={{ padding: '0.4rem 1rem', fontSize: '0.85rem' }}
            >
              <Zap size={14} /> Real Inference MVP
            </button>
            <button
              onClick={() => setActiveTab('stats')}
              className={activeTab === 'stats' ? 'btn-primary' : 'btn-secondary'}
              style={{ padding: '0.4rem 1rem', fontSize: '0.85rem' }}
            >
              <BarChart3 size={14} /> 10-Seed Statistics
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      {activeTab === 'inference' ? (
        <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: '1.5rem' }}>
          
          {/* Left Column: Sample Selector & File Upload */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            
            {/* Custom Upload Card */}
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Upload size={18} color="#38bdf8" /> Custom Wafer Upload
              </h3>
              <label 
                style={{ 
                  display: 'flex', 
                  flexDirection: 'column', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  border: '2px dashed rgba(56, 189, 248, 0.3)', 
                  borderRadius: '12px', 
                  padding: '1.5rem 1rem', 
                  cursor: 'pointer',
                  background: 'rgba(15, 23, 42, 0.5)',
                  transition: 'border-color 0.2s ease'
                }}
              >
                <Upload size={28} color="#94a3b8" style={{ marginBottom: '0.5rem' }} />
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  Click to Upload Wafer Map
                </span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  Supports PNG, JPG, or NPY matrix
                </span>
                <input type="file" accept="image/*,.npy" onChange={handleFileUpload} style={{ display: 'none' }} />
              </label>
            </div>

            {/* Test Sample Selector Grid */}
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <h3 style={{ fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Database size={18} color="#818cf8" /> Test Samples
                </h3>
                <button onClick={fetchSamples} className="btn-secondary" style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}>
                  <RefreshCw size={12} />
                </button>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem' }}>
                {samples.map((s) => (
                  <button
                    key={s.sample_id}
                    onClick={() => {
                      setSelectedSampleId(s.sample_id);
                      handleRunInference(s.sample_id);
                    }}
                    style={{
                      background: selectedSampleId === s.sample_id ? 'rgba(56, 189, 248, 0.2)' : 'rgba(30, 41, 59, 0.5)',
                      border: selectedSampleId === s.sample_id ? '1px solid var(--accent-cyan)' : '1px solid rgba(255, 255, 255, 0.08)',
                      borderRadius: '8px',
                      padding: '0.5rem',
                      cursor: 'pointer',
                      textAlign: 'center'
                    }}
                  >
                    <img src={s.wafer_image_b64} alt={s.class_name} style={{ width: '100%', height: '60px', objectFit: 'contain', borderRadius: '4px' }} />
                    <span style={{ fontSize: '0.7rem', display: 'block', marginTop: '0.25rem', fontWeight: 600, color: s.class_name === 'Donut' || s.class_name === 'Random' ? '#f59e0b' : '#94a3b8' }}>
                      {s.class_name}
                    </span>
                  </button>
                ))}
              </div>

              <button
                onClick={() => handleRunInference()}
                disabled={loading}
                className="btn-primary"
                style={{ width: '100%', marginTop: '1.25rem', justifyContent: 'center' }}
              >
                {loading ? <RefreshCw className="animate-spin" size={16} /> : <Zap size={16} />}
                Run Real Inference Pass
              </button>
            </div>

          </div>

          {/* Right Column: Dynamic Prediction & Grad-CAM & Divergence Results */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            
            {error && (
              <div className="glass-panel" style={{ borderLeft: '4px solid var(--accent-rose)', padding: '1rem 1.5rem', color: '#fca5a5', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            )}

            {!prediction && !loading && (
              <div className="glass-panel" style={{ padding: '4rem 2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                <Cpu size={48} color="#38bdf8" style={{ margin: '0 auto 1rem', opacity: 0.8 }} />
                <h3 style={{ color: 'var(--text-main)', fontSize: '1.2rem', marginBottom: '0.5rem' }}>No Inference Loaded</h3>
                <p style={{ maxWidth: '400px', margin: '0 auto', fontSize: '0.9rem' }}>
                  Select a wafer sample on the left or upload a custom wafer map to stream real EB-CNN forward inference, Grad-CAM heatmaps, and Saliency-Divergence branch selection.
                </p>
              </div>
            )}

            {prediction && (
              <>
                {/* Section 1: Decision Comparison (Baseline vs Proposed) */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                  
                  {/* Original EB-CNN (Baseline) */}
                  <div className="glass-panel" style={{ padding: '1.5rem', borderTop: '4px solid #3b82f6' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                      <div>
                        <span className="badge badge-cyan" style={{ marginBottom: '0.5rem' }}>Original EB-CNN (Baseline)</span>
                        <h3 style={{ fontSize: '1.2rem' }}>{prediction.baseline_selection.class_name}</h3>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>Confidence</span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '1.1rem', color: '#60a5fa' }}>
                          {(prediction.baseline_selection.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    <div style={{ background: 'rgba(15, 23, 42, 0.6)', borderRadius: '8px', padding: '0.85rem', marginBottom: '1rem' }}>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.35rem' }}>
                        Ensemble Rule (Accuracy Heuristic):
                      </span>
                      <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
                        {prediction.baseline_selection.selected_branches.map((b) => (
                          <span key={b} className="badge" style={{ background: 'rgba(59, 130, 246, 0.2)', color: '#93c5fd' }}>
                            {b}
                          </span>
                        ))}
                      </div>
                    </div>

                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      Original heuristic evaluates stacked combos C1..C5 based solely on validation accuracy without spatial explainability.
                    </p>
                  </div>

                  {/* Proposed Saliency-Divergence */}
                  <div className="glass-panel" style={{ padding: '1.5rem', borderTop: '4px solid #10b981', background: 'rgba(16, 185, 129, 0.05)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                      <div>
                        <span className="badge badge-green" style={{ marginBottom: '0.5rem' }}>Proposed (Saliency-Divergence)</span>
                        <h3 style={{ fontSize: '1.2rem', color: '#34d399' }}>{prediction.proposed_selection.class_name}</h3>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>Confidence</span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '1.1rem', color: '#34d399' }}>
                          {(prediction.proposed_selection.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    <div style={{ background: 'rgba(15, 23, 42, 0.6)', borderRadius: '8px', padding: '0.85rem', marginBottom: '1rem' }}>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.35rem' }}>
                        Selected Diverse Branches (Predictive Quality + Saliency Complementarity):
                      </span>
                      <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
                        {prediction.proposed_selection.selected_branches.map((b) => (
                          <span key={b} className="badge badge-green">
                            {b}
                          </span>
                        ))}
                      </div>
                    </div>

                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      Dynamically prunes redundant representations and selects spatially complementary feature maps ($D(i, j)$).
                    </p>
                  </div>

                </div>

                {/* Section 2: Real Grad-CAM Saliency Maps Across B1..B5 */}
                <div className="glass-panel" style={{ padding: '1.5rem' }}>
                  <h3 style={{ fontSize: '1.1rem', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Eye size={18} color="#38bdf8" /> Real Grad-CAM Saliency Maps Across VGG-16 Branches ($B_1 \dots B_5$)
                  </h3>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '1rem' }}>
                    {["B1", "B2", "B3", "B4", "B5"].map((b, idx) => {
                      const isSelectedBaseline = prediction.baseline_selection.selected_branches.includes(b);
                      const isSelectedProposed = prediction.proposed_selection.selected_branches.includes(b);

                      return (
                        <div key={b} className="glass-card" style={{ textAlign: 'center', padding: '0.75rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                            <span style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', fontSize: '0.9rem' }}>{b}</span>
                            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Stage {idx + 1}</span>
                          </div>

                          <img 
                            src={prediction.gradcam_heatmaps[b]} 
                            alt={`Grad-CAM ${b}`} 
                            style={{ width: '100%', borderRadius: '6px', marginBottom: '0.5rem', border: '1px solid rgba(255, 255, 255, 0.1)' }} 
                          />

                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                            {isSelectedProposed && (
                              <span className="badge badge-green" style={{ fontSize: '0.65rem', padding: '0.15rem 0.35rem' }}>
                                Proposed
                              </span>
                            )}
                            {isSelectedBaseline && (
                              <span className="badge badge-cyan" style={{ fontSize: '0.65rem', padding: '0.15rem 0.35rem' }}>
                                Baseline
                              </span>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Section 3: 5x5 Pairwise Spatial Saliency Divergence Matrix */}
                <div className="glass-panel" style={{ padding: '1.5rem' }}>
                  <h3 style={{ fontSize: '1.1rem', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Layers size={18} color="#f59e0b" /> Pairwise Spatial Saliency Divergence Matrix D(i, j) ∈ [0.0, 1.0] (5×5 Matrix)
                  </h3>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '0.5rem', maxWidth: '600px', margin: '0 auto' }}>
                    {/* Header Row */}
                    <div style={{ textAlign: 'center', fontWeight: 700, color: 'var(--text-muted)' }}></div>
                    {["B1", "B2", "B3", "B4", "B5"].map((b) => (
                      <div key={b} style={{ textAlign: 'center', fontWeight: 700, color: 'var(--text-muted)', fontSize: '0.85rem' }}>{b}</div>
                    ))}

                    {/* Matrix Rows */}
                    {prediction.divergence_matrix.map((row, i) => (
                      <React.Fragment key={i}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                          B{i + 1}
                        </div>
                        {row.map((val, j) => {
                          const bgAlpha = i === j ? 0.05 : val * 0.7;
                          const color = i === j ? '#475569' : val > 0.5 ? '#f59e0b' : '#38bdf8';
                          return (
                            <div 
                              key={j} 
                              className="heatmap-cell"
                              style={{ 
                                background: i === j ? 'rgba(30, 41, 59, 0.4)' : `rgba(245, 158, 11, ${bgAlpha})`,
                                color: color,
                                border: '1px solid rgba(255, 255, 255, 0.05)'
                              }}
                            >
                              {val.toFixed(2)}
                            </div>
                          );
                        })}
                      </React.Fragment>
                    ))}
                  </div>
                </div>

              </>
            )}

          </div>

        </div>
      ) : (
        /* Statistical Benchmark & Experimental Results Tab */
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          
          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <h2 style={{ fontSize: '1.3rem', marginBottom: '0.5rem' }}>10-Seed Paired Experimental Benchmark (Paper 3 Protocol)</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
              Multi-seed evaluation across 10 fixed seeds ($101 \dots 110$) with paired statistical tests (Friedman, Wilcoxon signed-rank with Holm correction, and McNemar test).
            </p>

            {/* Results Table */}
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.875rem' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid rgba(255, 255, 255, 0.1)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '0.75rem' }}>Method</th>
                    <th style={{ padding: '0.75rem' }}>Accuracy (Mean ± Std)</th>
                    <th style={{ padding: '0.75rem' }}>Macro F1 (Mean ± Std)</th>
                    <th style={{ padding: '0.75rem' }}>Donut F1 (Spatially Scattered)</th>
                    <th style={{ padding: '0.75rem' }}>Random F1 (Spatially Scattered)</th>
                  </tr>
                </thead>
                <tbody>
                  {statsData?.experiments_summary?.summary_table ? (
                    statsData.experiments_summary.summary_table.map((row, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)', background: row.Method.includes('Proposed') ? 'rgba(16, 185, 129, 0.05)' : 'transparent' }}>
                        <td style={{ padding: '0.75rem', fontWeight: 600, color: row.Method.includes('Proposed') ? '#34d399' : 'var(--text-main)' }}>
                          {row.Method.replace(/_/g, ' ')}
                        </td>
                        <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)' }}>{(row.Accuracy_Mean * 100).toFixed(2)}% ± {(row.Accuracy_Std * 100).toFixed(2)}%</td>
                        <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{row.Macro_F1_Mean.toFixed(4)} ± {row.Macro_F1_Std.toFixed(4)}</td>
                        <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)', color: '#f59e0b' }}>{row.Donut_F1_Mean.toFixed(4)}</td>
                        <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>{row.Random_F1_Mean.toFixed(4)}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="5" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                        Statistical results generated by full 10-seed experiment run.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

        </div>
      )}
    </div>
  );
}
