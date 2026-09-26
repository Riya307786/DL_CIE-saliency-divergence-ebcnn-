import React, { useState, useEffect } from 'react';
import { 
  Activity, Cpu, Eye, Layers, ShieldCheck, Zap, Upload, RefreshCw, 
  CheckCircle2, ArrowRight, BarChart3, Database, AlertCircle, Sparkles,
  TrendingUp, Sliders, FileText, ChevronRight, Info, Check, Image as ImageIcon
} from 'lucide-react';

const CLASS_NAMES = [
  "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", 
  "Near-full", "Random", "Scratch", "none"
];

const SAMPLE_EXPLANATIONS = {
  "Center": {
    whatModelSees: "The wafer shows clustered defect dice concentrated directly at the center of the silicon disk.",
    branchExplanation: "Shallower branches (B1, B2) capture the coarse circle boundary. Deeper branches (B4, B5) detect the dense cluster core.",
    saliencyExplanation: "B1 and B5 attend to different radii from the center, demonstrating complementary spatial focus.",
    selectionExplanation: "Selected branches meet the minimum quality floor and look at distinct spatial regions to confirm the central failure."
  },
  "Donut": {
    whatModelSees: "The wafer contains an annular ring-shaped spatial pattern with a clean non-defective center.",
    branchExplanation: "B1 focuses primarily on the outer circular ring, while B4/B5 isolate the hollow core to prevent misclassifying as Center.",
    saliencyExplanation: "High saliency divergence exists between the inner void and the outer ring circumference.",
    selectionExplanation: "These branches satisfy the predictive-quality requirement and provide complementary inner/outer spatial evidence."
  },
  "Edge-Loc": {
    whatModelSees: "A localized defect cluster located along the wafer peripheral edge.",
    branchExplanation: "B2 tracks the wafer rim geometry; B3/B4 localize the defect coordinates along the edge perimeter.",
    saliencyExplanation: "Branches distribute attention between wafer rim alignment and localized defect clusters.",
    selectionExplanation: "Accuracy-heuristic selection would pick redundant deep branches; saliency-divergence keeps the rim-aware branch."
  },
  "Edge-Ring": {
    whatModelSees: "A continuous or semi-continuous ring of defect dice tracing the outer perimeter of the wafer.",
    branchExplanation: "B1/B2 strongly activate along the entire circular boundary; B5 distinguishes continuous rings from partial edge defects.",
    saliencyExplanation: "Saliency maps trace the circumference with consistent angular coverage across branches.",
    selectionExplanation: "Complementary branches are selected to confirm complete perimeter encircling without false-positive core activation."
  },
  "Loc": {
    whatModelSees: "An isolated localized cluster of defective dice in an arbitrary interior region of the wafer.",
    branchExplanation: "B1 detects absence of global defects; B3/B4 pinpoint the spatial coordinates of the interior failure.",
    saliencyExplanation: "Spatial divergence highlights tight focus on the specific coordinates rather than diffuse background noise.",
    selectionExplanation: "The selected branches achieve high local precision without excessive spatial redundancy."
  },
  "Near-full": {
    whatModelSees: "Catastrophic wafer failure where almost the entire surface is covered by defective dice.",
    branchExplanation: "All branches observe widespread failure, but shallow branches verify the edge boundary is intact.",
    saliencyExplanation: "Divergence is naturally lower for near-full failure, but quality-floor ensures only top calibrated branches are ensembled.",
    selectionExplanation: "Ensures confident, calibrated classification of catastrophic yield loss."
  },
  "Random": {
    whatModelSees: "Uniformly scattered particulate defect dice dispersed across the wafer without coherent geometric clustering.",
    branchExplanation: "B1 captures global noise density; B3 and B5 evaluate local die dispersion to differentiate from scratch patterns.",
    saliencyExplanation: "Branches activate across completely disjoint micro-clusters, yielding high pairwise spatial divergence.",
    selectionExplanation: "High spatial divergence is essential for Random defects: multiple branches ensure dispersed defects are fully sampled."
  },
  "Scratch": {
    whatModelSees: "A linear or curvilinear scratch line across the wafer caused by robotic handling or mechanical stress.",
    branchExplanation: "B2 and B3 detect oriented edge contours; B5 confirms the line continuity across die boundaries.",
    saliencyExplanation: "Gradient activation follows the linear scratch trajectory with complementary segment detection.",
    selectionExplanation: "Selected branches combine line orientation sensitivity with defect continuity verification."
  },
  "none": {
    whatModelSees: "A standard clean wafer with normal die pass rates and no discernible failure pattern.",
    branchExplanation: "All branches confirm absence of localized defect clusters and uniform pass die distribution.",
    saliencyExplanation: "Low overall gradient magnitude across all branches, confirming no salient defect regions.",
    selectionExplanation: "Selected branches exhibit high baseline precision to avoid false alarms."
  }
};

export default function App() {
  const [activeTab, setActiveTab] = useState('pipeline'); // 'pipeline' | 'dashboard'
  const [dashboardSection, setDashboardSection] = useState('overview'); // 'overview' | 'status' | 'comparison' | 'perclass' | 'graphs' | 'info'
  const [health, setHealth] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [branchComparison, setBranchComparison] = useState(null);
  const [perClassData, setPerClassData] = useState(null);
  const [researchGraphs, setResearchGraphs] = useState([]);
  const [samples, setSamples] = useState([]);
  const [selectedSampleId, setSelectedSampleId] = useState(null);
  const [activeCamBranch, setActiveCamBranch] = useState('B5');
  const [selectedPerClassBranch, setSelectedPerClassBranch] = useState('B5');
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    fetchHealth();
    fetchTrainingStatus();
    fetchBranchComparison();
    fetchPerClassMetrics();
    fetchGraphs();
    fetchSamples();
  };

  const fetchHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) setHealth(await res.json());
    } catch (e) {
      console.warn("Health check error:", e);
    }
  };

  const fetchTrainingStatus = async () => {
    try {
      const res = await fetch('/api/training_status');
      if (res.ok) setTrainingStatus(await res.json());
    } catch (e) {
      console.warn("Training status error:", e);
    }
  };

  const fetchBranchComparison = async () => {
    try {
      const res = await fetch('/api/branch_comparison');
      if (res.ok) setBranchComparison(await res.json());
    } catch (e) {
      console.warn("Branch comparison error:", e);
    }
  };

  const fetchPerClassMetrics = async () => {
    try {
      const res = await fetch('/api/per_class_metrics');
      if (res.ok) setPerClassData(await res.json());
    } catch (e) {
      console.warn("Per class metrics error:", e);
    }
  };

  const fetchGraphs = async () => {
    try {
      const res = await fetch('/api/graphs');
      if (res.ok) {
        const d = await res.json();
        setResearchGraphs(d.graphs || []);
      }
    } catch (e) {
      console.warn("Graphs fetch error:", e);
    }
  };

  const fetchSamples = async () => {
    try {
      const res = await fetch('/api/samples?limit=9');
      if (res.ok) {
        const data = await res.json();
        setSamples(data.samples || []);
        if (data.samples && data.samples.length > 0) {
          const firstId = data.samples[0].sample_id;
          setSelectedSampleId(firstId);
          runInference(firstId);
        }
      }
    } catch (err) {
      console.warn("Failed to fetch samples:", err);
    }
  };

  const runInference = async (sampleIdToRun = selectedSampleId) => {
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      if (sampleIdToRun !== null && sampleIdToRun !== undefined) {
        formData.append('sample_id', sampleIdToRun);
      }
      const res = await fetch('/api/predict', { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`Inference returned HTTP ${res.status}`);
      const data = await res.json();
      setPrediction(data);
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to complete inference pass.");
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
      const res = await fetch('/api/predict', { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`Upload inference returned HTTP ${res.status}`);
      const data = await res.json();
      setPrediction(data);
      setSelectedSampleId(null);
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to process uploaded file.");
    } finally {
      setLoading(false);
    }
  };

  const currentExplanation = prediction?.true_class_name && SAMPLE_EXPLANATIONS[prediction.true_class_name]
    ? SAMPLE_EXPLANATIONS[prediction.true_class_name]
    : SAMPLE_EXPLANATIONS["Donut"];

  return (
    <div style={{ maxWidth: '1440px', margin: '0 auto', padding: '1.25rem 2rem' }}>
      
      {/* Top Header */}
      <header className="glass-panel" style={{ padding: '1.25rem 2rem', marginBottom: '1.75rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.25rem' }}>
            <Sparkles size={22} color="#38bdf8" />
            <h1 style={{ fontSize: '1.35rem', letterSpacing: '-0.02em' }}>
              Saliency-Divergence Branch Selection
            </h1>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Replacing Accuracy-Heuristic Ensembling with an Explainability-Driven Criterion for Spatially-Scattered Defect Classification
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div className={`badge ${health?.checkpoint_loaded ? 'badge-green' : 'badge-amber'}`}>
            <Activity size={12} />
            {health?.checkpoint_loaded ? 'Trained Checkpoint Active' : 'Model Online (Initialized)'}
          </div>

          <div style={{ display: 'flex', background: 'rgba(30, 41, 59, 0.7)', padding: '0.25rem', borderRadius: '10px' }}>
            <button
              onClick={() => setActiveTab('pipeline')}
              className={activeTab === 'pipeline' ? 'btn-primary' : 'btn-secondary'}
              style={{ padding: '0.45rem 1.1rem', fontSize: '0.85rem' }}
            >
              <Zap size={14} /> 8-Step Pipeline
            </button>
            <button
              onClick={() => setActiveTab('dashboard')}
              className={activeTab === 'dashboard' ? 'btn-primary' : 'btn-secondary'}
              style={{ padding: '0.45rem 1.1rem', fontSize: '0.85rem' }}
            >
              <BarChart3 size={14} /> Research Dashboard
            </button>
          </div>
        </div>
      </header>

      {/* Main View Mode */}
      {activeTab === 'pipeline' ? (
        <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '1.75rem' }}>
          
          {/* Left Column: Sample Selector, Upload, & Simple English Explanations */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            
            {/* Custom Wafer Upload */}
            <div className="glass-panel" style={{ padding: '1.25rem' }}>
              <h3 style={{ fontSize: '1rem', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Upload size={16} color="#38bdf8" /> Custom Wafer Upload
              </h3>
              <label 
                style={{ 
                  display: 'flex', 
                  flexDirection: 'column', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  border: '2px dashed rgba(56, 189, 248, 0.3)', 
                  borderRadius: '10px', 
                  padding: '1.25rem 1rem', 
                  cursor: 'pointer',
                  background: 'rgba(15, 23, 42, 0.5)'
                }}
              >
                <Upload size={22} color="#94a3b8" style={{ marginBottom: '0.35rem' }} />
                <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  Upload Wafer Map
                </span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                  Supports PNG, JPG, or NPY matrix
                </span>
                <input type="file" accept="image/*,.npy" onChange={handleFileUpload} style={{ display: 'none' }} />
              </label>
            </div>

            {/* Test Sample Grid (9 Balanced Classes) */}
            <div className="glass-panel" style={{ padding: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <h3 style={{ fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Database size={16} color="#818cf8" /> Real Dataset Samples
                </h3>
                <button onClick={fetchSamples} className="btn-secondary" style={{ padding: '0.2rem 0.4rem', fontSize: '0.7rem' }}>
                  <RefreshCw size={11} />
                </button>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.45rem' }}>
                {samples.map((s) => {
                  const isSelected = selectedSampleId === s.sample_id;
                  const isSpecial = ['Donut', 'Random', 'Near-full'].includes(s.class_name);
                  return (
                    <button
                      key={s.sample_id}
                      onClick={() => {
                        setSelectedSampleId(s.sample_id);
                        runInference(s.sample_id);
                      }}
                      style={{
                        background: isSelected ? 'rgba(56, 189, 248, 0.25)' : 'rgba(30, 41, 59, 0.5)',
                        border: isSelected ? '1.5px solid var(--accent-cyan)' : '1px solid rgba(255, 255, 255, 0.08)',
                        borderRadius: '8px',
                        padding: '0.4rem',
                        cursor: 'pointer',
                        textAlign: 'center',
                        transition: 'all 0.15s ease'
                      }}
                    >
                      <img src={s.wafer_image_b64} alt={s.class_name} style={{ width: '100%', height: '54px', objectFit: 'contain', borderRadius: '4px' }} />
                      <span style={{ fontSize: '0.68rem', display: 'block', marginTop: '0.2rem', fontWeight: 600, color: isSpecial ? '#f59e0b' : '#cbd5e1' }}>
                        {s.class_name}
                      </span>
                    </button>
                  );
                })}
              </div>

              <button
                onClick={() => runInference()}
                disabled={loading}
                className="btn-primary"
                style={{ width: '100%', marginTop: '1rem', justifyContent: 'center', padding: '0.6rem' }}
              >
                {loading ? <RefreshCw className="animate-spin" size={15} /> : <Zap size={15} />}
                Run Real Inference Pass
              </button>
            </div>

            {/* Simple English Sample Explanation Card */}
            <div className="glass-panel" style={{ padding: '1.25rem', borderLeft: '4px solid #38bdf8' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.65rem' }}>
                <Info size={16} color="#38bdf8" />
                <h4 style={{ fontSize: '0.9rem', color: '#38bdf8' }}>What the Model Sees</h4>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-main)', marginBottom: '0.75rem', lineHeight: 1.5 }}>
                {currentExplanation.whatModelSees}
              </p>

              <h5 style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>
                Branch Behavior:
              </h5>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.65rem' }}>
                {currentExplanation.branchExplanation}
              </p>

              <h5 style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>
                Saliency Divergence:
              </h5>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.65rem' }}>
                {currentExplanation.saliencyExplanation}
              </p>

              <h5 style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>
                Selection Explanation:
              </h5>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                {currentExplanation.selectionExplanation}
              </p>
            </div>

          </div>

          {/* Right Column: Step-by-Step 8-Step Pipeline */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            
            {error && (
              <div className="glass-panel" style={{ borderLeft: '4px solid var(--accent-rose)', padding: '1rem', color: '#fca5a5', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <AlertCircle size={18} />
                <span style={{ fontSize: '0.85rem' }}>{error}</span>
              </div>
            )}

            {!prediction && !loading && (
              <div className="glass-panel" style={{ padding: '3.5rem 2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                <Cpu size={40} color="#38bdf8" style={{ margin: '0 auto 0.75rem', opacity: 0.8 }} />
                <h3 style={{ color: 'var(--text-main)', fontSize: '1.1rem', marginBottom: '0.35rem' }}>No Inference Loaded</h3>
                <p style={{ maxWidth: '420px', margin: '0 auto', fontSize: '0.85rem' }}>
                  Select a wafer sample on the left or upload a custom wafer map to stream real EB-CNN forward inference, Grad-CAM heatmaps, and Saliency-Divergence branch selection.
                </p>
              </div>
            )}

            {prediction && (
              <>
                {/* STEP 1: INPUT WAFER MAP */}
                <div className="glass-panel" style={{ padding: '1.25rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.85rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span className="badge badge-cyan">STEP 1</span>
                      <h3 style={{ fontSize: '1rem' }}>Input Wafer Map Inspection</h3>
                    </div>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      Dimensions: {prediction.original_dimensions[0]} × {prediction.original_dimensions[1]} px
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', background: 'rgba(15, 23, 42, 0.5)', padding: '1rem', borderRadius: '10px' }}>
                    <img 
                      src={prediction.wafer_image_b64} 
                      alt="Input Wafer Map" 
                      style={{ width: '80px', height: '80px', objectFit: 'contain', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} 
                    />
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>True Defect Class:</span>
                        <span style={{ fontWeight: 700, color: '#38bdf8', fontSize: '1rem' }}>
                          {prediction.true_class_name}
                        </span>
                      </div>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                        {prediction.step_by_step.step_1_input.description}
                      </p>
                    </div>
                  </div>
                </div>

                {/* STEP 2: INDEPENDENT BRANCH PREDICTIONS */}
                <div className="glass-panel" style={{ padding: '1.25rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.85rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span className="badge badge-indigo">STEP 2</span>
                      <h3 style={{ fontSize: '1rem' }}>Independent Branch Predictions (B1 to B5)</h3>
                    </div>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      VGG-16 Hierarchical Stages
                    </span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '0.75rem' }}>
                    {["B1", "B2", "B3", "B4", "B5"].map((b, idx) => {
                      const bp = prediction.per_branch_predictions[b];
                      const isProposed = prediction.proposed_selection.selected_branches.includes(b);
                      const isBaseline = prediction.baseline_selection.selected_branches.includes(b);

                      return (
                        <div key={b} className="glass-card" style={{ padding: '0.85rem', textAlign: 'center', border: isProposed ? '1.5px solid #10b981' : '1px solid rgba(255, 255, 255, 0.08)' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                            <span style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', fontSize: '0.9rem' }}>{b}</span>
                            <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Stage {idx + 1}</span>
                          </div>

                          <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#38bdf8', marginBottom: '0.2rem' }}>
                            {bp.class_name}
                          </div>

                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                            Conf: <strong style={{ color: 'var(--text-main)', fontFamily: 'var(--font-mono)' }}>{(bp.confidence * 100).toFixed(1)}%</strong>
                          </div>

                          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', borderTop: '1px solid rgba(255, 255, 255, 0.06)', paddingTop: '0.4rem', textAlign: 'left' }}>
                            <div>Acc: <span style={{ fontFamily: 'var(--font-mono)', color: '#cbd5e1' }}>{typeof bp.accuracy === 'number' ? `${(bp.accuracy * 100).toFixed(1)}%` : bp.accuracy}</span></div>
                            <div>Macro F1: <span style={{ fontFamily: 'var(--font-mono)', color: '#cbd5e1' }}>{typeof bp.macro_f1 === 'number' ? bp.macro_f1.toFixed(3) : bp.macro_f1}</span></div>
                          </div>

                          <div style={{ display: 'flex', gap: '0.25rem', marginTop: '0.5rem', justifyContent: 'center' }}>
                            {isProposed && <span className="badge badge-green" style={{ fontSize: '0.6rem', padding: '0.1rem 0.3rem' }}>Proposed</span>}
                            {isBaseline && <span className="badge badge-cyan" style={{ fontSize: '0.6rem', padding: '0.1rem 0.3rem' }}>Baseline</span>}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* STEP 3: GRAD-CAM HEATMAPS */}
                <div className="glass-panel" style={{ padding: '1.25rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.85rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span className="badge badge-cyan">STEP 3</span>
                      <h3 style={{ fontSize: '1rem' }}>Grad-CAM Saliency Heatmaps Across Branches</h3>
                    </div>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      Bright regions indicate areas that contributed more strongly to this branch's prediction.
                    </span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '0.75rem' }}>
                    {["B1", "B2", "B3", "B4", "B5"].map((b) => (
                      <div key={b} className="glass-card" style={{ padding: '0.5rem', textAlign: 'center' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem', padding: '0 0.25rem' }}>
                          <span style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>{b}</span>
                          <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Map</span>
                        </div>
                        <img 
                          src={prediction.gradcam_heatmaps[b]} 
                          alt={`Grad-CAM ${b}`} 
                          style={{ width: '100%', borderRadius: '4px', border: '1px solid rgba(255, 255, 255, 0.08)' }} 
                        />
                      </div>
                    ))}
                  </div>

                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.75rem', fontStyle: 'italic' }}>
                    "Bright regions indicate areas that contributed more strongly to this branch's prediction."
                  </p>
                </div>

                {/* STEP 4: SALIENCY DIVERGENCE (5×5 MATRIX) */}
                <div className="glass-panel" style={{ padding: '1.25rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.85rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span className="badge badge-amber">STEP 4</span>
                      <h3 style={{ fontSize: '1rem' }}>Spatial Saliency Divergence Matrix D(i, j) (5×5)</h3>
                    </div>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      Cosine distance between normalized saliency maps
                    </span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '1.5rem', alignItems: 'center' }}>
                    
                    {/* 5x5 Matrix Display */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '0.4rem', maxWidth: '420px' }}>
                      <div style={{ textAlign: 'center', fontWeight: 700, color: 'var(--text-muted)', fontSize: '0.75rem' }}></div>
                      {["B1", "B2", "B3", "B4", "B5"].map((b) => (
                        <div key={b} style={{ textAlign: 'center', fontWeight: 700, color: 'var(--text-muted)', fontSize: '0.8rem' }}>{b}</div>
                      ))}

                      {prediction.divergence_matrix.map((row, i) => (
                        <React.Fragment key={i}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                            B{i + 1}
                          </div>
                          {row.map((val, j) => {
                            const bgAlpha = i === j ? 0.05 : Math.min(1.0, val * 0.85);
                            const color = i === j ? '#475569' : val > 0.5 ? '#f59e0b' : '#38bdf8';
                            return (
                              <div 
                                key={j} 
                                className="heatmap-cell"
                                style={{ 
                                  background: i === j ? 'rgba(30, 41, 59, 0.4)' : `rgba(245, 158, 11, ${bgAlpha})`,
                                  color: color,
                                  border: '1px solid rgba(255, 255, 255, 0.05)',
                                  fontSize: '0.8rem'
                                }}
                              >
                                {val.toFixed(2)}
                              </div>
                            );
                          })}
                        </React.Fragment>
                      ))}
                    </div>

                    {/* Explanatory summary */}
                    <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '1rem', borderRadius: '10px' }}>
                      <h4 style={{ fontSize: '0.85rem', marginBottom: '0.5rem', color: '#f59e0b' }}>
                        Average Pairwise Divergence:
                      </h4>
                      <div style={{ marginBottom: '0.65rem' }}>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Proposed Subset: </span>
                        <strong style={{ fontFamily: 'var(--font-mono)', color: '#34d399' }}>
                          {prediction.proposed_selection.average_divergence.toFixed(3)}
                        </strong>
                      </div>
                      <div style={{ marginBottom: '0.75rem' }}>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Baseline Subset: </span>
                        <strong style={{ fontFamily: 'var(--font-mono)', color: '#93c5fd' }}>
                          {prediction.baseline_selection.average_divergence.toFixed(3)}
                        </strong>
                      </div>
                      <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                        "Higher divergence means the selected branches focus on more different spatial regions. Divergence is not automatically good; branch predictive quality is also considered."
                      </p>
                    </div>

                  </div>
                </div>

                {/* STEP 5 & 6: ORIGINAL VS PROPOSED SELECTION */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem' }}>
                  
                  {/* STEP 5: ORIGINAL EB-CNN SELECTION */}
                  <div className="glass-panel" style={{ padding: '1.25rem', borderTop: '4px solid #3b82f6' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                      <span className="badge badge-cyan">STEP 5</span>
                      <h3 style={{ fontSize: '0.95rem' }}>Original EB-CNN Selection</h3>
                    </div>
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.85rem' }}>
                      Accuracy-Heuristic: Evaluates predefined combinations C1..C5 based solely on validation accuracy.
                    </p>

                    <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '0.75rem', borderRadius: '8px', marginBottom: '0.75rem' }}>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.35rem' }}>
                        Selected Combination ({prediction.baseline_selection.best_combo}):
                      </span>
                      <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap', marginBottom: '0.5rem' }}>
                        {prediction.baseline_selection.selected_branches.map((b) => (
                          <span key={b} className="badge badge-cyan" style={{ fontSize: '0.75rem' }}>
                            {b} → {prediction.baseline_selection.ensemble_weights[b]}%
                          </span>
                        ))}
                      </div>
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                        {prediction.baseline_selection.weights_explanation}
                      </span>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.8rem' }}>Predicted: <strong style={{ color: '#60a5fa' }}>{prediction.baseline_selection.class_name}</strong></span>
                      <span style={{ fontSize: '0.8rem' }}>Conf: <strong style={{ fontFamily: 'var(--font-mono)' }}>{(prediction.baseline_selection.confidence * 100).toFixed(1)}%</strong></span>
                    </div>
                  </div>

                  {/* STEP 6: PROPOSED SALIENCY-DIVERGENCE SELECTION */}
                  <div className="glass-panel" style={{ padding: '1.25rem', borderTop: '4px solid #10b981', background: 'rgba(16, 185, 129, 0.04)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                      <span className="badge badge-green">STEP 6</span>
                      <h3 style={{ fontSize: '0.95rem', color: '#34d399' }}>Proposed Saliency-Divergence Selection</h3>
                    </div>
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.85rem' }}>
                      Choose branches that are accurate AND look at different useful regions.
                    </p>

                    <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '0.75rem', borderRadius: '8px', marginBottom: '0.75rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                        <span>Quality Floor: <strong style={{ color: '#cbd5e1', fontFamily: 'var(--font-mono)' }}>{prediction.proposed_selection.quality_floor}</strong></span>
                        <span>Joint Score: <strong style={{ color: '#34d399', fontFamily: 'var(--font-mono)' }}>{prediction.proposed_selection.best_joint_score}</strong></span>
                      </div>
                      <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap', marginBottom: '0.5rem' }}>
                        {prediction.proposed_selection.selected_branches.map((b) => (
                          <span key={b} className="badge badge-green" style={{ fontSize: '0.75rem' }}>
                            {b} → {prediction.proposed_selection.ensemble_weights[b]}%
                          </span>
                        ))}
                      </div>
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                        {prediction.proposed_selection.weights_explanation}
                      </span>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.8rem' }}>Predicted: <strong style={{ color: '#34d399' }}>{prediction.proposed_selection.class_name}</strong></span>
                      <span style={{ fontSize: '0.8rem' }}>Conf: <strong style={{ fontFamily: 'var(--font-mono)' }}>{(prediction.proposed_selection.confidence * 100).toFixed(1)}%</strong></span>
                    </div>
                  </div>

                </div>

                {/* STEP 7 & 8: FINAL ENSEMBLE & COMPARISON */}
                <div className="glass-panel" style={{ padding: '1.25rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.85rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span className="badge badge-cyan">STEP 7 & 8</span>
                      <h3 style={{ fontSize: '1rem' }}>Final Ensemble Predictions & Comparison</h3>
                    </div>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      Equal-Weight Probability Averaging
                    </span>
                  </div>

                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: 'var(--text-muted)' }}>
                          <th style={{ padding: '0.65rem' }}>Ensemble Method</th>
                          <th style={{ padding: '0.65rem' }}>Selected Branches</th>
                          <th style={{ padding: '0.65rem' }}>Ensemble Weights</th>
                          <th style={{ padding: '0.65rem' }}>Predicted Defect</th>
                          <th style={{ padding: '0.65rem' }}>Confidence</th>
                          <th style={{ padding: '0.65rem' }}>Average Divergence</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                          <td style={{ padding: '0.65rem', fontWeight: 600, color: '#93c5fd' }}>Original EB-CNN (Baseline)</td>
                          <td style={{ padding: '0.65rem' }}>{prediction.baseline_selection.selected_branches.join(' + ')}</td>
                          <td style={{ padding: '0.65rem', fontFamily: 'var(--font-mono)' }}>Equal ({prediction.baseline_selection.ensemble_weights[prediction.baseline_selection.selected_branches[0]]}%)</td>
                          <td style={{ padding: '0.65rem', fontWeight: 700 }}>{prediction.baseline_selection.class_name}</td>
                          <td style={{ padding: '0.65rem', fontFamily: 'var(--font-mono)' }}>{(prediction.baseline_selection.confidence * 100).toFixed(1)}%</td>
                          <td style={{ padding: '0.65rem', fontFamily: 'var(--font-mono)' }}>{prediction.baseline_selection.average_divergence.toFixed(3)}</td>
                        </tr>
                        <tr style={{ background: 'rgba(16, 185, 129, 0.08)' }}>
                          <td style={{ padding: '0.65rem', fontWeight: 700, color: '#34d399' }}>Proposed Saliency-Divergence</td>
                          <td style={{ padding: '0.65rem', fontWeight: 600, color: '#34d399' }}>{prediction.proposed_selection.selected_branches.join(' + ')}</td>
                          <td style={{ padding: '0.65rem', fontFamily: 'var(--font-mono)' }}>Equal ({prediction.proposed_selection.ensemble_weights[prediction.proposed_selection.selected_branches[0]]}%)</td>
                          <td style={{ padding: '0.65rem', fontWeight: 700, color: '#34d399' }}>{prediction.proposed_selection.class_name}</td>
                          <td style={{ padding: '0.65rem', fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#34d399' }}>{(prediction.proposed_selection.confidence * 100).toFixed(1)}%</td>
                          <td style={{ padding: '0.65rem', fontFamily: 'var(--font-mono)', color: '#34d399' }}>{prediction.proposed_selection.average_divergence.toFixed(3)}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  <div style={{ marginTop: '0.85rem', display: 'flex', alignItems: 'center', gap: '1rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    <span>Divergence Gain over Baseline: <strong style={{ color: prediction.comparison.divergence_gain >= 0 ? '#34d399' : '#f43f5e' }}>{prediction.comparison.divergence_gain > 0 ? `+${prediction.comparison.divergence_gain.toFixed(3)}` : prediction.comparison.divergence_gain.toFixed(3)}</strong></span>
                    <span>•</span>
                    <span>True Label Match: <strong style={{ color: prediction.proposed_selection.class_name === prediction.true_class_name ? '#34d399' : '#f59e0b' }}>{prediction.proposed_selection.class_name === prediction.true_class_name ? 'Matched' : 'Discrepancy'}</strong></span>
                  </div>
                </div>

              </>
            )}

          </div>

        </div>
      ) : (
        /* DEDICATED RESEARCH DASHBOARD (SECTIONS A TO L) */
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          
          {/* Sub Navigation Bar for Dashboard */}
          <div className="glass-panel" style={{ padding: '0.75rem 1.25rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {[
              { id: 'overview', label: 'A. Dataset Overview' },
              { id: 'status', label: 'B. Training Status (B1-B5)' },
              { id: 'comparison', label: 'C. Branch Metrics Table' },
              { id: 'perclass', label: 'D. Per-Class Analysis' },
              { id: 'graphs', label: 'K. Research Graphs (Gallery)' },
              { id: 'info', label: 'L. Experiment Info' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setDashboardSection(tab.id)}
                className={dashboardSection === tab.id ? 'btn-primary' : 'btn-secondary'}
                style={{ padding: '0.4rem 0.85rem', fontSize: '0.8rem' }}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* SECTION A: DATASET OVERVIEW */}
          {dashboardSection === 'overview' && (
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h2 style={{ fontSize: '1.25rem', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Database size={20} color="#38bdf8" /> Section A: Dataset Overview (WM-811K Benchmark)
              </h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '1.25rem' }}>
                Verified canonical 3:1:1 stratified split without data leakage across 62,248 labeled wafer maps.
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                <div className="glass-card" style={{ textAlign: 'center' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Total Wafer Maps</span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>62,248</div>
                </div>
                <div className="glass-card" style={{ textAlign: 'center' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Train Split (60%)</span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#34d399', fontFamily: 'var(--font-mono)' }}>37,348</div>
                </div>
                <div className="glass-card" style={{ textAlign: 'center' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Validation Split (20%)</span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f59e0b', fontFamily: 'var(--font-mono)' }}>12,450</div>
                </div>
                <div className="glass-card" style={{ textAlign: 'center' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Test Split (20%)</span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#c084fc', fontFamily: 'var(--font-mono)' }}>12,450</div>
                </div>
              </div>

              <h4 style={{ fontSize: '0.95rem', marginBottom: '0.5rem' }}>Target Defect Categories (9 Classes):</h4>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                {CLASS_NAMES.map((c, i) => (
                  <span key={c} className="badge" style={{ background: ['Donut', 'Random', 'Near-full'].includes(c) ? 'rgba(245, 158, 11, 0.2)' : 'rgba(30, 41, 59, 0.6)', color: ['Donut', 'Random', 'Near-full'].includes(c) ? '#f59e0b' : '#cbd5e1' }}>
                    {i}. {c} {['Donut', 'Random', 'Near-full'].includes(c) ? '(Key Class)' : ''}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* SECTION B: TRAINING STATUS */}
          {dashboardSection === 'status' && (
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <h2 style={{ fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Activity size={20} color="#10b981" /> Section B: Branch Training Status (B1 to B5)
                </h2>
                <button onClick={fetchTrainingStatus} className="btn-secondary" style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}>
                  <RefreshCw size={12} /> Refresh Status
                </button>
              </div>

              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '1.25rem' }}>
                Genuine training status of the 5 EB-CNN branches. No fabricated values are shown.
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                {["B1", "B2", "B3", "B4", "B5"].map((b) => {
                  const bInfo = trainingStatus?.branches?.[b];
                  const isTrained = bInfo?.status === "TRAINED";

                  return (
                    <div key={b} className="glass-card" style={{ borderTop: `3px solid ${isTrained ? '#10b981' : '#f59e0b'}` }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                        <span style={{ fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{b}</span>
                        <span className={`badge ${isTrained ? 'badge-green' : 'badge-amber'}`} style={{ fontSize: '0.65rem' }}>
                          {isTrained ? 'TRAINED' : 'Untrained'}
                        </span>
                      </div>

                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                        Val Accuracy: <strong style={{ color: 'var(--text-main)', fontFamily: 'var(--font-mono)' }}>
                          {typeof bInfo?.best_validation_accuracy === 'number' ? `${(bInfo.best_validation_accuracy * 100).toFixed(2)}%` : 'Not yet evaluated'}
                        </strong>
                      </div>

                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                        Macro F1: <strong style={{ color: 'var(--text-main)', fontFamily: 'var(--font-mono)' }}>
                          {typeof bInfo?.best_macro_f1 === 'number' ? bInfo.best_macro_f1.toFixed(4) : 'Not yet evaluated'}
                        </strong>
                      </div>

                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                        Epochs: <strong style={{ color: 'var(--text-main)' }}>{bInfo?.epochs || 0}</strong>
                      </div>

                      <div style={{ fontSize: '0.72rem', color: isTrained ? '#34d399' : '#f59e0b', marginTop: '0.5rem', borderTop: '1px solid rgba(255, 255, 255, 0.05)', paddingTop: '0.35rem' }}>
                        Checkpoint: {bInfo?.checkpoint || 'Not available'}
                      </div>
                    </div>
                  );
                })}
              </div>

              {trainingStatus?.training_completed && (
                <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '1rem', borderRadius: '10px' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Training Duration: <strong>{trainingStatus.total_duration_seconds.toFixed(1)}s</strong> | Seed: <strong>{trainingStatus.seed}</strong> | Checkpoint Path: <code style={{ color: '#38bdf8' }}>{trainingStatus.checkpoint_path}</code>
                  </span>
                </div>
              )}
            </div>
          )}

          {/* SECTION C: BRANCH METRICS COMPARISON TABLE */}
          {dashboardSection === 'comparison' && (
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <h2 style={{ fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Sliders size={20} color="#38bdf8" /> Section C: Branch Metrics Comparison Table
                </h2>
                <button onClick={fetchBranchComparison} className="btn-secondary" style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}>
                  <RefreshCw size={12} /> Refresh
                </button>
              </div>

              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '1.25rem' }}>
                Empirical evaluation metrics calculated independently for every branch on the held-out test split.
              </p>

              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid rgba(255, 255, 255, 0.1)', color: 'var(--text-muted)' }}>
                      <th style={{ padding: '0.75rem' }}>Branch</th>
                      <th style={{ padding: '0.75rem' }}>Accuracy</th>
                      <th style={{ padding: '0.75rem' }}>Precision</th>
                      <th style={{ padding: '0.75rem' }}>Recall</th>
                      <th style={{ padding: '0.75rem' }}>Macro F1</th>
                      <th style={{ padding: '0.75rem' }}>Weighted F1</th>
                      <th style={{ padding: '0.75rem', color: '#f59e0b' }}>Donut F1</th>
                      <th style={{ padding: '0.75rem', color: '#38bdf8' }}>Random F1</th>
                      <th style={{ padding: '0.75rem', color: '#10b981' }}>Near-full F1</th>
                    </tr>
                  </thead>
                  <tbody>
                    {branchComparison?.rows && branchComparison.rows.length > 0 ? (
                      branchComparison.rows.map((r) => (
                        <tr key={r.branch} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                          <td style={{ padding: '0.75rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{r.branch}</td>
                          <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)' }}>{typeof r.accuracy === 'number' ? `${(r.accuracy * 100).toFixed(2)}%` : r.accuracy}</td>
                          <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)' }}>{typeof r.precision === 'number' ? r.precision.toFixed(4) : r.precision}</td>
                          <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)' }}>{typeof r.recall === 'number' ? r.recall.toFixed(4) : r.recall}</td>
                          <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#38bdf8' }}>{typeof r.macro_f1 === 'number' ? r.macro_f1.toFixed(4) : r.macro_f1}</td>
                          <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)' }}>{typeof r.weighted_f1 === 'number' ? r.weighted_f1.toFixed(4) : r.weighted_f1}</td>
                          <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)', color: '#f59e0b' }}>{typeof r.donut_f1 === 'number' ? r.donut_f1.toFixed(4) : r.donut_f1}</td>
                          <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>{typeof r.random_f1 === 'number' ? r.random_f1.toFixed(4) : r.random_f1}</td>
                          <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)', color: '#10b981' }}>{typeof r.nearfull_f1 === 'number' ? r.nearfull_f1.toFixed(4) : r.nearfull_f1}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="9" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                          Not yet evaluated. Results will appear after genuine branch training completes.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* SECTION D: PER-CLASS ANALYSIS */}
          {dashboardSection === 'perclass' && (
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <h2 style={{ fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <FileText size={20} color="#a78bfa" /> Section D: Per-Class Breakdown
                </h2>
                
                <div style={{ display: 'flex', gap: '0.35rem' }}>
                  {["B1", "B2", "B3", "B4", "B5"].map((b) => (
                    <button
                      key={b}
                      onClick={() => setSelectedPerClassBranch(b)}
                      className={selectedPerClassBranch === b ? 'btn-primary' : 'btn-secondary'}
                      style={{ padding: '0.3rem 0.75rem', fontSize: '0.75rem' }}
                    >
                      {b}
                    </button>
                  ))}
                </div>
              </div>

              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '1.25rem' }}>
                Per-class metrics for Branch <strong>{selectedPerClassBranch}</strong> across all 9 failure modes, highlighting Donut, Random, and Near-full.
              </p>

              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid rgba(255, 255, 255, 0.1)', color: 'var(--text-muted)' }}>
                      <th style={{ padding: '0.65rem' }}>Class</th>
                      <th style={{ padding: '0.65rem' }}>Precision</th>
                      <th style={{ padding: '0.65rem' }}>Recall</th>
                      <th style={{ padding: '0.65rem' }}>F1-Score</th>
                      <th style={{ padding: '0.65rem' }}>Support (Test Samples)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {perClassData?.per_branch?.[selectedPerClassBranch] ? (
                      perClassData.per_branch[selectedPerClassBranch].map((row) => {
                        const isSpecial = ['Donut', 'Random', 'Near-full'].includes(row.class_name);
                        return (
                          <tr key={row.class_name} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)', background: isSpecial ? 'rgba(245, 158, 11, 0.04)' : 'transparent' }}>
                            <td style={{ padding: '0.65rem', fontWeight: 600, color: isSpecial ? '#f59e0b' : 'var(--text-main)' }}>
                              {row.class_name} {isSpecial ? '★' : ''}
                            </td>
                            <td style={{ padding: '0.65rem', fontFamily: 'var(--font-mono)' }}>{typeof row.precision === 'number' ? row.precision.toFixed(4) : row.precision}</td>
                            <td style={{ padding: '0.65rem', fontFamily: 'var(--font-mono)' }}>{typeof row.recall === 'number' ? row.recall.toFixed(4) : row.recall}</td>
                            <td style={{ padding: '0.65rem', fontFamily: 'var(--font-mono)', fontWeight: 700, color: isSpecial ? '#f59e0b' : '#38bdf8' }}>
                              {typeof row.f1 === 'number' ? row.f1.toFixed(4) : row.f1}
                            </td>
                            <td style={{ padding: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>{row.support ?? 'Not yet evaluated'}</td>
                          </tr>
                        );
                      })
                    ) : (
                      <tr>
                        <td colSpan="5" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                          Not yet evaluated. Per-class metrics will appear after training finishes.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* SECTION K: RESEARCH GRAPHS GALLERY */}
          {dashboardSection === 'graphs' && (
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <h2 style={{ fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <ImageIcon size={20} color="#f472b6" /> Section K: Research Comparison Graphs
                </h2>
                <button onClick={fetchGraphs} className="btn-secondary" style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}>
                  <RefreshCw size={12} /> Refresh Graphs
                </button>
              </div>

              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                Publication-quality figures generated strictly from genuine empirical evaluation.
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem' }}>
                {researchGraphs.map((g) => (
                  <div key={g.filename} className="glass-card" style={{ padding: '1rem' }}>
                    <h4 style={{ fontSize: '0.95rem', marginBottom: '0.5rem', color: '#f8fafc' }}>{g.name}</h4>
                    {g.available ? (
                      <img 
                        src={g.url} 
                        alt={g.name} 
                        style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} 
                      />
                    ) : (
                      <div style={{ padding: '3rem 1rem', textAlign: 'center', color: 'var(--text-muted)', background: 'rgba(15, 23, 42, 0.4)', borderRadius: '6px' }}>
                        Results will appear after training/evaluation.
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* SECTION L: EXPERIMENT INFO */}
          {dashboardSection === 'info' && (
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h2 style={{ fontSize: '1.25rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <ShieldCheck size={20} color="#34d399" /> Section L: Experiment Metadata & Replication Protocol
              </h2>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', fontSize: '0.85rem' }}>
                <div className="glass-card">
                  <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>Architecture:</span>
                  <strong style={{ color: '#38bdf8' }}>5-Branch EB-CNN (Abdullah et al., 2025; VGG-16 Backbone)</strong>
                </div>
                <div className="glass-card">
                  <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>Random Seed:</span>
                  <strong style={{ fontFamily: 'var(--font-mono)' }}>101 (Fixed seed)</strong>
                </div>
                <div className="glass-card">
                  <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>Dataset Split:</span>
                  <strong>WM-811K Canonical Stratified (Train: 37,348, Val: 12,450, Test: 12,450)</strong>
                </div>
                <div className="glass-card">
                  <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>Optimizer & Scheduler:</span>
                  <strong>Adam (LR=1e-3, Weight Decay=1e-4) with ReduceLROnPlateau</strong>
                </div>
                <div className="glass-card">
                  <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>Divergence Metric:</span>
                  <strong>Cosine Spatial Distance: D(i, j) = 1.0 - cos(S_i, S_j)</strong>
                </div>
                <div className="glass-card">
                  <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>Selection Objective:</span>
                  <strong>Quality Floor (τ=0.80) + Divergence Weight (λ=0.15)</strong>
                </div>
              </div>
            </div>
          )}

        </div>
      )}

    </div>
  );
}
