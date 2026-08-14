import React, { useState, useEffect } from 'react';
import { thermoService, GPUMetrics, SystemMetrics, SchedulingDecision, ContextMetadata } from './services/api';

// GPU Card Component
const GPUCard: React.FC<{ gpu: GPUMetrics }> = ({ gpu }) => {
  const getTempClass = (temp: number): string => {
    if (temp < 50) return 'temp-cold';
    if (temp < 65) return 'temp-normal';
    if (temp < 75) return 'temp-warm';
    if (temp < 85) return 'temp-hot';
    return 'temp-critical';
  };

  const getThermalStatus = (temp: number): string => {
    if (temp < 50) return 'COLD';
    if (temp < 65) return 'NORMAL';
    if (temp < 75) return 'WARM';
    if (temp < 85) return 'HOT';
    return 'CRITICAL';
  };

  return (
    <div className="gpu-card">
      <div className="gpu-header">
        <span className="gpu-id">GPU {gpu.gpu_id}</span>
        <span className={`gpu-temp ${getTempClass(gpu.temperature)}`}>
          {gpu.temperature.toFixed(1)}°C - {getThermalStatus(gpu.temperature)}
        </span>
      </div>

      <div className="metric-row">
        <label>Utilization</label>
        <span>{gpu.utilization.toFixed(1)}%</span>
      </div>
      <div className="progress-bar">
        <div 
          className="progress-fill utilization-fill" 
          style={{ width: `${gpu.utilization}%` }}
        />
      </div>

      <div className="metric-row">
        <label>VRAM</label>
        <span>{gpu.vram_used.toFixed(1)} / {gpu.vram_total} GB</span>
      </div>
      <div className="progress-bar">
        <div 
          className="progress-fill vram-fill" 
          style={{ width: `${(gpu.vram_used / gpu.vram_total) * 100}%` }}
        />
      </div>

      <div className="metric-row">
        <label>Power</label>
        <span>{gpu.power_draw.toFixed(0)}W</span>
      </div>

      <div className="metric-row">
        <label>Thermal Headroom</label>
        <span>{(gpu.thermal_headroom * 100).toFixed(0)}%</span>
      </div>

      <div className="metric-row">
        <label>Trend</label>
        <span>{gpu.thermal_trend > 0 ? '+' : ''}{gpu.thermal_trend.toFixed(2)}°C/min</span>
      </div>

      {gpu.contexts.length > 0 && (
        <div className="context-list">
          <div className="metric-row">
            <label>Contexts ({gpu.contexts.length})</label>
          </div>
          {gpu.contexts.slice(0, 3).map((ctx, i) => (
            <span key={i} className="context-tag">{ctx}</span>
          ))}
          {gpu.contexts.length > 3 && (
            <span className="context-tag">+{gpu.contexts.length - 3} more</span>
          )}
        </div>
      )}
    </div>
  );
};

// Metrics Summary Component
const MetricsSummary: React.FC<{ metrics: SystemMetrics | null }> = ({ metrics }) => {
  if (!metrics) return null;

  return (
    <div className="metrics-grid">
      <div className="metric-card">
        <div className="metric-value">{metrics.average_temperature.toFixed(1)}°C</div>
        <div className="metric-label">Avg Temperature</div>
      </div>
      <div className="metric-card">
        <div className="metric-value">{metrics.peak_temperature.toFixed(1)}°C</div>
        <div className="metric-label">Peak Temperature</div>
      </div>
      <div className="metric-card">
        <div className="metric-value">{metrics.vram_utilization_percent.toFixed(1)}%</div>
        <div className="metric-label">VRAM Utilization</div>
      </div>
      <div className="metric-card">
        <div className="metric-value">{(metrics.context_hit_rate * 100).toFixed(1)}%</div>
        <div className="metric-label">Context Hit Rate</div>
      </div>
      <div className="metric-card">
        <div className="metric-value">{metrics.total_contexts}</div>
        <div className="metric-label">Cached Contexts</div>
      </div>
      <div className="metric-card">
        <div className="metric-value">{metrics.kv_cache_memory_saved.toFixed(1)} GB</div>
        <div className="metric-label">KV Cache Saved</div>
      </div>
      <div className="metric-card">
        <div className="metric-value">{metrics.requests_per_second.toFixed(1)}</div>
        <div className="metric-label">Requests/sec</div>
      </div>
      <div className="metric-card">
        <div className="metric-value">{metrics.estimated_power_kw.toFixed(2)} kW</div>
        <div className="metric-label">Power Consumption</div>
      </div>
    </div>
  );
};

// Decisions Table Component
const DecisionsTable: React.FC<{ decisions: SchedulingDecision[] }> = ({ decisions }) => {
  return (
    <table className="decisions-table">
      <thead>
        <tr>
          <th>Request ID</th>
          <th>GPU</th>
          <th>Temp</th>
          <th>Context Reused</th>
          <th>Score</th>
          <th>Reason</th>
        </tr>
      </thead>
      <tbody>
        {decisions.slice(0, 20).map((decision, i) => (
          <tr key={i}>
            <td>{decision.request_id}</td>
            <td>GPU {decision.selected_gpu}</td>
            <td>{decision.temperature.toFixed(1)}°C</td>
            <td>
              {decision.context_reused ? (
                <span className="badge badge-success">YES</span>
              ) : (
                <span className="badge badge-info">NO</span>
              )}
            </td>
            <td>{(decision.score * 100).toFixed(0)}%</td>
            <td>{decision.reason}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

// Comparison View Component
const ComparisonView: React.FC = () => {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await thermoService.runSimulation(1000, 0.7);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Simulation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="section">
      <div className="section-title">
        <span>📊</span> Baseline vs ThermoCache Comparison
      </div>
      
      <button 
        className="simulation-btn" 
        onClick={runSimulation}
        disabled={loading}
      >
        {loading ? 'Running Simulation...' : 'Run Comparison (1000 requests)'}
      </button>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="comparison-grid" style={{ marginTop: '20px' }}>
          <div className="comparison-card">
            <h3>Baseline Scheduler</h3>
            <div className="metric-row">
              <label>Avg Temperature</label>
              <span>{result.baseline_metrics.average_temperature.toFixed(1)}°C</span>
            </div>
            <div className="metric-row">
              <label>Peak Temperature</label>
              <span>{result.baseline_metrics.peak_temperature.toFixed(1)}°C</span>
            </div>
            <div className="metric-row">
              <label>VRAM Used</label>
              <span>{result.baseline_metrics.total_vram_used.toFixed(1)} GB</span>
            </div>
            <div className="metric-row">
              <label>Context Hit Rate</label>
              <span>{(result.baseline_metrics.context_hit_rate * 100).toFixed(1)}%</span>
            </div>
          </div>

          <div className="comparison-card">
            <h3>ThermoCache</h3>
            <div className="metric-row">
              <label>Avg Temperature</label>
              <span>{result.thermocache_metrics.average_temperature.toFixed(1)}°C</span>
            </div>
            <div className="metric-row">
              <label>Peak Temperature</label>
              <span>{result.thermocache_metrics.peak_temperature.toFixed(1)}°C</span>
            </div>
            <div className="metric-row">
              <label>VRAM Used</label>
              <span>{result.thermocache_metrics.total_vram_used.toFixed(1)} GB</span>
            </div>
            <div className="metric-row">
              <label>Context Hit Rate</label>
              <span>{(result.thermocache_metrics.context_hit_rate * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}

      {result && result.comparison && (
        <div className="comparison-card" style={{ marginTop: '20px' }}>
          <h3>Improvement Summary</h3>
          <div className="metrics-grid" style={{ marginTop: '15px' }}>
            <div className="metric-card">
              <div className="metric-value" style={{ color: '#22c55e' }}>
                {result.comparison.temperature_reduction?.toFixed(2) || 0}°C
              </div>
              <div className="metric-label">Avg Temp Reduction</div>
            </div>
            <div className="metric-card">
              <div className="metric-value" style={{ color: '#22c55e' }}>
                {result.comparison.peak_temperature_reduction?.toFixed(2) || 0}°C
              </div>
              <div className="metric-label">Peak Temp Reduction</div>
            </div>
            <div className="metric-card">
              <div className="metric-value" style={{ color: '#22c55e' }}>
                {result.comparison.vram_savings_percent?.toFixed(1) || 0}%
              </div>
              <div className="metric-label">VRAM Savings</div>
            </div>
            <div className="metric-card">
              <div className="metric-value" style={{ color: '#22c55e' }}>
                {result.comparison.context_reuse_rate ? (result.comparison.context_reuse_rate * 100).toFixed(1) : '0'}%
              </div>
              <div className="metric-label">Context Reuse Rate</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Main App Component
const App: React.FC = () => {
  const [gpus, setGpus] = useState<GPUMetrics[]>([]);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [decisions, setDecisions] = useState<SchedulingDecision[]>([]);
  const [contexts, setContexts] = useState<ContextMetadata[]>([]);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      const [gpusData, metricsData, decisionsData, contextsData] = await Promise.all([
        thermoService.getGPUs(),
        thermoService.getMetrics(),
        thermoService.getDecisions(50),
        thermoService.getContexts(),
      ]);
      setGpus(gpusData);
      setMetrics(metricsData);
      setDecisions(decisionsData);
      setContexts(contextsData);
      setLastUpdate(new Date());
      setError(null);
    } catch (err: any) {
      setError('Failed to fetch data. Is the backend running?');
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="dashboard">
      <header className="header">
        <div>
          <h1>🌡️ ThermoCache</h1>
          <p className="tagline">Share the context. Spread the heat.</p>
        </div>
        <div>
          {lastUpdate && (
            <div className="refresh-indicator">
              Last updated: {lastUpdate.toLocaleTimeString()}
            </div>
          )}
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <MetricsSummary metrics={metrics} />

      <div className="section">
        <div className="section-title">
          <span>🖥️</span> GPU Cluster Status
        </div>
        <div className="gpu-grid">
          {gpus.map((gpu) => (
            <GPUCard key={gpu.gpu_id} gpu={gpu} />
          ))}
        </div>
      </div>

      <ComparisonView />

      <div className="section">
        <div className="section-title">
          <span>📋</span> Recent Scheduling Decisions
        </div>
        <DecisionsTable decisions={decisions} />
      </div>

      <div className="section">
        <div className="section-title">
          <span>📦</span> Cached Contexts ({contexts.length})
        </div>
        {contexts.length > 0 ? (
          <table className="decisions-table">
            <thead>
              <tr>
                <th>Context ID</th>
                <th>GPU</th>
                <th>Tokens</th>
                <th>Memory</th>
                <th>References</th>
              </tr>
            </thead>
            <tbody>
              {contexts.slice(0, 10).map((ctx) => (
                <tr key={ctx.context_id}>
                  <td>{ctx.context_id}</td>
                  <td>GPU {ctx.gpu_location}</td>
                  <td>{ctx.token_count}</td>
                  <td>{ctx.memory_size.toFixed(3)} GB</td>
                  <td>{ctx.reference_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p style={{ color: 'var(--text-secondary)' }}>No contexts cached yet</p>
        )}
      </div>
    </div>
  );
};

export default App;
