// API service for ThermoCache backend

import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface GPUMetrics {
  gpu_id: number;
  temperature: number;
  utilization: number;
  vram_used: number;
  vram_total: number;
  power_draw: number;
  thermal_trend: number;
  thermal_headroom: number;
  memory_pressure: number;
  rack_id: number;
  contexts: string[];
}

export interface ContextMetadata {
  context_id: string;
  hash: string;
  token_count: number;
  gpu_location: number;
  memory_size: number;
  creation_time: string;
  last_accessed: string;
  reference_count: number;
}

export interface SchedulingDecision {
  request_id: string;
  selected_gpu: number;
  context_reused: boolean;
  context_id: string | null;
  temperature: number;
  thermal_headroom: number;
  migration_required: boolean;
  score: number;
  candidate_gpus: number[];
  reason: string;
  timestamp: string;
}

export interface SystemMetrics {
  total_gpus: number;
  active_gpus: number;
  average_temperature: number;
  peak_temperature: number;
  total_vram_used: number;
  total_vram_capacity: number;
  vram_utilization_percent: number;
  total_contexts: number;
  context_hit_rate: number;
  kv_cache_memory_saved: number;
  requests_per_second: number;
  average_latency_ms: number;
  estimated_power_kw: number;
  thermal_hotspots: number[];
}

export interface SimulationResult {
  baseline_metrics: SystemMetrics;
  thermocache_metrics: SystemMetrics;
  comparison: Record<string, number>;
}

export const thermoService = {
  // Get all GPUs
  getGPUs: async (): Promise<GPUMetrics[]> => {
    const response = await api.get('/gpus');
    return response.data;
  },

  // Get single GPU
  getGPU: async (gpuId: number): Promise<GPUMetrics> => {
    const response = await api.get(`/gpus/${gpuId}`);
    return response.data;
  },

  // Get system metrics
  getMetrics: async (): Promise<SystemMetrics> => {
    const response = await api.get('/metrics');
    return response.data;
  },

  // Get all contexts
  getContexts: async (): Promise<ContextMetadata[]> => {
    const response = await api.get('/contexts');
    return response.data;
  },

  // Get recent scheduling decisions
  getDecisions: async (limit: number = 100): Promise<SchedulingDecision[]> => {
    const response = await api.get(`/scheduler/decisions?limit=${limit}`);
    return response.data;
  },

  // Submit inference request
  submitInference: async (request: {
    prompt: string;
    context?: string;
    max_tokens?: number;
    priority?: number;
  }): Promise<SchedulingDecision> => {
    const response = await api.post('/inference', request);
    return response.data;
  },

  // Run simulation
  runSimulation: async (
    numRequests: number = 1000,
    contextSharingRatio: number = 0.7
  ): Promise<SimulationResult> => {
    const response = await api.get(
      `/simulation/run?num_requests=${numRequests}&context_sharing_ratio=${contextSharingRatio}`
    );
    return response.data;
  },

  // Get cache stats
  getCacheStats: async (): Promise<Record<string, any>> => {
    const response = await api.get('/cache/stats');
    return response.data;
  },

  // Get scheduler stats
  getSchedulerStats: async (): Promise<Record<string, any>> => {
    const response = await api.get('/scheduler/stats');
    return response.data;
  },
};

export default api;
