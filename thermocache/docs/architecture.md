# ThermoCache Architecture

## System Overview

ThermoCache is designed as an intelligent inference middleware that sits between client applications and LLM inference engines. The architecture enables thermal-aware scheduling combined with zero-copy context deduplication.

## High-Level Architecture

```
                ┌──────────────────────┐
                │   Client / API       │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Request Analyzer     │
                └──────────┬───────────┘
                           │
                ┌──────────▼───────────┐
                │ Context Fingerprint  │
                └──────────┬───────────┘
                           │
                  ┌────────▼────────┐
                  │ Context Index   │
                  └────────┬────────┘
                           │
              ┌────────────▼────────────┐
              │ Thermal-Aware Scheduler │
              └────────────┬────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
       GPU Worker      GPU Worker      GPU Worker
           │               │               │
           └───────────────┼───────────────┘
                           │
                    Telemetry Layer
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       GPU Metrics                 Thermal Model
```

## Component Details

### 1. Context Deduplication Engine (`context_cache/`)

**Purpose:** Detect and eliminate redundant context storage across inference requests.

**Key Classes:**
- `ContextFingerprinter`: Generates SHA-256 hashes and token prefixes for content
- `ContextIndex`: Maintains lookup tables for contexts by hash, ID, and prefix tokens
- `ContextCache`: High-level API combining fingerprinting and indexing

**Features:**
- Exact match detection via content hashing
- Prefix-based partial matching for shared prefixes
- Reference counting for lifecycle management
- GPU location tracking for locality-aware scheduling

**Data Structures:**
```python
ContextMetadata:
  - context_id: str
  - hash: str (SHA-256)
  - token_count: int
  - gpu_location: int
  - memory_size: float (GB)
  - reference_count: int
  - prefix_tokens: List[int]
```

### 2. Thermal Prediction Engine (`thermal/`)

**Purpose:** Track and predict GPU temperature trajectories.

**Key Classes:**
- `ThermalHistory`: Maintains timestamped temperature samples per GPU
- `ThermalPredictor`: Provides predictions at 30s, 60s, and 5min horizons
- `AdvancedThermalPredictor`: Extensible base for ML-based predictors

**Prediction Algorithm:**
- Linear regression on recent samples for trend calculation
- Exponential damping for longer prediction horizons
- Confidence estimation based on data quality and variance

**Output:**
```python
ThermalPrediction:
  - current_temp: float
  - temp_30s: float
  - temp_60s: float
  - temp_5min: float
  - confidence: float (0-1)
```

### 3. Thermal-Aware Scheduler (`scheduler/`)

**Purpose:** Make optimal GPU placement decisions balancing multiple factors.

**Key Classes:**
- `GPUScorer`: Calculates weighted scores for candidate GPUs
- `ThermalAwareScheduler`: Main scheduler with context awareness
- `BaselineScheduler`: Traditional scheduler for comparison

**Scoring Formula:**
```
Score = w1*ContextReuse + w2*MemoryAvail + w3*ComputeAvail 
        + w4*ThermalHeadroom - w5*ThermalStress

Default weights:
  - context_reuse: 0.35 (highest priority)
  - thermal_headroom: 0.20
  - memory_availability: 0.15
  - compute_availability: 0.15
  - thermal_stress: -0.15 (penalty)
```

**Decision Factors:**
1. **Context Locality:** Does the GPU already have the required context?
2. **Memory Availability:** Is there sufficient VRAM?
3. **Compute Headroom:** Is the GPU underutilized?
4. **Thermal State:** Is the GPU cool with room to heat up?
5. **Thermal Trend:** Is the GPU heating or cooling?

### 4. GPU Cluster Simulator (`simulator/`)

**Purpose:** Provide realistic testing environment without physical hardware.

**Key Classes:**
- `SimulatedGPU`: Models individual GPU with thermal dynamics
- `GPUClusterSimulator`: Manages cluster of simulated GPUs
- `WorkloadGenerator`: Creates synthetic inference requests
- `SimulationRunner`: Runs comparative experiments

**Thermal Model:**
```
dT/dt = (heat_from_load - cooling) / thermal_mass

where:
  heat_from_load = utilization * load_heating_coefficient
  cooling = (temp - ambient) * ambient_cooling_coefficient
```

### 5. API Layer (`api/`, `main.py`)

**Purpose:** RESTful interface for clients and dashboard.

**Endpoints:**
- `POST /inference` - Submit inference request
- `GET /gpus` - Get all GPU states
- `GET /contexts` - Get cached contexts
- `GET /metrics` - Get system metrics
- `GET /simulation/run` - Run comparison

## Data Flow

### Inference Request Processing

1. Client submits request with prompt and optional context ID
2. Request analyzer estimates workload size (SMALL/MEDIUM/LARGE/EXTREME)
3. Context fingerprinter checks if context exists in cache
4. Scheduler queries all GPU states from telemetry layer
5. GPUScorer calculates scores for each viable GPU
6. Best GPU selected based on composite score
7. Decision logged with reasoning
8. GPU state updated with new workload
9. Thermal predictor updated with new temperature reading

### Context Registration Flow

1. New context arrives
2. Compute SHA-256 hash of content
3. Check if identical context exists (by hash)
4. If exists: increment reference count, return existing
5. If new: create metadata, add to index, store on GPU

### Thermal Update Loop

1. Read GPU temperatures (real or simulated)
2. Add samples to ThermalHistory
3. Calculate trends using linear regression
4. Generate predictions for 30s, 60s, 5min horizons
5. Update scheduler's thermal model
6. Trigger migration if GPU approaching critical temp

## Extension Points

### For Production Deployment

1. **Real GPU Integration:**
   - Replace `SimulatedGPU` with NVML-based telemetry
   - Implement actual CUDA context management
   - Add vLLM/SGLang integration for KV-cache access

2. **Enhanced Tokenization:**
   - Replace simple word tokenizer with HuggingFace transformers
   - Support multiple model vocabularies
   - Cache tokenized representations

3. **ML-Based Prediction:**
   - Extend `AdvancedThermalPredictor` with neural network
   - Train on historical thermal data
   - Incorporate workload features

4. **Distributed Context Store:**
   - Replace in-memory index with Redis/PostgreSQL
   - Implement distributed hash table for multi-node
   - Add consistency protocols

5. **Advanced Migration:**
   - Implement actual GPU-to-GPU memory transfer
   - Consider NVLink bandwidth constraints
   - Batch migrations to minimize overhead

## Configuration

Key parameters tunable via environment or config:

```yaml
scheduler:
  weights:
    context_reuse: 0.35
    thermal_headroom: 0.20
    memory_availability: 0.15
    compute_availability: 0.15
    thermal_stress: -0.15
  
thresholds:
  thermal_critical: 85.0  # °C
  thermal_warning: 75.0   # °C
  vram_min_available: 2.0 # GB

simulation:
  num_gpus: 8
  num_racks: 4
  vram_per_gpu: 24.0  # GB
  context_sharing_ratio: 0.7
```

## Observability

### Logging

All scheduling decisions are logged with:
- Request ID
- Selected GPU and candidate GPUs
- Temperature and thermal trend
- Context match status
- Score components
- Selection reasoning

### Metrics Exported

- Per-GPU: temperature, utilization, VRAM, power
- System-wide: avg/peak temp, total VRAM, context hit rate
- Scheduler: decisions/sec, reuse rate, migration count
- Cache: total contexts, memory saved, lookup latency

## Testing Strategy

1. **Unit Tests:** Individual component functionality
2. **Integration Tests:** End-to-end request processing
3. **Simulation Tests:** Comparative performance analysis
4. **Load Tests:** High-throughput scenarios
5. **Chaos Tests:** GPU failures, thermal spikes
