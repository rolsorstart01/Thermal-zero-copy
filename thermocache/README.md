# ThermoCache - Thermal-Aware Zero-Copy LLM Inference Engine

**Tagline:** "Share the context. Spread the heat."

## Overview

ThermoCache is an intelligent inference middleware/scheduler that combines **thermal-aware workload scheduling** with **zero-copy context deduplication** for large-scale LLM inference.

Modern AI inference infrastructure wastes resources in two major ways:

1. **GPU memory duplication:** Thousands of users may send requests containing the same system prompt, documents, codebases, or other context. Traditional inference systems can end up maintaining separate KV-cache/context representations for overlapping requests.

2. **Thermal inefficiency:** GPU workloads create uneven thermal hotspots across servers and racks. Schedulers typically optimize for GPU availability, utilization, or VRAM rather than the physical temperature and thermal state of the hardware.

ThermoCache attacks both problems simultaneously.

## Architecture

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

## Key Components

### 1. Context Deduplication Engine

Detects identical or overlapping context across inference requests. Instead of treating every request as independent, the system identifies shared prefix/context and maintains reusable representations.

- **KV-cache/token-level representation reuse**
- **Context fingerprinting** using SHA-256 hashes
- **Reference counting** for context lifecycle management
- **Prefix matching** for partial context reuse

### 2. Thermal-Aware GPU Scheduler

Tracks the state of available GPUs with continuously updated profiles:

- Temperature
- GPU utilization
- VRAM utilization
- Power consumption
- Memory pressure
- Recent workload intensity
- Thermal trend
- Estimated thermal headroom

### 3. Unified Scheduling Algorithm

The scheduler calculates a placement score combining memory locality and thermal state:

```
Score = ContextReuseBenefit + MemoryAvailability + ComputeAvailability 
        + ThermalHeadroom - ThermalStress - ExpectedPowerCost
```

### 4. Thermal Prediction

Tracks temperature trajectory using weighted moving averages to estimate:
- Temperature in 30 seconds
- Temperature in 60 seconds
- Temperature in 5 minutes

### 5. Workload Classification

Classifies incoming inference requests based on expected resource requirements:
- SMALL, MEDIUM, LARGE, EXTREME

### 6. Context Migration

Basic mechanism for moving cached context between GPUs when thermal conditions require it.

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### Installation

```bash
# Backend
cd thermocache/backend
pip install -r requirements.txt

# Frontend
cd thermocache/frontend
npm install
```

### Running the Simulation

```bash
# Start the backend server
cd thermocache/backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run the simulation
python simulate.py

# Start the frontend
cd thermocache/frontend
npm run dev
```

### API Endpoints

- `POST /inference` - Submit an inference request
- `GET /gpus` - Get all GPU states
- `GET /gpus/{id}` - Get specific GPU state
- `GET /contexts` - Get all cached contexts
- `GET /contexts/{id}` - Get specific context info
- `GET /scheduler/decisions` - Get recent scheduling decisions
- `GET /metrics` - Get system metrics
- `GET /simulation/run` - Run simulation comparison

## Demo Scenario

The prototype includes a simulation demonstrating:

- 8 GPUs across 4 racks
- 10,000 inference requests
- 70% of requests share common context

Compare baseline scheduler vs ThermoCache:

| Metric | Baseline | ThermoCache |
|--------|----------|-------------|
| Context Duplication | HIGH | LOW |
| VRAM Usage | HIGH | LOWER |
| Temperature Imbalance | HIGH | BALANCED |
| Peak Temperature | HIGH | LOWER |

## Technical Details

### Context Fingerprinting

Each incoming request is fingerprinted:

```
Input → Tokenization → Context fingerprint → Search existing context index
                                                    ↓
                                    Match found? → YES: reuse / NO: create new
```

### GPU State Tracking

```python
{
    "gpu_id": 0,
    "temperature": 82.0,
    "utilization": 94.0,
    "vram_used": 21.84,  # GB
    "vram_total": 24.0,  # GB
    "power_draw": 285.0,  # Watts
    "thermal_trend": 1.5,  # °C/min
    "thermal_headroom": 0.15,
    "contexts": ["ctx_001", "ctx_002"]
}
```

### Scheduling Decision Log

Each decision logs:
- Request ID
- Selected GPU
- Candidate GPUs
- Temperature
- Thermal trend
- Context match
- VRAM
- Utilization
- Score
- Reason for selection

## Project Structure

```
thermocache/
├── backend/
│   ├── api/           # FastAPI endpoints
│   ├── scheduler/     # Thermal-aware scheduler
│   ├── context_cache/ # Context deduplication engine
│   ├── thermal/       # Thermal prediction models
│   ├── simulator/     # GPU cluster simulator
│   ├── models/        # Pydantic models
│   └── main.py        # Application entry point
├── frontend/
│   ├── src/
│   │   ├── components/ # React components
│   │   ├── hooks/      # Custom hooks
│   │   ├── services/   # API services
│   │   └── types/      # TypeScript types
│   └── package.json
├── docs/
│   ├── architecture.md
│   └── benchmark.md
├── tests/
│   └── test_*.py
└── README.md
```

## Engineering Principles

- **Modular:** Clean separation of concerns
- **Well documented:** Inline comments and docstrings
- **Observable:** Structured logging throughout
- **Testable:** Unit tests for core components
- **Reproducible:** Deterministic simulation mode

## Production Considerations

This prototype demonstrates the **architecture and optimization problem are plausible**. Production challenges include:

- Actual GPU memory allocator behavior
- CUDA stream synchronization
- KV-cache compatibility
- Tensor parallelism
- Distributed inference
- NUMA topology
- PCIe/NVLink bandwidth
- GPU-to-GPU transfer costs
- Rack-level thermal dynamics
- Cooling infrastructure
- Workload prediction
- Fault tolerance

## License

MIT
