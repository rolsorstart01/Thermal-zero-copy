"""
ThermoCache - Thermal-Aware Zero-Copy LLM Inference Engine

FastAPI application providing REST API for the scheduling system.
"""

import logging
import asyncio
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from models.schemas import (
    InferenceRequest, SchedulingDecision, GPUMetrics, 
    ContextMetadata, SystemMetrics, SimulationConfig, 
    SimulationResult, ThermalPrediction
)
from context_cache.engine import ContextCache
from thermal.predictor import ThermalPredictor
from scheduler.scheduler import ThermalAwareScheduler, BaselineScheduler
from simulator.cluster import GPUClusterSimulator, SimulationRunner


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize application
app = FastAPI(
    title="ThermoCache",
    description="Thermal-Aware Zero-Copy LLM Inference Engine",
    version="0.1.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state (in production, use proper state management)
context_cache: Optional[ContextCache] = None
thermal_predictor: Optional[ThermalPredictor] = None
scheduler: Optional[ThermalAwareScheduler] = None
cluster: Optional[GPUClusterSimulator] = None
simulation_runner: Optional[SimulationRunner] = None


@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup."""
    global context_cache, thermal_predictor, scheduler, cluster, simulation_runner
    
    logger.info("Initializing ThermoCache components...")
    
    context_cache = ContextCache()
    thermal_predictor = ThermalPredictor()
    scheduler = ThermalAwareScheduler(context_cache, thermal_predictor)
    cluster = GPUClusterSimulator(num_gpus=8, num_racks=4)
    simulation_runner = SimulationRunner(cluster, context_sharing_ratio=0.7, seed=42)
    
    # Initialize thermal predictor with current temps
    metrics = cluster.get_all_metrics()
    for m in metrics:
        thermal_predictor.update_gpu_temperature(m.gpu_id, m.temperature)
    
    logger.info("ThermoCache initialized successfully")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "ThermoCache",
        "tagline": "Share the context. Spread the heat.",
        "version": "0.1.0",
        "status": "running"
    }


@app.post("/inference", response_model=SchedulingDecision)
async def submit_inference(request: InferenceRequest):
    """
    Submit an inference request for scheduling.
    
    Returns the scheduling decision including selected GPU,
    context reuse status, and thermal information.
    """
    if not scheduler or not cluster:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Get current GPU states
    gpu_metrics = cluster.get_all_metrics()
    
    # Estimate workload size
    workload_size = request.estimate_workload_size()
    
    # Make scheduling decision
    decision = await scheduler.select_gpu(
        gpus=gpu_metrics,
        workload_size=workload_size,
        context_id=request.context,
        request_id=request.request_id or f"req_{datetime.now().timestamp()}"
    )
    
    # Update cluster state
    vram_increase = 0.05 if not decision.context_reused else 0.01
    cluster.update_gpu_workload(
        decision.selected_gpu,
        utilization_delta=2.0,
        vram_delta=vram_increase,
        contexts=[request.context] if request.context else None
    )
    
    # Update thermal predictor
    for m in gpu_metrics:
        thermal_predictor.update_gpu_temperature(m.gpu_id, m.temperature)
    
    cluster.request_count += 1
    
    return decision


@app.get("/gpus", response_model=List[GPUMetrics])
async def get_gpus():
    """Get status of all GPUs in the cluster."""
    if not cluster:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return cluster.get_all_metrics()


@app.get("/gpus/{gpu_id}", response_model=GPUMetrics)
async def get_gpu(gpu_id: int):
    """Get status of a specific GPU."""
    if not cluster:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    metrics = cluster.get_gpu_metrics(gpu_id)
    if not metrics:
        raise HTTPException(status_code=404, detail=f"GPU {gpu_id} not found")
    
    return metrics


@app.get("/gpus/{gpu_id}/prediction", response_model=ThermalPrediction)
async def get_gpu_prediction(gpu_id: int):
    """Get thermal prediction for a specific GPU."""
    if not thermal_predictor:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    prediction = thermal_predictor.get_prediction(gpu_id)
    if not prediction:
        raise HTTPException(status_code=404, detail=f"No prediction for GPU {gpu_id}")
    
    return prediction


@app.get("/contexts", response_model=List[ContextMetadata])
async def get_contexts():
    """Get all cached contexts."""
    if not context_cache:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return await context_cache.index.get_all_contexts()


@app.get("/contexts/{context_id}", response_model=ContextMetadata)
async def get_context(context_id: str):
    """Get details of a specific context."""
    if not context_cache:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    ctx = await context_cache.find_by_id(context_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")
    
    return ctx


@app.get("/scheduler/decisions", response_model=List[SchedulingDecision])
async def get_scheduler_decisions(limit: int = 100):
    """Get recent scheduling decisions."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return scheduler.get_recent_decisions(limit)


@app.get("/metrics", response_model=SystemMetrics)
async def get_metrics():
    """Get aggregated system metrics."""
    if not cluster or not context_cache:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return cluster.get_system_metrics(context_cache)


@app.get("/simulation/run", response_model=SimulationResult)
async def run_simulation(
    num_requests: int = 1000,
    context_sharing_ratio: float = 0.7,
    seed: Optional[int] = 42
):
    """
    Run a comparison simulation between baseline and ThermoCache.
    
    This runs both schedulers with identical workloads and returns
    comparative metrics demonstrating the benefits of thermal-aware
    scheduling with context deduplication.
    """
    if not simulation_runner:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    logger.info(f"Running simulation: {num_requests} requests, "
                f"{context_sharing_ratio*100:.0f}% context sharing")
    
    results = await simulation_runner.run_comparison(num_requests)
    
    # Convert to SimulationResult format
    return SimulationResult(
        baseline_metrics=results["baseline"],
        thermocache_metrics=results["thermocache"],
        comparison=results["comparison"],
        decisions_sample=[]  # Omit for brevity
    )


@app.get("/scheduler/stats")
async def get_scheduler_stats():
    """Get scheduler statistics."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return scheduler.get_scheduler_stats()


@app.get("/cache/stats")
async def get_cache_stats():
    """Get context cache statistics."""
    if not context_cache:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return await context_cache.get_cache_stats()


@app.get("/thermal/trends")
async def get_thermal_trends():
    """Get thermal trends summary across all GPUs."""
    if not thermal_predictor:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return thermal_predictor.get_trend_summary()


@app.post("/simulate/tick")
async def simulate_tick():
    """Advance simulation by one tick."""
    if not cluster:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    cluster.simulate_workload_tick()
    
    # Update thermal predictor
    metrics = cluster.get_all_metrics()
    for m in metrics:
        thermal_predictor.update_gpu_temperature(m.gpu_id, m.temperature)
    
    return {"status": "ok", "gpus": len(metrics)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
