"""
Pydantic models for ThermoCache API.

These models define the data structures used throughout the system,
including GPU states, context metadata, scheduling decisions, and API requests/responses.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from datetime import datetime
from enum import Enum


class WorkloadSize(str, Enum):
    """Classification of workload sizes."""
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    EXTREME = "EXTREME"


class ThermalStatus(str, Enum):
    """Thermal status classification."""
    COLD = "COLD"
    NORMAL = "NORMAL"
    WARM = "WARM"
    HOT = "HOT"
    CRITICAL = "CRITICAL"


class GPUMetrics(BaseModel):
    """Current metrics for a GPU."""
    gpu_id: int = Field(..., description="GPU identifier")
    temperature: float = Field(..., description="Current temperature in Celsius")
    utilization: float = Field(..., ge=0, le=100, description="GPU utilization percentage")
    vram_used: float = Field(..., ge=0, description="VRAM used in GB")
    vram_total: float = Field(..., ge=0, description="Total VRAM in GB")
    power_draw: float = Field(..., ge=0, description="Power draw in Watts")
    thermal_trend: float = Field(..., description="Temperature trend in °C/min")
    thermal_headroom: float = Field(..., ge=0, le=1, description="Available thermal headroom (0-1)")
    memory_pressure: float = Field(..., ge=0, le=1, description="Memory pressure (0-1)")
    rack_id: int = Field(..., description="Rack identifier")
    contexts: List[str] = Field(default_factory=list, description="List of context IDs on this GPU")
    
    @property
    def vram_available(self) -> float:
        """Calculate available VRAM in GB."""
        return self.vram_total - self.vram_used
    
    @property
    def vram_utilization(self) -> float:
        """Calculate VRAM utilization percentage."""
        return (self.vram_used / self.vram_total) * 100 if self.vram_total > 0 else 0
    
    def get_thermal_status(self) -> ThermalStatus:
        """Determine thermal status based on temperature."""
        if self.temperature < 50:
            return ThermalStatus.COLD
        elif self.temperature < 65:
            return ThermalStatus.NORMAL
        elif self.temperature < 75:
            return ThermalStatus.WARM
        elif self.temperature < 85:
            return ThermalStatus.HOT
        else:
            return ThermalStatus.CRITICAL


class ContextMetadata(BaseModel):
    """Metadata for a cached context."""
    context_id: str = Field(..., description="Unique context identifier")
    hash: str = Field(..., description="SHA-256 hash of context content")
    token_count: int = Field(..., ge=0, description="Number of tokens in context")
    gpu_location: int = Field(..., description="GPU ID where context is stored")
    memory_size: float = Field(..., ge=0, description="Memory size in GB")
    creation_time: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    last_accessed: datetime = Field(default_factory=datetime.now, description="Last access timestamp")
    reference_count: int = Field(default=1, ge=0, description="Number of active references")
    access_frequency: float = Field(default=0, ge=0, description="Accesses per minute")
    prefix_tokens: Optional[List[int]] = Field(default=None, description="Token prefix for partial matching")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class InferenceRequest(BaseModel):
    """Request for LLM inference."""
    request_id: Optional[str] = Field(None, description="Unique request ID (auto-generated if not provided)")
    prompt: str = Field(..., description="The prompt to process")
    context: Optional[str] = Field(None, description="Context identifier for reuse")
    max_tokens: int = Field(default=512, ge=1, le=8192, description="Maximum tokens to generate")
    priority: int = Field(default=5, ge=1, le=10, description="Request priority (1=lowest, 10=highest)")
    user_id: Optional[str] = Field(None, description="User identifier")
    
    def estimate_workload_size(self) -> WorkloadSize:
        """Estimate workload size based on prompt length and max_tokens."""
        estimated_tokens = len(self.prompt.split()) + self.max_tokens
        if estimated_tokens < 256:
            return WorkloadSize.SMALL
        elif estimated_tokens < 1024:
            return WorkloadSize.MEDIUM
        elif estimated_tokens < 4096:
            return WorkloadSize.LARGE
        else:
            return WorkloadSize.EXTREME


class SchedulingDecision(BaseModel):
    """Result of a scheduling decision."""
    request_id: str = Field(..., description="Request identifier")
    selected_gpu: int = Field(..., description="Selected GPU ID")
    context_reused: bool = Field(..., description="Whether existing context was reused")
    context_id: Optional[str] = Field(None, description="Context ID if reused or created")
    temperature: float = Field(..., description="GPU temperature at scheduling time")
    thermal_headroom: float = Field(..., description="Thermal headroom at scheduling time")
    migration_required: bool = Field(default=False, description="Whether migration is needed")
    score: float = Field(..., description="Scheduling score")
    candidate_gpus: List[int] = Field(default_factory=list, description="Considered GPU IDs")
    reason: str = Field(..., description="Reason for selection")
    timestamp: datetime = Field(default_factory=datetime.now, description="Decision timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ThermalPrediction(BaseModel):
    """Predicted thermal state."""
    gpu_id: int = Field(..., description="GPU identifier")
    current_temp: float = Field(..., description="Current temperature")
    temp_30s: float = Field(..., description="Predicted temperature in 30 seconds")
    temp_60s: float = Field(..., description="Predicted temperature in 60 seconds")
    temp_5min: float = Field(..., description="Predicted temperature in 5 minutes")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")


class SystemMetrics(BaseModel):
    """Aggregated system metrics."""
    total_gpus: int = Field(..., description="Total number of GPUs")
    active_gpus: int = Field(..., description="Number of active GPUs")
    average_temperature: float = Field(..., description="Average GPU temperature")
    peak_temperature: float = Field(..., description="Peak GPU temperature")
    total_vram_used: float = Field(..., description="Total VRAM used across all GPUs (GB)")
    total_vram_capacity: float = Field(..., description="Total VRAM capacity (GB)")
    vram_utilization_percent: float = Field(..., description="Overall VRAM utilization percentage")
    total_contexts: int = Field(..., description="Total cached contexts")
    context_hit_rate: float = Field(..., ge=0, le=1, description="Context cache hit rate")
    kv_cache_memory_saved: float = Field(..., description="KV-cache memory saved (GB)")
    requests_per_second: float = Field(..., description="Current request throughput")
    average_latency_ms: float = Field(..., description="Average request latency")
    estimated_power_kw: float = Field(..., description="Estimated power consumption (kW)")
    thermal_hotspots: List[int] = Field(default_factory=list, description="GPUs with thermal issues")


class SimulationConfig(BaseModel):
    """Configuration for simulation runs."""
    num_gpus: int = Field(default=8, ge=1, le=64, description="Number of GPUs to simulate")
    num_racks: int = Field(default=4, ge=1, description="Number of racks")
    num_requests: int = Field(default=10000, ge=1, description="Number of requests to simulate")
    context_sharing_ratio: float = Field(default=0.7, ge=0, le=1, description="Ratio of requests with shared context")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class SimulationResult(BaseModel):
    """Results from a simulation run."""
    baseline_metrics: SystemMetrics = Field(..., description="Baseline scheduler metrics")
    thermocache_metrics: SystemMetrics = Field(..., description="ThermoCache metrics")
    comparison: Dict[str, float] = Field(..., description="Comparison metrics")
    decisions_sample: List[SchedulingDecision] = Field(default_factory=list, description="Sample decisions")


class GPUState(GPUMetrics):
    """Extended GPU state with additional runtime info."""
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    workload_queue: List[str] = Field(default_factory=list, description="Queued request IDs")
    recent_decisions: List[str] = Field(default_factory=list, description="Recent decision IDs")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
