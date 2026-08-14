"""
Thermal-Aware GPU Scheduler for ThermoCache.

This module implements the core scheduling algorithm that combines context locality,
thermal state, and compute availability to make optimal GPU placement decisions.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import asyncio

from models.schemas import (
    GPUMetrics, SchedulingDecision, ContextMetadata, 
    WorkloadSize, ThermalStatus
)
from context_cache.engine import ContextCache
from thermal.predictor import ThermalPredictor


logger = logging.getLogger(__name__)


# Scheduling weight coefficients
WEIGHTS = {
    "context_reuse": 0.35,       # High priority for context reuse
    "memory_availability": 0.15, # VRAM headroom
    "compute_availability": 0.15, # GPU utilization headroom
    "thermal_headroom": 0.20,    # Thermal capacity
    "thermal_stress": -0.15,     # Penalize hot GPUs
}

# Thresholds
THERMAL_CRITICAL_TEMP = 85.0  # °C - do not schedule on GPUs above this
THERMAL_WARNING_TEMP = 75.0   # °C - apply penalty
VRAM_MIN_AVAILABLE = 2.0      # GB - minimum free VRAM required


class GPUScorer:
    """
    Calculates scheduling scores for GPU candidates.
    
    Implements the unified scoring algorithm that balances:
    - Context reuse benefit
    - Memory availability
    - Compute availability
    - Thermal headroom
    - Thermal stress
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the scorer.
        
        Args:
            weights: Custom weight coefficients (uses defaults if None)
        """
        self.weights = weights or WEIGHTS.copy()
    
    def calculate_context_reuse_score(
        self,
        gpu: GPUMetrics,
        required_context_id: Optional[str],
        context_cache: ContextCache
    ) -> float:
        """
        Calculate score component for context reuse.
        
        Args:
            gpu: GPU metrics
            required_context_id: Context ID needed for request
            context_cache: Context cache instance
            
        Returns:
            Score between 0 and 1
        """
        if not required_context_id:
            return 0.5  # Neutral if no context required
        
        # Check if context exists on this GPU
        # gpu.contexts is a list of context_id strings
        contexts_on_gpu = gpu.contexts
        
        if required_context_id in contexts_on_gpu:
            return 1.0  # Perfect match - context already on GPU
        
        # Partial credit if GPU has some contexts (potential for batch optimization)
        if contexts_on_gpu:
            return 0.3
        
        return 0.0
    
    def calculate_memory_score(self, gpu: GPUMetrics, workload_size: WorkloadSize) -> float:
        """
        Calculate score component for memory availability.
        
        Args:
            gpu: GPU metrics
            workload_size: Estimated workload size
            
        Returns:
            Score between 0 and 1
        """
        available_vram = gpu.vram_available
        
        # Define VRAM requirements by workload size
        vram_requirements = {
            WorkloadSize.SMALL: 1.0,
            WorkloadSize.MEDIUM: 4.0,
            WorkloadSize.LARGE: 8.0,
            WorkloadSize.EXTREME: 16.0
        }
        
        required = vram_requirements.get(workload_size, 4.0)
        
        if available_vram < VRAM_MIN_AVAILABLE:
            return 0.0  # Not enough space
        
        if available_vram >= required * 1.5:
            return 1.0  # Plenty of space
        
        # Linear interpolation
        return min(1.0, (available_vram - VRAM_MIN_AVAILABLE) / (required * 1.5 - VRAM_MIN_AVAILABLE))
    
    def calculate_compute_score(self, gpu: GPUMetrics, workload_size: WorkloadSize) -> float:
        """
        Calculate score component for compute availability.
        
        Args:
            gpu: GPU metrics
            workload_size: Estimated workload size
            
        Returns:
            Score between 0 and 1
        """
        utilization = gpu.utilization / 100.0
        
        # Higher utilization = lower score
        if utilization > 0.95:
            return 0.0  # Nearly saturated
        
        if utilization < 0.3:
            return 1.0  # Lots of headroom
        
        # Linear interpolation
        return 1.0 - utilization
    
    def calculate_thermal_score(self, gpu: GPUMetrics, predictor: ThermalPredictor) -> float:
        """
        Calculate combined thermal score (headroom - stress).
        
        Args:
            gpu: GPU metrics
            predictor: Thermal predictor instance
            
        Returns:
            Combined thermal score between 0 and 1
        """
        temp = gpu.temperature
        
        # Critical temperature check
        if temp >= THERMAL_CRITICAL_TEMP:
            return 0.0  # Do not schedule on critically hot GPUs
        
        # Base thermal headroom score
        # 20°C = perfect (1.0), 85°C = zero (0.0)
        headroom_score = max(0, min(1, (85 - temp) / 65))
        
        # Get thermal trend
        prediction = predictor.get_prediction(gpu.gpu_id)
        if prediction:
            # Penalize if temperature is rising rapidly
            if prediction.temp_30s > temp + 2:
                headroom_score *= 0.7
            if prediction.temp_60s > temp + 5:
                headroom_score *= 0.7
            if prediction.temp_5min > THERMAL_WARNING_TEMP:
                headroom_score *= 0.5
        
        # Additional penalty for warning zone temperatures
        if temp >= THERMAL_WARNING_TEMP:
            headroom_score *= 0.5
        
        return headroom_score
    
    def calculate_total_score(
        self,
        gpu: GPUMetrics,
        workload_size: WorkloadSize,
        required_context_id: Optional[str],
        context_cache: ContextCache,
        predictor: ThermalPredictor
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total scheduling score for a GPU.
        
        Args:
            gpu: GPU metrics
            workload_size: Estimated workload size
            required_context_id: Context ID needed
            context_cache: Context cache instance
            predictor: Thermal predictor instance
            
        Returns:
            Tuple of (total_score, component_scores)
        """
        components = {
            "context_reuse": self.calculate_context_reuse_score(
                gpu, required_context_id, context_cache
            ),
            "memory_availability": self.calculate_memory_score(gpu, workload_size),
            "compute_availability": self.calculate_compute_score(gpu, workload_size),
            "thermal_headroom": self.calculate_thermal_score(gpu, predictor),
        }
        
        # Thermal stress is inverse of thermal headroom
        components["thermal_stress"] = 1 - components["thermal_headroom"]
        
        # Weighted sum
        total = sum(
            components[key] * self.weights.get(key, 0)
            for key in components
        )
        
        # Normalize to 0-1 range (weights may not sum to 1)
        weight_sum = sum(abs(w) for w in self.weights.values())
        if weight_sum > 0:
            total = total / weight_sum
        
        return max(0, min(1, total)), components


class ThermalAwareScheduler:
    """
    Main scheduler that makes GPU placement decisions.
    
    Combines context caching awareness with thermal prediction
    to optimize both memory efficiency and thermal distribution.
    """
    
    def __init__(
        self,
        context_cache: ContextCache,
        thermal_predictor: ThermalPredictor,
        enable_migration: bool = True
    ):
        """
        Initialize the scheduler.
        
        Args:
            context_cache: Context deduplication engine
            thermal_predictor: Thermal prediction engine
            enable_migration: Enable context migration when needed
        """
        self.context_cache = context_cache
        self.thermal_predictor = thermal_predictor
        self.enable_migration = enable_migration
        self.scorer = GPUScorer()
        
        # Decision history
        self.decisions: List[SchedulingDecision] = []
        self.max_decisions = 1000
        
        # Migration tracking
        self.migrations_performed = 0
        self.migrations_skipped = 0
    
    async def select_gpu(
        self,
        gpus: List[GPUMetrics],
        workload_size: WorkloadSize,
        context_id: Optional[str] = None,
        request_id: str = ""
    ) -> SchedulingDecision:
        """
        Select the best GPU for a workload.
        
        Args:
            gpus: Available GPU states
            workload_size: Estimated workload size
            context_id: Required context ID (if any)
            request_id: Request identifier for logging
            
        Returns:
            SchedulingDecision with selected GPU and reasoning
        """
        if not gpus:
            raise ValueError("No GPUs available")
        
        candidate_gpus = [g.gpu_id for g in gpus]
        
        # Filter out GPUs with insufficient resources
        viable_gpus = []
        for gpu in gpus:
            if gpu.vram_available < VRAM_MIN_AVAILABLE:
                continue
            if gpu.temperature >= THERMAL_CRITICAL_TEMP:
                continue
            viable_gpus.append(gpu)
        
        if not viable_gpus:
            # Fall back to all GPUs if none are viable (emergency mode)
            viable_gpus = [g for g in gpus if g.vram_available > 0]
        
        if not viable_gpus:
            raise RuntimeError("No GPUs have available VRAM")
        
        # Score each viable GPU
        scores: Dict[int, Tuple[float, Dict[str, float]]] = {}
        for gpu in viable_gpus:
            score, components = self.scorer.calculate_total_score(
                gpu=gpu,
                workload_size=workload_size,
                required_context_id=context_id,
                context_cache=self.context_cache,
                predictor=self.thermal_predictor
            )
            scores[gpu.gpu_id] = (score, components)
        
        # Select highest scoring GPU
        best_gpu_id = max(scores.keys(), key=lambda x: scores[x][0])
        best_gpu = next(g for g in viable_gpus if g.gpu_id == best_gpu_id)
        best_score, best_components = scores[best_gpu_id]
        
        # Determine if context will be reused
        context_reused = False
        if context_id:
            # best_gpu.contexts is a list of context_id strings
            contexts_on_gpu = best_gpu.contexts
            context_reused = context_id in contexts_on_gpu
        
        # Check if migration might be needed
        migration_required = False
        if self.enable_migration and best_gpu.temperature >= THERMAL_WARNING_TEMP:
            # Consider migrating some contexts away
            migration_required = await self._evaluate_migration_need(best_gpu)
        
        # Generate reason string
        reason = self._generate_reason(
            best_gpu=best_gpu,
            score_components=best_components,
            context_reused=context_reused,
            all_gpus=viable_gpus
        )
        
        decision = SchedulingDecision(
            request_id=request_id,
            selected_gpu=best_gpu_id,
            context_reused=context_reused,
            context_id=context_id,
            temperature=best_gpu.temperature,
            thermal_headroom=best_gpu.thermal_headroom,
            migration_required=migration_required,
            score=best_score,
            candidate_gpus=candidate_gpus,
            reason=reason
        )
        
        # Record decision
        self._record_decision(decision)
        
        logger.info(
            f"Scheduled request {request_id} on GPU {best_gpu_id} "
            f"(score={best_score:.3f}, temp={best_gpu.temperature:.1f}°C)"
        )
        
        return decision
    
    async def _evaluate_migration_need(self, gpu: GPUMetrics) -> bool:
        """
        Evaluate if contexts should be migrated from a GPU.
        
        Args:
            gpu: GPU to evaluate
            
        Returns:
            True if migration recommended
        """
        if gpu.temperature < THERMAL_WARNING_TEMP:
            return False
        
        # Check thermal trajectory
        prediction = self.thermal_predictor.get_prediction(gpu.gpu_id)
        if prediction and prediction.temp_5min > THERMAL_CRITICAL_TEMP:
            return True
        
        # Check if cooling
        if self.thermal_predictor.is_cooling(gpu.gpu_id):
            return False  # Let it cool naturally
        
        return True
    
    def _generate_reason(
        self,
        best_gpu: GPUMetrics,
        score_components: Dict[str, float],
        context_reused: bool,
        all_gpus: List[GPUMetrics]
    ) -> str:
        """Generate human-readable reason for selection."""
        reasons = []
        
        if context_reused:
            reasons.append("Context reuse on GPU")
        
        if score_components.get("thermal_headroom", 0) > 0.8:
            reasons.append("Good thermal headroom")
        elif best_gpu.temperature > THERMAL_WARNING_TEMP:
            reasons.append("Despite elevated temperature")
        
        if score_components.get("memory_availability", 0) > 0.8:
            reasons.append("Ample VRAM available")
        
        if score_components.get("compute_availability", 0) > 0.7:
            reasons.append("Low utilization")
        
        # Find why other GPUs weren't selected
        if len(all_gpus) > 1:
            cooler_gpus = [g for g in all_gpus if g.temperature < best_gpu.temperature - 5]
            if cooler_gpus and context_reused:
                reasons.append("Preferred over cooler GPUs due to context locality")
        
        return "; ".join(reasons) if reasons else "Best overall score"
    
    def _record_decision(self, decision: SchedulingDecision):
        """Record decision in history."""
        self.decisions.append(decision)
        if len(self.decisions) > self.max_decisions:
            self.decisions.pop(0)
    
    def get_recent_decisions(self, limit: int = 100) -> List[SchedulingDecision]:
        """Get recent scheduling decisions."""
        return self.decisions[-limit:]
    
    async def migrate_contexts_from_gpu(
        self,
        gpu_id: int,
        target_gpus: List[GPUMetrics],
        max_contexts: int = 3
    ) -> List[Tuple[str, int]]:
        """
        Migrate contexts away from an overheating GPU.
        
        Args:
            gpu_id: Source GPU ID
            target_gpus: Candidate destination GPUs
            max_contexts: Maximum contexts to migrate
            
        Returns:
            List of (context_id, target_gpu_id) tuples
        """
        migrations = []
        
        # Get contexts on the GPU
        contexts = await self.context_cache.index.get_contexts_on_gpu(gpu_id)
        
        if not contexts:
            return migrations
        
        # Sort by access frequency (migrate least accessed first)
        contexts_sorted = sorted(contexts, key=lambda c: c.access_frequency)
        
        # Find best target GPU for each context
        for ctx in contexts_sorted[:max_contexts]:
            # Skip high-reference contexts
            if ctx.reference_count > 10:
                self.migrations_skipped += 1
                continue
            
            # Find coolest GPU with available VRAM
            best_target = None
            best_temp = float('inf')
            
            for target in target_gpus:
                if target.gpu_id == gpu_id:
                    continue
                if target.vram_available < ctx.memory_size + 1:
                    continue
                if target.temperature < best_temp:
                    best_temp = target.temperature
                    best_target = target
            
            if best_target:
                success = await self.context_cache.migrate_context(
                    ctx.context_id, best_target.gpu_id
                )
                if success:
                    migrations.append((ctx.context_id, best_target.gpu_id))
                    self.migrations_performed += 1
                    logger.info(
                        f"Migrated context {ctx.context_id} from GPU {gpu_id} "
                        f"to GPU {best_target.gpu_id}"
                    )
        
        return migrations
    
    def get_scheduler_stats(self) -> dict:
        """Get scheduler statistics."""
        if not self.decisions:
            return {
                "total_decisions": 0,
                "context_reuse_rate": 0,
                "average_score": 0,
                "migrations_performed": 0
            }
        
        context_reuses = sum(1 for d in self.decisions if d.context_reused)
        avg_score = sum(d.score for d in self.decisions) / len(self.decisions)
        
        return {
            "total_decisions": len(self.decisions),
            "context_reuse_rate": context_reuses / len(self.decisions),
            "average_score": avg_score,
            "migrations_performed": self.migrations_performed,
            "migrations_skipped": self.migrations_skipped
        }


class BaselineScheduler:
    """
    Baseline scheduler for comparison.
    
    Uses traditional scheduling based only on:
    - VRAM availability
    - GPU utilization
    
    Does NOT consider context locality or thermal state.
    """
    
    def __init__(self):
        """Initialize baseline scheduler."""
        self.decisions: List[SchedulingDecision] = []
    
    def select_gpu(
        self,
        gpus: List[GPUMetrics],
        workload_size: WorkloadSize,
        context_id: Optional[str] = None,
        request_id: str = ""
    ) -> SchedulingDecision:
        """
        Select GPU using baseline algorithm.
        
        Args:
            gpus: Available GPU states
            workload_size: Estimated workload size
            context_id: Ignored in baseline
            request_id: Request identifier
            
        Returns:
            SchedulingDecision
        """
        if not gpus:
            raise ValueError("No GPUs available")
        
        # Simple scoring: prefer GPUs with more VRAM and lower utilization
        best_gpu = None
        best_score = -float('inf')
        
        for gpu in gpus:
            if gpu.vram_available < VRAM_MIN_AVAILABLE:
                continue
            
            # Score based on VRAM and utilization only
            score = (
                gpu.vram_available * 0.7 -  # Prefer more available VRAM
                gpu.utilization * 0.3       # Prefer lower utilization
            )
            
            if score > best_score:
                best_score = score
                best_gpu = gpu
        
        if best_gpu is None:
            # Fallback to first GPU with any VRAM
            for gpu in gpus:
                if gpu.vram_available > 0:
                    best_gpu = gpu
                    break
        
        if best_gpu is None:
            raise RuntimeError("No GPUs have available VRAM")
        
        decision = SchedulingDecision(
            request_id=request_id,
            selected_gpu=best_gpu.gpu_id,
            context_reused=False,  # Baseline doesn't track context
            context_id=None,
            temperature=best_gpu.temperature,
            thermal_headroom=best_gpu.thermal_headroom,
            migration_required=False,
            score=best_score / 100,  # Normalize
            candidate_gpus=[g.gpu_id for g in gpus],
            reason=f"Best VRAM ({best_gpu.vram_available:.1f}GB) and utilization ({best_gpu.utilization:.1f}%)"
        )
        
        self.decisions.append(decision)
        return decision
    
    def get_recent_decisions(self, limit: int = 100) -> List[SchedulingDecision]:
        """Get recent decisions."""
        return self.decisions[-limit:]
