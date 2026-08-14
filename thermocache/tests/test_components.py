"""
Unit tests for ThermoCache components.
"""

import pytest
import asyncio
from models.schemas import GPUMetrics, WorkloadSize
from context_cache.engine import ContextFingerprinter, ContextIndex, ContextCache
from thermal.predictor import ThermalHistory, ThermalPredictor
from scheduler.scheduler import GPUScorer, BaselineScheduler


class TestContextFingerprinter:
    """Tests for context fingerprinting."""
    
    def test_hash_computation(self):
        fp = ContextFingerprinter()
        content = "test content"
        hash1 = fp.compute_hash(content)
        hash2 = fp.compute_hash(content)
        assert hash1 == hash2  # Same content = same hash
    
    def test_different_content_different_hash(self):
        fp = ContextFingerprinter()
        hash1 = fp.compute_hash("content A")
        hash2 = fp.compute_hash("content B")
        assert hash1 != hash2
    
    def test_tokenization(self):
        fp = ContextFingerprinter()
        tokens = fp.tokenize("hello world test")
        assert len(tokens) == 3
        assert all(isinstance(t, int) for t in tokens)


class TestThermalHistory:
    """Tests for thermal history tracking."""
    
    def test_add_sample(self):
        history = ThermalHistory()
        history.add_sample(45.0)
        assert history.current_temperature == 45.0
    
    def test_trend_calculation(self):
        history = ThermalHistory()
        # Add increasing temperatures
        for i in range(10):
            history.add_sample(40 + i * 2)
        trend = history.get_trend()
        assert trend > 0  # Should be increasing
    
    def test_prediction(self):
        history = ThermalHistory()
        history.add_sample(50.0)
        history.add_sample(52.0)
        history.add_sample(54.0)
        pred = history.predict_temperature(30)
        assert pred > 54.0  # Should predict higher if trend is up


class TestGPUScorer:
    """Tests for GPU scoring algorithm."""
    
    def test_context_reuse_score_with_match(self):
        scorer = GPUScorer()
        gpu = GPUMetrics(
            gpu_id=0,
            temperature=60.0,
            utilization=50.0,
            vram_used=10.0,
            vram_total=24.0,
            power_draw=200.0,
            thermal_trend=0.5,
            thermal_headroom=0.5,
            memory_pressure=0.4,
            rack_id=0,
            contexts=["ctx_123"]
        )
        score = scorer.calculate_context_reuse_score(gpu, "ctx_123", None)
        assert score == 1.0  # Perfect match
    
    def test_context_reuse_score_without_match(self):
        scorer = GPUScorer()
        gpu = GPUMetrics(
            gpu_id=0,
            temperature=60.0,
            utilization=50.0,
            vram_used=10.0,
            vram_total=24.0,
            power_draw=200.0,
            thermal_trend=0.5,
            thermal_headroom=0.5,
            memory_pressure=0.4,
            rack_id=0,
            contexts=[]
        )
        score = scorer.calculate_context_reuse_score(gpu, "ctx_456", None)
        assert score == 0.0  # No match


class TestBaselineScheduler:
    """Tests for baseline scheduler."""
    
    def test_select_gpu(self):
        scheduler = BaselineScheduler()
        gpus = [
            GPUMetrics(
                gpu_id=0,
                temperature=70.0,
                utilization=80.0,
                vram_used=20.0,
                vram_total=24.0,
                power_draw=250.0,
                thermal_trend=1.0,
                thermal_headroom=0.2,
                memory_pressure=0.83,
                rack_id=0,
                contexts=[]
            ),
            GPUMetrics(
                gpu_id=1,
                temperature=50.0,
                utilization=30.0,
                vram_used=5.0,
                vram_total=24.0,
                power_draw=150.0,
                thermal_trend=-0.5,
                thermal_headroom=0.8,
                memory_pressure=0.21,
                rack_id=0,
                contexts=[]
            )
        ]
        decision = scheduler.select_gpu(
            gpus=gpus,
            workload_size=WorkloadSize.MEDIUM,
            request_id="test_req"
        )
        assert decision.selected_gpu == 1  # Should prefer cooler, less utilized GPU


@pytest.mark.asyncio
async def test_context_cache():
    """Test context cache registration and lookup."""
    cache = ContextCache()
    
    # Register a context
    ctx, is_new = await cache.register_or_reuse(
        content="test context content",
        gpu_location=0,
        context_id="test_ctx"
    )
    assert is_new
    assert ctx.context_id == "test_ctx"
    
    # Try to register same content again
    ctx2, is_new2 = await cache.register_or_reuse(
        content="test context content",
        gpu_location=1,
        context_id="test_ctx_2"
    )
    assert not is_new2  # Should reuse existing
    assert ctx2.context_id == "test_ctx"  # Same ID as first


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
