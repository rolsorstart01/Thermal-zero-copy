"""
Thermal Prediction Engine for ThermoCache.

This module implements thermal prediction models that track temperature trajectories
and estimate future GPU temperatures. Uses weighted moving averages for prediction,
with an architecture designed to support future ML-based predictors.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import numpy as np

from models.schemas import ThermalPrediction


logger = logging.getLogger(__name__)


class ThermalHistory:
    """
    Maintains temperature history for a GPU.
    
    Stores timestamped temperature readings and provides methods
    for trend analysis and prediction.
    """
    
    def __init__(self, max_samples: int = 300):
        """
        Initialize thermal history.
        
        Args:
            max_samples: Maximum number of samples to retain (5 min at 1 sample/sec)
        """
        self.max_samples = max_samples
        self._samples: deque[Tuple[datetime, float]] = deque(maxlen=max_samples)
    
    def add_sample(self, temperature: float, timestamp: Optional[datetime] = None):
        """
        Add a temperature sample.
        
        Args:
            temperature: Temperature reading in Celsius
            timestamp: Sample timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        self._samples.append((timestamp, temperature))
    
    def get_trend(self, window_seconds: int = 60) -> float:
        """
        Calculate temperature trend over a time window.
        
        Args:
            window_seconds: Time window for trend calculation
            
        Returns:
            Temperature change rate in °C/minute
        """
        if len(self._samples) < 2:
            return 0.0
        
        now = datetime.now()
        cutoff = now - timedelta(seconds=window_seconds)
        
        # Filter samples within window
        recent = [(ts, temp) for ts, temp in self._samples if ts >= cutoff]
        
        if len(recent) < 2:
            recent = list(self._samples)[-10:]  # Fall back to last 10 samples
            if len(recent) < 2:
                return 0.0
        
        # Linear regression for trend
        times = [(ts - recent[0][0]).total_seconds() for ts, _ in recent]
        temps = [temp for _, temp in recent]
        
        if max(times) - min(times) < 1:  # Less than 1 second span
            return 0.0
        
        # Simple linear fit: slope = Δy / Δx
        slope = (temps[-1] - temps[0]) / (times[-1] - times[0]) if times[-1] != times[0] else 0
        
        # Convert to °C/minute
        return slope * 60
    
    def predict_temperature(self, seconds_ahead: int) -> float:
        """
        Predict temperature at a future time.
        
        Args:
            seconds_ahead: How far ahead to predict
            
        Returns:
            Predicted temperature in Celsius
        """
        if len(self._samples) < 2:
            return self._samples[0][1] if self._samples else 25.0  # Default ambient
        
        current_temp = self._samples[-1][1]
        trend = self.get_trend(window_seconds=min(60, len(self._samples)))
        
        # Simple linear extrapolation with damping
        # Damping factor reduces prediction confidence for longer horizons
        damping = 0.8 ** (seconds_ahead / 60)  # Exponential damping
        predicted_change = trend * (seconds_ahead / 60) * damping
        
        return current_temp + predicted_change
    
    def get_recent_temps(self, count: int = 10) -> List[float]:
        """Get most recent temperature readings."""
        return [temp for _, temp in list(self._samples)[-count:]]
    
    @property
    def current_temperature(self) -> float:
        """Get most recent temperature."""
        return self._samples[-1][1] if self._samples else 25.0
    
    @property
    def average_temperature(self) -> float:
        """Get average temperature over history."""
        if not self._samples:
            return 25.0
        return sum(temp for _, temp in self._samples) / len(self._samples)
    
    @property
    def max_temperature(self) -> float:
        """Get maximum recorded temperature."""
        if not self._samples:
            return 25.0
        return max(temp for _, temp in self._samples)


class ThermalPredictor:
    """
    Main thermal prediction engine.
    
    Manages thermal histories for multiple GPUs and provides
    predictions with confidence estimates.
    """
    
    def __init__(self):
        """Initialize the thermal predictor."""
        self._histories: Dict[int, ThermalHistory] = {}
        self._model_weights: Dict[str, float] = {
            "linear": 0.6,
            "exponential": 0.3,
            "seasonal": 0.1
        }
    
    def update_gpu_temperature(self, gpu_id: int, temperature: float):
        """
        Update temperature reading for a GPU.
        
        Args:
            gpu_id: GPU identifier
            temperature: Temperature in Celsius
        """
        if gpu_id not in self._histories:
            self._histories[gpu_id] = ThermalHistory()
        self._histories[gpu_id].add_sample(temperature)
    
    def get_prediction(self, gpu_id: int) -> Optional[ThermalPrediction]:
        """
        Get thermal prediction for a GPU.
        
        Args:
            gpu_id: GPU identifier
            
        Returns:
            ThermalPrediction or None if no history
        """
        if gpu_id not in self._histories:
            return None
        
        history = self._histories[gpu_id]
        
        if len(history._samples) < 2:
            current = history.current_temperature
            return ThermalPrediction(
                gpu_id=gpu_id,
                current_temp=current,
                temp_30s=current,
                temp_60s=current,
                temp_5min=current,
                confidence=0.1
            )
        
        current_temp = history.current_temperature
        trend = history.get_trend()
        
        # Calculate predictions with different confidence based on data quality
        data_quality = min(1.0, len(history._samples) / 100)  # Confidence based on samples
        
        # Apply damping for longer predictions
        pred_30s = history.predict_temperature(30)
        pred_60s = history.predict_temperature(60)
        pred_5min = history.predict_temperature(300)
        
        # Clamp predictions to reasonable range
        pred_30s = max(20, min(100, pred_30s))
        pred_60s = max(20, min(100, pred_60s))
        pred_5min = max(20, min(100, pred_5min))
        
        # Overall confidence based on data quality and trend stability
        recent_temps = history.get_recent_temps(10)
        if len(recent_temps) >= 2:
            temp_variance = np.var(recent_temps)
            stability_confidence = max(0.5, 1.0 - (temp_variance / 50))  # Lower variance = higher confidence
        else:
            stability_confidence = 0.5
        
        confidence = data_quality * stability_confidence
        
        return ThermalPrediction(
            gpu_id=gpu_id,
            current_temp=current_temp,
            temp_30s=pred_30s,
            temp_60s=pred_60s,
            temp_5min=pred_5min,
            confidence=confidence
        )
    
    def get_thermal_trajectory(self, gpu_id: int) -> str:
        """
        Classify thermal trajectory for a GPU.
        
        Args:
            gpu_id: GPU identifier
            
        Returns:
            Trajectory classification: "RAPIDLY_INCREASING", "INCREASING", 
            "STABLE", "DECREASING", "RAPIDLY_DECREASING"
        """
        if gpu_id not in self._histories:
            return "UNKNOWN"
        
        trend = self._histories[gpu_id].get_trend()
        
        if trend > 3.0:
            return "RAPIDLY_INCREASING"
        elif trend > 0.5:
            return "INCREASING"
        elif trend > -0.5:
            return "STABLE"
        elif trend > -3.0:
            return "DECREASING"
        else:
            return "RAPIDLY_DECREASING"
    
    def is_cooling(self, gpu_id: int) -> bool:
        """Check if GPU is cooling down."""
        return self.get_thermal_trajectory(gpu_id) in ["DECREASING", "RAPIDLY_DECREASING"]
    
    def is_heating(self, gpu_id: int) -> bool:
        """Check if GPU is heating up."""
        return self.get_thermal_trajectory(gpu_id) in ["INCREASING", "RAPIDLY_INCREASING"]
    
    def will_exceed_threshold(
        self,
        gpu_id: int,
        threshold: float,
        time_horizon: int = 300
    ) -> bool:
        """
        Predict if GPU will exceed temperature threshold.
        
        Args:
            gpu_id: GPU identifier
            threshold: Temperature threshold in Celsius
            time_horizon: Prediction horizon in seconds
            
        Returns:
            True if threshold likely to be exceeded
        """
        prediction = self.get_prediction(gpu_id)
        if not prediction:
            return False
        
        # Check if any predicted temperature exceeds threshold with margin
        margin = 3.0  # Safety margin
        return (
            prediction.temp_30s > threshold - margin or
            prediction.temp_60s > threshold - margin or
            prediction.temp_5min > threshold - margin
        )
    
    def get_all_predictions(self) -> List[ThermalPrediction]:
        """Get predictions for all tracked GPUs."""
        predictions = []
        for gpu_id in self._histories:
            pred = self.get_prediction(gpu_id)
            if pred:
                predictions.append(pred)
        return predictions
    
    def reset_history(self, gpu_id: int):
        """Reset thermal history for a GPU."""
        if gpu_id in self._histories:
            self._histories[gpu_id] = ThermalHistory()
    
    def get_trend_summary(self) -> Dict[str, int]:
        """Get summary of thermal trends across all GPUs."""
        summary = {
            "RAPIDLY_INCREASING": 0,
            "INCREASING": 0,
            "STABLE": 0,
            "DECREASING": 0,
            "RAPIDLY_DECREASING": 0,
            "UNKNOWN": 0
        }
        for gpu_id in self._histories:
            trajectory = self.get_thermal_trajectory(gpu_id)
            summary[trajectory] = summary.get(trajectory, 0) + 1
        return summary


class AdvancedThermalPredictor(ThermalPredictor):
    """
    Extended thermal predictor with support for more sophisticated models.
    
    This class demonstrates the architecture for future ML-based prediction.
    Currently uses enhanced statistical methods but maintains the same interface.
    """
    
    def __init__(self, enable_exponential_smoothing: bool = True):
        """
        Initialize advanced predictor.
        
        Args:
            enable_exponential_smoothing: Use exponential smoothing for predictions
        """
        super().__init__()
        self.enable_exponential_smoothing = enable_exponential_smoothing
        self._alpha = 0.3  # Smoothing factor
    
    def predict_with_exponential_smoothing(
        self,
        gpu_id: int,
        seconds_ahead: int
    ) -> Optional[float]:
        """
        Predict using exponential smoothing.
        
        Args:
            gpu_id: GPU identifier
            seconds_ahead: Prediction horizon
            
        Returns:
            Predicted temperature or None
        """
        if gpu_id not in self._histories:
            return None
        
        history = self._histories[gpu_id]
        recent = history.get_recent_temps(20)
        
        if len(recent) < 3:
            return history.current_temperature
        
        # Simple exponential smoothing forecast
        smoothed = recent[0]
        for temp in recent[1:]:
            smoothed = self._alpha * temp + (1 - self._alpha) * smoothed
        
        # Extrapolate with smoothed value
        trend = (recent[-1] - recent[0]) / len(recent)
        steps_ahead = seconds_ahead // 3  # Assume ~3 second intervals
        
        return smoothed + trend * steps_ahead
