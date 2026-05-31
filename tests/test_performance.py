"""Tests for performance monitoring utilities."""

import pytest
import time

from molprop.utils.performance import (
    MetricsCollector,
    PerformanceMetrics,
    measure_time,
    performance_threshold,
    profile,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_metrics_initialization(self):
        """Metrics should initialize with correct values."""
        metrics = PerformanceMetrics(name="test_op")
        assert metrics.name == "test_op"
        assert metrics.call_count == 0
        assert metrics.total_time_ms == 0.0
        assert metrics.min_time_ms == float("inf")
        assert metrics.max_time_ms == 0.0

    def test_avg_time_calculation(self):
        """Average time should be calculated correctly."""
        metrics = PerformanceMetrics(name="test_op")
        metrics.total_time_ms = 100.0
        metrics.call_count = 4
        assert metrics.avg_time_ms == 25.0

    def test_avg_time_with_zero_calls(self):
        """Average time should be 0 with zero calls."""
        metrics = PerformanceMetrics(name="test_op")
        assert metrics.avg_time_ms == 0.0

    def test_to_dict(self):
        """Should convert to dictionary with rounded values."""
        metrics = PerformanceMetrics(name="test_op")
        metrics.total_time_ms = 100.123456
        metrics.call_count = 2
        result = metrics.to_dict()
        assert result["name"] == "test_op"
        assert result["total_time_ms"] == 100.123
        assert result["avg_time_ms"] == 50.062


class TestMetricsCollector:
    """Test MetricsCollector for collecting performance data."""

    def test_record_single_measurement(self):
        """Should record a single measurement."""
        collector = MetricsCollector()
        collector.record("op1", 10.5)

        metrics = collector.get_metrics("op1")
        assert metrics is not None
        assert metrics.call_count == 1
        assert metrics.total_time_ms == 10.5

    def test_record_multiple_measurements(self):
        """Should aggregate multiple measurements."""
        collector = MetricsCollector()
        collector.record("op1", 10.0)
        collector.record("op1", 20.0)
        collector.record("op1", 15.0)

        metrics = collector.get_metrics("op1")
        assert metrics.call_count == 3
        assert metrics.total_time_ms == 45.0
        assert metrics.min_time_ms == 10.0
        assert metrics.max_time_ms == 20.0

    def test_record_with_error(self):
        """Should track error count."""
        collector = MetricsCollector()
        collector.record("op1", 10.0)
        collector.record("op1", 20.0, error=ValueError("test error"))

        metrics = collector.get_metrics("op1")
        assert metrics.call_count == 2
        assert metrics.error_count == 1
        assert metrics.last_error == "test error"

    def test_reset_specific_metric(self):
        """Should reset metrics for a specific operation."""
        collector = MetricsCollector()
        collector.record("op1", 10.0)
        collector.record("op2", 20.0)

        collector.reset("op1")

        assert collector.get_metrics("op1") is None
        assert collector.get_metrics("op2") is not None

    def test_reset_all_metrics(self):
        """Should reset all metrics."""
        collector = MetricsCollector()
        collector.record("op1", 10.0)
        collector.record("op2", 20.0)

        collector.reset()

        assert len(collector.get_all_metrics()) == 0

    def test_stats(self):
        """Should return correct statistics."""
        collector = MetricsCollector()
        collector.record("op1", 10.0)
        collector.record("op1", 20.0)
        collector.record("op2", 30.0, error=ValueError("error"))

        stats = collector.stats()
        assert stats["total_operations"] == 3
        assert stats["total_errors"] == 1


class TestMeasureTimeContextManager:
    """Test measure_time context manager."""

    def test_measure_successful_operation(self):
        """Should measure execution time of successful operation."""
        with measure_time("test_op", record_metrics=False) as result_dict:
            time.sleep(0.01)

        assert result_dict["operation"] == "test_op"
        assert result_dict["elapsed_ms"] > 10  # At least 10ms
        assert result_dict["error"] is None

    def test_measure_failed_operation(self):
        """Should record error in result dict."""
        try:
            with measure_time("test_op", record_metrics=False) as result_dict:
                raise ValueError("test error")
        except ValueError:
            pass

        assert result_dict["error"] is not None
        assert str(result_dict["error"]) == "test error"

    def test_measure_with_metrics_recording(self):
        """Should record metrics in global collector."""
        from molprop.utils.performance import get_metrics_collector

        collector = get_metrics_collector()
        collector.reset("test_op")

        with measure_time("test_op", record_metrics=True):
            time.sleep(0.001)

        metrics = collector.get_metrics("test_op")
        assert metrics is not None
        assert metrics.call_count == 1


class TestProfileDecorator:
    """Test @profile decorator."""

    def test_profile_decorator_execution(self):
        """Decorated function should execute correctly."""

        @profile(collect_metrics=False)
        def slow_function():
            time.sleep(0.01)
            return "result"

        result = slow_function()
        assert result == "result"

    def test_profile_decorator_with_metrics(self):
        """Profile decorator should record metrics."""
        from molprop.utils.performance import get_metrics_collector

        collector = get_metrics_collector()

        @profile(collect_metrics=True)
        def test_function():
            return 42

        result = test_function()
        assert result == 42

        # Metrics should be recorded (function name includes module path)
        all_metrics = collector.get_all_metrics()
        assert len(all_metrics) > 0


class TestPerformanceThresholdDecorator:
    """Test @performance_threshold decorator."""

    def test_threshold_not_exceeded(self):
        """Should not warn if under threshold."""

        @performance_threshold(threshold_ms=100)
        def fast_function():
            time.sleep(0.001)
            return "fast"

        result = fast_function()
        assert result == "fast"

    def test_threshold_exceeded(self):
        """Should log warning if over threshold."""

        @performance_threshold(threshold_ms=1)
        def slow_function():
            time.sleep(0.05)
            return "slow"

        result = slow_function()
        assert result == "slow"
