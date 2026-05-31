"""
Performance monitoring and metrics collection for molprop.

Provides decorators, context managers, and utilities for collecting
performance metrics, profiling, and bottleneck detection.
"""

from __future__ import annotations

import functools
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

log = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance measurement data."""

    name: str
    total_time_ms: float = 0.0
    call_count: int = 0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    error_count: int = 0
    last_call_time_ms: float = 0.0
    last_error: Optional[str] = None

    @property
    def avg_time_ms(self) -> float:
        """Average execution time in milliseconds."""
        return self.total_time_ms / self.call_count if self.call_count > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "call_count": self.call_count,
            "total_time_ms": round(self.total_time_ms, 3),
            "avg_time_ms": round(self.avg_time_ms, 3),
            "min_time_ms": round(self.min_time_ms, 3),
            "max_time_ms": round(self.max_time_ms, 3),
            "error_count": self.error_count,
            "last_call_time_ms": round(self.last_call_time_ms, 3),
        }


class MetricsCollector:
    """Centralized metrics collection for performance monitoring."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._metrics: dict[str, PerformanceMetrics] = defaultdict(
            lambda: PerformanceMetrics(name="unknown")
        )
        self._lock = False

    def record(
        self,
        name: str,
        elapsed_ms: float,
        error: Optional[Exception] = None,
    ) -> None:
        """Record a performance measurement."""
        if name not in self._metrics:
            self._metrics[name] = PerformanceMetrics(name=name)

        metrics = self._metrics[name]
        metrics.total_time_ms += elapsed_ms
        metrics.call_count += 1
        metrics.last_call_time_ms = elapsed_ms
        metrics.min_time_ms = min(metrics.min_time_ms, elapsed_ms)
        metrics.max_time_ms = max(metrics.max_time_ms, elapsed_ms)

        if error:
            metrics.error_count += 1
            metrics.last_error = str(error)

    def get_metrics(self, name: str) -> Optional[PerformanceMetrics]:
        """Get metrics for a specific operation."""
        return self._metrics.get(name)

    def get_all_metrics(self) -> dict[str, PerformanceMetrics]:
        """Get all collected metrics."""
        return dict(self._metrics)

    def reset(self, name: Optional[str] = None) -> None:
        """Reset metrics for a specific operation or all."""
        if name:
            if name in self._metrics:
                del self._metrics[name]
        else:
            self._metrics.clear()

    def stats(self) -> dict[str, Any]:
        """Get summary statistics."""
        if not self._metrics:
            return {"total_operations": 0, "metrics": {}}

        return {
            "total_operations": sum(m.call_count for m in self._metrics.values()),
            "total_errors": sum(m.error_count for m in self._metrics.values()),
            "total_time_ms": sum(m.total_time_ms for m in self._metrics.values()),
            "metrics": {name: m.to_dict() for name, m in self._metrics.items()},
        }


# Global metrics collector
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics_collector


@contextmanager
def measure_time(
    operation_name: str,
    record_metrics: bool = True,
    log_result: bool = True,
) -> Generator[dict[str, Any], None, None]:
    """
    Context manager to measure operation execution time.

    Args:
        operation_name: Name of the operation being measured.
        record_metrics: Whether to record in global metrics collector.
        log_result: Whether to log the result.

    Example:
        with measure_time("database_query"):
            result = db.query(...)
    """
    start_time = time.perf_counter()
    result_dict: dict[str, Any] = {"operation": operation_name, "error": None}

    try:
        yield result_dict
    except Exception as e:
        result_dict["error"] = e
        raise
    finally:
        elapsed = time.perf_counter() - start_time
        elapsed_ms = elapsed * 1000

        if record_metrics:
            _metrics_collector.record(
                operation_name,
                elapsed_ms,
                error=result_dict["error"],
            )

        result_dict["elapsed_ms"] = elapsed_ms
        result_dict["timestamp"] = time.time()

        if log_result:
            if result_dict["error"]:
                log.error(
                    f"{operation_name} failed after {elapsed_ms:.1f}ms: {result_dict['error']}",
                    exc_info=False,
                )
            else:
                log.debug(f"{operation_name} completed in {elapsed_ms:.1f}ms")


def profile(
    func: F,
    collect_metrics: bool = True,
) -> F:
    """
    Decorator to profile function execution time.

    Example:
        @profile
        def expensive_operation():
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with measure_time(f"{func.__module__}.{func.__name__}", record_metrics=collect_metrics):
            return func(*args, **kwargs)

    return wrapper  # type: ignore


def performance_threshold(threshold_ms: float = 1000.0) -> Callable[[F], F]:
    """
    Decorator to warn if function exceeds execution time threshold.

    Args:
        threshold_ms: Threshold in milliseconds.

    Example:
        @performance_threshold(threshold_ms=500)
        def should_be_fast():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                if elapsed_ms > threshold_ms:
                    log.warning(
                        f"{func.__name__} exceeded threshold: {elapsed_ms:.1f}ms > {threshold_ms:.1f}ms"
                    )

        return wrapper  # type: ignore

    return decorator


def cache_stats(func: F) -> F:
    """
    Decorator to collect cache hit/miss statistics.

    Example:
        @cache_stats
        @functools.lru_cache(maxsize=128)
        def cached_operation():
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # This decorator is typically used before @lru_cache
        # It will collect stats if the function has cache_info
        result = func(*args, **kwargs)

        if hasattr(func, "cache_info"):
            info = func.cache_info()  # type: ignore
            if info.hits + info.misses > 0:
                hit_rate = info.hits / (info.hits + info.misses)
                if hit_rate < 0.5:
                    log.debug(
                        f"{func.__name__} cache hit rate: {hit_rate:.1%} "
                        f"(hits={info.hits}, misses={info.misses})"
                    )

        return result

    return wrapper  # type: ignore
