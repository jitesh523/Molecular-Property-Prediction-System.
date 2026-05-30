"""
Decorators for common patterns in molprop.

Includes decorators for timing, error handling, and validation.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional, TypeVar, cast

from molprop.exceptions import MolpropError

F = TypeVar("F", bound=Callable[..., Any])

log = logging.getLogger(__name__)


def timing(func: F) -> F:
    """
    Decorator to log execution time of a function.

    Example:
        @timing
        def slow_operation():
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            log.debug(f"{func.__name__} took {elapsed:.3f}s")

    return cast(F, wrapper)


def retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator to retry a function on failure.

    Args:
        max_attempts: Maximum number of attempts.
        delay_seconds: Initial delay between retries.
        backoff_factor: Multiplier for delay after each retry.
        exceptions: Tuple of exception types to catch and retry on.

    Example:
        @retry(max_attempts=3, delay_seconds=0.5)
        def flaky_operation():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            delay = delay_seconds

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        log.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_attempts}), "
                            f"retrying in {delay:.1f}s: {str(e)}"
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        log.error(f"{func.__name__} failed after {max_attempts} attempts")

            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected failure in retry decorator for {func.__name__}")

        return cast(F, wrapper)

    return decorator


def handle_errors(
    default_return: Any = None,
    log_traceback: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to catch and log exceptions.

    Args:
        default_return: Value to return if exception occurs.
        log_traceback: Whether to log full traceback.

    Example:
        @handle_errors(default_return=[])
        def parse_data():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except MolpropError as e:
                log.error(f"{func.__name__} raised MolpropError: {e.message}", exc_info=log_traceback)
                return default_return
            except Exception as e:
                log.error(f"{func.__name__} raised {type(e).__name__}: {str(e)}", exc_info=log_traceback)
                return default_return

        return cast(F, wrapper)

    return decorator


def validate_input(**validators: Callable[[Any], Any]) -> Callable[[F], F]:
    """
    Decorator to validate function arguments.

    Args:
        **validators: Mapping of argument name to validator function.

    Example:
        @validate_input(
            smiles=validate_smiles,
            batch_size=lambda x: validate_batch_size(x)
        )
        def predict(smiles: str, batch_size: int):
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Validate keyword arguments
            for arg_name, validator in validators.items():
                if arg_name in kwargs:
                    try:
                        kwargs[arg_name] = validator(kwargs[arg_name])
                    except Exception as e:
                        log.error(f"Validation failed for {arg_name}: {str(e)}")
                        raise

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
