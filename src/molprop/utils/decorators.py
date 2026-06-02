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
    """Decorator to log execution time of a function.
    
    Measures wall-clock time from function start to completion (including exceptions)
    and logs elapsed time at debug level. Useful for profiling performance-critical
    code paths.

    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with timing logged

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
    """Decorator to retry a function on failure with exponential backoff.
    
    Retries the decorated function up to max_attempts times if it raises an
    exception matching the specified exception types. Between retries, waits
    for delay_seconds * (backoff_factor ^ retry_count) seconds.
    
    All exceptions are caught and re-raised after final attempt; the last
    exception encountered is the one raised.

    Args:
        max_attempts: Maximum number of attempts (default 3).
        delay_seconds: Initial delay between retries in seconds (default 1.0).
        backoff_factor: Multiplier for delay after each retry (default 2.0).
        exceptions: Tuple of exception types to catch and retry on (default Exception).
        
    Returns:
        Decorator function

    Example:
        @retry(max_attempts=3, delay_seconds=0.5, exceptions=(ConnectionError, TimeoutError))
        def flaky_api_call():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
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

            # Always have a last_exception here (at least from final attempt)
            if last_exception:
                raise last_exception
            # This should never happen, but if it does, it's a logic error
            raise RuntimeError(f"Unexpected state in retry decorator for {func.__name__}")  # pragma: no cover

        return cast(F, wrapper)

    return decorator


def handle_errors(
    default_return: Any = None,
    log_traceback: bool = True,
) -> Callable[[F], F]:
    """Decorator to catch and log exceptions, returning a default value.
    
    Wraps function execution with a try/except that catches both MolpropError
    and generic exceptions, logging them appropriately before returning a
    default value. Useful for functions that should fail gracefully.

    Args:
        default_return: Value to return if any exception occurs.
        log_traceback: Whether to include full traceback in log output (default True).
        
    Returns:
        Decorator function

    Example:
        @handle_errors(default_return=[], log_traceback=True)
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
    """Decorator to validate function keyword arguments before execution.
    
    Applies validator functions to specified keyword arguments, raising an
    exception if any validator fails. Useful for enforcing constraints on
    function inputs (e.g., SMILES format, numeric bounds).
    
    Note: Currently only validates keyword arguments; positional arguments
    are not validated.

    Args:
        **validators: Mapping of argument name to validator function.
                     Validator should accept one argument and return validated value
                     or raise an exception.
        
    Returns:
        Decorator function

    Example:
        @validate_input(
            smiles=validate_smiles,
            batch_size=lambda x: x if x > 0 else error('batch_size must be positive')
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
