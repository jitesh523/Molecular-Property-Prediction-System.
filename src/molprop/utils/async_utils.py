"""
Async and concurrency utilities for molprop.

Provides helpers for async operations, thread pooling, and
concurrent processing patterns.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, AsyncIterator, Callable, Coroutine, Generic, Optional, TypeVar

T = TypeVar("T")
P = TypeVar("P")

log = logging.getLogger(__name__)


class AsyncBatchProcessor(Generic[T, P]):
    """Process items asynchronously in batches."""

    def __init__(
        self,
        batch_size: int = 32,
        max_concurrent: int = 4,
    ) -> None:
        """
        Initialize async batch processor.

        Args:
            batch_size: Number of items per batch.
            max_concurrent: Maximum concurrent operations.
        """
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent

    async def process(
        self,
        items: list[T],
        processor: Callable[[list[T]], Coroutine[Any, Any, list[P]]],
    ) -> list[P]:
        """
        Process items in batches asynchronously.

        Args:
            items: List of items to process.
            processor: Async function to process a batch.

        Returns:
            List of processed results in order.
        """
        results: list[P] = []
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_batch(batch: list[T]) -> list[P]:
            async with semaphore:
                return await processor(batch)

        # Create batches
        batches = [
            items[i : i + self.batch_size] for i in range(0, len(items), self.batch_size)
        ]

        # Process batches
        batch_results = await asyncio.gather(*[process_batch(b) for b in batches])

        # Flatten results
        for batch_result in batch_results:
            results.extend(batch_result)

        return results


class ThreadPoolBatchProcessor(Generic[T, P]):
    """Process items concurrently using thread pool."""

    def __init__(
        self,
        batch_size: int = 32,
        max_workers: int = 4,
    ) -> None:
        """
        Initialize thread pool batch processor.

        Args:
            batch_size: Number of items per batch.
            max_workers: Maximum worker threads.
        """
        self.batch_size = batch_size
        self.max_workers = max_workers

    def process(
        self,
        items: list[T],
        processor: Callable[[T], P],
    ) -> list[P]:
        """
        Process items concurrently using thread pool.

        Args:
            items: List of items to process.
            processor: Function to process a single item.

        Returns:
            List of processed results in order.
        """
        results: list[Optional[P]] = [None] * len(items)
        item_to_index = {id(item): i for i, item in enumerate(items)}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(processor, item): i for i, item in enumerate(items)
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    idx = futures[future]
                    results[idx] = result
                except Exception as e:
                    idx = futures[future]
                    log.error(f"Error processing item {idx}: {str(e)}")
                    results[idx] = None

        return [r for r in results if r is not None]  # type: ignore


async def gather_with_limit(
    *coroutines: Coroutine[Any, Any, T],
    limit: int = 10,
) -> list[T]:
    """
    Run coroutines concurrently with a concurrency limit.

    Args:
        coroutines: Coroutines to run.
        limit: Maximum concurrent coroutines.

    Returns:
        List of results in order.
    """
    semaphore = asyncio.Semaphore(limit)

    async def bounded_coro(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*[bounded_coro(c) for c in coroutines])


async def async_retry(
    coro_func: Callable[[], Coroutine[Any, Any, T]],
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """
    Retry an async operation with exponential backoff.

    Args:
        coro_func: Callable that returns the coroutine to execute.
        max_attempts: Maximum retry attempts.
        delay_seconds: Initial delay between retries.
        backoff_factor: Multiplier for delay after each retry.
        exceptions: Tuple of exceptions to catch and retry on.

    Returns:
        Result from successful execution.

    Raises:
        The last exception if all retries fail.
    """
    last_exception = None
    delay = delay_seconds

    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts:
                log.warning(
                    f"Async operation failed (attempt {attempt}/{max_attempts}), "
                    f"retrying in {delay:.1f}s: {str(e)}"
                )
                await asyncio.sleep(delay)
                delay *= backoff_factor
            else:
                log.error(f"Async operation failed after {max_attempts} attempts")

    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected failure in async_retry")


def sync_to_async(func: Callable[..., T], executor: Optional[ThreadPoolExecutor] = None) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Convert a synchronous function to async.

    Args:
        func: Synchronous function to wrap.
        executor: Optional executor for the blocking function.

    Returns:
        Async wrapper function.
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, functools.partial(func, *args, **kwargs))

    return wrapper


class AsyncGenerator(Generic[T]):
    """Async generator for streamed processing."""

    def __init__(
        self,
        items: list[T],
        batch_size: int = 32,
    ) -> None:
        """
        Initialize async generator.

        Args:
            items: Items to iterate over.
            batch_size: Batch size for yielding.
        """
        self.items = items
        self.batch_size = batch_size

    async def __aiter__(self) -> AsyncIterator[list[T]]:
        """Iterate over batches asynchronously."""
        for i in range(0, len(self.items), self.batch_size):
            batch = self.items[i : i + self.batch_size]
            # Simulate async work
            await asyncio.sleep(0)
            yield batch
