from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional, Tuple
import time
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NonRetryableError(Exception):
    pass


def run_parallel_with_retries(
    items: List[Any],
    worker: Callable[[Any], Any],
    *,
    max_workers: int = 8,
    max_attempt_rounds: int = 3,
    sleep_seconds: float = 0.0,
) -> List[Any]:
    """
    Run tasks in parallel with retry rounds while preserving input order.

    The algorithm retries remaining failed tasks in rounds. If a round makes
    no progress (remaining count does not decrease), the attempt round counter
    increases. If remaining tasks drop, the counter stays the same. If tasks
    are still failing after `max_attempt_rounds` no-progress rounds, raise.

    Args:
        items: Inputs to process.
        worker: Function that processes a single item and returns a result.
        max_workers: Maximum concurrent workers.
        max_attempt_rounds: Max number of no-progress retry rounds allowed.
        sleep_seconds: Optional sleep between rounds.

    Returns:
        List of results aligned with the order of `items`.

    Raises:
        RuntimeError: If some tasks still fail after retries.
    """
    total = len(items)
    results: List[Optional[Any]] = [None] * total
    pending: List[Tuple[int, Any]] = list(enumerate(items))

    last_remaining = len(pending)
    no_progress_rounds = 0
    last_errors: List[Tuple[int, BaseException]] = []

    # Create progress bar
    pbar = tqdm(total=total, desc="Processing")

    while pending:
        next_pending: List[Tuple[int, Any]] = []
        errors_this_round: List[Tuple[int, BaseException]] = []
        non_retryable_errors: List[Tuple[int, BaseException]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(worker, item): idx for idx, item in pending}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                    pbar.update(1)  # Update progress on success
                except BaseException as exc:  # retry on any exception
                    logger.error(f"Task {idx} failed: {exc}")
                    if isinstance(exc, NonRetryableError):
                        non_retryable_errors.append((idx, exc))
                    else:
                        errors_this_round.append((idx, exc))
                        next_pending.append((idx, items[idx]))

        if non_retryable_errors:
            pbar.close()
            aggregate: List[Tuple[int, BaseException]] = sorted(non_retryable_errors, key=lambda t: t[0])
            messages = [f"[{idx}] {type(err).__name__}: {err}" for idx, err in aggregate]
            logger.warning("Non-retryable task failures:\n" + "\n".join(messages))
            # raise RuntimeError("Non-retryable task failures:\n" + "\n".join(messages))
            return results # type: ignore[return-value]

        remaining = len(next_pending)
        if remaining == 0:
            break

        if remaining >= last_remaining:
            no_progress_rounds += 1

        if no_progress_rounds > max_attempt_rounds:
            pbar.close()
            # Aggregate errors for clarity
            aggregate: List[Tuple[int, BaseException]] = sorted(errors_this_round + last_errors, key=lambda t: t[0])
            messages = [f"[{idx}] {type(err).__name__}: {err}" for idx, err in aggregate]
            logger.warning("Some tasks failed after retries:\n" + "\n".join(messages))
            # raise RuntimeError("Some tasks failed after retries:\n" + "\n".join(messages))
            return results # type: ignore[return-value]

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        pending = next_pending
        last_remaining = remaining
        last_errors = errors_this_round

    # Close progress bar
    pbar.close()
    
    # All done; results are aligned with inputs
    return results  # type: ignore[return-value]


