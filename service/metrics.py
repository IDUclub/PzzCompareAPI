from __future__ import annotations

from prometheus_client import Counter, Histogram

queue_wait_seconds = Histogram(
    "queue_wait_seconds",
    "Time spent in queue before worker starts task.",
)

task_run_seconds = Histogram(
    "task_run_seconds",
    "Task execution duration in worker.",
)

task_fail_total = Counter(
    "task_fail_total",
    "Total failed task executions.",
)

task_retry_total = Counter(
    "task_retry_total",
    "Total task retries due to capacity limits.",
)
