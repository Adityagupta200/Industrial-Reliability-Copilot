import time
from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["path", "method", "status"],
)

LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["path", "method"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)


class Timer:
    def __init__(self, path: str, method: str):
        self.path = path
        self.method = method
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.start
        LATENCY.labels(self.path, self.method).observe(elapsed)
