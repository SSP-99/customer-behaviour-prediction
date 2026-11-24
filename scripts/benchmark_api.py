"""Latency benchmark script sending 500 single-customer prediction requests to FastAPI."""

import time

import numpy as np
from fastapi.testclient import TestClient

from api.main import app


def run_benchmark(n_requests: int = 500) -> None:
    """Measure p50, p95, and p99 latency against the < 400 ms SLA target."""
    latencies_ms: list[float] = []
    payload = {"recency_days": 45.0, "frequency": 6.0, "monetary_gross": 1450.0}
    with TestClient(app) as client:
        client.post("/predict", json=payload)  # warm-up
        for _ in range(n_requests):
            t0 = time.perf_counter()
            resp = client.post("/predict", json=payload)
            assert resp.status_code == 200
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)
    print(
        f"Benchmark ({n_requests} requests) | "
        f"p50={np.percentile(arr, 50):.2f}ms | "
        f"p95={np.percentile(arr, 95):.2f}ms | "
        f"p99={np.percentile(arr, 99):.2f}ms (SLA < 400ms)"
    )


if __name__ == "__main__":
    run_benchmark()
