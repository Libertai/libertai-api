# Cadence of the background job loop: keys, server health, pricing and model metadata.
# The /openrouter/models Retry-After hint intentionally shares this value: it is the
# worst-case wait until the next model-metadata load attempt.
JOB_INTERVAL_SECONDS = 30

# Free-tier admission gate (src/proxy.py) pacing: how long a free-tier request
# waits for the pool to drain and how often the soft-load wait polls the
# inflight loads. The thresholds themselves are env-configurable in config.py.
FREE_GATE_MAX_WAIT = 5.0
FREE_GATE_POLL_INTERVAL = 0.5
