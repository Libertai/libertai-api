# Cadence of the background job loop: keys, server health, pricing and model metadata.
# The /openrouter/models Retry-After hint intentionally shares this value: it is the
# worst-case wait until the next model-metadata load attempt.
JOB_INTERVAL_SECONDS = 30
