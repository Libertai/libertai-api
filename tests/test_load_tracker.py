import asyncio
from unittest.mock import patch

from src import load_tracker
from src.load_tracker import _prune_and_count, acquire, get_model_loads, release


def test_prune_and_count_counts_live_drops_expired_and_malformed():
    now = 1000.0
    entries = {
        "live1": f"{now + 100}",
        "live2": f"{now + 5}",
        "expired": f"{now - 1}",  # past deadline — must not count
        "garbage": "not-a-lease",  # malformed — must not count
    }
    live, expired = _prune_and_count(entries, now)
    assert live == 2
    assert set(expired) == {"expired", "garbage"}


class _FakePipe:
    def __init__(self, store):
        self._store = store
        self._ops: list[tuple] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def hgetall(self, key):
        self._ops.append(("hgetall", key))
        return self

    def hset(self, key, field, value):
        self._ops.append(("hset", key, field, value))
        return self

    def expire(self, key, ttl):
        self._ops.append(("expire", key, ttl))
        return self

    def hdel(self, key, *fields):
        self._ops.append(("hdel", key, fields))
        return self

    async def execute(self):
        res = []
        for op in self._ops:
            if op[0] == "hgetall":
                res.append(dict(self._store.get(op[1], {})))
            elif op[0] == "hset":
                self._store.setdefault(op[1], {})[op[2]] = op[3]
                res.append(1)
            elif op[0] == "expire":
                res.append(True)
            elif op[0] == "hdel":
                h = self._store.get(op[1], {})
                for f in op[2]:
                    h.pop(f, None)
                res.append(1)
        self._ops = []
        return res


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, dict[str, str]] = {}

    def pipeline(self, transaction=False):
        return _FakePipe(self.store)

    async def hdel(self, key, field):
        self.store.get(key, {}).pop(field, None)
        return 1


def _run(fake, models, coro_factory, clock=None):
    with (
        patch.object(load_tracker, "get_redis", return_value=fake),
        patch.object(load_tracker.config, "MODELS", models),
    ):
        if clock is not None:
            with patch.object(load_tracker.time, "time", clock):
                return asyncio.run(coro_factory())
        return asyncio.run(coro_factory())


def test_acquire_release_roundtrip():
    fake = _FakeRedis()
    models = {"m": ["A", "B"]}

    async def scenario():
        await acquire("A", "r1")
        await acquire("A", "r2")
        await acquire("B", "r3")
        loads = await get_model_loads("m")
        assert loads == {"A": 2, "B": 1}

        await release("A", "r1")
        loads = await get_model_loads("m")
        assert loads == {"A": 1, "B": 1}

    _run(fake, models, scenario)


def test_unreleased_lease_self_heals_after_ttl():
    """A request that never releases (cancelled / process killed / Redis blip on
    release) must stop counting once its lease expires — the leak fix."""
    fake = _FakeRedis()
    models = {"m": ["A"]}
    t = {"now": 1000.0}

    async def acquire_and_check_present():
        await acquire("A", "leaked")  # deliberately never released
        return await get_model_loads("m")

    present = _run(fake, models, acquire_and_check_present, clock=lambda: t["now"])
    assert present == {"A": 1}

    # Jump past the lease TTL without ever releasing.
    t["now"] = 1000.0 + load_tracker.LEASE_TTL + 1
    healed = _run(fake, models, lambda: get_model_loads("m"), clock=lambda: t["now"])
    assert healed == {"A": 0}
    # And the stale field was pruned from Redis.
    assert fake.store.get(load_tracker._key("A"), {}) == {}


def test_reacquire_keeps_long_stream_counted_past_original_ttl():
    """A stream that re-acquires (heartbeat) before its deadline stays counted
    even when total wall time exceeds LEASE_TTL."""
    fake = _FakeRedis()
    models = {"m": ["A"]}
    t = {"now": 1000.0}

    async def acquire_once():
        await acquire("A", "long")

    _run(fake, models, acquire_once, clock=lambda: t["now"])

    # Refresh just before expiry, then advance past the ORIGINAL deadline.
    t["now"] = 1000.0 + load_tracker.LEASE_TTL - 10
    _run(fake, models, acquire_once, clock=lambda: t["now"])  # re-acquire == heartbeat

    t["now"] = 1000.0 + load_tracker.LEASE_TTL + 10  # past original deadline
    loads = _run(fake, models, lambda: get_model_loads("m"), clock=lambda: t["now"])
    assert loads == {"A": 1}  # still counted thanks to the refresh


def test_get_model_loads_reads_only_the_models_servers():
    """The proxy fetches this per request: it must not HGETALL every configured
    server of every model, and duplicate entries must not double-read."""
    fake = _FakeRedis()
    models = {"m": ["A", "B", "A"], "other": ["C"]}  # duplicate "A" entry

    async def scenario():
        await acquire("A", "r1")
        await acquire("B", "r2")
        await acquire("C", "r3")
        return await get_model_loads("m")

    hgetall_keys: list[str] = []

    with (
        patch.object(load_tracker, "get_redis", return_value=fake),
        patch.object(load_tracker.config, "MODELS", models),
    ):
        base_pipeline = fake.pipeline

        def recording_pipeline(transaction=False):
            pipe = base_pipeline(transaction)
            orig_hgetall = pipe.hgetall

            def hgetall(key):
                hgetall_keys.append(key)
                return orig_hgetall(key)

            pipe.hgetall = hgetall
            return pipe

        fake.pipeline = recording_pipeline
        loads = asyncio.run(scenario())

    # Only the model's servers were read (the duplicate "A" deduped), and the
    # other model's server was never touched.
    assert len(hgetall_keys) == 2
    assert load_tracker._key("C") not in hgetall_keys
    assert loads == {"A": 1, "B": 1}


def test_get_model_loads_deduplicates_duplicate_model_entries():
    fake = _FakeRedis()
    models = {"m": ["A", "A"]}

    async def scenario():
        await acquire("A", "r1")
        return await get_model_loads("m")

    loads = _run(fake, models, scenario)
    assert loads == {"A": 1}
