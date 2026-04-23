import asyncio
import json
from http import HTTPStatus

import httpx

from src.config import config
from src.logger import setup_logger
from src.redis_client import get_redis, k

logger = setup_logger(__name__)

REDIS_KEY = k("health", "snapshot")


class ServerMetrics:
    """Represents metrics for a server."""

    def __init__(
        self,
        requests_processing: int = 0,
        requests_deferred: int = 0,
        is_healthy: bool = True,
        is_loaded: bool = False,
    ):
        self.requests_processing = requests_processing
        self.requests_deferred = requests_deferred
        self.is_healthy = is_healthy
        self.is_loaded = is_loaded

    @property
    def load_score(self) -> int:
        """Calculate load score for load balancing. Lower is better."""
        return self.requests_processing + self.requests_deferred


class ServerHealthMonitor:
    def __init__(self) -> None:
        """
        Initialize the health monitor for LLM servers.
        """
        # Map of model name to list of URL strings
        self.model_urls: dict[str, list[str]] = dict(config.MODELS.items())

        # Map of model name to list of healthy URL strings (model loaded and ready)
        self.healthy_model_urls: dict[str, list[str]] = dict(config.MODELS.items())

        # Map of model name to list of capable URL strings (server up, model not loaded)
        self.capable_model_urls: dict[str, list[str]] = {}

        # Map of URL to metrics
        self.server_metrics: dict[str, ServerMetrics] = {}

    def get_healthy_model_urls(self) -> dict[str, list[str]]:
        """Get a dictionary of healthy servers grouped by model."""
        return self.healthy_model_urls

    def get_server_metrics(self, url: str) -> ServerMetrics:
        """Get metrics for a specific server URL."""
        return self.server_metrics.get(url, ServerMetrics(is_healthy=False, is_loaded=False))

    def get_least_busy_server(self, model_name: str, preferred_server: str | None = None) -> str | None:
        """
        Get the least busy server for a model. Prefers loaded, falls back to capable.

        Args:
            model_name: The model to find a server for
            preferred_server: Optional preferred server URL (gets priority if available)

        Returns:
            URL of the least busy server or None if no servers available
        """
        healthy_urls = self.healthy_model_urls.get(model_name, [])
        capable_urls = self.capable_model_urls.get(model_name, [])
        urls = healthy_urls if healthy_urls else capable_urls

        if not urls:
            return None

        # If preferred server is available use it
        if preferred_server is not None and preferred_server in urls:
            return preferred_server

        # Find the least busy server
        best_server = None
        best_load = float("inf")

        for url in urls:
            metrics = self.get_server_metrics(url)
            if metrics.load_score < best_load:
                best_load = metrics.load_score
                best_server = url

        return best_server

    async def check_server_metrics_async(self, url: str, model: str) -> ServerMetrics:
        """
        Asynchronously check server health via /health endpoint.

        Args:
            url: The server URL to check
            model: The model name

        Returns:
            ServerMetrics object with health status and load information
        """
        try:
            health_url = f"{url}/health/{model}"
            if model == "hermes-3-8b-tee":
                # Hardcoded healthcheck for Hermes which is in an isolated TEE with an old version
                return ServerMetrics(is_healthy=True, is_loaded=True)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(health_url)
                if response.status_code == HTTPStatus.OK:
                    return ServerMetrics(is_healthy=True, is_loaded=True)
                elif response.status_code == HTTPStatus.ACCEPTED:
                    return ServerMetrics(is_healthy=True, is_loaded=False)
                else:
                    logger.warning(f"Health status error for {url}: {response.status_code}")
                    return ServerMetrics(is_healthy=False, is_loaded=False)
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(f"Health check error for {url}: {type(e).__name__}: {e or 'No error message'}")
            return ServerMetrics(is_healthy=False, is_loaded=False)

    async def check_all_servers(self) -> None:
        """Check health of all registered servers and update healthy/capable URLs per model."""
        new_healthy_model_urls: dict[str, list[str]] = {model: [] for model in self.model_urls}
        new_capable_model_urls: dict[str, list[str]] = {model: [] for model in self.model_urls}
        new_server_metrics: dict[str, ServerMetrics] = {}

        for model, urls in self.model_urls.items():
            tasks = [self.check_server_metrics_async(url, model) for url in urls]

            if tasks:
                results = await asyncio.gather(*tasks)

                for i, url in enumerate(urls):
                    if i < len(results):
                        metrics = results[i]
                        new_server_metrics[url] = metrics
                        if metrics.is_loaded:
                            new_healthy_model_urls[model].append(url)
                        elif metrics.is_healthy:
                            new_capable_model_urls[model].append(url)

        self.healthy_model_urls = new_healthy_model_urls
        self.capable_model_urls = new_capable_model_urls
        self.server_metrics = new_server_metrics

        try:
            snapshot = {
                "healthy_model_urls": new_healthy_model_urls,
                "capable_model_urls": new_capable_model_urls,
                "server_metrics": {
                    url: {
                        "requests_processing": m.requests_processing,
                        "requests_deferred": m.requests_deferred,
                        "is_healthy": m.is_healthy,
                        "is_loaded": m.is_loaded,
                    }
                    for url, m in new_server_metrics.items()
                },
            }
            await get_redis().set(REDIS_KEY, json.dumps(snapshot))
        except Exception as e:
            logger.error(f"Failed to publish health snapshot to Redis: {e}", exc_info=True)

    async def sync_from_redis(self) -> None:
        """All replicas: refresh local snapshot from Redis."""
        try:
            raw = await get_redis().get(REDIS_KEY)
            if not raw:
                return
            snap = json.loads(raw)
            self.healthy_model_urls = {m: list(urls) for m, urls in snap.get("healthy_model_urls", {}).items()}
            self.capable_model_urls = {m: list(urls) for m, urls in snap.get("capable_model_urls", {}).items()}
            self.server_metrics = {
                url: ServerMetrics(
                    requests_processing=m.get("requests_processing", 0),
                    requests_deferred=m.get("requests_deferred", 0),
                    is_healthy=m.get("is_healthy", False),
                    is_loaded=m.get("is_loaded", False),
                )
                for url, m in snap.get("server_metrics", {}).items()
            }
        except Exception as e:
            logger.error(f"Failed to sync health snapshot from Redis: {e}", exc_info=True)


server_health_monitor = ServerHealthMonitor()
