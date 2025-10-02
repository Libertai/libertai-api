import asyncio
import re
from http import HTTPStatus

import aiohttp

from src.config import config


class ServerMetrics:
    """Represents metrics for a server."""

    def __init__(self, requests_processing: int = 0, requests_deferred: int = 0, is_healthy: bool = True):
        self.requests_processing = requests_processing
        self.requests_deferred = requests_deferred
        self.is_healthy = is_healthy

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

        # Map of model name to list of healthy URL strings
        self.healthy_model_urls: dict[str, list[str]] = dict(config.MODELS.items())

        # Map of URL to metrics
        self.server_metrics: dict[str, ServerMetrics] = {}

    def get_healthy_model_urls(self) -> dict[str, list[str]]:
        """Get a dictionary of healthy servers grouped by model."""
        return self.healthy_model_urls

    def get_server_metrics(self, url: str) -> ServerMetrics:
        """Get metrics for a specific server URL."""
        return self.server_metrics.get(url, ServerMetrics(is_healthy=False))

    def get_least_busy_server(self, model_name: str, preferred_server: str | None = None) -> str | None:
        """
        Get the least busy healthy server for a model.

        Args:
            model_name: The model to find a server for
            preferred_server: Optional preferred server URL (gets priority if healthy)

        Returns:
            URL of the least busy server or None if no healthy servers
        """
        if model_name not in self.healthy_model_urls:
            return None

        healthy_urls = self.healthy_model_urls[model_name]
        if not healthy_urls:
            return None

        # If preferred server is healthy use it
        if preferred_server is not None and preferred_server in healthy_urls:
            return preferred_server

        # Find the least busy server
        best_server = None
        best_load = float("inf")

        for url in healthy_urls:
            metrics = self.get_server_metrics(url)
            if metrics.load_score < best_load:
                best_load = metrics.load_score
                best_server = url

        return best_server

    @staticmethod
    def parse_metrics(metrics_text: str) -> dict[str, float]:
        """
        Parse Prometheus-style metrics text.

        Args:
            metrics_text: Raw metrics text from /metrics endpoint

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        for line in metrics_text.strip().split("\n"):
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            # Match lines like "llamacpp:requests_processing 0"
            match = re.match(r"^(\S+)\s+([\d.]+|nan)$", line)
            if match:
                metric_name = match.group(1)
                value_str = match.group(2)

                try:
                    if value_str == "nan":
                        value = 0.0
                    else:
                        value = float(value_str)
                    metrics[metric_name] = value
                except ValueError:
                    continue

        return metrics

    async def check_server_metrics_async(self, url: str) -> ServerMetrics:
        """
        Asynchronously check server health via /metrics endpoint.

        Args:
            url: The server URL to check

        Returns:
            ServerMetrics object with health status and load information
        """
        try:
            metrics_url = f"{url}/metrics"
            async with aiohttp.ClientSession() as session:
                async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == HTTPStatus.OK:
                        metrics_text = await response.text()
                        metrics = self.parse_metrics(metrics_text)

                        # Extract the specific metrics we need
                        requests_processing = int(metrics.get("llamacpp:requests_processing", 0))
                        requests_deferred = int(metrics.get("llamacpp:requests_deferred", 0))

                        return ServerMetrics(
                            requests_processing=requests_processing,
                            requests_deferred=requests_deferred,
                            is_healthy=True,
                        )
                    else:
                        return ServerMetrics(is_healthy=False)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return ServerMetrics(is_healthy=False)

    async def check_all_servers(self) -> None:
        """Check health of all registered servers and update healthy URLs per model with metrics."""
        new_healthy_model_urls: dict[str, list[str]] = {model: [] for model in self.model_urls}
        new_server_metrics: dict[str, ServerMetrics] = {}

        # Check each model's servers
        for model, urls in self.model_urls.items():
            tasks = [self.check_server_metrics_async(url) for url in urls]

            if tasks:
                results = await asyncio.gather(*tasks)

                # Update healthy servers and metrics
                for i, url in enumerate(urls):
                    if i < len(results):
                        metrics = results[i]
                        new_server_metrics[url] = metrics
                        if metrics.is_healthy:
                            new_healthy_model_urls[model].append(url)

        self.healthy_model_urls = new_healthy_model_urls
        self.server_metrics = new_server_metrics


server_health_monitor = ServerHealthMonitor()
