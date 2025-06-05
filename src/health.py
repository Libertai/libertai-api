import asyncio
from http import HTTPStatus

import aiohttp

from src.config import config


class ServerHealthMonitor:
    def __init__(self) -> None:
        """
        Initialize the health monitor for LLM servers.
        """
        # Map of model name to list of URL strings
        self.model_urls: dict[str, list[str]] = {
            model: [server.url for server in servers] for model, servers in config.MODELS.items()
        }

        # Map of model name to list of healthy URL strings
        self.healthy_model_urls: dict[str, list[str]] = {
            model: [server.url for server in servers] for model, servers in config.MODELS.items()
        }

    def get_healthy_model_urls(self) -> dict[str, list[str]]:
        """Get a dictionary of healthy servers grouped by model."""
        return self.healthy_model_urls

    @staticmethod
    async def check_server_health_async(url: str) -> bool:
        """
        Asynchronously check if a server is responding.

        Args:
            url: The server URL to check

        Returns:
            True if server is responding, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == HTTPStatus.METHOD_NOT_ALLOWED:
                        return True  # Method not allowed but server is up
                    return False
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def check_all_servers(self) -> None:
        """Check health of all registered servers and update healthy URLs per model."""
        new_healthy_model_urls: dict[str, list[str]] = {model: [] for model in self.model_urls}

        # Check each model's servers
        for model, urls in self.model_urls.items():
            tasks = [self.check_server_health_async(url) for url in urls]

            if tasks:
                results = await asyncio.gather(*tasks)

                # Update healthy servers for this model
                new_healthy_model_urls[model] = [url for i, url in enumerate(urls) if i < len(results) and results[i]]
        self.healthy_model_urls = new_healthy_model_urls


server_health_monitor = ServerHealthMonitor()
