import asyncio
from typing import List, Set

import aiohttp

from src.config import config


class ServerHealthMonitor:
    def __init__(self) -> None:
        """
        Initialize the health monitor for LLM servers.
        """
        self.server_urls: List[str] = [server.url for servers in config.MODELS.values() for server in servers]
        self.healthy_servers: Set[str] = set()
        self.timeout = 1  # Default timeout for health checks

    def get_healthy_servers(self) -> Set[str]:
        """Get the set of currently healthy servers."""
        return self.healthy_servers

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
                    return response.status < 500  # server responded, not a server error
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def check_all_servers(self) -> None:
        """Check health of all registered servers."""
        tasks = []

        for url in self.server_urls:
            # Only check if enough time has passed since last check
            tasks.append(self.check_server_health_async(url))

        if tasks:
            results = await asyncio.gather(*tasks)

            # Update healthy servers set with URLs that are healthy
            self.healthy_servers = {url for i, url in enumerate(self.server_urls) if i < len(results) and results[i]}


server_health_monitor = ServerHealthMonitor()
