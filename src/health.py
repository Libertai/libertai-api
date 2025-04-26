import asyncio
import time
from typing import Dict, List, Set

import aiohttp


class ServerHealthMonitor:
    def __init__(self, check_interval: int = 60):
        """
        Initialize the health monitor for LLM servers.

        Args:
            check_interval: How often to check server health (in seconds)
        """
        self.server_urls: List[str] = [
            "https://hermes-8b-1.tee.api.libertai.io",
            "https://hermes-8b-2.tee.api.libertai.io",
        ]
        self.healthy_servers: Set[str] = set()
        self.last_check: Dict[str, float] = {}
        self.check_interval = check_interval
        self.timeout = 1  # Default timeout for health checks

    def get_healthy_servers(self) -> Set[str]:
        """Get the set of currently healthy servers."""
        return self.healthy_servers

    def is_server_healthy(self, url: str) -> bool:
        """Check if a specific server is healthy."""
        return url in self.healthy_servers

    async def check_server_health_async(self, url: str) -> bool:
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
        current_time = time.time()
        tasks = []

        for url in self.server_urls:
            # Only check if enough time has passed since last check
            if current_time - self.last_check[url] >= self.check_interval:
                tasks.append(self.check_server_health_async(url))
                self.last_check[url] = current_time

        if tasks:
            results = await asyncio.gather(*tasks)

            # Update healthy servers set with URLs that are healthy
            self.healthy_servers = {url for i, url in enumerate(self.server_urls) if i < len(results) and results[i]}

    async def run_health_checks(self) -> None:
        """Run periodic health checks in the background."""
        while True:
            await self.check_all_servers()
            await asyncio.sleep(self.check_interval)


server_health_monitor = ServerHealthMonitor()
