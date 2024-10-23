from src.token_manager import TokenManager
from src.utils.common import current_timestamp_5min_interval

token_manager = TokenManager()


def add_token_task(token):
    print("task called")
    token_manager.add_token(token)


def call_event_task(token):
    token_manager.increment_requests(token)
    print("collect calls...")

async def sync_metrics():
    current_unix_timestamp = current_timestamp_5min_interval().strftime("%s")

    metrics = token_manager.get_metrics()
