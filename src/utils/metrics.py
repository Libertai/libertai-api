import asyncio

from aleph.sdk import AlephHttpClient, AuthenticatedAlephHttpClient
from aleph.sdk.chains.ethereum import ETHAccount
from aleph.sdk.query.filters import MessageFilter
from aleph_message.models import MessageType
from src.account_manager import AccountManager
from src.config import config
from src.interfaces.metrics import Metric, Metrics
from src.utils.common import current_timestamp_5min_interval

account_manager = AccountManager()

async def sync_metrics():
    current_unix_timestamp = int(current_timestamp_5min_interval().strftime("%s"))

    metrics = account_manager.get_metrics()
    for metric in metrics.values:
        """Sync after current interval"""
        if metric.ts <= current_unix_timestamp:
            print("Dont sync current interval")
            continue

        print("metric", metric)
        exists = await metric_exists(metric)
        if exists:
            continue

        await post_metric(metric)
        await asyncio.sleep(5)


async def metric_exists(metric) -> bool:
    async with AlephHttpClient(
            api_server=config.ALEPH_API_URL
    ) as client:
        items = await client.get_messages(
            page=1,
            page_size=1,
            message_filter=MessageFilter(
                addresses=[config.LTAI_SENDER_ADDRESS],
                channels=[config.LTAI_AUTH_METRICS_POST_CHANNEL],
                content_types=[config.LTAI_AUTH_METRICS_POST_TYPE],
                message_types=[MessageType.post],
                tags=[f"{metric.key}.{metric.ts}"]
            )
        )
        if items.pagination_total > 0:
            return True
        else:
            return False


async def post_metric(metric: Metric):
    print("syncing metric", metric.owner)
    aleph_account = ETHAccount(config.LTAI_SENDER_SK)
    content = {
        "owner": metric.owner,
        "count": metric.count,
        "tags": [f"{metric.key}.{metric.ts}", metric.key],
    }

    async with AuthenticatedAlephHttpClient(
        account=aleph_account, api_server=config.ALEPH_API_URL
    ) as client:
        post_message, _ = await client.create_post(
            post_content=content,
            channel=config.LTAI_AUTH_METRICS_POST_CHANNEL,
            post_type=config.LTAI_AUTH_METRICS_POST_TYPE,
            sync=True
        )


async def get_user_metrics(key: str, since: int = None, until: int = None) -> Metrics:
    metrics = []
    print("filter", since, until)
    async with AlephHttpClient(
            api_server=config.ALEPH_API_URL
    ) as client:
        items = await client.get_messages(
            page=1,
            page_size=1000,
            message_filter=MessageFilter(
                addresses=[config.LTAI_SENDER_ADDRESS],
                channels=[config.LTAI_AUTH_METRICS_POST_CHANNEL],
                content_types=[config.LTAI_AUTH_METRICS_POST_TYPE],
                message_types=[MessageType.post],
                tags=[key]
            )
        )

        for item in items.messages:
            data = item.content.content
            tag = [tag for tag in data["tags"] if f"{key}." in tag]
            if not tag:
                continue
            ts = int(tag[0].split('.')[1])

            if since and ts < since:
                continue

            if until and ts > until:
                continue

            metric = {
                "owner": data["owner"],
                "key": key,
                "ts": ts,
                "count": data["count"]
            }

            metrics.append(metric)

    if metrics:
        return Metrics(values=metrics, owner=metrics[0]["owner"])
