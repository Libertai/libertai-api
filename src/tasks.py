import aiohttp

from src.config import config


async def report_usage_event_task(usage: dict):
    print("collect usage...", usage)
    try:
        async with aiohttp.ClientSession() as session:
            session.headers["x-admin-token"] = config.BACKEND_SECRET_TOKEN
            path = "api-keys/admin/usage"
            async with session.post(f"{config.BACKEND_API_URL}/{path}", json=usage.dict()) as response:
                if response.status == 200:
                    pass
                else:
                    print(f"Error reporting usage: {response.status}")

    except Exception as e:
        print(f"Exception occured during usage report {str(e)}")
