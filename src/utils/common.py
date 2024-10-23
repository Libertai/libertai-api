from datetime import datetime

"""Return 5min cron job like scheduling behaviour"""
def current_timestamp_5min_interval():
    now = datetime.now()
    rounded_minutes = (now.minute // 5) * 5
    return now.replace(minute=rounded_minutes, second=0, microsecond=0)
