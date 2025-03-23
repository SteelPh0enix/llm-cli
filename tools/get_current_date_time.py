import datetime
from typing import Any


def get_current_date_time() -> dict[str, Any]:
    """Returns current date and time"""

    current_datetime = datetime.datetime.now()
    return {
        "year": current_datetime.year,
        "month": current_datetime.month,
        "day": current_datetime.day,
        "hour": current_datetime.hour,
        "minute": current_datetime.minute,
        "second": current_datetime.second,
    }
