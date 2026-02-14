from datetime import datetime, timedelta

FREE_LIMIT = 5  # free uploads per day
_usage = {}     # {user_id: {"count": int, "date": date}}

def can_use(user_id: str):
    today = datetime.utcnow().date()

    if user_id not in _usage or _usage[user_id]["date"] != today:
        _usage[user_id] = {"count": 0, "date": today}

    if _usage[user_id]["count"] >= FREE_LIMIT:
        return False

    _usage[user_id]["count"] += 1
    return True
