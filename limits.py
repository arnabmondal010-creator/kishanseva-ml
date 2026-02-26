# limits.py
from collections import defaultdict
from datetime import datetime, timedelta

# In-memory demo storage (replace with DB in prod)
_usage = defaultdict(int)
_plans = defaultdict(lambda: "basic")

FREE_LIMIT = 10

def get_user_plan(user_id: str) -> str:
    return _plans[user_id]

def set_user_plan(user_id: str, plan: str):
    _plans[user_id] = plan

def can_use(user_id: str) -> bool:
    plan = get_user_plan(user_id)
    if plan == "pro":
        return True
    return _usage[user_id] < FREE_LIMIT

def mark_used(user_id: str):
    _usage[user_id] += 1
