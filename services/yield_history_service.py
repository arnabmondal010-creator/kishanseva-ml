# services/yield_history_service.py

_YIELD_HISTORY = {}

def add_yield_record(user_id: str, field_id: str, record: dict):
    key = f"{user_id}:{field_id}"
    _YIELD_HISTORY.setdefault(key, []).append(record)
    return True

def get_history(user_id: str, field_id: str):
    key = f"{user_id}:{field_id}"
    return _YIELD_HISTORY.get(key, [])