USAGE = {}

FREE_LIMIT = 5

def can_use(user_id: str) -> bool:
    if user_id not in USAGE:
        USAGE[user_id] = 0

    if USAGE[user_id] >= FREE_LIMIT:
        return False

    USAGE[user_id] += 1
    return True
