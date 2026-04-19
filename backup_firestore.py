# -*- coding: utf-8 -*-
import firebase_admin
from firebase_admin import credentials, firestore
import json

# 🔥 INIT
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()


# 🔥 CONVERT FIRESTORE TYPES
def serialize(obj):
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


docs = db.collection("farmers").stream()

backup = []

for d in docs:
    data = d.to_dict()

    # 🔥 convert all values
    clean_data = json.loads(json.dumps(data, default=serialize))

    clean_data["old_id"] = d.id
    backup.append(clean_data)


with open("backup.json", "w", encoding="utf-8") as f:
    json.dump(backup, f, indent=2, ensure_ascii=False)

print("✅ Backup done:", len(backup))
