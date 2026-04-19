# -*- coding: utf-8 -*-
import firebase_admin
from firebase_admin import credentials, firestore

# INIT
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

docs = db.collection("farmers").stream()

deleted = 0
kept = 0

for d in docs:
    data = d.to_dict()

    email = data.get("personal", {}).get("email")

    # ❌ delete if no email OR token-like id
    if (not email) or (":" in d.id) or (len(d.id) > 40):
        db.collection("farmers").document(d.id).delete()
        print("🧹 Deleted:", d.id)
        deleted += 1
    else:
        kept += 1

print("\nDONE")
print("Deleted:", deleted)
print("Kept:", kept)
