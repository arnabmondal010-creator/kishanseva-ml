# -*- coding: utf-8 -*-
import firebase_admin
from firebase_admin import credentials, firestore, auth

# 🔥 INIT FIREBASE
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

docs = db.collection("farmers").stream()

deleted = 0
kept = 0

for d in docs:
    try:
        # ✅ valid UID → exists in Firebase Auth
        auth.get_user(d.id)
        kept += 1
    except:
        # ❌ not a UID → delete
        db.collection("farmers").document(d.id).delete()
        print("🧹 Deleted:", d.id)
        deleted += 1

print("\nDONE")
print("Deleted:", deleted)
print("Kept:", kept)
