# -*- coding: utf-8 -*-
import firebase_admin
from firebase_admin import credentials, firestore, auth

# 🔥 INIT FIREBASE
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()   # ✅ THIS WAS MISSING

docs = db.collection("farmers").stream()

migrated = 0
skipped = 0

for d in docs:
    data = d.to_dict()

    email = data.get("personal", {}).get("email")

    if not email:
        print("❌ No email → skip:", d.id)
        skipped += 1
        continue

    try:
        user = auth.get_user_by_email(email)
        uid = user.uid
    except:
        print("❌ Auth user not found:", email)
        skipped += 1
        continue

    # 🚨 prevent overwrite
    if db.collection("farmers").document(uid).get().exists:
        print("⚠️ Already exists:", email)
        continue

    db.collection("farmers").document(uid).set(data)

    token = data.get("fcm_token")
    if token:
        db.collection("alerts_state").document(uid).set({
            "fcm_token": token
        })

    print("✅ Migrated:", email)
    migrated += 1

print("\nDONE")
print("Migrated:", migrated)
print("Skipped:", skipped)
