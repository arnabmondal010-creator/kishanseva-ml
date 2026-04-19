# -*- coding: utf-8 -*-
import firebase_admin
from firebase_admin import credentials, firestore, auth

# 🔥 INIT
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

docs = db.collection("farmers").stream()

created = 0
skipped = 0

for d in docs:
    data = d.to_dict()
    email = data.get("personal", {}).get("email")

    if not email:
        skipped += 1
        continue

    try:
        auth.get_user_by_email(email)
        print("⚠️ Already exists:", email)
        continue
    except:
        try:
            auth.create_user(
                email=email,
                password="Temp@1234"   # temp password
            )
            print("✅ Created:", email)
            created += 1
        except Exception as e:
            print("❌ Failed:", email, e)
            skipped += 1

print("\nDONE")
print("Created:", created)
print("Skipped:", skipped)
