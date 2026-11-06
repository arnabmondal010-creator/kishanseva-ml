#!/usr/bin/env python3
"""One-time migration for missing/placeholder KishanSeva quiz display names.

Uses the same FIREBASE_KEY JSON environment variable as api.py.
Default mode is read-only. Pass --apply to write to Firestore.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from typing import Any

from google.api_core.retry import Retry
from google.cloud import firestore
from google.oauth2 import service_account

PLACEHOLDER_NAMES = {"newfarmer", "farmer", "user"}
ADJECTIVES = (
    "Green",
    "Golden",
    "Harvest",
    "Monsoon",
    "Mango",
    "Bamboo",
    "River",
    "Field",
)


def make_display_name() -> str:
    # Same naming pattern as the existing api.py helper.
    return f"Farmer_{secrets.choice(ADJECTIVES)}{secrets.randbelow(900000) + 100000}"


def is_missing_or_placeholder(value: Any) -> bool:
    name = str(value or "").strip()
    return not name or name.casefold() in PLACEHOLDER_NAMES


def get_firestore_client() -> firestore.Client:
    raw_key = os.getenv("FIREBASE_KEY")
    if not raw_key:
        raise RuntimeError(
            "FIREBASE_KEY environment variable is missing. "
            "Set it to the service-account JSON used by api.py."
        )

    try:
        firebase_key = json.loads(raw_key)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"FIREBASE_KEY is not valid JSON: {exc}") from exc

    if not isinstance(firebase_key, dict) or not firebase_key.get("project_id"):
        raise RuntimeError("FIREBASE_KEY JSON must contain project_id.")

    credentials = service_account.Credentials.from_service_account_info(firebase_key)
    return firestore.Client(
        credentials=credentials,
        project=firebase_key["project_id"],
    )


def assign_name_transactionally(db: firestore.Client, uid: str) -> tuple[str | None, bool]:
    """Assign a unique name, without overwriting a valid existing name."""
    profile_ref = db.collection("farmers").document(uid)
    transaction = db.transaction()

    @firestore.transactional
    def apply(transaction):
        profile_snapshot = profile_ref.get(transaction=transaction)
        if not profile_snapshot.exists:
            return None, False

        profile = profile_snapshot.to_dict() or {}
        current_name = str(profile.get("quizDisplayName") or "").strip()

        if not is_missing_or_placeholder(current_name):
            return current_name, False

        # Remove an old placeholder reservation only if this account owns it.
        old_alias_ref = None
        old_alias_snapshot = None
        if current_name:
            old_alias_ref = db.collection("quiz_display_names").document(
                current_name.casefold()
            )
            old_alias_snapshot = old_alias_ref.get(transaction=transaction)

        # Check each candidate inside the transaction so two concurrent
        # migrations cannot reserve the same alias.
        for _ in range(100):
            generated_name = make_display_name()
            normalized = generated_name.casefold()
            alias_ref = db.collection("quiz_display_names").document(normalized)
            alias_snapshot = alias_ref.get(transaction=transaction)

            if alias_snapshot.exists:
                continue

            # All transaction reads happen before these writes.
            transaction.set(alias_ref, {
                "userId": uid,
                "displayName": generated_name,
                "createdAt": firestore.SERVER_TIMESTAMP,
                "source": "legacy_account_backfill",
            })
            transaction.set(profile_ref, {
                "quizDisplayName": generated_name,
                "quizDisplayNameUpdatedAt": firestore.SERVER_TIMESTAMP,
            }, merge=True)

            if (
                old_alias_ref is not None
                and old_alias_snapshot is not None
                and old_alias_snapshot.exists
                and (old_alias_snapshot.to_dict() or {}).get("userId") == uid
            ):
                transaction.delete(old_alias_ref)

            return generated_name, True

        raise RuntimeError(f"Could not find an available name for UID {uid} after 100 attempts")

    return apply(transaction)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill missing or placeholder farmer quiz display names."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually update Firestore. Without this flag, the script is read-only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of farmer documents to inspect (0 = all).",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be zero or a positive integer")

    db = get_firestore_client()
    # Keep the collection stream retryable for longer migrations. Without an
    # explicit Retry object, some Firestore/gRPC timeout paths fail while
    # trying to read the callable's private _retry attribute.
    farmers = db.collection("farmers").stream(retry=Retry(timeout=1800))

    scanned = 0
    needs_name = 0
    assigned = 0
    unchanged = 0
    failed = 0
    leaderboard_synced = 0

    mode = "APPLY" if args.apply else "DRY RUN (read-only)"
    print(f"Mode: {mode}")
    print(f"Firestore project: {db.project}")

    for farmer_doc in farmers:
        if args.limit and scanned >= args.limit:
            break

        scanned += 1
        profile = farmer_doc.to_dict() or {}
        old_name = str(profile.get("quizDisplayName") or "").strip()

        if not is_missing_or_placeholder(old_name):
            unchanged += 1
            continue

        needs_name += 1

        if not args.apply:
            print(f"WOULD ASSIGN: uid={farmer_doc.id}, current={old_name or '<missing>'}")
            continue

        try:
            assigned_name, changed = assign_name_transactionally(db, farmer_doc.id)
            if changed and assigned_name:
                assigned += 1
                print(f"ASSIGNED: uid={farmer_doc.id}, quizDisplayName={assigned_name}")

                # Update the visible name only for users who already have a
                # leaderboard document; do not create empty leaderboard rows.
                leaderboard_ref = db.collection("farmer_quiz_leaderboard").document(
                    farmer_doc.id
                )
                leaderboard_snapshot = leaderboard_ref.get()
                if leaderboard_snapshot.exists:
                    leaderboard_ref.set({
                        "farmerName": assigned_name,
                        "updatedAt": firestore.SERVER_TIMESTAMP,
                    }, merge=True)
                    leaderboard_synced += 1
            else:
                unchanged += 1

        except Exception as exc:  # Continue migrating the remaining accounts.
            failed += 1
            print(f"FAILED: uid={farmer_doc.id}: {exc}", file=sys.stderr)

        if scanned % 100 == 0:
            print(
                f"Progress: scanned={scanned}, assigned={assigned}, "
                f"unchanged={unchanged}, failed={failed}"
            )

    print("\nMigration summary")
    print(f"Scanned: {scanned}")
    print(f"Missing/placeholder names: {needs_name}")
    print(f"Assigned: {assigned}")
    print(f"Unchanged: {unchanged}")
    print(f"Existing leaderboard names synchronized: {leaderboard_synced}")
    print(f"Failed: {failed}")

    if not args.apply:
        print("\nNo Firestore data was changed. Re-run with --apply to commit the migration.")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"MIGRATION STOPPED: {error}", file=sys.stderr)
        raise SystemExit(2)
