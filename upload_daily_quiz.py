from pathlib import Path
import argparse
import json
import os
import sys
from datetime import date

from google.cloud import firestore
from google.oauth2 import service_account


def load_firebase_credentials():
    credentials_path = Path(__file__).parent / "serviceAccountKey.json"

    if not credentials_path.is_file():
        raise FileNotFoundError(
            f"Firebase credentials not found: {credentials_path}"
        )

    with credentials_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_quiz(data):
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object.")

    questions = data.get("questions")

    if not isinstance(questions, list) or len(questions) != 10:
        raise ValueError("The quiz must contain exactly 10 questions.")

    if not isinstance(data.get("title"), str) or not data["title"].strip():
        raise ValueError("Quiz title is missing.")

    question_ids = set()

    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"Question {index} must be an object.")

        question_id = question.get("id")
        question_text = question.get("question")
        options = question.get("options")
        correct_id = question.get("correctOptionId")
        explanation = question.get("explanation")

        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"Question {index}: invalid ID.")

        if question_id in question_ids:
            raise ValueError(f"Duplicate question ID: {question_id}")

        question_ids.add(question_id)

        if not isinstance(question_text, str) or not question_text.strip():
            raise ValueError(f"Question {index}: question text is missing.")

        if not isinstance(options, list) or len(options) != 4:
            raise ValueError(f"Question {index}: exactly 4 options required.")

        option_ids = set()

        for option in options:
            if not isinstance(option, dict):
                raise ValueError(f"Question {index}: invalid option.")

            option_id = option.get("id")
            option_text = option.get("text")

            if not isinstance(option_id, str) or not option_id.strip():
                raise ValueError(f"Question {index}: invalid option ID.")

            if option_id in option_ids:
                raise ValueError(f"Question {index}: duplicate option ID.")

            if not isinstance(option_text, str) or not option_text.strip():
                raise ValueError(f"Question {index}: option text is missing.")

            option_ids.add(option_id)

        if correct_id not in option_ids:
            raise ValueError(
                f"Question {index}: correctOptionId does not match an option."
            )

        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError(f"Question {index}: explanation is missing.")


def main():
    parser = argparse.ArgumentParser(
        description="Upload a daily farmer quiz to Firestore."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Quiz date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing quiz document."
    )
    args = parser.parse_args()

    try:
        quiz_date = date.fromisoformat(args.date)
    except ValueError:
        print("ERROR: Date must use YYYY-MM-DD format.")
        sys.exit(1)

    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "daily_quiz_questions.json"
    )

    if not os.path.isfile(json_path):
        print(f"ERROR: JSON file not found: {json_path}")
        sys.exit(1)

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            quiz_data = json.load(file)

        validate_quiz(quiz_data)

        firebase_key = load_firebase_credentials()
        credentials = service_account.Credentials.from_service_account_info(
            firebase_key
        )

        db = firestore.Client(
            credentials=credentials,
            project=firebase_key["project_id"]
        )

        document_id = quiz_date.isoformat()
        quiz_ref = db.collection("daily_quizzes").document(document_id)

        if quiz_ref.get().exists and not args.overwrite:
            print(
                f"ERROR: Quiz {document_id} already exists. "
                "Use --overwrite only if you intend to replace it."
            )
            sys.exit(1)

        upload_data = {
            "title": quiz_data["title"].strip(),
            "status": "draft",
            "questions": quiz_data["questions"],
        }

        quiz_ref.set(upload_data)

        print("\nSUCCESS: Quiz uploaded to Firestore.")
        print(f"Collection: daily_quizzes")
        print(f"Document: {document_id}")
        print(f"Questions: {len(quiz_data['questions'])}")
        print("Status: draft")

    except Exception as exc:
        print(f"\nUPLOAD FAILED: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
