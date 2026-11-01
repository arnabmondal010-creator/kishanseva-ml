
from pathlib import Path

path = Path("upload_daily_quiz.py")
source = path.read_text(encoding="utf-8")

start = source.index("def load_firebase_credentials():")
end = source.index("\ndef validate_quiz(data):", start)

replacement = '''def load_firebase_credentials():
    credentials_path = Path(__file__).parent / "serviceAccountKey.json"

    if not credentials_path.is_file():
        raise FileNotFoundError(
            f"Firebase credentials not found: {credentials_path}"
        )

    with credentials_path.open("r", encoding="utf-8") as file:
        return json.load(file)


'''

updated = source[:start] + replacement + source[end + 1:]
path.write_text(updated, encoding="utf-8")

print("Credential function replaced successfully.")
