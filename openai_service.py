import os
import base64
import requests

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

def diagnose_crop(image_bytes: bytes, crop: str):
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert agricultural plant pathologist."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Detect disease for this {crop} leaf and give treatment."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        "max_tokens": 200
    }

    res = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=60)
    res.raise_for_status()
    data = res.json()

    return data["choices"][0]["message"]["content"]
