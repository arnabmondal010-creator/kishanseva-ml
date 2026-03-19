# ai_service.py

import base64
import os
import json
import io
from PIL import Image
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def compress_image(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((768, 768))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=80, optimize=True)

    return buffer.getvalue()


def validate_image(image_bytes: bytes):
    if not image_bytes:
        raise ValueError("Empty image")

    if len(image_bytes) > 5 * 1024 * 1024:
        raise ValueError("Image too large```python
                         
async def analyze_image(image_bytes: bytes, crop: str):

    try:
        validate_image(image_bytes)

        compressed = compress_image(image_bytes)
        print("Compressed size:", len(compressed))

        base64_image = base64.b64encode(compressed).decode()

        prompt = f"""
Crop: {crop}

Identify the disease in this crop leaf image.

Return JSON:
{{
"disease":"",
"confidence":0.0,
"advice":""
}}
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a plant disease expert."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )

        raw = response.choices[0].message.content
        print("RAW:", raw)

        try:
            return json.loads(raw)
        except:
            return {
                "disease": raw,
                "confidence": 0.5,
                "advice": "Parsed from raw response"
            }

    except Exception as e:
        print("ERROR:", e)

        return {
            "disease": "Unknown",
            "confidence": 0.0,
            "advice": "Server error"
        }
