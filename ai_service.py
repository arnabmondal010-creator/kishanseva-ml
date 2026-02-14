import base64
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def analyze_image(crop: str, image_bytes: bytes):
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = f"""
You are an agriculture expert AI.
Analyze this {crop} leaf image and respond strictly in JSON:

{{
  "disease": "string",
  "confidence": 0.0,
  "advice": "clear treatment advice"
}}

If healthy, say disease = "Healthy".
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )

    raw = response.choices[0].message.content

    try:
        import json
        return json.loads(raw)
    except Exception:
        return {
            "disease": "Unknown",
            "confidence": 0.0,
            "advice": raw
        }
