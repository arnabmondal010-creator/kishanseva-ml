```python
# ai_service.py

import base64
import os
import json
from PIL import Image
import io
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------
# IMAGE COMPRESSION
# -----------------------------
def compress_image(image_bytes: bytes) -> bytes:
    """
    Resize and compress image before sending to OpenAI
    Target size ~100–250 KB
    """

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize large images
    img.thumbnail((512, 512))

    buffer = io.BytesIO()

    img.save(
        buffer,
        format="JPEG",
        quality=70,
        optimize=True
    )

    return buffer.getvalue()


# -----------------------------
# IMAGE VALIDATION
# -----------------------------
def validate_image(image_bytes: bytes):

    if not image_bytes:
        raise ValueError("Empty image received")

    if len(image_bytes) > 5 * 1024 * 1024:
        raise ValueError("Image too large. Max size 5MB.")


# -----------------------------
# DISEASE ANALYSIS
# -----------------------------
async def analyze_image(image_bytes: bytes, crop: str):

    try:

        # Validate image
        validate_image(image_bytes)

        # Compress image
        compressed = compress_image(image_bytes)

        base64_image = base64.b64encode(compressed).decode("utf-8")

        prompt = f"""
Crop: {crop}

Task:
Identify disease symptoms visible on the crop leaf.

Rules:
- If the leaf is healthy return disease = "Healthy"
- If the disease cannot be identified return disease = "Unknown"
- Do NOT guess diseases
- Confidence must be between 0 and 1

Return JSON only:

{{
"disease": "",
"confidence": 0.0,
"advice": ""
}}
"""

        response = client.chat.completions.create(

            model="gpt-4o-mini",

            temperature=0,

            response_format={"type": "json_object"},

            messages=[

                {
                    "role": "system",
                    "content": """
You are a strict agricultural plant disease diagnostician.

Rules:
- Only analyze visible symptoms
- Never hallucinate diseases
- If uncertain return "Unknown"
- Output valid JSON only
"""
                },

                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
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

        result = response.choices[0].message.content

        parsed = json.loads(result)

        # Safety filter
        if parsed.get("confidence", 0) < 0.3:
            parsed["disease"] = "Unknown"
            parsed["advice"] = "Image unclear. Please capture leaf in natural light."

        return parsed

    except Exception as e:

        print("Disease detection error:", str(e))

        return {
            "disease": "Unknown",
            "confidence": 0.0,
            "advice": "Image analysis failed. Try again with a clearer leaf photo."
        }
```
