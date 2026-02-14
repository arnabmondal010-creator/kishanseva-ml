import base64
import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Fallback advice for common diseases (in case AI fails)
FALLBACK_ADVICE = {
    "Bacterial_leaf_blight": "Remove infected leaves. Avoid overhead irrigation. Apply copper-based bactericide.",
    "Brown_spot": "Ensure good drainage. Apply fungicide such as mancozeb. Avoid excess nitrogen.",
    "Leaf_blast": "Use resistant varieties. Apply tricyclazole fungicide. Avoid late nitrogen application.",
    "Healthy": "Crop looks healthy. Maintain proper irrigation and balanced fertilization."
}

def _encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

async def analyze_image(crop: str, image_bytes: bytes) -> dict:
    """
    Returns:
    {
        "disease": str,
        "confidence": float (0 to 1),
        "advice": str
    }
    """

    image_b64 = _encode_image_to_base64(image_bytes)

    prompt = f"""
You are an expert agriculture plant pathologist.

Analyze this {crop} leaf image and detect possible disease.

Return ONLY valid JSON with ALL fields filled:

{{
  "disease": "short disease name or Healthy",
  "confidence": 0.00,
  "advice": "clear, practical treatment advice for Indian farmers"
}}

Rules:
- Confidence must be between 0 and 1.
- If image is unclear, still guess and advise better photo.
- Never return null.
- Keep advice simple and actionable.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
                    ],
                }
            ],
            max_tokens=300,
        )

        raw_text = response.choices[0].message.content.strip()

        # Attempt to parse JSON
        data = json.loads(raw_text)

        disease = data.get("disease") or "Unknown"
        confidence = float(data.get("confidence") or 0.5)
        advice = data.get("advice") or FALLBACK_ADVICE.get(disease, "Consult a local agriculture officer.")

        return {
            "disease": disease.replace(" ", "_"),
            "confidence": max(0.0, min(confidence, 1.0)),
            "advice": advice
        }

    except Exception as e:
        print("AI Error:", e)

        return {
            "disease": "Unknown",
            "confidence": 0.30,
            "advice": "Image unclear. Please upload a clear close-up of the affected leaf in daylight."
        }
