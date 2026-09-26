from __future__ import annotations
import base64
import json
import os
import re
import urllib.error
import urllib.request

from fastapi import HTTPException

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}

# The model that reads the screenshot. Set OPENAI_VISION_MODEL to whatever
# vision-capable model your OpenAI account actually has access to -- this
# default is just a starting point, not a guarantee it exists on your plan.
DEFAULT_MODEL = "gpt-4o"

INSTRUCTION = """
You are a strictly educational assistant that reads ONE screenshot of a
trading chart. You are not connected to any broker and you cannot see
live prices -- you can only describe what is visibly present in this
single image.

Rules:
- Only use what is clearly visible: candles/price action, direction,
  any visible indicator panels (RSI, MACD, moving averages, etc.),
  support/resistance levels drawn on the chart.
- If the asset, timeframe, or enough recent candles are not clearly
  visible, or the image is blurry/cropped/ambiguous, choose WAIT.
- If visible signals conflict with each other, choose WAIT.
- Never claim certainty. Confidence must be conservative, not promotional.
- A single screenshot has no way to verify what happens next -- do not
  imply a guaranteed outcome anywhere in "reason".
- If you choose BUY or SELL, pick exactly one duration from: 30s, 1m,
  2m, 3m, 5m, based on the timeframe visible on the chart. If you can't
  tell, choose WAIT instead.

Return ONLY JSON, no markdown, in exactly this shape:
{"action":"BUY|SELL|WAIT","asset":"symbol or null","duration":"30s|1m|2m|3m|5m|-",
 "confidence":0,"reasons":["short reason 1","short reason 2"],"warning":"short caveat or empty string"}
""".strip()


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("Could not parse the analysis result")
        return json.loads(match.group(0))


def _normalize(data: dict) -> dict:
    action = str(data.get("action", "WAIT")).upper().strip()
    if action not in {"BUY", "SELL"}:
        action = "WAIT"

    try:
        confidence = max(0, min(100, int(round(float(data.get("confidence", 0))))))
    except Exception:
        confidence = 0

    duration = str(data.get("duration", "-")).strip()
    allowed_durations = {"30s", "1m", "2m", "3m", "5m", "-"}
    if duration not in allowed_durations:
        duration = "-" if action == "WAIT" else "1m"

    reasons = data.get("reasons")
    if not isinstance(reasons, list):
        reasons = [str(data.get("reason", "Image was not conclusive enough."))]
    reasons = [str(r).strip()[:160] for r in reasons][:6]

    warning = str(data.get("warning", "")).strip()[:180]
    asset = data.get("asset")
    asset = str(asset).strip() if asset else None

    # Conservative guardrail: below this confidence, never render a
    # directional call -- fall back to WAIT regardless of what the model said.
    MIN_CONFIDENCE_FOR_SIGNAL = 65
    if confidence < MIN_CONFIDENCE_FOR_SIGNAL:
        action = "WAIT"
        duration = "-"

    return {
        "action": action,
        "asset": asset,
        "duration": duration,
        "confidence": confidence,
        "reasons": reasons,
        "warning": warning,
        "source": "SCREENSHOT",
    }


def analyze_screenshot(image_bytes: bytes, content_type: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not set on the server, so screenshot analysis is disabled.",
        )

    model = os.getenv("OPENAI_VISION_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    mime = content_type if content_type in ALLOWED_CONTENT_TYPES else "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{encoded}"

    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": INSTRUCTION},
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                ],
            }
        ],
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:800]
        raise HTTPException(status_code=502, detail=f"Vision provider error: {detail}") from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not reach the analysis service: {e}") from e

    output_text = raw.get("output_text")
    if not output_text:
        pieces = []
        for item in raw.get("output", []):
            for content in (item.get("content", []) if isinstance(item, dict) else []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    pieces.append(content.get("text", ""))
        output_text = "\n".join(pieces)

    try:
        parsed = _extract_json(output_text or "")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not interpret the analysis result: {e}") from e

    return _normalize(parsed)
