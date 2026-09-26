from __future__ import annotations
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .engine import PaperEngine
from .history_store import HistoryStore
from .risk import RiskInput, compute_risk
from .vision import analyze_screenshot, ALLOWED_CONTENT_TYPES

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(title="AI Trade Coach", version="3.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

history = HistoryStore()
engine = PaperEngine(history)


class TradeRequest(BaseModel):
    side: str
    amount: float | None = None


class OutcomeRequest(BaseModel):
    outcome: str  # WIN | LOSS | SKIPPED


# ---------- App shell / PWA plumbing ----------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(STATIC_DIR / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    return FileResponse(
        STATIC_DIR / "sw.js",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
    )


# ---------- Demo live chart / paper trading ----------

@app.get("/api/state")
def state():
    return engine.snapshot()


@app.post("/api/tick")
def tick():
    return engine.tick()


@app.post("/api/paper-trade")
def paper_trade(req: TradeRequest):
    return engine.place_paper_trade(req.side.upper(), req.amount)


# ---------- Screenshot analysis ----------

@app.post("/api/analyze-screen")
async def analyze_screen(image: UploadFile = File(...)):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPG, WEBP or GIF image.")
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="The image is empty.")
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large; 12MB max.")

    result = analyze_screenshot(data, image.content_type)
    logged = history.add(
        source="SCREENSHOT",
        asset=result.get("asset") or "Unknown (from screenshot)",
        price=None,
        action=result["action"],
        duration=result["duration"],
        confidence=result["confidence"],
        reasons=result["reasons"],
        warning=result.get("warning", ""),
    )
    result["history_id"] = logged["id"]
    return {"ok": True, "signal": result}


@app.get("/api/ai-status")
def ai_status():
    return {
        "configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "model": os.getenv("OPENAI_VISION_MODEL", "gpt-4o"),
    }


# ---------- History ----------

@app.get("/api/history")
def get_history(limit: int = 200):
    return {"records": history.list(limit=limit), "stats": history.stats()}


@app.get("/api/history/stats")
def get_history_stats():
    return history.stats()


@app.post("/api/history/{record_id}/outcome")
def set_history_outcome(record_id: str, req: OutcomeRequest):
    rec = history.set_outcome(record_id, req.outcome)
    if rec is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"ok": True, "record": rec, "stats": history.stats()}


@app.delete("/api/history")
def clear_history():
    history.clear()
    return {"ok": True}


# ---------- Risk manager ----------

@app.post("/api/risk/calc")
def risk_calc(payload: RiskInput):
    return compute_risk(payload)
