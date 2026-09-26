from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_FILE = DATA_DIR / "history.json"
MAX_RECORDS = 2000  # keep the file bounded

VALID_OUTCOMES = {"PENDING", "WIN", "LOSS", "SKIPPED"}


@dataclass
class SignalRecord:
    id: str
    time: str
    source: str          # LIVE_DATA | SCREENSHOT | MANUAL_DATA
    asset: str
    price: float | None
    action: str           # BUY | SELL | WAIT
    duration: str
    confidence: int
    reasons: list[str] = field(default_factory=list)
    warning: str = ""
    outcome: str = "PENDING"  # WIN | LOSS | SKIPPED | PENDING
    outcome_time: str | None = None


class HistoryStore:
    def __init__(self):
        DATA_DIR.mkdir(exist_ok=True)
        self._lock = Lock()
        self._records: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if HISTORY_FILE.exists():
            try:
                return json.loads(HISTORY_FILE.read_text())
            except Exception:
                return []
        return []

    def _save(self):
        HISTORY_FILE.write_text(json.dumps(self._records[-MAX_RECORDS:], indent=2, ensure_ascii=False))

    def add(self, *, source: str, asset: str, price: float | None, action: str,
            duration: str, confidence: int, reasons: list[str], warning: str = "") -> dict:
        rec = SignalRecord(
            id=uuid.uuid4().hex[:10],
            time=datetime.now(timezone.utc).isoformat(),
            source=source,
            asset=asset,
            price=price,
            action=action,
            duration=duration,
            confidence=confidence,
            reasons=reasons or [],
            warning=warning,
            outcome="PENDING",
        )
        with self._lock:
            self._records.append(asdict(rec))
            self._save()
        return asdict(rec)

    def set_outcome(self, record_id: str, outcome: str) -> dict | None:
        outcome = outcome.upper().strip()
        if outcome not in VALID_OUTCOMES:
            raise ValueError(f"Invalid outcome: {outcome}")
        with self._lock:
            for r in self._records:
                if r["id"] == record_id:
                    r["outcome"] = outcome
                    r["outcome_time"] = datetime.now(timezone.utc).isoformat()
                    self._save()
                    return r
        return None

    def list(self, limit: int = 200) -> list[dict]:
        return list(reversed(self._records[-limit:]))

    def stats(self) -> dict:
        """Real counts from recorded history only -- never a made-up win rate."""
        total = len(self._records)
        wins = sum(1 for r in self._records if r["outcome"] == "WIN")
        losses = sum(1 for r in self._records if r["outcome"] == "LOSS")
        skipped = sum(1 for r in self._records if r["outcome"] == "SKIPPED")
        pending = sum(1 for r in self._records if r["outcome"] == "PENDING")
        waits = sum(1 for r in self._records if r["action"] == "WAIT")
        decided = wins + losses
        win_rate = round((wins / decided) * 100, 1) if decided > 0 else None
        return {
            "total_signals": total,
            "wins": wins,
            "losses": losses,
            "skipped": skipped,
            "pending": pending,
            "wait_signals": waits,
            "win_rate": win_rate,  # null until there is at least one recorded WIN/LOSS
        }

    def clear(self):
        with self._lock:
            self._records = []
            self._save()
