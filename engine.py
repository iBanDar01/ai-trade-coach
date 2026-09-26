from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .strategy import score_signal
from .history_store import HistoryStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATE_FILE = DATA_DIR / "state.json"
TRADES_FILE = DATA_DIR / "trades.csv"

DEMO_ASSET = "EUR/USD (DEMO)"


@dataclass
class RiskConfig:
    starting_balance: float = 1000.0
    risk_per_trade_pct: float = 1.0
    daily_loss_limit_pct: float = 5.0
    max_trades_per_day: int = 12
    max_consecutive_losses: int = 3
    payout_pct: float = 80.0


class PaperEngine:
    """Drives the on-screen demo chart with a SIMULATED random-walk feed.

    This is intentionally not connected to ExpertOption or any live market
    data provider -- it exists so the UI (chart, signal card, paper
    trading, risk manager) has something to render out of the box. Wire a
    real market-data source into `_seed_market`/`tick` if you want signals
    computed from real prices instead of the simulation.
    """

    def __init__(self, history: HistoryStore):
        DATA_DIR.mkdir(exist_ok=True)
        self.history = history
        self.risk = RiskConfig()
        self.balance = self.risk.starting_balance
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.loss_streak = 0
        self.df = self._seed_market()
        self._last_logged_key = ""
        self._load_state()

    def _seed_market(self):
        rng = np.random.default_rng(7)
        n = 180
        ret = rng.normal(0, 0.00045, n)
        close = 1.085 + np.cumsum(ret)
        open_ = np.r_[close[0], close[:-1]]
        high = np.maximum(open_, close) + rng.uniform(0.00005, 0.0005, n)
        low = np.minimum(open_, close) - rng.uniform(0.00005, 0.0005, n)
        ts = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="1min")
        return pd.DataFrame({"time": ts, "open": open_, "high": high, "low": low, "close": close})

    def _load_state(self):
        if STATE_FILE.exists():
            try:
                s = json.loads(STATE_FILE.read_text())
                self.balance = float(s.get("balance", self.balance))
                self.daily_pnl = float(s.get("daily_pnl", 0))
                self.trades_today = int(s.get("trades_today", 0))
                self.loss_streak = int(s.get("loss_streak", 0))
            except Exception:
                pass

    def _save_state(self):
        STATE_FILE.write_text(json.dumps({
            "balance": self.balance,
            "daily_pnl": self.daily_pnl,
            "trades_today": self.trades_today,
            "loss_streak": self.loss_streak,
        }, indent=2))

    def _maybe_log_signal(self, sig: dict):
        """Log to history only when the signal actually changes, so WAIT
        ticks every few seconds don't spam the history list."""
        action = "BUY" if sig["side"].startswith("BUY") else ("SELL" if sig["side"].startswith("SELL") else "WAIT")
        key = f"{action}|{sig.get('price')}"
        if action == "WAIT" or key == self._last_logged_key:
            return
        self._last_logged_key = key
        self.history.add(
            source="LIVE_DATA",
            asset=DEMO_ASSET,
            price=sig.get("price"),
            action=action,
            duration="1m",
            confidence=sig.get("confidence", 0),
            reasons=sig.get("reasons", []),
        )

    def tick(self):
        rng = np.random.default_rng()
        o = float(self.df.iloc[-1].close)
        c = o + float(rng.normal(0, 0.0004))
        h = max(o, c) + float(rng.uniform(0.00002, 0.00035))
        l = min(o, c) - float(rng.uniform(0.00002, 0.00035))
        row = pd.DataFrame([{"time": pd.Timestamp.utcnow(), "open": o, "high": h, "low": l, "close": c}])
        self.df = pd.concat([self.df.iloc[-299:], row], ignore_index=True)
        return self.snapshot()

    def snapshot(self):
        sig = score_signal(self.df)
        self._maybe_log_signal(sig)
        candles = self.df.tail(60)[["time", "open", "high", "low", "close"]].copy()
        candles["time"] = candles["time"].astype(str)
        return {
            "asset": DEMO_ASSET,
            "source": "LIVE_DATA",
            "balance": round(self.balance, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "trades_today": self.trades_today,
            "loss_streak": self.loss_streak,
            "signal": sig,
            "risk": asdict(self.risk),
            "candles": candles.to_dict("records"),
        }

    def can_trade(self):
        if self.trades_today >= self.risk.max_trades_per_day:
            return False, "Daily trade limit reached"
        if self.loss_streak >= self.risk.max_consecutive_losses:
            return False, "Stopped after consecutive losses"
        if self.daily_pnl <= -(self.risk.starting_balance * self.risk.daily_loss_limit_pct / 100):
            return False, "Daily loss limit reached"
        return True, "OK"

    def place_paper_trade(self, side: str, amount: float | None = None):
        ok, reason = self.can_trade()
        if not ok:
            return {"ok": False, "message": reason}
        if side not in {"BUY", "SELL"}:
            return {"ok": False, "message": "Invalid trade side"}
        if amount is None:
            amount = max(1.0, self.balance * self.risk.risk_per_trade_pct / 100)
        amount = min(float(amount), self.balance)
        if amount <= 0:
            return {"ok": False, "message": "Invalid amount"}

        entry = float(self.df.iloc[-1].close)
        for _ in range(3):
            self.tick()
        exit_price = float(self.df.iloc[-1].close)
        won = (side == "BUY" and exit_price > entry) or (side == "SELL" and exit_price < entry)
        pnl = amount * (self.risk.payout_pct / 100) if won else -amount
        self.balance += pnl
        self.daily_pnl += pnl
        self.trades_today += 1
        self.loss_streak = 0 if won else self.loss_streak + 1
        self._save_state()

        rec = {
            "time": datetime.now(timezone.utc).isoformat(),
            "side": side, "entry": round(entry, 5), "exit": round(exit_price, 5),
            "amount": round(amount, 2), "result": "WIN" if won else "LOSS",
            "pnl": round(pnl, 2), "balance": round(self.balance, 2),
        }
        pd.DataFrame([rec]).to_csv(TRADES_FILE, mode="a", header=not TRADES_FILE.exists(), index=False)
        return {"ok": True, "trade": rec, "snapshot": self.snapshot()}
