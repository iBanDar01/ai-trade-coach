from __future__ import annotations
from pydantic import BaseModel


class RiskInput(BaseModel):
    balance: float
    risk_per_trade_pct: float = 1.0
    daily_loss_limit_pct: float = 5.0
    trades_today: int = 0
    daily_pnl: float = 0.0


def compute_risk(inp: RiskInput) -> dict:
    balance = max(0.0, inp.balance)
    stake = round(balance * (inp.risk_per_trade_pct / 100), 2)
    daily_loss_limit_amount = round(balance * (inp.daily_loss_limit_pct / 100), 2)
    remaining_before_limit = round(daily_loss_limit_amount + min(0.0, inp.daily_pnl), 2)
    limit_hit = inp.daily_pnl <= -daily_loss_limit_amount
    return {
        "suggested_stake": stake,
        "daily_loss_limit_amount": daily_loss_limit_amount,
        "remaining_before_limit": max(0.0, remaining_before_limit),
        "limit_hit": bool(limit_hit),
        "trades_today": inp.trades_today,
    }
