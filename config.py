"""
Configuración del trading scanner de crypto.
"""
import os

# ── Activos a escanear (formato Alpaca: BASE/QUOTE) ───────────────────────────
SYMBOLS = ["BTC/USD", "ETH/USD"]

# ── Timeframes ────────────────────────────────────────────────────────────────
TIMEFRAMES = ["H1", "4H"]

# ── Setup: parámetros ─────────────────────────────────────────────────────────
MIN_CONFLUENCES    = 4
FIB_LOW_LEVEL      = 0.382
FIB_HIGH_LEVEL     = 0.618
EMA_PERIODS        = [4, 9, 18]
MIN_PULLBACK_BARS  = 3
SWING_LOOKBACK     = 5
BARS_LIMIT         = 120

# ── Credenciales ──────────────────────────────────────────────────────────────
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
