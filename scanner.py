"""
Trading Scanner — Crypto

Detecta setups de impulso + retroceso Fibonacci en H1 y D1.
Opera 24/7 (sin chequeo de horario de mercado).
Envía alertas por Telegram cuando se dan ≥4 confluencias.

Uso manual:
  python scanner.py                          # escanea BTC/USD y ETH/USD
  python scanner.py --symbol BTC             # acepta BTC o BTC/USD
  python scanner.py --symbol ETH --timeframe H1
  python scanner.py --symbol SOL --timeframe D1  # cualquier crypto de Alpaca
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pytz
import pandas as pd

from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import (
    SYMBOLS, TIMEFRAMES, BARS_LIMIT,
    ALPACA_API_KEY, ALPACA_API_SECRET,
    MIN_CONFLUENCES,
)
from indicators import analyze_setup
from alerts import format_alert, format_daily_summary, send_telegram_alert

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
_TF_MAP = {
    "H1": TimeFrame(1, TimeFrameUnit.Hour),
    "4H": TimeFrame(4, TimeFrameUnit.Hour),
    "D1": TimeFrame(1, TimeFrameUnit.Day),
}

_TF_LABEL = {
    "H1": "1H",
    "4H": "4H",
    "D1": "D",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_symbol(symbol: str) -> str:
    """Convierte BTC → BTC/USD, ETH → ETH/USD. Deja BTC/USD sin cambios."""
    symbol = symbol.upper().strip()
    if "/" not in symbol:
        symbol = f"{symbol}/USD"
    return symbol


def get_alpaca_client() -> CryptoHistoricalDataClient:
    """
    CryptoHistoricalDataClient no requiere credenciales para datos históricos,
    pero las pasamos si están disponibles.
    """
    if ALPACA_API_KEY and ALPACA_API_SECRET:
        return CryptoHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    return CryptoHistoricalDataClient()


def fetch_bars(
    client: CryptoHistoricalDataClient, symbol: str, timeframe: str
) -> Optional[pd.DataFrame]:
    """Descarga las últimas BARS_LIMIT velas para el símbolo y timeframe dados."""
    tf = _TF_MAP.get(timeframe)
    if tf is None:
        log.error(f"Timeframe '{timeframe}' no soportado. Usar H1 o D1.")
        return None

    try:
        start = datetime.utcnow() - timedelta(days=60)
        req   = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            limit=BARS_LIMIT,
        )
        data = client.get_crypto_bars(req)

        bars_data = data.data
        log.debug(f"{symbol} — símbolos en respuesta: {list(bars_data.keys())}")

        key = symbol if symbol in bars_data else (list(bars_data)[0] if bars_data else None)
        if not key or not bars_data[key]:
            log.warning(f"{symbol} — sin datos en la respuesta de Alpaca")
            return None

        records = [
            {
                "timestamp": b.timestamp,
                "open":      float(b.open),
                "high":      float(b.high),
                "low":       float(b.low),
                "close":     float(b.close),
                "volume":    float(b.volume),
            }
            for b in bars_data[key]
        ]
        df = pd.DataFrame(records).set_index("timestamp")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        return df[["open", "high", "low", "close", "volume"]]

    except Exception as e:
        log.warning(f"No se pudieron obtener datos para {symbol} ({timeframe}): {e}")
        return None


# ── Escaneo de un símbolo / timeframe ────────────────────────────────────────

def scan_one(
    client: CryptoHistoricalDataClient, symbol: str, timeframe: str
) -> Dict:
    """
    Analiza bullish y bearish para un símbolo + timeframe.
    Manda alerta de Telegram si hay setup (≥4 confluencias).
    Retorna un dict con los scores para el resumen diario.
    """
    log.info(f"Escaneando {symbol} [{timeframe}]...")

    summary = {
        "symbol":        symbol,
        "bullish_score": 0,
        "bearish_score": 0,
        "bullish_valid": False,
        "bearish_valid": False,
    }

    df = fetch_bars(client, symbol, timeframe)
    if df is None or len(df) < 30:
        log.warning(f"{symbol} [{timeframe}] — datos insuficientes, saltando.")
        return summary

    tf_label = _TF_LABEL.get(timeframe, timeframe)

    for direction in ("bullish", "bearish"):
        try:
            result = analyze_setup(df, direction)
        except Exception as e:
            log.error(f"{symbol} [{timeframe}] {direction} — error en análisis: {e}")
            continue

        confluences = result.get("confluences", 0)
        valid       = result.get("valid", False)

        log.info(
            f"  {symbol} {tf_label} {direction}: "
            f"{confluences}/6 confluencias — {'✅ ALERTA' if valid else '❌ no setup'}"
        )

        if direction == "bullish":
            summary["bullish_score"] = confluences
            summary["bullish_valid"] = valid
        else:
            summary["bearish_score"] = confluences
            summary["bearish_valid"] = valid

        if valid:
            message = format_alert(symbol, tf_label, result)
            log.info(f"\n{message}\n")
            send_telegram_alert(message)

    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trading Scanner — Crypto (24/7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python scanner.py\n"
            "  python scanner.py --symbol BTC\n"
            "  python scanner.py --symbol BTC/USD --timeframe H1\n"
            "  python scanner.py --symbol SOL --timeframe D1\n"
        ),
    )
    p.add_argument("--symbol",    type=str, help="Símbolo a escanear (ej: BTC o BTC/USD)")
    p.add_argument("--timeframe", type=str, default=None,
                   choices=["H1", "4H", "D1"],
                   help="Timeframe: H1, 4H o D1. Default: ambos (H1 + 4H)")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    log.info("=" * 55)
    log.info(f"🔍 Crypto Scanner — {now_utc}")
    log.info("=" * 55)

    symbols    = [normalize_symbol(args.symbol)] if args.symbol else SYMBOLS
    timeframes = [args.timeframe] if args.timeframe else TIMEFRAMES

    client = get_alpaca_client()

    for tf in timeframes:
        rows = [scan_one(client, sym, tf) for sym in symbols]

        # Resumen de 4H: solo cuando se escanea 4H con todos los activos
        if tf == "4H" and not args.symbol:
            tf_label = _TF_LABEL.get(tf, tf)
            summary_msg = format_daily_summary(rows, tf_label)
            log.info(f"\n{summary_msg}\n")
            send_telegram_alert(summary_msg)

    log.info("=" * 55)
    log.info("✔  Scan completado")


if __name__ == "__main__":
    main()
