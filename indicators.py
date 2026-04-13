"""
Indicadores técnicos para el trading scanner.

Condiciones del setup (bullish / bearish):
  1. Impulso previo (swing high + swing low identificables)
  2. Retroceso de mínimo 3 velas sin hacer nuevo extremo
  3. Retroceso en zona Fibonacci 38.2%–61.8%
  4. EMAs 4, 9 y 18 alineadas en dirección del impulso
  5. Soporte/resistencia coincidente con la zona de retroceso
  6. Patrón de vela de reversión (engulfing, hammer, shooting star)
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

from config import (
    EMA_PERIODS, SWING_LOOKBACK, FIB_LOW_LEVEL, FIB_HIGH_LEVEL,
    MIN_PULLBACK_BARS, MIN_CONFLUENCES,
)


# ─────────────────────────────────────────────────────────────────────────────
# EMAs
# ─────────────────────────────────────────────────────────────────────────────

def calculate_emas(df: pd.DataFrame) -> Dict[int, pd.Series]:
    """Calcula EMAs para los períodos configurados."""
    return {p: df["close"].ewm(span=p, adjust=False).mean() for p in EMA_PERIODS}


def check_ema_alignment(emas: Dict[int, pd.Series], direction: str) -> bool:
    """
    Bullish : EMA4 > EMA9 > EMA18
    Bearish : EMA4 < EMA9 < EMA18
    """
    e4  = emas[4].iloc[-1]
    e9  = emas[9].iloc[-1]
    e18 = emas[18].iloc[-1]
    if direction == "bullish":
        return e4 > e9 > e18
    return e4 < e9 < e18


# ─────────────────────────────────────────────────────────────────────────────
# Swing highs / lows
# ─────────────────────────────────────────────────────────────────────────────

def find_swing_points(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> pd.DataFrame:
    """
    Marca swing highs y swing lows.
    Un swing high tiene el máximo más alto que las `lookback` velas de cada lado.
    Un swing low tiene el mínimo más bajo.
    """
    df = df.copy()
    df["swing_high"] = np.nan
    df["swing_low"]  = np.nan
    n = len(df)

    for i in range(lookback, n - lookback):
        hi_window = df["high"].iloc[i - lookback : i + lookback + 1]
        lo_window = df["low"].iloc[i - lookback : i + lookback + 1]

        if df["high"].iloc[i] == hi_window.max():
            df.iloc[i, df.columns.get_loc("swing_high")] = df["high"].iloc[i]
        if df["low"].iloc[i] == lo_window.min():
            df.iloc[i, df.columns.get_loc("swing_low")] = df["low"].iloc[i]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Impulso previo
# ─────────────────────────────────────────────────────────────────────────────

def find_last_impulse(df: pd.DataFrame, direction: str) -> Optional[Dict[str, Any]]:
    """
    Busca el impulso más reciente en la dirección indicada.

    Bullish: swing low → swing high (movimiento alcista previo).
    Bearish: swing high → swing low (movimiento bajista previo).

    Retorna None si no hay impulso válido.
    """
    highs = df[df["swing_high"].notna()]
    lows  = df[df["swing_low"].notna()]

    if direction == "bullish":
        if highs.empty or lows.empty:
            return None
        # Último swing high
        sh_idx   = highs.index[-1]
        sh_price = highs["swing_high"].iloc[-1]
        # Swing low que lo precede
        prev_lows = lows[lows.index < sh_idx]
        if prev_lows.empty:
            return None
        sl_idx   = prev_lows.index[-1]
        sl_price = prev_lows["swing_low"].iloc[-1]
        # El impulso debe ser al menos 1%
        if (sh_price - sl_price) / sl_price < 0.01:
            return None
        return {
            "direction":   "bullish",
            "start_idx":   sl_idx,
            "end_idx":     sh_idx,
            "start_price": sl_price,   # swing low
            "end_price":   sh_price,   # swing high
        }

    else:  # bearish
        if highs.empty or lows.empty:
            return None
        # Último swing low
        sl_idx   = lows.index[-1]
        sl_price = lows["swing_low"].iloc[-1]
        # Swing high que lo precede
        prev_highs = highs[highs.index < sl_idx]
        if prev_highs.empty:
            return None
        sh_idx   = prev_highs.index[-1]
        sh_price = prev_highs["swing_high"].iloc[-1]
        if (sh_price - sl_price) / sh_price < 0.01:
            return None
        return {
            "direction":   "bearish",
            "start_idx":   sh_idx,
            "end_idx":     sl_idx,
            "start_price": sh_price,   # swing high
            "end_price":   sl_price,   # swing low
        }


# ─────────────────────────────────────────────────────────────────────────────
# Retroceso (pullback)
# ─────────────────────────────────────────────────────────────────────────────

def check_pullback(df: pd.DataFrame, impulse: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verifica que, después del impulso, haya un retroceso válido:
    - Al menos MIN_PULLBACK_BARS velas
    - Sin hacer nuevo extremo más allá del inicio del impulso
    - Precio actual va en dirección opuesta al impulso
    """
    post = df[df.index > impulse["end_idx"]]
    n    = len(post)

    result: Dict[str, Any] = {
        "valid":           False,
        "bar_count":       n,
        "pullback_extreme": None,
        "no_new_extreme":  False,
    }

    if n < MIN_PULLBACK_BARS:
        return result

    current_close = df["close"].iloc[-1]

    if impulse["direction"] == "bullish":
        pb_low = post["low"].min()
        result["pullback_extreme"] = pb_low
        result["no_new_extreme"]   = pb_low > impulse["start_price"]
        result["valid"] = (
            result["no_new_extreme"]
            and current_close < impulse["end_price"]
        )
    else:  # bearish
        pb_high = post["high"].max()
        result["pullback_extreme"] = pb_high
        result["no_new_extreme"]   = pb_high < impulse["start_price"]
        result["valid"] = (
            result["no_new_extreme"]
            and current_close > impulse["end_price"]
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Fibonacci
# ─────────────────────────────────────────────────────────────────────────────

def get_fib_zone(impulse: Dict[str, Any]) -> Dict[str, float]:
    """
    Calcula la zona de retroceso Fibonacci 38.2%–61.8% del impulso.
    """
    if impulse["direction"] == "bullish":
        hi, lo = impulse["end_price"], impulse["start_price"]
        rng    = hi - lo
        return {
            "zone_low":  hi - rng * FIB_HIGH_LEVEL,   # 61.8% (más profundo)
            "zone_high": hi - rng * FIB_LOW_LEVEL,    # 38.2% (menos profundo)
            "fib_382":   hi - rng * 0.382,
            "fib_500":   hi - rng * 0.500,
            "fib_618":   hi - rng * 0.618,
        }
    else:  # bearish
        hi, lo = impulse["start_price"], impulse["end_price"]
        rng    = hi - lo
        return {
            "zone_low":  lo + rng * FIB_LOW_LEVEL,    # 38.2%
            "zone_high": lo + rng * FIB_HIGH_LEVEL,   # 61.8%
            "fib_382":   lo + rng * 0.382,
            "fib_500":   lo + rng * 0.500,
            "fib_618":   lo + rng * 0.618,
        }


def price_in_fib_zone(price: float, fib_zone: Dict[str, float]) -> bool:
    return fib_zone["zone_low"] <= price <= fib_zone["zone_high"]


# ─────────────────────────────────────────────────────────────────────────────
# Soporte / Resistencia
# ─────────────────────────────────────────────────────────────────────────────

def check_sr_in_zone(df: pd.DataFrame, fib_zone: Dict[str, float]) -> bool:
    """
    Detecta si hay un nivel previo de soporte/resistencia dentro de la zona Fibonacci.
    Busca swing highs o lows históricos (excluyendo las últimas barras del retroceso)
    que coincidan con la zona.
    """
    # Excluir las velas más recientes del retroceso para no hacer trampa
    history = df.iloc[:-MIN_PULLBACK_BARS].copy()
    if len(history) < SWING_LOOKBACK * 2 + 1:
        return False

    swings   = find_swing_points(history)
    lo, hi   = fib_zone["zone_low"], fib_zone["zone_high"]

    sh_vals  = swings["swing_high"].dropna()
    sl_vals  = swings["swing_low"].dropna()

    in_zone  = (
        ((sh_vals >= lo) & (sh_vals <= hi)).any()
        or ((sl_vals >= lo) & (sl_vals <= hi)).any()
    )
    return bool(in_zone)


# ─────────────────────────────────────────────────────────────────────────────
# Patrones de vela de reversión
# ─────────────────────────────────────────────────────────────────────────────

def detect_reversal_candle(df: pd.DataFrame, direction: str) -> Tuple[bool, str]:
    """
    Detecta patrones de reversión en la última vela.

    Bullish : Hammer, Bullish Engulfing, Bullish Pinbar
    Bearish : Shooting Star, Bearish Engulfing, Bearish Pinbar

    Retorna (detectado, nombre_del_patrón).
    """
    if len(df) < 2:
        return False, ""

    cur  = df.iloc[-1]
    prev = df.iloc[-2]

    body        = abs(cur["close"] - cur["open"])
    total_range = cur["high"] - cur["low"]
    if total_range == 0:
        return False, ""

    upper_wick  = cur["high"] - max(cur["close"], cur["open"])
    lower_wick  = min(cur["close"], cur["open"]) - cur["low"]
    body_ratio  = body / total_range

    is_bull_cur  = cur["close"]  > cur["open"]
    is_bear_cur  = cur["close"]  < cur["open"]
    is_bear_prev = prev["close"] < prev["open"]
    is_bull_prev = prev["close"] > prev["open"]

    if direction == "bullish":
        # Hammer: cuerpo pequeño arriba, mecha inferior larga (≥2× cuerpo)
        if (body > 0
                and lower_wick >= 2 * body
                and upper_wick <= body * 0.5
                and lower_wick >= 0.55 * total_range):
            return True, "Hammer"

        # Bullish Engulfing
        if (is_bull_cur and is_bear_prev
                and cur["close"] > prev["open"]
                and cur["open"]  < prev["close"]):
            return True, "Bullish Engulfing"

        # Bullish Pinbar (cierre alcista con mecha inferior significativa)
        if (is_bull_cur
                and lower_wick >= 1.5 * body
                and body_ratio > 0.1):
            return True, "Bullish Pinbar"

    else:  # bearish
        # Shooting Star: cuerpo pequeño abajo, mecha superior larga
        if (body > 0
                and upper_wick >= 2 * body
                and lower_wick <= body * 0.5
                and upper_wick >= 0.55 * total_range):
            return True, "Shooting Star"

        # Bearish Engulfing
        if (is_bear_cur and is_bull_prev
                and cur["close"] < prev["open"]
                and cur["open"]  > prev["close"]):
            return True, "Bearish Engulfing"

        # Bearish Pinbar
        if (is_bear_cur
                and upper_wick >= 1.5 * body
                and body_ratio > 0.1):
            return True, "Bearish Pinbar"

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Análisis completo del setup
# ─────────────────────────────────────────────────────────────────────────────

def analyze_setup(df: pd.DataFrame, direction: str) -> Dict[str, Any]:
    """
    Evalúa el setup completo para una dirección dada.

    Retorna un dict con:
      - valid        : bool — al menos MIN_CONFLUENCES condiciones cumplen
      - confluences  : int  — cuántas condiciones se cumplen
      - conditions   : dict — detalle de cada condición
      - entry_low/high, stop_loss, emas, fib_zone, impulse, pullback
    """
    base: Dict[str, Any] = {
        "valid":      False,
        "direction":  direction,
        "confluences": 0,
        "conditions": {},
    }

    if len(df) < 30:
        base["reason"] = "Datos insuficientes"
        return base

    df_swings = find_swing_points(df)
    cond: Dict[str, Any] = {}

    # 1. Impulso previo
    impulse = find_last_impulse(df_swings, direction)
    cond["impulso"] = impulse is not None
    if not impulse:
        base["conditions"] = cond
        return base

    # 2. Retroceso válido
    pullback = check_pullback(df, impulse)
    cond["retroceso"] = pullback["valid"]

    # 3. Fibonacci
    fib_zone      = get_fib_zone(impulse)
    current_price = df["close"].iloc[-1]
    cond["fibonacci"] = price_in_fib_zone(current_price, fib_zone)

    # 4. EMAs alineadas
    emas = calculate_emas(df)
    cond["emas_alineadas"] = check_ema_alignment(emas, direction)

    # 5. Soporte/Resistencia en zona
    cond["soporte_resistencia"] = check_sr_in_zone(df, fib_zone)

    # 6. Vela de reversión
    rev_detected, rev_pattern = detect_reversal_candle(df, direction)
    cond["vela_reversion"] = rev_detected
    cond["patron_vela"]    = rev_pattern

    # ── Confluencias ──────────────────────────────────────────────────────────
    score_keys  = ["impulso", "retroceso", "fibonacci", "emas_alineadas",
                   "soporte_resistencia", "vela_reversion"]
    confluences = sum(1 for k in score_keys if cond.get(k))

    # ── Zonas de entrada y stop loss ─────────────────────────────────────────
    if direction == "bullish":
        entry_low  = fib_zone["fib_618"]
        entry_high = fib_zone["fib_382"]
        pb_ext     = pullback.get("pullback_extreme") or impulse["start_price"]
        stop_loss  = round(pb_ext * 0.997, 4)   # 0.3% por debajo del mínimo
    else:
        entry_low  = fib_zone["fib_382"]
        entry_high = fib_zone["fib_618"]
        pb_ext     = pullback.get("pullback_extreme") or impulse["start_price"]
        stop_loss  = round(pb_ext * 1.003, 4)   # 0.3% por encima del máximo

    return {
        "valid":       confluences >= MIN_CONFLUENCES,
        "direction":   direction,
        "confluences": confluences,
        "conditions":  cond,
        "fib_zone":    fib_zone,
        "impulse":     impulse,
        "pullback":    pullback,
        "current_price": current_price,
        "entry_low":   round(entry_low,  4),
        "entry_high":  round(entry_high, 4),
        "stop_loss":   stop_loss,
        "emas_values": {p: round(emas[p].iloc[-1], 4) for p in EMA_PERIODS},
    }
