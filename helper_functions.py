# ---------------------- Modules ----------------------
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from functools import lru_cache
from math import exp, log, sqrt
from pathlib import Path
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)


# ---------------------- Helper Functions ----------------------
def bs_d1_d2(
    S: float, K: float, T: float, r: float, sigma: float
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute the Black-Scholes d1 and d2 terms.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (annualised).

    Returns
    -------
    tuple
        (d1, d2) as floats, or (None, None) if T <= 0 or sigma <= 0.
    """
    if T <= 0 or sigma <= 0:
        return None, None

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "Call",
) -> float:
    """
    Black-Scholes option price (Call or Put).

    Handles edge cases:
      - If T <= 0, returns intrinsic value.
      - If sigma <= 0, returns forward intrinsic (discounted cash) form.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (annualised).
    option_type : str, optional
        'Call' or 'Put' (default: 'Call').

    Returns
    -------
    float
        Option price.
    """
    S = float(S)
    K = float(K)

    if T <= 0:
        return max(S - K, 0.0) if option_type == "Call" else max(K - S, 0.0)

    if sigma <= 0:
        return (
            max(S - K * exp(-r * T), 0.0)
            if option_type == "Call"
            else max(K * exp(-r * T) - S, 0.0)
        )

    d1, d2 = bs_d1_d2(S, K, T, r, sigma)

    if option_type == "Call":
        return float(S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_price_vectorized(
    S_array: Sequence[float],
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "Call",
) -> np.ndarray:
    """
    Vectorized wrapper for bs_price over a sequence/array of spot values.

    Parameters
    ----------
    S_array : sequence of float
        Spots to evaluate.
    K, T, r, sigma : see bs_price
    option_type : str, optional
        'Call' or 'Put' (default: 'Call').

    Returns
    -------
    np.ndarray
        Array of option prices (float).
    """
    Sarr = np.array(S_array, dtype=float)
    return np.vectorize(
        lambda s: bs_price(s, K, T, r, sigma, option_type)
    )(Sarr)


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "Call",
) -> Dict[str, float]:
    """
    Compute basic Black-Scholes greeks: delta, gamma, vega, theta, rho.

    Edge-cases:
      - If T <= 0, returns payoff-based delta and zeros for other greeks.
      - If sigma <= 0, returns zeros for greeks.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (annualised).
    option_type : str, optional
        'Call' or 'Put' (default: 'Call').

    Returns
    -------
    dict
        {'delta': float, 'gamma': float, 'vega': float, 'theta': float, 'rho': float}
    """
    if T <= 0:
        if option_type == "Call":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0

        return {
            "delta": delta,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
        }

    if sigma <= 0:
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
        }

    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    delta = cdf_d1 if option_type == "Call" else cdf_d1 - 1.0
    gamma = pdf_d1 / (S * sigma * sqrt(T))
    vega = S * pdf_d1 * sqrt(T)
    term1 = -(S * pdf_d1 * sigma) / (2 * sqrt(T))

    if option_type == "Call":
        theta = term1 - r * K * exp(-r * T) * cdf_d2
    else:
        theta = term1 + r * K * exp(-r * T) * (1 - cdf_d2)

    rho = (
        K * T * exp(-r * T) * cdf_d2
        if option_type == "Call"
        else -K * T * exp(-r * T) * (1 - cdf_d2)
    )

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def weighted_return_percent(
    prices: Union[Sequence[float], np.ndarray, float],
    entry: float,
    mult: float,
    fx_rate: float,
    fund_nav: float,
    sign: float = 1.0,
) -> np.ndarray:
    """
    Compute weighted return as percentage of NAV.

    The function accepts either a scalar or an array-like `prices`. It returns an
    ndarray of weighted returns (in percent).

    Parameters
    ----------
    prices : sequence of float or float
        Price(s) to evaluate (option contract value).
    entry : float
        Entry price (per contract).
    mult : float
        Multiplier (qty * lot size).
    fx_rate : float
        FX rate used to convert contract P&L to NAV currency.
    fund_nav : float
        Fund NAV (denominator).
    sign : float, optional
        +1 for long, -1 for short (default: 1.0).

    Returns
    -------
    np.ndarray
        Weighted returns as percentages. If fx_rate or fund_nav is zero, returns
        an array of NaNs of the appropriate shape.
    """
    ret_per_contract = (np.asarray(prices, dtype=float) - entry) * sign
    total_ret = ret_per_contract * mult

    if fx_rate == 0 or fund_nav == 0:
        return np.full_like(np.asarray(prices, dtype=float), np.nan, dtype=float)

    weighted = (total_ret / fx_rate) / fund_nav
    return weighted * 100.0


def surface_value_normalisation(v: float, vmin: float, vmax: float) -> float:
    """
    Normalize a value `v` into [0,1] based on vmin..vmax.

    Parameters
    ----------
    v : float
        Value to normalise.
    vmin : float
        Minimum of normalization range.
    vmax : float
        Maximum of normalization range.

    Returns
    -------
    float
        Normalised value in floating point.
    """
    return float((v - vmin) / (vmax - vmin))


def find_zero_crossings(
    x_arr: Sequence[float],
    y_arr: Sequence[float],
    func_exact: Optional[Callable[[float], float]] = None,
    brent_xtol: float = 1e-10,
    brent_rtol: float = 1e-12,
    brent_maxiter: int = 200,
    min_search_tol: float = 1e-10,
    min_search_maxiter: int = 100,
) -> List[float]:
    """
    Find zero-crossings (roots) in a sampled function given by (x_arr, y_arr).

    Behavior:
      - Any exact grid points where |y| < 1e-14 are taken as roots.
      - Sign changes between adjacent grid points attempt refinement via brentq on
        an `func_exact` function (if provided). If `func_exact` is None, an
        exact function is built from `st.session_state` and the current legs
        (this preserves existing behaviour).
      - If brentq fails or doesn't bracket, falls back to linear interpolation.
      - Detects potential tangent/root-touch using `minimize_scalar` when both
        adjacent values are small but have no sign change.
      - Results are deduplicated and sorted.

    Parameters
    ----------
    x_arr : sequence of float
        Sampled x values (monotonic).
    y_arr : sequence of float
        Sampled y values corresponding to x_arr.
    func_exact : callable, optional
        Exact function f(S) used to refine roots; must accept float and return float.
        If None (default), the function will compute `f` from `st.session_state` legs.
    brent_xtol, brent_rtol, brent_maxiter : solver tolerances/limits for brentq.
    min_search_tol, min_search_maxiter : options passed to minimize_scalar for tangent detection.

    Returns
    -------
    list of float
        Sorted list of distinct root x-values.
    """
    if func_exact is None:

        def func_exact_local(S_val: float) -> float:
            """
            Exact function built from st.session_state legs.

            This mirrors the existing behaviour where find_zero_crossings
            uses the session_state to compute the weighted-return at a given S.
            """
            try:
                legs = st.session_state.get("legs", [])
                r = float(st.session_state.get("r", 0.0))
                fx_rate = float(st.session_state.get("fx_rate", 0.0))
                fund_nav = float(st.session_state.get("fund_nav", 28560000.0))
                current_days = int(st.session_state.get("current_days", 0))
                close_days = int(st.session_state.get("close_days", 0))
                total = 0.0

                for leg in legs:
                    days_leg = int(leg.get("days", current_days))
                    T_close_leg = max(0.0, (days_leg - int(close_days)) / 365.0)
                    price_close = bs_price(
                        S_val, leg["K"], T_close_leg, r, leg["vol"], leg["type"]
                    )
                    wr = weighted_return_percent(
                        np.array([price_close]),
                        leg["entry"],
                        leg.get("mult", leg.get("qty", 0) * leg.get("size", 0)),
                        fx_rate,
                        fund_nav,
                        leg.get("sign", 1.0),
                    )[0]
                    total += float(wr)

                return total
            except Exception:
                return float("nan")

        func_exact = func_exact_local

    # caching wrapper to avoid recomputing same S many times in brentq
    @lru_cache(maxsize=8192)
    def f_cached(S_rounded: float) -> float:
        return float(func_exact(float(S_rounded)))

    def f(S: float) -> float:
        # quantize S before cache-key so floating differences map to same cached bucket
        key = round(float(S), 12)
        return f_cached(key)

    zeros: List[float] = []
    n = len(x_arr)

    for i in range(n):
        yi = y_arr[i]
        if np.isnan(yi):
            continue
        if abs(yi) < 1e-14:  # exact (or numerically tiny) on the sampled grid
            zeros.append(float(x_arr[i]))

    # find sign-change brackets and refine with brentq
    for i in range(n - 1):
        a, b = float(x_arr[i]), float(x_arr[i + 1])
        y1, y2 = y_arr[i], y_arr[i + 1]
        if np.isnan(y1) or np.isnan(y2):
            continue

        # If sign change on the sampled y-array, attempt brentq on the exact function
        if y1 * y2 < 0:
            try:
                # ensure endpoints bracket with the exact function
                fa, fb = f(a), f(b)
                if np.isnan(fa) or np.isnan(fb) or fa * fb > 0:
                    # if exact doesn't bracket, fall back to linear interpolation root on grid
                    root_lin = a - y1 * (b - a) / (y2 - y1)
                    zeros.append(root_lin)
                    continue

                root = brentq(
                    lambda S: f(S),
                    a,
                    b,
                    xtol=brent_xtol,
                    rtol=brent_rtol,
                    maxiter=brent_maxiter,
                )
                zeros.append(float(root))
            except Exception:
                # fallback: linear interp
                try:
                    root_lin = a - y1 * (b - a) / (y2 - y1)
                    zeros.append(root_lin)
                except Exception:
                    pass
            continue

        # No sign change but the grid value is very small in magnitude -> consider it a root
        if abs(y1) < 1e-8:
            zeros.append(a)
            continue
        if abs(y2) < 1e-8:
            zeros.append(b)
            continue

        # Detect potential tangent/root-touch where function may dip to zero without sign change.
        if min(abs(y1), abs(y2)) < 1e-2:  # heuristic threshold; adjust for more/less aggressive detection
            try:
                res = minimize_scalar(
                    lambda S: abs(f(S)),
                    bounds=(a, b),
                    method="bounded",
                    options={"xatol": min_search_tol, "maxiter": min_search_maxiter},
                )
                if res.success and abs(res.fun) < 1e-10:
                    zeros.append(float(res.x))
            except Exception:
                pass

    # dedupe & sort results
    zeros_sorted = sorted(zeros)
    final: List[float] = []
    for z in zeros_sorted:
        if not final:
            final.append(z)
        else:
            if abs(z - final[-1]) > 1e-9:  # distinct roots
                final.append(z)

    return final