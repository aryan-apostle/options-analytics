# ---------------------- Modules ----------------------
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from math import exp, log, sqrt
from scipy.stats import norm



# ---------------------- Helper Functions ----------------------
def bs_d1_d2(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return None, None
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type='Call'):
    S = float(S)
    K = float(K)

    if T <= 0:
        return max(S - K, 0.0) if option_type == 'Call' else max(K - S, 0.0)

    if sigma <= 0:
        return max(S - K * exp(-r * T), 0.0) if option_type == 'Call' else max(K * exp(-r * T) - S, 0.0)

    d1, d2 = bs_d1_d2(S, K, T, r, sigma)

    if option_type == 'Call':
        return float(S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_price_vectorized(S_array, K, T, r, sigma, option_type='Call'):
    Sarr = np.array(S_array, dtype=float)
    return np.vectorize(lambda s: bs_price(s, K, T, r, sigma, option_type))(Sarr)


def bs_greeks(S, K, T, r, sigma, option_type='Call'):
    if T <= 0:
        if option_type == 'Call':
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {'delta': delta, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}

    if sigma <= 0:
        return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}

    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    delta = cdf_d1 if option_type == 'Call' else cdf_d1 - 1.0
    
    gamma = pdf_d1 / (S * sigma * sqrt(T))
    
    vega = S * pdf_d1 * sqrt(T)
    
    term1 = - (S * pdf_d1 * sigma) / (2 * sqrt(T))
    
    if option_type == 'Call':
        theta = term1 - r * K * exp(-r * T) * cdf_d2
    else:
        theta = term1 + r * K * exp(-r * T) * (1 - cdf_d2)
    
    rho = K * T * exp(-r * T) * cdf_d2 if option_type == 'Call' else -K * T * exp(-r * T) * (1 - cdf_d2)

    return {'delta': float(delta), 'gamma': float(gamma), 'vega': float(vega), 'theta': float(theta), 'rho': float(rho)}


def weighted_return_percent(prices, entry, mult, fx_rate, fund_nav, sign=1.0):
    ret_per_contract = (prices - entry) * sign
    total_ret = ret_per_contract * mult

    if fx_rate == 0 or fund_nav == 0:
        return np.full_like(np.asarray(prices, dtype=float), np.nan, dtype=float)
    
    weighted = (total_ret / fx_rate) / fund_nav
    return weighted * 100.0


def find_zero_crossings(x_arr, y_arr):
    zeros = []
    for i in range(len(y_arr)-1):
        y1 = y_arr[i]; y2 = y_arr[i+1]
        if np.isnan(y1) or np.isnan(y2):
            continue
        if y1 == 0.0:
            zeros.append(x_arr[i])
        elif y1 * y2 < 0:
            x1 = x_arr[i]; x2 = x_arr[i+1]
            x_zero = x1 - y1 * (x2 - x1) / (y2 - y1)
            zeros.append(x_zero)
    return zeros