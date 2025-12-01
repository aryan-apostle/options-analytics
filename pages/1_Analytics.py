# ---------------------- Modules ----------------------
from helper_functions import *


# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="ACCF Analytics", layout="wide")


# ---------------------- Sidebar Parameters ----------------------
st.sidebar.header("Parameter Details")

if "option_market" not in st.session_state:
    st.session_state.option_market = "EUA"
option_market = st.sidebar.selectbox(
    "Option Market", ["CCA", "UKA", "EUA"],
    index=["CCA", "UKA", "EUA"].index(st.session_state.option_market),
)
st.session_state.option_market = option_market

try:
    df_sec = pd.read_csv("ACCF PM Model Copy.csv")
    match = df_sec[df_sec["Underlying"] == option_market]
    if not match.empty:
        spot_from_file = float(match.iloc[0]["Spot"])
        fx_from_file = float(match.iloc[0]["FX to AUD"])
        r_from_file = float(match.iloc[0]["Risk-Free Rate"])

        st.session_state.spot = spot_from_file
        st.session_state.fx_rate = fx_from_file
        st.session_state.r = r_from_file
except Exception:
    st.write("File not being read in properly")
    pass

if "spot" not in st.session_state:
    st.session_state.spot = 100
spot = st.session_state.spot

if "r" not in st.session_state:
    st.session_state.r = 0.0191
r = st.session_state.r

if "fx_rate" not in st.session_state:
    st.session_state.fx_rate = 0.56
fx_rate = st.session_state.fx_rate

if "expected_spot" not in st.session_state:
    st.session_state.expected_spot = spot + 1
expected_spot = st.sidebar.number_input(
    "Expected Spot",
    min_value=0.01,
    value=float(st.session_state.expected_spot),
    step=0.01,
    format="%.2f",
)
st.session_state.expected_spot = expected_spot

if "fund_nav" not in st.session_state:
    st.session_state.fund_nav = 31000000
fund_nav = st.sidebar.number_input(
    "Fund NAV", min_value=1, value=int(st.session_state.fund_nav), step=1
)
st.session_state.fund_nav = fund_nav

if "current_days" not in st.session_state:
    st.session_state.current_days = 180
    
if "close_days" not in st.session_state:
    st.session_state.close_days = int(st.session_state.current_days // 4)
close_days = st.sidebar.number_input(
    "Days until close", min_value=0, max_value=3650, value=st.session_state.close_days
)
st.session_state.close_days = close_days

st.sidebar.markdown("---")
st.sidebar.header("Option Details")

if "num_legs" not in st.session_state:
    st.session_state.num_legs = 6
num_legs = st.sidebar.number_input(
    "Number of legs", min_value=1, max_value=8, value=st.session_state.num_legs, step=1
)
st.session_state.num_legs = num_legs

if "legs" not in st.session_state:
    st.session_state.legs = []

MAX_LEGS = 8
for i in range(MAX_LEGS):
    default_offset = (i - (num_legs - 1) / 2) * 5.0

    if f"options_type_{i}" not in st.session_state:
        st.session_state[f"options_type_{i}"] = "Call"
    if f"options_side_{i}" not in st.session_state:
        st.session_state[f"options_side_{i}"] = "Short" if i % 2 == 0 else "Long"
    if f"options_strike_{i}" not in st.session_state:
        st.session_state[f"options_strike_{i}"] = float(100.0 + default_offset)
    if f"options_vol_{i}" not in st.session_state:
        st.session_state[f"options_vol_{i}"] = 0.25
    if f"options_entry_{i}" not in st.session_state:
        st.session_state[f"options_entry_{i}"] = float(1.0 + abs(default_offset) / 10.0)
    if f"options_qty_{i}" not in st.session_state:
        st.session_state[f"options_qty_{i}"] = 25
    if f"options_size_{i}" not in st.session_state:
        st.session_state[f"options_size_{i}"] = 1000
    if f"options_days_{i}" not in st.session_state:
        st.session_state[f"options_days_{i}"] = 90
    if f"options_delta_{i}" not in st.session_state:
        st.session_state[f"options_delta_{i}"] = 0.5

legs = []
MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

for i in range(int(num_legs)):
    default_offset = (i - (num_legs - 1) / 2) * 5.0

    with st.sidebar.expander(f"Leg {i + 1}"):
        opt_code = st.selectbox(f"Opt Type {i + 1}", ["C", "P"], key=f"options_opt_{i}")
        month = st.selectbox(
            f"Expiry Month {i + 1}", MONTHS, index=0, key=f"options_month_{i}"
        )
        year_sel = st.selectbox(
            f"Expiry Year {i + 1}", ["2025", "2026"], index=0, key=f"options_year_{i}"
        )
        l_strike = st.number_input(f"Strike {i + 1}", key=f"options_strike_{i}", format="%.2f")
        l_qty = st.number_input(f"Quantity {i + 1}", min_value=1, key=f"options_qty_{i}", step=1)

        l_vol_val = float(st.session_state.get(f"options_vol_{i}", 0.25))
        l_entry_val = float(st.session_state.get(f"options_entry_{i}", 1.0))
        l_days_val = int(st.session_state.get(f"options_days_{i}", 90))
        l_delta_val = float(st.session_state.get(f"options_delta_{i}", 0.5))

        try:
            df_options_source = match.copy() if "match" in locals() else pd.DataFrame()
            if not df_options_source.empty:
                opt_type_col = (
                    df_options_source.get("Opt Type")
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )
                month_col = df_options_source.get("Month").astype(str).str.strip()
                year_col = (
                    df_options_source.get("Year").astype(str).str.strip().str[-2:]
                )
                strike_col = pd.to_numeric(df_options_source.get("Strike"), errors="coerce")

                opt_code_norm = str(opt_code).strip().upper()[0]
                month_norm = str(month).strip()
                year_norm = str(year_sel).strip()[-2:]
                strike_val = float(l_strike)

                mask = (
                    opt_type_col.str.startswith(opt_code_norm)
                ) & (month_col == month_norm) & (year_col == year_norm) & (
                    strike_col.notna()
                    & np.isclose(strike_col.values, strike_val, atol=1e-6)
                )

                matched_rows = df_options_source[mask]

                if not matched_rows.empty:
                    row = matched_rows.iloc[0]
                    if "IVOL" in row.index and pd.notna(row["IVOL"]):
                        l_vol_val = float(row["IVOL"])
                    if "Opt Price" in row.index and pd.notna(row["Opt Price"]):
                        l_entry_val = float(row["Opt Price"])
                    if "Days to Expiry" in row.index and pd.notna(
                        row["Days to Expiry"]
                    ):
                        l_days_val = int(row["Days to Expiry"])
                    if "Delta" in row.index and pd.notna(row["Delta"]):
                        l_delta_val = float(row["Delta"])
                    try:
                        T_years = float(l_days_val) / 365.0
                        if "Notional" in row.index and pd.notna(row["Notional"]):
                            notional_spot = float(row["Notional"])
                        else:
                            notional_spot = float(spot)
                        carry = 0.0
                        if "Carry p.a." in row.index and pd.notna(row["Carry p.a."]):
                            try:
                                carry = float(row["Carry p.a."])
                            except Exception:
                                carry = 0.0
                        days_to_calc = float(l_days_val)
                        adjusted_notional = notional_spot * (1.0 + (carry * (days_to_calc / 365.0)))
                        K_adj = float(strike_val) * (1 + (float(r) * T_years))
                        opt_type_calc = "Call" if opt_code_norm == "C" else "Put"
                        l_entry_val = float(bs_price(adjusted_notional, K_adj, T_years, r, l_vol_val, opt_type_calc))
                    except Exception:
                        pass
                else:
                    st.write(f"No match for leg {i + 1}")
        except Exception as e:
            st.write(f"Error matching option row for leg {i + 1}: {e}")

        l_type = "Call" if str(opt_code).upper() == "C" else "Put"
        l_size = 1000

        legs.append(
            {
                "type": l_type,
                "side": st.selectbox(f"Side {i + 1}", ["Long", "Short"], key=f"options_side_{i}"),
                "K": float(l_strike),
                "vol": float(l_vol_val),
                "entry": float(l_entry_val),
                "qty": int(l_qty),
                "size": int(l_size),
                "days": int(l_days_val),
                "delta": float(l_delta_val),
            }
        )

if st.sidebar.button("Reset"):
    st.session_state.clear()
    st.experimental_rerun()

for leg in legs:
    leg["sign"] = 1.0 if leg["side"] == "Long" else -1.0
    leg["mult"] = leg["qty"] * leg["size"]

st.session_state.legs = legs

st.sidebar.markdown("---")
st.sidebar.header("Heatmap")

if "show_heatmap" not in st.session_state:
    st.session_state.show_heatmap = False
show_heatmap = st.sidebar.checkbox(
    "Show heatmap", value=st.session_state.show_heatmap
)
st.session_state.show_heatmap = show_heatmap

st.sidebar.markdown("---")
st.sidebar.header("Surface")

if "show_surface" not in st.session_state:
    st.session_state.show_surface = False
show_surface = st.sidebar.checkbox(
    "Show surface", value=st.session_state.show_surface
)
st.session_state.show_surface = show_surface

if len(legs) > 0:
    _default_surface_days = int(max([l.get("days", st.session_state.current_days) for l in legs]))
else:
    _default_surface_days = int(st.session_state.current_days)

if "surface_days" not in st.session_state:
    st.session_state.surface_days = _default_surface_days
surface_days = st.sidebar.number_input(
    "Surface days", min_value=1, max_value=10000, value=int(st.session_state.surface_days), step=1
)
st.session_state.surface_days = surface_days

_default_S_min = max(0.01, spot * 0.85)
_default_S_max = spot * 1.15

if "spot_min_override" not in st.session_state:
    st.session_state.spot_min_override = float(_default_S_min)
if "spot_max_override" not in st.session_state:
    st.session_state.spot_max_override = float(_default_S_max)

min_spot = st.sidebar.number_input(
    "Minimum spot", min_value=0.01, value=float(st.session_state.spot_min_override), step=0.01, format="%.2f"
)
max_spot = st.sidebar.number_input(
    "Maximum spot", min_value=0.01, value=float(st.session_state.spot_max_override), step=0.01, format="%.2f"
)
st.session_state.spot_min_override = float(min_spot)
st.session_state.spot_max_override = float(max_spot)

st.sidebar.markdown("---")
st.sidebar.header("Greeks")

if "show_greeks" not in st.session_state:
    st.session_state.show_greeks = False
show_greeks = st.sidebar.checkbox(
    "Show greeks", value=st.session_state.show_greeks
)
st.session_state.show_greeks = show_greeks

if "use_expected_spot" not in st.session_state:
    st.session_state.use_expected_spot = False
use_expected_spot = st.sidebar.checkbox(
    "Use expected spot", value=st.session_state.use_expected_spot
)
st.session_state.use_expected_spot = use_expected_spot

st.sidebar.markdown("---")
st.sidebar.header("Monte Carlo")

if "show_monte_carlo" not in st.session_state: 
    st.session_state.show_monte_carlo = False 
show_monte_carlo = st.sidebar.checkbox(
    "Show Monte Carlo", value = st.session_state.show_monte_carlo
)
st.session_state.show_monte_carlo = show_monte_carlo

if "mc_hist_years" not in st.session_state:
    st.session_state.mc_hist_years = 2
mc_hist_years = st.sidebar.number_input(
    "Years of historical data", min_value=1, max_value=2, value=int(st.session_state.mc_hist_years), step=1
)
st.session_state.mc_hist_years = mc_hist_years

if "mc_days" not in st.session_state:
    st.session_state.mc_days = 30
mc_days = st.sidebar.number_input(
    "Days to simulate", min_value=1, max_value=730, value=int(st.session_state.mc_days), step=1
)
st.session_state.mc_days = mc_days

if "mc_paths" not in st.session_state:
    st.session_state.mc_paths = 1000
mc_paths = st.sidebar.number_input(
    "Number of simulations", min_value=1, max_value=1000000, value=int(st.session_state.mc_paths), step=100
)
st.session_state.mc_paths = mc_paths

st.session_state.mc_seed = 42


# ---------------------- Market Parameters Table ----------------------
st.subheader("Market Parameters Summary")
params_df = pd.DataFrame(
    [
        {
            "Current Spot": spot,
            "Risk-Free Rate": r,
            "FX to AUD": fx_rate,
        }
    ]
)
st.table(params_df)


# ---------------------- Option Legs Table ----------------------
st.subheader("Option Legs Summary")
legs_df = pd.DataFrame(
    [
        {
            "Type": l["type"],
            "Side": l["side"],
            "Strike": l["K"],
            # "Adjusted Strike": l["K_adj"],
            "Vol": l["vol"],
            "Price": l["entry"],
            "Qty": l["qty"],
            "Lot Size": l["size"],
            "Days to Expiry": l["days"],
            "Delta": l["delta"],
        }
        for l in legs
    ]
)
st.dataframe(legs_df)


# ---------------------- Weighted Return at Expiry ----------------------
st.subheader("Weighted Return at Expiry")

default_S_min = max(0.01, spot * 0.85)
default_S_max = spot * 1.15

S_min = float(st.session_state.get("spot_min_override", default_S_min))
S_max = float(st.session_state.get("spot_max_override", default_S_max))

S_pts = 100             # reduce or increase this based on computing power
S_range = np.linspace(S_min, S_max, S_pts)

wr_close_total = np.zeros_like(S_range)
wr_exp_total = np.zeros_like(S_range)

for leg in legs:
    T_close_leg = max(
        0.0,
        (leg.get("days", st.session_state.current_days) - st.session_state.close_days)
        / 365.0,
    )
    prices_close = bs_price_vectorized(
        S_range, leg["K"], T_close_leg, r, leg["vol"], leg["type"]
    )
    prices_exp = bs_price_vectorized(S_range, leg["K"], 0.0, r, leg["vol"], leg["type"])
    sign = leg["sign"]

    wr_close_total += weighted_return_percent(
        prices_close, leg["entry"], leg["mult"], fx_rate, fund_nav, sign
    )
    wr_exp_total += weighted_return_percent(
        prices_exp, leg["entry"], leg["mult"], fx_rate, fund_nav, sign
    )

breakevens = sorted(
    set(find_zero_crossings(S_range, wr_exp_total) + find_zero_crossings(S_range, wr_close_total))
)
breakevens = [float(np.round(b, 10)) for b in breakevens]

combined = np.hstack((wr_close_total, wr_exp_total))
finite_vals = combined[~np.isnan(combined)]
y_min, y_max = (
    (-1.0, 1.0)
    if finite_vals.size == 0
    else (
        float(np.nanmin(finite_vals) - 0.05 * (np.nanmax(finite_vals) - np.nanmin(finite_vals))),
        float(np.nanmax(finite_vals) + 0.05 * (np.nanmax(finite_vals) - np.nanmin(finite_vals))),
    )
)

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=S_range,
        y=wr_close_total,
        mode="lines",
        name=f"Close in {st.session_state.close_days}d",
        hovertemplate="S %{x:.2f}<br>Weighted return: %{y:.6f}%<extra></extra>",
    )
)
fig2.add_trace(
    go.Scatter(
        x=S_range,
        y=wr_exp_total,
        mode="lines",
        name="Expiry",
        line=dict(dash="dash"),
        hovertemplate="S %{x:.2f}<br>Weighted return: %{y:.6f}%<extra></extra>",
    )
)
fig2.add_trace(
    go.Scatter(
        x=breakevens,
        y=[0] * len(breakevens),
        mode="markers",
        name="Breakeven",
        marker=dict(color="red", size=8),
        hovertemplate="Breakeven at S %{x:.2f}<extra></extra>",
    )
)
fig2.add_trace(
    go.Scatter(
        x=[spot, spot],
        y=[y_min, y_max],
        mode="lines",
        name=f"Spot = {spot:.2f}",
        line=dict(color="gray", dash="dot"),
    )
)
fig2.add_trace(
    go.Scatter(
        x=[st.session_state.expected_spot, st.session_state.expected_spot],
        y=[y_min, y_max],
        mode="lines",
        name=f"Expected Spot = {st.session_state.expected_spot:.2f}",
        line=dict(color="green", dash="dot"),
    )
)
fig2.add_shape(
    type="line", x0=S_min, x1=S_max, y0=0, y1=0, line=dict(color="black", width=1)
)
fig2.update_layout(
    xaxis_title="Spot", yaxis_title="Weighted return (% of NAV)", height=420
)
fig2.update_xaxes(range=[S_min, S_max])
fig2.update_yaxes(range=[y_min, y_max])
st.plotly_chart(fig2, use_container_width=True)


# ---------------------- Weighted Return Over Time (Current & Expected) ----------------------
if len(legs) > 0:
    overall_max_days = int(max([l.get("days", st.session_state.current_days) for l in legs]))
else:
    overall_max_days = int(st.session_state.current_days)

st.subheader("Weighted Return at Current Spot")
days_forward = np.arange(0, overall_max_days)

wr_time_total_current = np.zeros_like(days_forward, dtype=float)
for leg in legs:
    prices_t = np.array(
        [
            bs_price(
                spot,
                leg["K"],
                max(0.0, (leg.get("days", st.session_state.current_days) - int(d)) / 365.0),
                r,
                leg["vol"],
                leg["type"],
            )
            for d in days_forward
        ]
    )
    wr_time_total_current += weighted_return_percent(
        prices_t, leg["entry"], leg["mult"], fx_rate, fund_nav, leg["sign"]
    )

today = pd.to_datetime("today").normalize()
dates_str = (today + pd.to_timedelta(days_forward, unit="D")).strftime("%Y-%m-%d")

fig_time_current = go.Figure()
fig_time_current.add_trace(
    go.Scatter(
        x=days_forward,
        y=wr_time_total_current,
        mode="lines+markers",
        name="Weighted return (%NAV)",
        customdata=dates_str,
        hovertemplate="Date: %{customdata}<br>Weighted return: %{y:.6f}%<extra></extra>",
    )
)
if 0 <= st.session_state.close_days <= overall_max_days:
    finite_vals = wr_time_total_current[~np.isnan(wr_time_total_current)]
    y_min_c, y_max_c = (
        (-1.0, 1.0)
        if finite_vals.size == 0
        else (
            float(np.nanmin(finite_vals) - 0.05 * (np.nanmax(finite_vals) - np.nanmin(finite_vals))),
            float(np.nanmax(finite_vals) + 0.05 * (np.nanmax(finite_vals) - np.nanmin(finite_vals))),
        )
    )
    fig_time_current.add_shape(
        type="line",
        x0=st.session_state.close_days,
        x1=st.session_state.close_days,
        y0=y_min_c,
        y1=y_max_c,
        line=dict(color="black", dash="dash"),
    )
    fig_time_current.add_annotation(
        x=st.session_state.close_days,
        y=y_max_c,
        text=f"Close in {st.session_state.close_days}d",
        showarrow=False,
        yanchor="bottom",
    )

fig_time_current.update_layout(
    xaxis_title="Days", yaxis_title="Weighted Return (% of NAV)", height=420
)
fig_time_current.update_xaxes(range=[0, overall_max_days])
st.plotly_chart(fig_time_current, use_container_width=True)

st.subheader("Weighted Return at Expected Spot")
days_forward_expected = np.arange(0, overall_max_days)

wr_time_total_expected = np.zeros_like(days_forward_expected, dtype=float)
for leg in legs:
    prices_t_e = np.array(
        [
            bs_price(
                st.session_state.expected_spot,
                leg["K"],
                max(0.0, (leg.get("days", st.session_state.current_days) - int(d)) / 365.0),
                r,
                leg["vol"],
                leg["type"],
            )
            for d in days_forward_expected
        ]
    )
    wr_time_total_expected += weighted_return_percent(
        prices_t_e, leg["entry"], leg["mult"], fx_rate, fund_nav, leg["sign"]
    )

dates_str_e = (today + pd.to_timedelta(days_forward_expected, unit="D")).strftime("%Y-%m-%d")

fig_time_expected = go.Figure()
fig_time_expected.add_trace(
    go.Scatter(
        x=days_forward_expected,
        y=wr_time_total_expected,
        mode="lines+markers",
        name="Weighted return (%NAV)",
        customdata=dates_str_e,
        hovertemplate="Date: %{customdata}<br>Weighted return: %{y:.6f}%<extra></extra>",
    )
)
if 0 <= st.session_state.close_days <= overall_max_days:
    finite_vals_e = wr_time_total_expected[~np.isnan(wr_time_total_expected)]
    y_min_e, y_max_e = (
        (-1.0, 1.0)
        if finite_vals_e.size == 0
        else (
            float(np.nanmin(finite_vals_e) - 0.05 * (np.nanmax(finite_vals_e) - np.nanmin(finite_vals_e))),
            float(np.nanmax(finite_vals_e) + 0.05 * (np.nanmax(finite_vals_e) - np.nanmin(finite_vals_e))),
        )
    )
    fig_time_expected.add_shape(
        type="line",
        x0=st.session_state.close_days,
        x1=st.session_state.close_days,
        y0=y_min_e,
        y1=y_max_e,
        line=dict(color="black", dash="dash"),
    )
    fig_time_expected.add_annotation(
        x=st.session_state.close_days,
        y=y_max_e,
        text=f"Close in {st.session_state.close_days}d",
        showarrow=False,
        yanchor="bottom",
    )

fig_time_expected.update_layout(
    xaxis_title="Days", yaxis_title="Weighted Return (% of NAV)", height=420
)
fig_time_expected.update_xaxes(range=[0, overall_max_days])
st.plotly_chart(fig_time_expected, use_container_width=True)


# ---------------------- Weighted Return Heatmap ----------------------
if st.session_state.get("show_heatmap", False):
    st.subheader("Weighted Return Heatmap")

    surface_days_input = int(st.session_state.surface_days)
    max_possible_day = days_forward[-1] if len(days_forward) > 0 else surface_days_input

    if surface_days_input >= max_possible_day:
        if len(days_forward) == 0:
            days_surface = np.array([0], dtype=int)
        else:
            days_surface = days_forward
    else:
        max_day = min(surface_days_input, max_possible_day)
        days_surface = np.arange(0, max_day + 1, dtype=int)

    n_days, n_spots = len(days_surface), len(S_range)
    wr_grid = np.zeros((n_days, n_spots), dtype=float)

    for i, day in enumerate(days_surface):
        wr_day_total = np.zeros(n_spots)
        for leg in legs:
            T_leg = max(0.0, (leg.get("days", st.session_state.current_days) - int(day)) / 365.0)
            prices_day = bs_price_vectorized(S_range, leg["K"], T_leg, r, leg["vol"], leg["type"])
            wr_day_total += weighted_return_percent(
                prices_day, leg["entry"], leg["mult"], fx_rate, fund_nav, leg["sign"]
            )
        wr_grid[i, :] = wr_day_total

    halfband = 0.0005
    thr_neg = -halfband
    thr_zero = 0.0
    thr_pos = halfband

    vmin = float(np.nanmin(wr_grid))
    vmax = float(np.nanmax(wr_grid))
    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6

    if vmax <= 0.0:
        colorscale = [
            (0.0, "#8B0000"),
            (0.25, "#ff6666"),
            (0.5, "#ff9999"),
            (0.75, "#ffcccc"),
            (1.0, "#ffe6e6"),
        ]
    elif vmin >= 0.0:
        colorscale = [
            (0.0, "#e6ffea"),
            (0.25, "#b3ffcc"),
            (0.5, "#80ffa6"),
            (0.75, "#33cc33"),
            (1.0, "#006400"),
        ]
    else:
        p_neg = min(max(surface_value_normalisation(thr_neg, vmin, vmax), 0.0), 1.0)
        p_zero = min(max(surface_value_normalisation(thr_zero, vmin, vmax), 0.0), 1.0)
        p_pos = min(max(surface_value_normalisation(thr_pos, vmin, vmax), 0.0), 1.0)

        eps = 1e-4
        p_before_neg = max(0.0, p_neg - eps)
        p_after_neg = min(1.0, p_neg + eps)
        p_before_pos = max(0.0, p_pos - eps)
        p_after_pos = min(1.0, p_pos + eps)

        colorscale = [
            (0.0, "#8B0000"),           # deep negative
            (p_before_neg, "#ff6666"),  # towards light pink before white
            (p_neg, "#ffe6e6"),         # light pink at negative threshold
            (p_after_neg, "#ffffff"),   # start white band
            (p_before_pos, "#ffffff"),  # end white band
            (p_pos, "#e6ffea"),         # light green at positive threshold
            (p_after_pos, "#00b300"),   # green beyond threshold
            (1.0, "#006400"),           # deep positive
        ]

    fig_heat = go.Figure()
    fig_heat.add_trace(
        go.Heatmap(
            x=days_surface,
            y=S_range,
            z=wr_grid.T,
            hovertemplate="Day %{x}<br>Spot %{y:.2f}<br>Weighted return: %{z:.6f}%<extra></extra>",
            showscale=False,
            colorscale=colorscale,
            zmin=vmin,
            zmax=vmax,
        )
    )

    if 0 <= st.session_state.close_days <= st.session_state.current_days:
        fig_heat.add_shape(
            type="line",
            x0=st.session_state.close_days,
            x1=st.session_state.close_days,
            y0=S_min,
            y1=S_max,
            line=dict(color="black", dash="dash"),
            xref="x",
            yref="y",
        )
        fig_heat.add_annotation(
            x=st.session_state.close_days,
            y=S_max,
            text=f"Close in {st.session_state.close_days}d",
            showarrow=False,
            yanchor="bottom",
        )

    fig_heat.update_layout(
        xaxis_title="Days",
        yaxis_title="Spot",
        height=700,
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ---------------------- Weighted Return Surface ----------------------
if st.session_state.get("show_surface", False):
    st.subheader("Weighted Return Surface")

    surface_days_input = int(st.session_state.surface_days)
    max_possible_day = days_forward[-1] if len(days_forward) > 0 else surface_days_input

    if surface_days_input >= max_possible_day:
        if len(days_forward) == 0:
            days_surface = np.array([0], dtype=int)
        else:
            days_surface = days_forward
    else:
        max_day = min(surface_days_input, max_possible_day)
        days_surface = np.arange(0, max_day + 1, dtype=int)

    n_days, n_spots = len(days_surface), len(S_range)
    wr_grid = np.zeros((n_days, n_spots), dtype=float)

    for i, day in enumerate(days_surface):
        wr_day_total = np.zeros(n_spots)
        for leg in legs:
            T_leg = max(0.0, (leg.get("days", st.session_state.current_days) - int(day)) / 365.0)
            prices_day = bs_price_vectorized(S_range, leg["K"], T_leg, r, leg["vol"], leg["type"])
            wr_day_total += weighted_return_percent(
                prices_day, leg["entry"], leg["mult"], fx_rate, fund_nav, leg["sign"]
            )
        wr_grid[i, :] = wr_day_total

    halfband = 0.0005
    thr_neg = -halfband
    thr_zero = 0.0
    thr_pos = halfband

    vmin = float(np.nanmin(wr_grid))
    vmax = float(np.nanmax(wr_grid))
    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6

    if vmax <= 0.0:
        colorscale = [
            (0.0, "#8B0000"),
            (0.25, "#ff6666"),
            (0.5, "#ff9999"),
            (0.75, "#ffcccc"),
            (1.0, "#ffe6e6"),
        ]
    elif vmin >= 0.0:
        colorscale = [
            (0.0, "#e6ffea"),
            (0.25, "#b3ffcc"),
            (0.5, "#80ffa6"),
            (0.75, "#33cc33"),
            (1.0, "#006400"),
        ]
    else:
        p_neg = min(max(surface_value_normalisation(thr_neg, vmin, vmax), 0.0), 1.0)
        p_zero = min(max(surface_value_normalisation(thr_zero, vmin, vmax), 0.0), 1.0)
        p_pos = min(max(surface_value_normalisation(thr_pos, vmin, vmax), 0.0), 1.0)

        eps = 1e-4
        p_before_neg = max(0.0, p_neg - eps)
        p_after_neg = min(1.0, p_neg + eps)
        p_before_pos = max(0.0, p_pos - eps)
        p_after_pos = min(1.0, p_pos + eps)

        colorscale = [
            (0.0, "#8B0000"),           # deep negative
            (p_before_neg, "#ff6666"),  # towards light pink before white
            (p_neg, "#ffe6e6"),         # light pink at negative threshold
            (p_after_neg, "#ffffff"),   # start white band
            (p_before_pos, "#ffffff"),  # end white band
            (p_pos, "#e6ffea"),         # light green at positive threshold
            (p_after_pos, "#00b300"),   # green beyond threshold
            (1.0, "#006400"),           # deep positive
        ]

    fig_3d = go.Figure()
    fig_3d.add_trace(
        go.Surface(
            x=days_surface,
            y=S_range,
            z=wr_grid.T,
            hovertemplate="Day %{x}<br>Spot %{y:.2f}<br>Weighted return: %{z:.6f}%<extra></extra>",
            showscale=False,
            colorscale=colorscale,
            cmin=vmin,
            cmax=vmax,
        )
    )

    if 0 <= st.session_state.close_days <= st.session_state.current_days:
        close_idx = int(np.argmin(np.abs(days_surface - st.session_state.close_days)))
        fig_3d.add_trace(
            go.Scatter3d(
                x=[st.session_state.close_days] * n_spots,
                y=S_range,
                z=wr_grid[close_idx, :],
                mode="lines",
                line=dict(color="blue", width=2, dash="dash"),
                name=f"Close in {st.session_state.close_days}d",
            )
        )

    fig_3d.update_layout(
        scene=dict(
            xaxis_title="Days", yaxis_title="Spot", zaxis_title="Weighted return (% of NAV)"
        ),
        height=700,
    )
    st.plotly_chart(fig_3d, use_container_width=True)


# ---------------------- Greeks ----------------------
if st.session_state.get("show_greeks", True):
    spot_for_greeks = (
        st.session_state.expected_spot
        if st.session_state.get("use_expected_spot", False)
        else st.session_state.get("spot", 110.0)
    )

    r = st.session_state.get("r", 0.0191)
    fx_rate = st.session_state.get("fx_rate", 0.56)
    fund_nav = st.session_state.get("fund_nav", 31000000.0)
    current_days = st.session_state.get("current_days", 180)
    close_days = st.session_state.get("close_days", int(current_days // 4))
    num_legs = st.session_state.get("num_legs", 6)
    legs = st.session_state.get("legs", [])

    if len(legs) > 0:
        max_days_greeks = int(max([l.get("days", current_days) for l in legs]))
    else:
        max_days_greeks = int(current_days)

    days_forward = np.arange(0, max_days_greeks)

    today = pd.to_datetime("today").normalize()
    dates_greeks = (today + pd.to_timedelta(days_forward, unit="D")).strftime("%Y-%m-%d")

    greeks_total = {
        "delta": np.zeros_like(days_forward, dtype=float),
        "gamma": np.zeros_like(days_forward, dtype=float),
        "vega": np.zeros_like(days_forward, dtype=float),
        "theta": np.zeros_like(days_forward, dtype=float),
        "rho": np.zeros_like(days_forward, dtype=float),
    }

    for i, day in enumerate(days_forward):
        greek_day_total = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        for leg in legs:
            T_leg = max(0.0, (leg.get("days", current_days) - int(day)) / 365.0)
            g = bs_greeks(spot_for_greeks, leg["K"], T_leg, r, leg["vol"], leg["type"])

            for k in greek_day_total:
                greek_day_total[k] += (
                    g[k]
                    * leg.get("mult", leg.get("qty", 0) * leg.get("size", 0))
                    * leg.get("sign", (1.0 if leg.get("side", "Long") == "Long" else -1.0))
                )

        for k in greeks_total:
            greeks_total[k][i] = greek_day_total[k]

    colors = {
        "delta": "blue",
        "gamma": "red",
        "vega": "green",
        "theta": "orange",
        "rho": "purple",
    }

    for k in greeks_total:
        st.subheader(f"{k.capitalize()}")
        fig_greek = go.Figure()
        fig_greek.add_trace(
            go.Scatter(
                x=days_forward,
                y=greeks_total[k],
                mode="lines+markers",
                name=k.capitalize(),
                line=dict(color=colors[k]),
                customdata=dates_greeks,
                hovertemplate=f"Date: %{{customdata}}<br>{k.capitalize()}: %{{y:.6f}}<extra></extra>",
            )
        )

        try:
            if 0 <= close_days < max_days_greeks:
                finite_vals = greeks_total[k][~np.isnan(greeks_total[k])]
                if finite_vals.size == 0:
                    y_min_k, y_max_k = -1.0, 1.0
                else:
                    y_min_k = float(np.nanmin(finite_vals))
                    y_max_k = float(np.nanmax(finite_vals))
                    span_k = max(0.0001, y_max_k - y_min_k)
                    y_min_k -= 0.05 * span_k
                    y_max_k += 0.05 * span_k

                fig_greek.add_shape(
                    type="line",
                    x0=close_days,
                    x1=close_days,
                    y0=y_min_k,
                    y1=y_max_k,
                    line=dict(color="black", dash="dash"),
                    xref="x",
                    yref="y",
                )
                fig_greek.add_annotation(
                    x=close_days,
                    y=y_max_k,
                    text=f"Close in {close_days}d",
                    showarrow=False,
                    yanchor="bottom",
                )
        except Exception:
            pass

        fig_greek.update_layout(xaxis_title="Days", yaxis_title=f"{k.capitalize()}", height=420)
        st.plotly_chart(fig_greek, use_container_width=True)
        
        
# ---------------------- Monte Carlo ----------------------
if st.session_state.get("show_monte_carlo", False):
    st.subheader("Monte Carlo Price Paths")

    price_data = pd.read_csv("Historical Spot Price.csv")
    price_data = price_data.sort_index()

    if st.session_state.option_market == 'EUA': 
        price_series = price_data['EUA FUTURES - EUA - DEC25']
    if st.session_state.option_market == 'UKA':
        price_series = price_data['UKA FUTURES - UKA - DEC25']
    if st.session_state.option_market == 'CCA':
        price_series = price_data['CCA FUTURES - CCA V25 - DEC25']

    hist_days = int(
        max(10, min(len(price_series) - 1, int(mc_hist_years * 252)))
    )
    hist_series = price_series.iloc[-(hist_days + 1) :]
    returns = np.log(hist_series / hist_series.shift(1)).dropna().values

    mu_d = float(np.nanmean(returns))
    sigma_d = float(np.nanstd(returns, ddof=1))

    mu_annual = mu_d * 252.0
    sigma_annual = sigma_d * np.sqrt(252.0)

    mu_calendar_daily = mu_annual / 365.0
    sigma_calendar_daily = sigma_annual / np.sqrt(365.0)

    rng = np.random.default_rng(st.session_state.mc_seed)

    zs = (returns - mu_d) / (sigma_d if sigma_d != 0.0 else 1.0)

    idxs = rng.integers(0, len(zs), size=(int(mc_paths), int(mc_days)))
    z_samples = zs[idxs]

    increments = mu_calendar_daily + sigma_calendar_daily * z_samples

    log_paths = np.cumsum(increments, axis=1)
    s0 = float(spot)
    s_paths = s0 * np.exp(log_paths)

    percentiles = [1, 10, 25, 50, 75, 90, 99]
    pct_prices = np.percentile(s_paths, percentiles, axis=0)
    mean_path = np.mean(s_paths, axis=0)

    days_idx = np.arange(0, mc_days + 1)
    dates_mc = (today + pd.to_timedelta(days_idx, unit="D")).strftime("%Y-%m-%d")

    pct_with_zero = np.zeros((len(percentiles), mc_days + 1), dtype=float)
    pct_with_zero[:, 0] = s0
    pct_with_zero[:, 1:] = pct_prices

    mean_with_zero = np.concatenate(([s0], mean_path))

    fig_mc = go.Figure()

    for i, p in enumerate(percentiles):
        name = f"{p}th percentile"
        fig_mc.add_trace(
            go.Scatter(
                x=days_idx,
                y=pct_with_zero[i, :],
                mode="lines",
                name=name,
                line=dict(dash="dot", width=2),
                hovertemplate="Date: %{x}<br>Price: %{y:.4f}<extra></extra>",
            )
        )

    fig_mc.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[np.min(s_paths), np.max(s_paths)],
            mode="lines",
            name=f"Spot = {s0:.2f}",
            line=dict(color="gray", dash="dot"),
        )
    )

    fig_mc.update_layout(
        xaxis_title="Days",
        yaxis_title="Price",
        height=520,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    st.subheader("Monte Carlo Densities")
    final_prices = np.array(s_paths[:, -1])
    bins = 80
    hist_vals, bin_edges = np.histogram(final_prices, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    fig_dist = go.Figure()
    fig_dist.add_trace(
        go.Scatter(
            x=bin_centers,
            y=hist_vals,
            mode="lines",
            name="Density",
            fill="tozeroy",
            hovertemplate="Price: %{x:.4f}<br>Density: %{y:.6f}<extra></extra>",
        )
    )
    
    fig_dist.update_layout(
        xaxis_title="Price",
        yaxis_title="Density",
        height=300,
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig_dist, use_container_width=True)