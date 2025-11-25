# ---------------------- Modules ----------------------
from helper_functions import *


# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="ACCF Trade Ideas", layout="wide")


# ---------------------- Parameters ----------------------
st.title("EUA Automated Trade Ideas")

try:
    df_sec = pd.read_csv("ACCF PM Model Copy.csv")
    df_market = df_sec[df_sec["Underlying"] == "EUA"].copy()
    
    if df_market.empty:
        st.error("No EUA data found in the CSV.")
        st.stop()

    spot_from_file = float(df_market.iloc[0]["Spot"])
    fx_from_file = float(df_market.iloc[0]["FX to AUD"])
    r_from_file = float(df_market.iloc[0]["Risk-Free Rate"])

    st.session_state.spot = spot_from_file
    st.session_state.fx_rate = fx_from_file
    st.session_state.r = r_from_file

except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

spot = st.session_state.spot
fx_rate = st.session_state.fx_rate
r = st.session_state.r
fund_nav = st.session_state.get("fund_nav", 31000000)

st.metric("EUA Reference Spot", f"€{spot:.2f}")


# ---------------------- Next Quarter Scanner ----------------------
today = datetime.now()
curr_month = today.month
curr_year = today.year

if curr_month < 3:
    target_m_num = 3
    target_year = curr_year
elif curr_month < 6:
    target_m_num = 6
    target_year = curr_year
elif curr_month < 9:
    target_m_num = 9
    target_year = curr_year
elif curr_month < 12:
    target_m_num = 12
    target_year = curr_year
else:
    # If it is Dec (12), look for Mar (3) of next year
    target_m_num = 3
    target_year = curr_year + 1

month_map = {3: "Mar", 6: "Jun", 9: "Sep", 12: "Dec"}
target_month_str = month_map[target_m_num]
target_year_str = str(target_year)

st.write(f"**Scanning for Next Quarter Expiry:** {target_month_str} {target_year_str}")

# Filter the dataframe for this specific Year first (to differentiate Mar 25 vs Mar 26) - match the last 2 digits of the year to be safe (e.g. '2026' matches '26')
df_expiry_year = df_market[
    df_market["Year"].astype(str).str.strip().str[-2:] == target_year_str[-2:]
].copy()


# ---------------------- Logic: Find Options by Delta ----------------------

def get_option_by_delta(df, target_delta_abs, opt_type, expiry_filter=None, tolerance=0.05):
    """
    Scans the dataframe for an option nearest to the target absolute delta.
    """
    temp_df = df.copy()
    
    type_char = "C" if opt_type == "Call" else "P"
    temp_df = temp_df[temp_df["Opt Type"].str.strip().str.upper().str.startswith(type_char)]

    if expiry_filter:
        temp_df = temp_df[temp_df["Month"] == expiry_filter]

    temp_df["AbsDelta"] = pd.to_numeric(temp_df["Delta"], errors='coerce').abs()
    
    mask = (temp_df["AbsDelta"] >= (target_delta_abs - tolerance)) & \
           (temp_df["AbsDelta"] <= (target_delta_abs + tolerance))
    
    candidates = temp_df[mask]
    
    if candidates.empty:
        return None
    
    candidates["dist"] = abs(candidates["AbsDelta"] - target_delta_abs)
    best_match = candidates.sort_values("dist").iloc[0]
    
    return best_match


# ---------------------- Strategy Builder ----------------------
strategies = []

qty_per_leg = 25
lot_size = 1000
S_min = spot * 0.85
S_max = spot * 1.15
S_range = np.linspace(S_min, S_max, 100)

# --- 1. Bear Put Spread ---
# Buy 0.45 Delta Put, Sell 0.20 Delta Put
long_leg_bp = get_option_by_delta(df_expiry_year, 0.45, "Put", target_month_str)
short_leg_bp = get_option_by_delta(df_expiry_year, 0.20, "Put", target_month_str)

if long_leg_bp is not None and short_leg_bp is not None:
    strategies.append({
        "name": "Bear Put Spread",
        "desc": f"Long {long_leg_bp['Strike']} Put / Short {short_leg_bp['Strike']} Put ({target_month_str} {target_year_str})",
        "legs": [
            {"side": "Long", "row": long_leg_bp, "sign": 1.0},
            {"side": "Short", "row": short_leg_bp, "sign": -1.0}
        ]
    })
else:
    st.warning(f"Could not find matching options for Bear Put Spread in {target_month_str} {target_year_str}")

# --- 2. Bull Call Spread ---
# Buy 0.45 Delta Call, Sell 0.20 Delta Call
long_leg_bc = get_option_by_delta(df_expiry_year, 0.45, "Call", target_month_str)
short_leg_bc = get_option_by_delta(df_expiry_year, 0.20, "Call", target_month_str)

if long_leg_bc is not None and short_leg_bc is not None:
    strategies.append({
        "name": "Bull Call Spread",
        "desc": f"Long {long_leg_bc['Strike']} Call / Short {short_leg_bc['Strike']} Call ({target_month_str} {target_year_str})",
        "legs": [
            {"side": "Long", "row": long_leg_bc, "sign": 1.0},
            {"side": "Short", "row": short_leg_bc, "sign": -1.0}
        ]
    })
else:
    st.warning(f"Could not find matching options for Bull Call Spread in {target_month_str} {target_year_str}")


# ---------------------- Weighted Return at Expiry ----------------------
for strat in strategies:
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader(strat["name"])
        st.caption(strat["desc"])

        leg_data = []
        for l in strat["legs"]:
            row = l['row']
            leg_data.append({
                "Side": l['side'],
                "Strike": row['Strike'],
                "Type": row['Opt Type'],
                "Delta": row['Delta'],
                "Price": row['Opt Price'],
                "IVOL": row['IVOL']
            })
        st.table(pd.DataFrame(leg_data))
        
        net_premium = (leg_data[0]['Price'] - leg_data[1]['Price'])
        st.info(f"Est. Net Debit: €{net_premium:.2f}")

    with c2:
        wr_exp_total = np.zeros_like(S_range)
        wr_current_total = np.zeros_like(S_range)
        
        for leg_info in strat["legs"]:
            row = leg_info["row"]
            sign = leg_info["sign"]
            
            K = float(row["Strike"])
            vol = float(row["IVOL"])
            days = float(row["Days to Expiry"])
            entry_price = float(row["Opt Price"])
            opt_type = "Call" if str(row["Opt Type"]).upper().startswith("C") else "Put"
            
            T_exp = days / 365.0
            
            prices_exp = bs_price_vectorized(S_range, K, 0.0, r, vol, opt_type)
            
            prices_curr = bs_price_vectorized(S_range, K, T_exp, r, vol, opt_type)

            mult = qty_per_leg * lot_size
            
            wr_exp_total += weighted_return_percent(
                prices_exp, entry_price, mult, fx_rate, fund_nav, sign
            )
            wr_current_total += weighted_return_percent(
                prices_curr, entry_price, mult, fx_rate, fund_nav, sign
            )

        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=S_range, y=wr_current_total,
            mode='lines', name='Current (T-0)',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=S_range, y=wr_exp_total,
            mode='lines', name='At Expiry',
            line=dict(color='gray', dash='dash')
        ))
        
        fig.add_vline(x=spot, line_dash="dot", line_color="black", annotation_text="Spot")
        fig.add_hline(y=0, line_color="black", line_width=1)

        fig.update_layout(
            title=f"{strat['name']} - Weighted Return vs Spot",
            xaxis_title="EUA Spot Price",
            yaxis_title="Weighted Return (% NAV)",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)