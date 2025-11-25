# ---------------------- Modules ----------------------
from helper_functions import *


# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="ACCF Trade Ideas", layout="wide")


# ---------------------- Parameters ----------------------
st.title("EUA Trade Ideas")

HISTORY_FOLDER = Path("Trade Ideas History")

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

st.metric("EUA Spot", f"€{spot:.2f}")


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
    target_m_num = 3
    target_year = curr_year + 1

month_map = {3: "Mar", 6: "Jun", 9: "Sep", 12: "Dec"}
target_month_str = month_map[target_m_num]
target_year_str = str(target_year)
current_date_str = today.strftime("%Y-%m-%d")

st.write(f"**Scanning for Next Quarter Expiry:** {target_month_str} {target_year_str}")

df_expiry_year = df_market[
    df_market["Year"].astype(str).str.strip().str[-2:] == target_year_str[-2:]
].copy()


# ---------------------- Strategy Builder ----------------------
strategies = []

qty_per_leg = 25
lot_size = 1000
S_min = spot * 0.85
S_max = spot * 1.15
S_range = np.linspace(S_min, S_max, 100)

# --- 1. Bear Put Spread ---
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

# --- 3. Iron Condor ---
long_leg_p = get_option_by_delta(df_expiry_year, 0.20, "Put", target_month_str)
short_leg_p = get_option_by_delta(df_expiry_year, 0.30, "Put", target_month_str)
long_leg_c = get_option_by_delta(df_expiry_year, 0.20, "Call", target_month_str)
short_leg_c = get_option_by_delta(df_expiry_year, 0.30, "Call", target_month_str)

if long_leg_p is not None and short_leg_p is not None and long_leg_c is not None and short_leg_c is not None:
    strategies.append({
        "name": "Iron Condor",
        "desc": f"Long {long_leg_p['Strike']} Put + Long {long_leg_c['Strike']} Call / Short {short_leg_p['Strike']} Put + Short {short_leg_c['Strike']} Call ({target_month_str} {target_year_str})",
        "legs": [
            {"side": "Long", "row": long_leg_p, "sign": 1.0},
            {"side": "Short", "row": short_leg_p, "sign": -1.0},
            {"side": "Long", "row": long_leg_c, "sign": 1.0},
            {"side": "Short", "row": short_leg_c, "sign": -1.0},
        ]
    })
else:
    st.warning(f"Could not find matching options for Iron Condor in {target_month_str} {target_year_str}")


# ---------------------- Weighted Return at Expiry ----------------------
for strat in strategies:
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader(strat["name"])
        st.caption(strat["desc"])

        leg_data_display = []
        total_cost = 0.0
        
        for l in strat["legs"]:
            row = l['row']
            price = float(row['Opt Price'])
            total_cost += price if l['side'] == "Long" else -price
            
            leg_data_display.append({
                "Side": l['side'],
                "Strike": row['Strike'],
                "Type": row['Opt Type'],
                "Delta": row['Delta'],
                "Price": row['Opt Price'],
                "IVOL": row['IVOL']
            })
        st.table(pd.DataFrame(leg_data_display))
        
        cost_label = "Net Debit" if total_cost > 0 else "Net Credit"
        st.info(f"{cost_label}: €{abs(total_cost):.2f}")

        st.markdown("##### Return Scenarios (At Expiry)")
        
        scenarios_pct = np.array([-0.15, -0.10, -0.05, 0.05, 0.10, 0.15])
        scenario_spots = spot * (1 + scenarios_pct)
        scenario_returns = np.zeros_like(scenario_spots)

        for leg_info in strat["legs"]:
            row = leg_info["row"]
            sign = leg_info["sign"]
            
            K = float(row["Strike"])
            vol = float(row["IVOL"])
            entry_price = float(row["Opt Price"])
            opt_type = "Call" if str(row["Opt Type"]).upper().startswith("C") else "Put"
            
            mult = qty_per_leg * lot_size

            prices_scen = bs_price_vectorized(scenario_spots, K, 0.0, r, vol, opt_type)
            
            scenario_returns += weighted_return_percent(
                prices_scen, entry_price, mult, fx_rate, fund_nav, sign
            )

        df_scenarios = pd.DataFrame({
            "Move": [f"{p*100:+.0f}%" for p in scenarios_pct],
            "Spot (€)": scenario_spots,
            "Net Return (% NAV)": scenario_returns
        })
        
        st.dataframe(
            df_scenarios.style.format({
                "Spot (€)": "{:.2f}",
                "Net Return (% NAV)": "{:.4f}%"
            }),
            hide_index=True,
            use_container_width=True
        )

        try:
            saved_file = save_strategy_data(
                strat_name=strat["name"], 
                current_spot=spot, 
                date_str=current_date_str, 
                leg_data=strat["legs"], 
                df_scenarios=df_scenarios, 
                total_cost=total_cost,
                fx_rate=fx_rate,
                r_rate=r,
                qty_per_leg=qty_per_leg,
                lot_size=lot_size,
                history_folder=HISTORY_FOLDER
            )
            st.success(f"Data for **{current_date_str}** saved successfully to: `{saved_file}`")
        except Exception as e:
            st.error(f"Error saving data locally: {e}")

        st.markdown("---")
        st.markdown("##### 🕰️ Compare to Historic Data")
        
        compare_date_str = st.date_input(
            f"Select Historic Date for {strat['name']}",
            value=None,
            min_value=datetime(2023, 1, 1),
            max_value=today.date(),
            key=f"date_input_{strat['name']}"
        )
        
        if compare_date_str:
            historic_date_str = compare_date_str.strftime("%Y-%m-%d")
            historic_filename = HISTORY_FOLDER / f"{strat['name'].replace(' ', '_')}_{historic_date_str}.csv"
            
            if historic_filename.exists():
                try:
                    df_historic = pd.read_csv(historic_filename)
                    
                    st.success(f"Loaded historic data from {historic_date_str}!")
                    
                    historic_spot = df_historic['EUA_Spot'].iloc[0]
                    
                    comparison_summary = pd.DataFrame({
                        'Metric': ['EUA Spot', f'{cost_label} (€)'],
                        'Current': [f"€{spot:.2f}", f"€{abs(total_cost):.2f}"],
                        f'Historic ({historic_date_str})': [f"€{historic_spot:.2f}", f"€{df_historic['Net_Cost_EUR'].abs().iloc[0]:.2f}"],
                        'Difference': [f"€{spot - historic_spot:.2f}", f"€{abs(total_cost) - df_historic['Net_Cost_EUR'].abs().iloc[0]:.2f}"]
                    })
                    st.table(comparison_summary.set_index('Metric'))
                    
                    df_current_returns = df_scenarios.set_index('Move')['Net Return (% NAV)']
                    df_historic_returns = df_historic.groupby('Move')['Net Return (% NAV)'].first()
                    
                    df_return_comp = pd.DataFrame({
                        'Move': df_current_returns.index,
                        'Current Return': df_current_returns.values,
                        f'Historic Return ({historic_date_str})': df_historic_returns.values,
                        'Difference': df_current_returns.values - df_historic_returns.values
                    })

                    st.markdown("###### Scenario Return Comparison")
                    st.dataframe(
                        df_return_comp.style.format({
                            'Current Return': '{:.4f}%',
                            f'Historic Return ({historic_date_str})': '{:.4f}%',
                            'Difference': '{:+.4f}%'
                        }),
                        hide_index=True,
                        use_container_width=True
                    )

                except Exception as e:
                    st.error(f"Error reading historic file: {e}")
            else:
                st.warning(f"No historic data found for {historic_date_str} in the 'Trade Ideas History' folder.")

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
            height=500, 
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)