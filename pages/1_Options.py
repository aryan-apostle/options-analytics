# ---------------------- Modules ----------------------
from helper_functions import *



# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="ACCF Options", layout="wide")
st.title("ACCF Options")



# ---------------------- Sidebar Parameters ----------------------
st.sidebar.header("Parameter Details")
if 'spot' not in st.session_state:
    st.session_state.spot = 100.0
spot = st.sidebar.number_input("Spot", value=st.session_state.spot, step=0.1, format="%.2f")
st.session_state.spot = spot

if 'r' not in st.session_state:
    st.session_state.r = 0.0191
r = st.sidebar.number_input("Rate", value=st.session_state.r, step=0.001, format="%.4f")
st.session_state.r = r

if 'fx_rate' not in st.session_state:
    st.session_state.fx_rate = 0.56
fx_rate = st.sidebar.number_input("FX to AUD", value=st.session_state.fx_rate, step=0.01, format="%.4f")
st.session_state.fx_rate = fx_rate

if 'fund_nav' not in st.session_state:
    st.session_state.fund_nav = 28560000.0
fund_nav = st.sidebar.number_input("Fund NAV (base currency)", value=st.session_state.fund_nav, step=100000.0, format="%.2f")
st.session_state.fund_nav = fund_nav

st.sidebar.markdown("---")
st.sidebar.header("Time Details")
if 'current_days' not in st.session_state:
    st.session_state.current_days = 180
current_days = st.sidebar.number_input("Days to expiry", min_value=0, max_value=3650, value=st.session_state.current_days)
st.session_state.current_days = current_days

if 'close_days' not in st.session_state:
    st.session_state.close_days = int(current_days // 4)
close_days = st.sidebar.number_input("Days until close", min_value=0, max_value=int(current_days), value=st.session_state.close_days)
st.session_state.close_days = close_days

T_today = current_days / 365.0
T_close = close_days / 365.0

st.sidebar.markdown("---")
st.sidebar.header("Option Details")
if 'num_legs' not in st.session_state:
    st.session_state.num_legs = 4
num_legs = st.sidebar.number_input("Number of legs", min_value=1, max_value=8, value=st.session_state.num_legs, step=1)
st.session_state.num_legs = num_legs



# ---------------------- Option Legs Input ----------------------
if 'legs' not in st.session_state:
    st.session_state.legs = []

MAX_LEGS = 8
for i in range(MAX_LEGS):
    default_offset = (i - (num_legs-1)/2) * 5.0
    if f'options_type_{i}' not in st.session_state:
        st.session_state[f'options_type_{i}'] = 'Call'
    if f'options_side_{i}' not in st.session_state:
        st.session_state[f'options_side_{i}'] = 'Short' if i % 2 == 0 else 'Long'
    if f'options_strike_{i}' not in st.session_state:
        st.session_state[f'options_strike_{i}'] = float(100.0 + default_offset)
    if f'options_vol_{i}' not in st.session_state:
        st.session_state[f'options_vol_{i}'] = 0.25
    if f'options_entry_{i}' not in st.session_state:
        st.session_state[f'options_entry_{i}'] = float(1.0 + abs(default_offset)/10.0)
    if f'options_qty_{i}' not in st.session_state:
        st.session_state[f'options_qty_{i}'] = 25
    if f'options_size_{i}' not in st.session_state:
        st.session_state[f'options_size_{i}'] = 1000

legs = []
for i in range(int(num_legs)):
    default_offset = (i - (num_legs-1)/2) * 5.0
    with st.sidebar.expander(f"Leg {i+1}"):
        l_type = st.selectbox(f"Type {i+1}", ['Call','Put'], key=f'options_type_{i}')
        l_side = st.selectbox(f"Side {i+1}", ['Long','Short'], key=f'options_side_{i}')
        l_strike = st.number_input(f"Strike {i+1}", value=st.session_state[f'options_strike_{i}'], key=f'options_strike_{i}', format="%.2f")
        l_vol = st.number_input(f"Vol (σ) {i+1}", min_value=0.0001, max_value=5.0, value=st.session_state[f'options_vol_{i}'], step=0.01, format="%.4f", key=f'options_vol_{i}')
        l_entry = st.number_input(f"Entry premium {i+1}", value=st.session_state[f'options_entry_{i}'], format="%.4f", key=f'options_entry_{i}')
        l_qty = st.number_input(f"Quantity (contracts) {i+1}", min_value=1, value=st.session_state[f'options_qty_{i}'], step=1, key=f'options_qty_{i}')
        l_size = st.number_input(f"Lot size (per contract) {i+1}", min_value=1, value=st.session_state[f'options_size_{i}'], step=1, key=f'options_size_{i}')
        legs.append({
            'type': l_type, 'side': l_side, 'K': float(l_strike),
            'vol': float(l_vol), 'entry': float(l_entry),
            'qty': int(l_qty), 'size': int(l_size)
        })

if st.sidebar.button("Reset"):
    st.session_state.clear()
    st.experimental_rerun()

for leg in legs:
    leg['sign'] = 1.0 if leg['side'] == 'Long' else -1.0
    leg['mult'] = leg['qty'] * leg['size']

st.session_state.legs = legs



# ---------------------- Option Legs Table ----------------------
st.subheader("Option Legs Summary")
legs_df = pd.DataFrame([{
    'Type': l['type'], 'Side': l['side'], 'Strike': l['K'], 'Vol': l['vol'], 
    'Price': l['entry'], 'Qty': l['qty'], 'Lot Size': l['size']
} for l in legs])
st.dataframe(legs_df)



# ---------------------- Weighted Return at Expiry ----------------------
st.subheader("Weighted Return at Expiry")
S_min = max(0.01, spot * 0.5)
S_max = spot * 1.5 + 1.0
S_range = np.linspace(S_min, S_max, 400)

wr_close_total = np.zeros_like(S_range)
wr_exp_total = np.zeros_like(S_range)

for leg in legs:
    prices_close = bs_price_vectorized(S_range, leg['K'], T_close, r, leg['vol'], leg['type'])
    prices_exp = bs_price_vectorized(S_range, leg['K'], 0.0, r, leg['vol'], leg['type'])
    sign = leg['sign']
    wr_close_total += weighted_return_percent(prices_close, leg['entry'], leg['mult'], fx_rate, fund_nav, sign)
    wr_exp_total += weighted_return_percent(prices_exp, leg['entry'], leg['mult'], fx_rate, fund_nav, sign)

breakevens = sorted(set(find_zero_crossings(S_range, wr_exp_total) + find_zero_crossings(S_range, wr_close_total)))
breakevens = [float(np.round(b, 10)) for b in breakevens]

combined = np.hstack((wr_close_total, wr_exp_total))
finite_vals = combined[~np.isnan(combined)]
y_min, y_max = (-1.0, 1.0) if finite_vals.size==0 else (float(np.nanmin(finite_vals)-0.05*(np.nanmax(finite_vals)-np.nanmin(finite_vals))),
                                                         float(np.nanmax(finite_vals)+0.05*(np.nanmax(finite_vals)-np.nanmin(finite_vals))))

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=S_range, y=wr_close_total, mode='lines', name=f'Close in {close_days}d',
                          hovertemplate='S %{x:.2f}<br>Weighted return: %{y:.6f}%<extra></extra>'))
fig2.add_trace(go.Scatter(x=S_range, y=wr_exp_total, mode='lines', name='Expiry', line=dict(dash='dash'),
                          hovertemplate='S %{x:.2f}<br>Weighted return: %{y:.6f}%<extra></extra>'))
fig2.add_trace(go.Scatter(x=breakevens, y=[0]*len(breakevens), mode='markers', name='Breakeven',
                          marker=dict(color='red', size=8), hovertemplate='Breakeven at S %{x:.2f}<extra></extra>'))
fig2.add_trace(go.Scatter(x=[spot, spot], y=[y_min, y_max], mode='lines', name='Spot',
                          line=dict(color='gray', dash='dot')))
fig2.add_shape(type='line', x0=S_min, x1=S_max, y0=0, y1=0, line=dict(color='black', width=1))
fig2.update_layout(xaxis_title='Spot', yaxis_title='Weighted return (% of NAV)', height=420)
fig2.update_xaxes(range=[S_min, S_max])
fig2.update_yaxes(range=[y_min, y_max])
st.plotly_chart(fig2, use_container_width=True)



# ---------------------- Weighted Return Over Time ----------------------
st.subheader("Weighted Return at Selected Spot")
days_forward = np.arange(0, int(current_days)+1)
T_grid = (current_days - days_forward)/365.0

wr_time_total = np.zeros_like(days_forward, dtype=float)
for leg in legs:
    prices_t = np.array([bs_price(spot, leg['K'], T, r, leg['vol'], leg['type']) for T in T_grid])
    wr_time_total += weighted_return_percent(prices_t, leg['entry'], leg['mult'], fx_rate, fund_nav, leg['sign'])

today = pd.to_datetime('today').normalize()
dates_str = (today + pd.to_timedelta(days_forward, unit='D')).strftime('%Y-%m-%d')

fig_time = go.Figure()
fig_time.add_trace(go.Scatter(x=days_forward, y=wr_time_total, mode='lines+markers', name='Weighted return (%NAV)',
                              customdata=dates_str, hovertemplate='Date %{customdata}<br>Weighted return: %{y:.6f}%<extra></extra>'))

if 0 <= close_days <= current_days:
    finite_vals = wr_time_total[~np.isnan(wr_time_total)]
    y_min, y_max = (-1.0,1.0) if finite_vals.size==0 else (float(np.nanmin(finite_vals)-0.05*(np.nanmax(finite_vals)-np.nanmin(finite_vals))),
                                                         float(np.nanmax(finite_vals)+0.05*(np.nanmax(finite_vals)-np.nanmin(finite_vals))))
    fig_time.add_shape(type='line', x0=close_days, x1=close_days, y0=y_min, y1=y_max, line=dict(color='black', dash='dash'))
    fig_time.add_annotation(x=close_days, y=y_max, text=f'Close in {close_days}d', showarrow=False, yanchor='bottom')

fig_time.update_layout(xaxis_title='Days', yaxis_title='Weighted Return (% of NAV)', height=420)
st.plotly_chart(fig_time, use_container_width=True)



# ---------------------- Weighted Return Surface ----------------------
st.subheader("Weighted Return Surface")
n_days, n_spots = len(days_forward), len(S_range)
wr_grid = np.zeros((n_days, n_spots), dtype=float)

for i, T in enumerate(T_grid):
    wr_day_total = np.zeros(n_spots)
    for leg in legs:
        prices_day = bs_price_vectorized(S_range, leg['K'], T, r, leg['vol'], leg['type'])
        wr_day_total += weighted_return_percent(prices_day, leg['entry'], leg['mult'], fx_rate, fund_nav, leg['sign'])
    wr_grid[i,:] = wr_day_total

fig_3d = go.Figure()
fig_3d.add_trace(go.Surface(x=days_forward, y=S_range, z=wr_grid.T,
                            hovertemplate='Day %{x}<br>Spot %{y:.2f}<br>Weighted return: %{z:.6f}%<extra></extra>',
                            showscale=False, colorscale='Viridis', cmin=-5, cmax=5))
if 0 <= close_days <= current_days:
    close_idx = int(np.argmin(np.abs(days_forward - close_days)))
    fig_3d.add_trace(go.Scatter3d(x=[close_days]*n_spots, y=S_range, z=wr_grid[close_idx,:], mode='lines',
                                  line=dict(color='blue', width=2, dash='dash'),
                                  name=f'Close in {close_days}d'))

fig_3d.update_layout(scene=dict(xaxis_title='Days', yaxis_title='Spot', zaxis_title='Weighted return (% of NAV)'), height=700)
st.plotly_chart(fig_3d, use_container_width=True)