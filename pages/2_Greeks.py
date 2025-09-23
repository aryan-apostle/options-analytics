# ---------------------- Modules ----------------------
from helper_functions import *



# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="ACCF Greeks", layout="wide")
st.title("ACCF Greeks")

spot = st.session_state.get('spot', 100.0)
r = st.session_state.get('r', 0.0191)
fx_rate = st.session_state.get('fx_rate', 0.56)
fund_nav = st.session_state.get('fund_nav', 28560000.0)
current_days = st.session_state.get('current_days', 180)
close_days = st.session_state.get('close_days', int(current_days // 4))
num_legs = st.session_state.get('num_legs', 4)
legs = st.session_state.get('legs', [])



# ---------------------- Greeks Over Time ----------------------
days_forward = np.arange(0, int(current_days) + 1)
T_grid = (current_days - days_forward) / 365.0

greeks_total = {
    'delta': np.zeros_like(days_forward, dtype=float),
    'gamma': np.zeros_like(days_forward, dtype=float),
    'vega': np.zeros_like(days_forward, dtype=float),
    'theta': np.zeros_like(days_forward, dtype=float),
    'rho': np.zeros_like(days_forward, dtype=float)
}

for i, T in enumerate(T_grid):
    greek_day_total = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
    for leg in legs:
        g = bs_greeks(spot, leg['K'], T, r, leg['vol'], leg['type'])
        for k in greek_day_total:
            greek_day_total[k] += g[k] * leg.get('mult', leg.get('qty', 0) * leg.get('size', 0)) * leg.get('sign', (1.0 if leg.get('side','Long')=='Long' else -1.0))
    for k in greeks_total:
        greeks_total[k][i] = greek_day_total[k]

colors = {'delta': 'blue', 'gamma': 'red', 'vega': 'green', 'theta': 'orange', 'rho': 'purple'}

for k in greeks_total:
    st.subheader(f"{k.capitalize()}")
    fig_greek = go.Figure()
    fig_greek.add_trace(go.Scatter(
        x=days_forward,
        y=greeks_total[k],
        mode='lines+markers',
        name=k.capitalize(),
        line=dict(color=colors[k]),
        hovertemplate=f'Day {{x}}<br>{k.capitalize()}: {{y:.6f}}<extra></extra>'
    ))
    try:
        if 0 <= close_days <= current_days:
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
                type='line',
                x0=close_days, x1=close_days, y0=y_min_k, y1=y_max_k,
                line=dict(color='black', dash='dash'),
                xref='x', yref='y'
            )
            fig_greek.add_annotation(x=close_days, y=y_max_k, text=f'Close in {close_days}d', showarrow=False, yanchor='bottom')
    except Exception:
        pass

    fig_greek.update_layout(
        xaxis_title='Days',
        yaxis_title=f'{k.capitalize()}',
        height=420
    )
    st.plotly_chart(fig_greek, use_container_width=True)