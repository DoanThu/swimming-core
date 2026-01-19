import streamlit as st
import pandas as pd
import time
from ui.shared_state import get_shared_metrics
from ui.config_ui import NUM_CHARTS, CELL_FONT_SIZE, HEADER_FONT_SIZE

def render_athlete_view():
    c1, c2 = st.columns([1, 8])
    with c1:
        if st.button("← Back"):
            st.session_state.page = "landing"
            st.rerun()
    with c2:
        st.markdown(f"<h1 style='font-size: {HEADER_FONT_SIZE}'>Session Leaderboard</h1>", unsafe_allow_html=True)

    table_placeholder = st.empty()
    st.caption("Live ranking updates based on video data.")

    shared_metrics = get_shared_metrics()

    while True:
        data = shared_metrics.athlete_data
        
        if not data:
            table_placeholder.info("Waiting for data from Coach View...")
            time.sleep(1)
            continue

        rows = []
        for uid, metrics in data.items():
            rows.append({
                "Lane": metrics["lane"],
                "Avg Speed (m/s)": round(metrics.get("avg_speed", 0), 2),
                "Avg Stroke Rate (SPM)": round(metrics.get("avg_stroke_rate", 0), 1),
                "Avg DPS (m)": round(metrics.get("avg_dps", 0), 2)
            })
        
        if rows:
            df = pd.DataFrame(rows)
            sorted_df = df.sort_values("Lane")
            
            # Use HTML rendering for bigger text and exact row count
            styler = sorted_df.style.format(precision=2)
            styler = styler.highlight_max(subset=["Avg Speed (m/s)", "Avg Stroke Rate (SPM)", "Avg DPS (m)"], color='#d1fae5')
            
            # Apply styles for bigger text
            styler.set_properties(**{
                'font-size': CELL_FONT_SIZE,
                'padding': '24px',
                'text-align': 'center'
            })
            styler.set_table_attributes('style="width: 100%; border-collapse: collapse;"')
            styler.set_table_styles([
                {'selector': 'th', 'props': [('font-size', HEADER_FONT_SIZE), ('text-align', 'center'), ('background-color', '#f0f2f6')]}
            ])
            styler.hide(axis='index')
            
            table_placeholder.markdown(styler.to_html(), unsafe_allow_html=True)
        
        time.sleep(0.1)
