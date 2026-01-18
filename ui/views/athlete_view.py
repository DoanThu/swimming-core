import streamlit as st
import pandas as pd
import time
from ui.shared_state import get_shared_metrics
from ui.config_ui import NUM_CHARTS

def render_athlete_view():
    c1, c2 = st.columns([1, 8])
    with c1:
        if st.button("← Back"):
            st.session_state.page = "landing"
            st.rerun()
    with c2:
        st.subheader("Session Leaderboard")

    # Dropdown for ranking
    rank_metric = st.selectbox(
        "Rank by:",
        ["Speed", "Stroke Rate", "Distance Per Stroke", "Swim Time"]
    )

    # Map selection to column name and sort order (True=Ascending/Lower is better, False=Descending/Higher is better)
    metric_map = {
        "Speed": ("Speed (m/s)", False),
        "Stroke Rate": ("Stroke Rate (SPM)", True),
        "Distance Per Stroke": ("DPS (m)", False),
        "Swim Time": ("Swim Time (s)", True)
    }

    st.markdown(f"### Top {rank_metric}")
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
                "Athlete #": uid,
                "Lane": metrics["lane"],
                "Speed (m/s)": round(metrics["speed"], 2),
                "Stroke Rate (SPM)": round(metrics["stroke_rate"], 1),
                "DPS (m)": round(metrics["dps"], 2),
                "Swim Time (s)": round(metrics["swim_time"], 1)
            })
        
        if rows:
            df = pd.DataFrame(rows)
            col_name, ascending = metric_map[rank_metric]
            limit = getattr(shared_metrics, 'num_lanes', NUM_CHARTS)
            sorted_df = df.sort_values(col_name, ascending=ascending).head(limit)
            
            table_placeholder.dataframe(
                sorted_df.style.highlight_min(subset=[col_name], color='#d1fae5') if ascending else sorted_df.style.highlight_max(subset=[col_name], color='#d1fae5'),
                use_container_width=True,
                height=300,
                hide_index=True
            )
        
        time.sleep(0.1)
