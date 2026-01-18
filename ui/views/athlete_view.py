import streamlit as st
import pandas as pd

leaderboard_df = pd.DataFrame(
    {
        "Athlete #": [101, 102, 103, 104, 105, 106],
        "Lane": [1, 2, 3, 4, 5, 6],
        "Speed (m/s)": [2.05, 1.98, 1.96, 1.92, 1.90, 1.85],
        "Stroke Rate (SPM)": [30, 32, 33, 35, 36, 38],
        "DPS (m)": [2.4, 2.3, 2.2, 2.0, 1.9, 1.8],
        "Swim Time (s)": [24.2, 25.1, 25.5, 26.8, 27.0, 28.5],
    }
)

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
    
    col_name, ascending = metric_map[rank_metric]
    sorted_df = leaderboard_df.sort_values(col_name, ascending=ascending)

    st.markdown(f"### Top {rank_metric}")
    st.dataframe(
        sorted_df.style.highlight_min(subset=[col_name], color='#d1fae5') if ascending else sorted_df.style.highlight_max(subset=[col_name], color='#d1fae5'),
        use_container_width=True,
        height=300,
        hide_index=True
    )
    st.caption("Live ranking updates based on video data.")
