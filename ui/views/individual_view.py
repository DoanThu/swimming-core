import streamlit as st
import pandas as pd
import altair as alt
import os
from ui.config_ui import DATA_FOLDER

def render_individual_view():
    c1, c2 = st.columns([1, 8])
    with c1:
        if st.button("← Back"):
            st.session_state.page = "landing"
            st.rerun()
    with c2:
        st.title("Individual Performance View")

    file_path = os.path.join(DATA_FOLDER, "swimming_session_data.csv")
    if not os.path.exists(file_path):
        st.info("No session data found. Please run a session in Coach View first.")
        return

    try:
        df = pd.read_csv(file_path)
        
        # Filter by Lane
        lanes = sorted(df['Lane'].unique())
        selected_lane = st.selectbox("Select Lane to Analyze", lanes)
        
        lane_df = df[df['Lane'] == selected_lane]
        
        # Group by SessionID to get averages per session
        # We take the mean of all data points logged for that session
        session_stats = lane_df.groupby('SessionID').agg({
            'AvgSpeed': 'mean',
            'AvgStrokeRate': 'mean',
            'AvgDPS': 'mean',
            'Date': 'first',
            'Time': 'first' 
        }).reset_index()
        
        # Create a DateTime column for sorting
        session_stats['DateTime'] = pd.to_datetime(session_stats['Date'] + ' ' + session_stats['Time'])
        session_stats = session_stats.sort_values('DateTime')

        st.subheader(f"Performance Trend - Lane {selected_lane}")
        
        # Charts
        base = alt.Chart(session_stats).encode(x=alt.X('SessionID', sort=None))

        c_speed = base.mark_line(point=True).encode(y='AvgSpeed', tooltip=['SessionID', 'Date', 'AvgSpeed']).properties(title="Average Speed (m/s)")
        c_stroke = base.mark_line(point=True, color='green').encode(y='AvgStrokeRate', tooltip=['SessionID', 'Date', 'AvgStrokeRate']).properties(title="Average Stroke Rate (SPM)")
        c_dps = base.mark_line(point=True, color='red').encode(y='AvgDPS', tooltip=['SessionID', 'Date', 'AvgDPS']).properties(title="Average DPS (m)")
        
        st.altair_chart(c_speed | c_stroke | c_dps, use_container_width=True)
        
        st.dataframe(session_stats[['Date', 'Time', 'SessionID', 'AvgSpeed', 'AvgStrokeRate', 'AvgDPS']], use_container_width=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")
