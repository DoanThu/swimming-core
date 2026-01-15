import streamlit as st
import os
import pandas as pd
import numpy as np
import altair as alt
import time
import socket
import struct
import pickle
import cv2
import sys
import torch
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.general import SERVER_HOST, SERVER_PORT
from postprocess.multiple_data import FrameMultipleData

st.set_page_config(page_title="Swim Performance Dashboard", layout="wide")

# -------------------------
# Fake data setup
# -------------------------
athlete_ids = [1, 2, 3, 4]

# Generate base trend data
time_idx = np.arange(0, 60, 5)
base_speed = 1.9 + 0.1 * np.sin(time_idx / 10)

# Create specific metrics and chart data for each athlete
athlete_data = {}
for a in athlete_ids:
    # Randomize chart data slightly per athlete so they look different
    noise = np.random.normal(0, 0.05, len(time_idx))
    athlete_speed_trend = base_speed + (0.05 * a) + noise
    
    athlete_data[a] = {
        "current_speed": athlete_speed_trend[-1],
        "prev_speed": athlete_speed_trend[-2],
        "stroke": "Freestyle" if a % 2 != 0 else "Freestyle",
        "lane": a,
        "chart_data": pd.DataFrame({"time": time_idx, "speed": athlete_speed_trend}).set_index("time"),
        "dps": np.random.uniform(1.8, 2.5),
        "stroke_count": np.random.randint(30, 45)
    }

leaderboard_df = pd.DataFrame(
    {
        "Athlete #": [101, 102, 103, 104, 105, 106],
        "Lane": [1, 2, 3, 4, 5, 6],
        "Speed (m/s)": [2.05, 1.98, 1.96, 1.92, 1.90, 1.85],
        "Stroke Count": [30, 32, 33, 35, 36, 38],
        "DPS (m)": [2.4, 2.3, 2.2, 2.0, 1.9, 1.8],
        "Swim Time (s)": [24.2, 25.1, 25.5, 26.8, 27.0, 28.5],
    }
)

# -------------------------
# Styling
# -------------------------
st.markdown(
    """
    <style>
    .main { padding-top: 1rem; }
    h1 { font-size: 1.8rem; font-weight: 700; }
    
    /* Coach Card Styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
    }
    .big-metric {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }
    .small-label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    .stroke-badge {
        background-color: #e0e7ff;
        color: #4338ca;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Video Container */
    .video-wrapper {
        background-color: #000;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Router
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "landing"

# -------------------------
# LANDING PAGE
# -------------------------
if st.session_state.page == "landing":
    st.markdown("<h1 style='text-align: center;'>Swim Performance Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Select a view to continue.</p>", unsafe_allow_html=True)

    _, col1, col2, _ = st.columns([1, 1, 1, 1])

    with col1:
        if st.button("Coach View", use_container_width=True):
            st.session_state.page = "coach"
            st.rerun()
        st.caption("Live video + 4-lane analysis")

    with col2:
        if st.button("Athlete View", use_container_width=True):
            st.session_state.page = "athlete"
            st.rerun()
        st.caption("Leaderboard & Rankings")

# -------------------------
# COACH VIEW
# -------------------------
elif st.session_state.page == "coach":
    def recvall(sock, n):
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def receive_data(sock):
        # 1. Get number of items (pickled)
        payload_size = struct.calcsize("L")
        packed_msg_size = recvall(sock, payload_size)
        if not packed_msg_size: return None
        msg_size = struct.unpack("L", packed_msg_size)[0]
        
        data = recvall(sock, msg_size)
        if not data: return None
        num_items = pickle.loads(data)
        
        items = []
        for _ in range(num_items):
            packed_msg_size = recvall(sock, payload_size)
            if not packed_msg_size: return None
            msg_size = struct.unpack("L", packed_msg_size)[0]
            
            data = recvall(sock, msg_size)
            if not data: return None
            items.append(pickle.loads(data))
        return items

    # 1. Top Navigation
    c1, c2 = st.columns([1, 8])
    with c1:
        if st.button("← Back"):
            st.session_state.page = "landing"
            st.rerun()
    with c2:
        st.subheader("Coach Dashboard: Live Monitor")

    # 2. Main Video Feed (Single)
    c_left, c_center, c_right = st.columns([1, 2, 1])
    with c_center:
        st.markdown('<div class="video-wrapper">', unsafe_allow_html=True)
        video_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Four Charts Side-by-Side
    st.markdown("### Lane Performance")
    perf_placeholder = st.empty()

    if 'athlete_history' not in st.session_state:
        st.session_state.athlete_history = {}
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0

    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_HOST, SERVER_PORT))
        st.toast(f"Connected to {SERVER_HOST}:{SERVER_PORT}")
    except Exception as e:
        st.error(f"Connection failed: {e}")
        st.stop()

    while True:
        try:
            print('Waiting for data from server...', flush=True)
            data_list = receive_data(client_socket)
            if not data_list: break
            print(f"Received data from server: {len(data_list)} items. Timestamp: {data_list[0]}", flush=True)
            
            timestamp_str, annotated_frame, raw_frame, frame_data = data_list
            
            # Display Video
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            

            # Process Data
            ids = getattr(frame_data, 'swimmer_id_list', [])
            if hasattr(ids, 'tolist'): ids = ids.tolist()
            
            speeds = getattr(frame_data, 'speed_m_list', [])
            strokes = getattr(frame_data, 'stroke_count_list', [])
            dps_list = getattr(frame_data, 'distance_per_stroke_list', [])
            
            st.session_state.frame_count += 1
            t_idx = st.session_state.frame_count
            
            # Update History
            for i, uid in enumerate(ids):
                uid = int(uid)
                if uid not in st.session_state.athlete_history:
                    st.session_state.athlete_history[uid] = {
                        "times": [], "speeds": []
                    }
                hist = st.session_state.athlete_history[uid]
                hist["times"].append(t_idx)
                hist["speeds"].append(speeds[i])
                if len(hist["times"]) > 60:
                    hist["times"] = hist["times"][-60:]
                    hist["speeds"] = hist["speeds"][-60:]
            
            with perf_placeholder.container():
                num_athletes = len(ids)
                if num_athletes == 0:
                    st.info("No swimmers detected.")
                else:
                    cols_per_row = num_athletes if num_athletes <= 4 else 4
                    for i in range(0, num_athletes, cols_per_row):
                        row_indices = range(i, min(i + cols_per_row, num_athletes))
                        cols = st.columns(cols_per_row)
                        
                        for idx, col in zip(row_indices, cols):
                            uid = int(ids[idx])
                            spd = speeds[idx]
                            strk = strokes[idx]
                            dps = dps_list[idx]
                            
                            hist = st.session_state.athlete_history[uid]
                            prev_speed = hist["speeds"][-2] if len(hist["speeds"]) > 1 else spd
                            speed_delta = spd - prev_speed
                            
                            chart_df = pd.DataFrame({"time": hist["times"], "speed": hist["speeds"]})
                            
                            with col:
                                with st.container():
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                                            <span style="font-weight:bold;">ID {uid}</span>
                                            <span class="stroke-badge">Freestyle</span>
                                        </div>
                                        <div class="small-label">Current Speed</div>
                                        <div class="big-metric">{spd:.2f} m/s</div>
                                        <div style="color: {'green' if speed_delta >= 0 else 'red'}; font-size: 0.9rem; margin-bottom:15px;">
                                            {speed_delta:+.2f} vs prev
                                        </div>
                                        <div style="display:flex; justify-content:space-around; margin-top:10px; border-top:1px solid #eee; padding-top:10px;">
                                            <div>
                                                <div class="small-label" style="font-size:0.7rem;">DPS</div>
                                                <div style="font-weight:600;">{dps:.2f} m</div>
                                            </div>
                                            <div>
                                                <div class="small-label" style="font-size:0.7rem;">Stroke Count</div>
                                                <div style="font-weight:600;">{strk}</div>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    color_hex = ["#2563eb", "#16a34a", "#dc2626", "#d97706"][uid % 4]
                                    chart = alt.Chart(chart_df).mark_line(color=color_hex).encode(
                                        x=alt.X('time', title='Time'),
                                        y=alt.Y('speed', title='Speed (m/s)', scale=alt.Scale(domain=[0, 3.0]))
                                    ).properties(height=150)
                                    st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Stream error: {e}")
            print(f"Stream error: {e}")
            traceback.print_exc()
            break

# -------------------------
# ATHLETE VIEW
# -------------------------
elif st.session_state.page == "athlete":
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
        ["Speed", "Stroke Count", "Distance Per Stroke", "Swim Time"]
    )

    # Map selection to column name and sort order (True=Ascending/Lower is better, False=Descending/Higher is better)
    metric_map = {
        "Speed": ("Speed (m/s)", False),
        "Stroke Count": ("Stroke Count", True),
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
