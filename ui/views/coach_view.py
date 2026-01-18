import streamlit as st
import time
import cv2
import pandas as pd
import altair as alt
import traceback
from config.general import SERVER_HOST, SERVER_PORT
from ui.socket_receiver import SocketReceiver

# Constants for UI rendering
UI_FPS = 30
CHART_UPDATE_INTERVAL = 1 # seconds
FRAMES_PER_UPDATE = UI_FPS * CHART_UPDATE_INTERVAL
MAX_HISTORY_FRAMES = UI_FPS * 10 # Keep 10 seconds of history
TIMEOUT = 5.0 # seconds before considering a swimmer inactive
SMOOTH_WINDOW = 10
NUM_CHARTS = 3

def render_coach_view():
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
        show_skeletons = st.toggle("Show Skeletons", value=True)
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

    if 'socket_receiver' not in st.session_state:
        st.session_state.socket_receiver = SocketReceiver(SERVER_HOST, SERVER_PORT)
        result = st.session_state.socket_receiver.start()
        if result is not True:
            st.error(f"Connection failed: {result}")
            st.stop()
        else:
            st.toast(f"Connected to {SERVER_HOST}:{SERVER_PORT}")

    receiver = st.session_state.socket_receiver

    while True:
        try:
            data_list = receiver.get_latest()
            if data_list is None:
                if not receiver.is_running: break
                time.sleep(0.01)
                continue
            
            timestamp_str, annotated_frame, raw_frame, frame_data = data_list
            
            # Display Video
            frame_to_show = annotated_frame if show_skeletons else raw_frame
            frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            

            # Process Data
            ids = getattr(frame_data, 'swimmer_id_list', [])
            if hasattr(ids, 'tolist'): ids = ids.tolist()
            
            speeds = getattr(frame_data, 'speed_m_list', [])
            strokes = getattr(frame_data, 'stroke_rate_list', [])
            dps_list = getattr(frame_data, 'distance_per_stroke_list', [])
            
            st.session_state.frame_count += 1
            current_sim_time = st.session_state.frame_count / UI_FPS
            current_time = time.time()
            
            # Update History
            active_uids = []
            for i, uid in enumerate(ids):
                uid = int(uid)
                active_uids.append(uid)
                if uid not in st.session_state.athlete_history:
                    st.session_state.athlete_history[uid] = {
                        "times": [], "speeds": [], "strokes": [], "dps": [], "last_seen": current_time,
                        "metrics": {"speed": 0, "stroke": 0, "dps": 0}
                    }
                hist = st.session_state.athlete_history[uid]
                hist["times"].append(current_sim_time)
                hist["speeds"].append(speeds[i])
                hist["strokes"].append(strokes[i])
                hist["dps"].append(dps_list[i])
                hist["last_seen"] = current_time
                hist["metrics"] = {
                    "speed": speeds[i],
                    "stroke": strokes[i],
                    "dps": dps_list[i]
                }
                if len(hist["times"]) > MAX_HISTORY_FRAMES:
                    hist["times"] = hist["times"][-MAX_HISTORY_FRAMES:]
                    hist["speeds"] = hist["speeds"][-MAX_HISTORY_FRAMES:]
                    hist["strokes"] = hist["strokes"][-MAX_HISTORY_FRAMES:]
                    hist["dps"] = hist["dps"][-MAX_HISTORY_FRAMES:]
            
            # Clear expired data (timeout after 5 seconds)
            expired_ids = [k for k, v in st.session_state.athlete_history.items() 
                           if current_time - v.get("last_seen", 0) > TIMEOUT]
            for uid in expired_ids:
                del st.session_state.athlete_history[uid]

            # Sort swimmers by number of data points (descending)
            sorted_uids = sorted(
                st.session_state.athlete_history.keys(),
                key=lambda k: len(st.session_state.athlete_history[k]["times"]),
                reverse=True
            )

            if st.session_state.frame_count % FRAMES_PER_UPDATE == 0:
                with perf_placeholder.container():
                    # Create grid based on NUM_CHARTS
                    cols = []
                    for _ in range((NUM_CHARTS + 2) // 3):
                        cols.extend(st.columns(3))
                    cols = cols[:NUM_CHARTS]

                    for slot_idx, col in enumerate(cols):
                        uid = sorted_uids[slot_idx] if slot_idx < len(sorted_uids) else None
                        
                        with col:
                            if uid is not None and uid in st.session_state.athlete_history:
                                hist = st.session_state.athlete_history[uid]
                                
                                speed_series = pd.Series(hist["speeds"]).rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
                                spd = speed_series.iloc[-1]
                                prev_speed = speed_series.iloc[-(FRAMES_PER_UPDATE + 1)] if len(speed_series) > FRAMES_PER_UPDATE else spd
                                speed_delta = spd - prev_speed
                                
                                strk = pd.Series(hist["strokes"]).rolling(window=SMOOTH_WINDOW, min_periods=1).mean().iloc[-1]
                                dps = pd.Series(hist["dps"]).rolling(window=SMOOTH_WINDOW, min_periods=1).mean().iloc[-1]
                                
                                chart_df = pd.DataFrame({"time": hist["times"], "speed": hist["speeds"]})
                                chart_df["speed"] = chart_df["speed"].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
                                
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
                                                <div class="small-label" style="font-size:0.7rem;">Stroke Rate (SPM)</div>
                                                <div style="font-weight:600;">{strk:.1f}</div>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    color_hex = ["#2563eb", "#16a34a", "#dc2626", "#d97706"][uid % 4]
                                    chart = alt.Chart(chart_df).mark_line(color=color_hex).encode(
                                        x=alt.X('time', title='Time (s)'),
                                        y=alt.Y('speed', title='Speed (m/s)', scale=alt.Scale(domain=[0, 3.0]))
                                    ).properties(height=150)
                                    st.altair_chart(chart, use_container_width=True)
                            else:
                                # Empty slot placeholder
                                st.markdown(f"""
                                <div class="metric-card" style="opacity: 0.3; min-height: 300px; display: flex; align-items: center; justify-content: center;">
                                    <div style="color: #888;">Slot {slot_idx + 1}<br>Waiting...</div>
                                </div>
                                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Stream error: {e}")
            print(f"Stream error: {e}")
            traceback.print_exc()
            break
