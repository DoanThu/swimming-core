import streamlit as st
import os
import pandas as pd
import altair as alt
import time
import socket
import cv2
import sys
import traceback
import threading
import queue

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.general import SERVER_HOST, SERVER_PORT
from postprocess.multiple_data import FrameMultipleData
from ui.utils import receive_data, CSS_STYLE

# Constants for UI rendering
UI_FPS = 30
CHART_UPDATE_INTERVAL = 1 # seconds
FRAMES_PER_UPDATE = UI_FPS * CHART_UPDATE_INTERVAL
MAX_HISTORY_FRAMES = UI_FPS * 10 # Keep 10 seconds of history

st.set_page_config(page_title="Swim Performance Dashboard", layout="wide")

# -------------------------
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

class SocketReceiver:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.sock = None
        self.is_running = False

    def start(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.is_running = True
            t = threading.Thread(target=self._receive_loop, daemon=True)
            t.start()
            return True
        except Exception as e:
            return str(e)

    def _receive_loop(self):
        while self.is_running and not self.stop_event.is_set():
            try:
                data = receive_data(self.sock)
                if data is None: break
                if self.queue.full():
                    try: self.queue.get_nowait()
                    except queue.Empty: pass
                self.queue.put(data)
            except Exception:
                break
        self.is_running = False
        if self.sock: 
            try: self.sock.close()
            except: pass

    def get_latest(self):
        try: return self.queue.get_nowait()
        except queue.Empty: return None
    
    def stop(self):
        self.stop_event.set()
        self.is_running = False
        if self.sock: 
            try: self.sock.close()
            except: pass

# -------------------------
# Styling
# -------------------------
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# -------------------------
# Router
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "landing"

# -------------------------
# LANDING PAGE
# -------------------------
if st.session_state.page == "landing":
    if 'socket_receiver' in st.session_state:
        st.session_state.socket_receiver.stop()
        del st.session_state.socket_receiver

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
    if 'slot_mapping' not in st.session_state:
        st.session_state.slot_mapping = {}

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
            strokes = getattr(frame_data, 'stroke_count_list', [])
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
                        "times": [], "speeds": [], "last_seen": current_time,
                        "metrics": {"speed": 0, "stroke": 0, "dps": 0}
                    }
                hist = st.session_state.athlete_history[uid]
                hist["times"].append(current_sim_time)
                hist["speeds"].append(speeds[i])
                hist["last_seen"] = current_time
                hist["metrics"] = {
                    "speed": speeds[i],
                    "stroke": strokes[i],
                    "dps": dps_list[i]
                }
                if len(hist["times"]) > MAX_HISTORY_FRAMES:
                    hist["times"] = hist["times"][-MAX_HISTORY_FRAMES:]
                    hist["speeds"] = hist["speeds"][-MAX_HISTORY_FRAMES:]
            
            # Clear expired data (timeout after 5 seconds)
            TIMEOUT = 5.0
            expired_ids = [k for k, v in st.session_state.athlete_history.items() 
                           if current_time - v.get("last_seen", 0) > TIMEOUT]
            for uid in expired_ids:
                del st.session_state.athlete_history[uid]
                if uid in st.session_state.slot_mapping:
                    del st.session_state.slot_mapping[uid]

            # Assign slots to new active swimmers
            used_slots = set(st.session_state.slot_mapping.values())
            all_slots = set(range(6)) # 6 slots for 2x3 grid
            available_slots = sorted(list(all_slots - used_slots))
            
            for uid in active_uids:
                if uid not in st.session_state.slot_mapping:
                    if available_slots:
                        slot = available_slots.pop(0)
                        st.session_state.slot_mapping[uid] = slot

            if st.session_state.frame_count % FRAMES_PER_UPDATE == 0:
                with perf_placeholder.container():
                    # Create fixed 2x3 grid
                    rows = [st.columns(3), st.columns(3)]
                    cols = rows[0] + rows[1] # Flatten to list of 6 columns
                    
                    # Reverse mapping to find which UID is in which slot
                    slot_to_uid = {v: k for k, v in st.session_state.slot_mapping.items()}

                    for slot_idx, col in enumerate(cols):
                        uid = slot_to_uid.get(slot_idx)
                        
                        with col:
                            if uid is not None and uid in st.session_state.athlete_history:
                                hist = st.session_state.athlete_history[uid]
                                metrics = hist["metrics"]
                                
                                spd = metrics["speed"]
                                strk = metrics["stroke"]
                                dps = metrics["dps"]
                                
                                prev_speed = hist["speeds"][-2] if len(hist["speeds"]) > 1 else spd
                                speed_delta = spd - prev_speed
                                
                                chart_df = pd.DataFrame({"time": hist["times"], "speed": hist["speeds"]})
                                
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
