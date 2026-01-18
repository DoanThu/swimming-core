import streamlit as st
import time
import cv2
import pandas as pd
import altair as alt
import traceback
from ui.shared_state import get_shared_metrics, get_stream_manager
from postprocess.const import FrameDataConst
from ui.config_ui import UI_FPS, FRAMES_PER_UPDATE, MAX_HISTORY_FRAMES, TIMEOUT, SMOOTH_WINDOW, NUM_CHARTS


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
        c_tog, c_num = st.columns([1, 1])
        with c_tog: show_skeletons = st.toggle("Show Skeletons", value=True)
        with c_num: num_lanes = st.number_input("Lanes", min_value=1, max_value=3, value=3)
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

    # Use shared stream manager instead of local socket receiver
    manager = get_stream_manager()
    result = manager.start()
    if result is not True and result is not None:
        st.error(f"Connection failed: {result}")
        st.stop()

    shared_metrics = get_shared_metrics()
    last_frame_idx = -1

    while True:
        try:
            data_list = manager.get_latest_frame()
            if data_list is None:
                time.sleep(0.01)
                continue
            
            timestamp_str, annotated_frame, raw_frame, frame_data = data_list
            current_idx = getattr(frame_data, 'frame_idx', -1)
            if current_idx == last_frame_idx:
                time.sleep(0.01)
                continue
            last_frame_idx = current_idx
            
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
            bboxes = getattr(frame_data, 'bbox_list', [])
            orientation = getattr(frame_data, 'frame_orientation', FrameDataConst.UNKNOWN)

            # Determine Lanes based on position
            lane_map = {}
            if ids and bboxes and len(ids) == len(bboxes):
                h, w = raw_frame.shape[:2]
                
                for i, uid in enumerate(ids):
                    if i < len(bboxes):
                        box = bboxes[i]
                        center_x = box[0] + box[2] / 2
                        center_y = box[1] + box[3] / 2
                        
                        if orientation == FrameDataConst.VERTICAL:
                            lane_map[int(uid)] = int(center_x / (w / num_lanes)) + 1
                        else:
                            lane_map[int(uid)] = int(center_y / (h / num_lanes)) + 1
                        
                        lane_map[int(uid)] = max(1, min(lane_map[int(uid)], num_lanes))
            
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
                        "times": [], "speeds": [], "strokes": [], "dps": [], 
                        "last_seen": current_time, "first_seen": current_time,
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
            all_uids = sorted(
                st.session_state.athlete_history.keys(),
                key=lambda k: len(st.session_state.athlete_history[k]["times"]),
                reverse=True
            )
            
            # Take top NUM_CHARTS and sort by lane (ascending)
            sorted_uids = sorted(
                all_uids[:NUM_CHARTS],
                key=lambda k: lane_map.get(k, 999)
            )
            
            if st.session_state.frame_count % FRAMES_PER_UPDATE == 0:
                # Update Shared Metrics for Athlete View
                current_shared_data = {}
                for uid in sorted_uids:
                    if uid in st.session_state.athlete_history:
                        hist = st.session_state.athlete_history[uid]
                        # Calculate simple averages for the leaderboard using the smoothing window
                        s_window = min(len(hist["speeds"]), SMOOTH_WINDOW)
                        current_shared_data[uid] = {
                            "speed": sum(hist["speeds"][-s_window:]) / s_window if s_window > 0 else 0,
                            "stroke_rate": sum(hist["strokes"][-s_window:]) / s_window if s_window > 0 else 0,
                            "dps": sum(hist["dps"][-s_window:]) / s_window if s_window > 0 else 0,
                            "swim_time": current_time - hist.get("first_seen", current_time),
                            "lane": lane_map.get(uid, (uid % 8) + 1)
                        }
                shared_metrics.athlete_data = current_shared_data
                shared_metrics.last_updated = current_time
                shared_metrics.num_lanes = NUM_CHARTS

                with perf_placeholder.container():
                    # Create grid based on NUM_CHARTS
                    cols = st.columns(NUM_CHARTS)

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
                                
                                lane_num = lane_map.get(uid, '?')
                                with st.container():
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                                            <span style="font-weight:bold;">ID {uid} - Lane {lane_num}</span>
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
