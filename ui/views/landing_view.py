import streamlit as st

def render_landing_view():
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
