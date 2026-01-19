import streamlit as st

def render_landing_view():
    if 'socket_receiver' in st.session_state:
        st.session_state.socket_receiver.stop()
        del st.session_state.socket_receiver

    st.markdown("<h1 style='text-align: center;'>Swim Performance Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Select a view to continue.</p>", unsafe_allow_html=True)

    _, col1, col2, col3, _ = st.columns([0.2, 1, 1, 1, 0.2])

    with col1:
        if st.button("Coach View", width='stretch'):
            st.session_state.page = "coach"
            st.rerun()

    with col2:
        if st.button("Athlete View", width='stretch'):
            st.session_state.page = "athlete"
            st.rerun()
            
    with col3:
        if st.button("Individual View", width='stretch'):
            st.session_state.page = "individual"
            st.rerun()
