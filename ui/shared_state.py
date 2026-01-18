import streamlit as st

class SharedMetrics:
    def __init__(self):
        self.athlete_data = {}
        self.last_updated = 0

@st.cache_resource
def get_shared_metrics():
    return SharedMetrics()
