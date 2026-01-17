import streamlit as st
import os
import pandas as pd
import altair as alt
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui.utils import CSS_STYLE
from ui.views.landing_view import render_landing_view
from ui.views.coach_view import render_coach_view
from ui.views.athlete_view import render_athlete_view

st.set_page_config(page_title="Swim Performance Dashboard", layout="wide")

# -------------------------
# Styling
# -------------------------
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# -------------------------
# Router
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "landing"

if st.session_state.page == "landing":
    render_landing_view()
elif st.session_state.page == "coach":
    render_coach_view()
elif st.session_state.page == "athlete":
    render_athlete_view()
