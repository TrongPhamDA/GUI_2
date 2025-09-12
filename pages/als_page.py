import streamlit as st
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from myfunctions import fn_render_footer
from app_config import *

# ALS model
st.markdown("# Page 4 ❄️")
st.sidebar.markdown("# Page 4 ❄️")

# ------------------------------------------------------------------------------
# Render footer
fn_render_footer(OWNER, PROJECT_INFO, SHOW_FOOTER)