# ------------------------------------------------------------------------------
# Import libraries
import streamlit as st
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from myfunctions import *
from app_config import *


# ------------------------------------------------------------------------------
# Display main hotel platform image
# st.image("static/agoda_mainpage.jpg", use_container_width=True)

# Render main page header
fn_render_mainpage_header(
    img_src=IMG_SRC,
    page_title=PAGE_TITLE,
    description_1=DESCRIPTION_1,
    description_2=DESCRIPTION_finalreport,
)


# ------------------------------------------------------------------------------
# Display PDF report
fn_display_report(FINAL_REPORT, FINAL_REPORT_IMAGES_DIR, COL_NUM)


# ------------------------------------------------------------------------------
# Render footer
fn_render_footer(OWNER, PROJECT_INFO, SHOW_FOOTER)