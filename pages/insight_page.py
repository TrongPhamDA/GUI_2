# ------------------------------------------------------------------------------
# Import libraries
import streamlit as st
import pandas as pd
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from myfunctions import fn_render_mainpage_header, fn_render_footer, fn_display_hotel_insights
from app_config import *


# ------------------------------------------------------------------------------
# Load hotel dataset
df_hotels = pd.read_csv(HOTEL_INFO_CSV, header=0)
df_hotels["hotel_id"] = df_hotels["hotel_id"].astype(str)
st.session_state[SESSION_KEYS["random_hotels"]] = df_hotels

# Load Gensim TF-IDF similarity matrix
df_matrix_gensim = pd.read_csv(GENSIM_MATRIX_CSV, header=None, index_col=False)

# Load Cosine similarity matrix
df_matrix_cosine = pd.read_csv(COSINE_MATRIX_CSV, header=None, index_col=False)


# ------------------------------------------------------------------------------
# Display main hotel platform image
# st.image("static/agoda_mainpage.jpg", use_container_width=True)

# Render main page header
fn_render_mainpage_header(
    img_src=IMG_SRC,
    page_title=PAGE_TITLE,
    description_1=DESCRIPTION_1,
    description_2=DESCRIPTION_insight,
)

# ------------------------------------------------------------------------------
# Create hotel dropdown options
hotel_options = [
    (row["hotel_name"], row["hotel_id"])
    for _, row in st.session_state[SESSION_KEYS["random_hotels"]].iterrows()
]


# ------------------------------------------------------------------------------
# User interface controls
# Create hotel selection dropdown
selected_hotel = st.selectbox(
    "Select your hotel for insight analysis!", options=hotel_options, format_func=lambda x: x[0]
)

# Create sidebar controls
with st.sidebar:
    st.markdown("### Chart Settings")
    
    # Slider for figure size
    figsize_width = st.slider(
        "Chart width", 
        min_value=8, 
        max_value=15, 
        value=DEFAULT_FIGSIZE[0], 
        step=1
    )
    
    figsize_height = st.slider(
        "Chart height", 
        min_value=4, 
        max_value=8, 
        value=DEFAULT_FIGSIZE[1], 
        step=1
    )
    
    # Slider for bins time
    bins_time = st.slider(
        "Bins per unit score", 
        min_value=1, 
        max_value=5, 
        value=DEFAULT_BINS_TIME, 
        step=1
    )
    
    # Slider for description limit
    desc_limit = st.slider(
        "Description limit (words)", 
        min_value=DESC_LIMIT_MIN, 
        max_value=DESC_LIMIT_MAX, 
        value=INSIGHT_DESC_LIMIT, 
        step=DESC_LIMIT_STEP
    )
    
    # Slider for word count limit
    word_count_limit = st.slider(
        "Word count limit for word cloud", 
        min_value=WORD_COUNT_LIMIT_MIN, 
        max_value=WORD_COUNT_LIMIT_MAX, 
        value=WORD_COUNT_LIMIT_DEFAULT, 
        step=WORD_COUNT_LIMIT_STEP
    )


# ------------------------------------------------------------------------------
# main program

# Initialize session state
if SESSION_KEYS["selected_hotel_id"] not in st.session_state:
    st.session_state[SESSION_KEYS["selected_hotel_id"]] = None

# Update session state
st.session_state[SESSION_KEYS["selected_hotel_id"]] = selected_hotel[1]

# Execute insight analysis
if st.session_state[SESSION_KEYS["selected_hotel_id"]] is not None:
    fn_display_hotel_insights(
        selected_hotel_id=st.session_state[SESSION_KEYS["selected_hotel_id"]],
        df_hotels=df_hotels,
        figsize=(figsize_width, figsize_height),
        bins_time=bins_time,
        rating_cols=RATING_COLS,
        score_classify_dict=SCORE_CLASSIFY_DICT,
        show_radar=True,
        show_customer=True,
        show_wordcloud=True,
        word_count_limit=word_count_limit,
        df_comments=None  # Placeholder for future comment data
    )

# ------------------------------------------------------------------------------
# Render footer
fn_render_footer(OWNER, PROJECT_INFO, SHOW_FOOTER)
