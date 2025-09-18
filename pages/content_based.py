# ------------------------------------------------------------------------------
# Import libraries
import streamlit as st
import pandas as pd
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from myfunctions import fn_main_display_selected_hotel, fn_render_mainpage_header, fn_render_footer
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
    description_2=DESCRIPTION_name,
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
    "Select a hotel!", options=hotel_options, format_func=lambda x: x[0]
)

# Create sidebar controls
with st.sidebar:
    # Radio button for model selection
    selected_model = st.radio(
        "Model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
        horizontal=True,
    )

    # Slider for recommendations
    # top_k = st.slider(
    #     "Top N hotels", 
    #     min_value=TOP_K_MIN, 
    #     max_value=TOP_K_MAX, 
    #     value=DEFAULT_TOP_K, 
    #     step=TOP_K_STEP
    # )
    top_k = DEFAULT_TOP_K

    # Slider for description limit
    # desc_limit = st.slider(
    #     "Description limit (words)", 
    #     min_value=DESC_LIMIT_MIN, 
    #     max_value=DESC_LIMIT_MAX, 
    #     value=DEFAULT_DESC_LIMIT, 
    #     step=DESC_LIMIT_STEP
    # )
    desc_limit = DEFAULT_DESC_LIMIT

# Select similarity matrix
df_matrix = df_matrix_gensim if selected_model == "gensim" else df_matrix_cosine


# ------------------------------------------------------------------------------
# main program

# Initialize session state
if SESSION_KEYS["selected_hotel_id"] not in st.session_state:
    st.session_state[SESSION_KEYS["selected_hotel_id"]] = None

# Update session state
st.session_state[SESSION_KEYS["selected_hotel_id"]] = selected_hotel[1]

# Execute recommendation display
if st.session_state[SESSION_KEYS["selected_hotel_id"]] is not None:
    fn_main_display_selected_hotel(
        selected_hotel_id=st.session_state[SESSION_KEYS["selected_hotel_id"]],
        df_hotels=df_hotels,
        df_matrix=df_matrix,
        top_k=top_k,
        desc_limit=desc_limit,
    )

# ------------------------------------------------------------------------------
# Render footer
fn_render_footer(OWNER, PROJECT_INFO, SHOW_FOOTER)
