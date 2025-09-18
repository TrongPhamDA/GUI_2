# ------------------------------------------------------------------------------
# Import libraries
import streamlit as st
import pandas as pd
import joblib
from gensim import corpora
from gensim.similarities import SparseMatrixSimilarity
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from myfunctions import (
    fn_clean_tokens,
    fn_read_txt,
    fn_read_dict,
    fn_main_display_search_results,
    fn_render_mainpage_header,
    fn_render_footer,
)
from app_config import *

# ------------------------------------------------------------------------------
# Load NLP dictionaries
emoji = fn_read_dict(path=EMOJI_DICT_PATH)
engvie = fn_read_dict(path=ENGVIE_DICT_PATH)
teencode = fn_read_dict(path=TEENCODE_DICT_PATH)
stopword_vie = fn_read_txt(path=STOPWORD_PATH)
wrongword = fn_read_txt(path=WRONGWORD_PATH)

# ------------------------------------------------------------------------------
# Load preprocessed hotel data
df_hotels = pd.read_csv(HOTEL_INFO_TOKEN_CSV, header=0)
df_hotels["hotel_id"] = df_hotels["hotel_id"].astype(str)

# ------------------------------------------------------------------------------
# Load Gensim models
loaded_model_tfidf = joblib.load(TFIDF_MODEL_PKL)
loaded_dictionary = corpora.Dictionary.load(DICTIONARY_DICT)
loaded_matrix_gensim = SparseMatrixSimilarity.load(GENSIM_MATRIX_INDEX)

# ------------------------------------------------------------------------------
# Display main hotel platform image
# st.image("static/agoda_mainpage.jpg", use_container_width=True)

# Render main page header
fn_render_mainpage_header(
    img_src=IMG_SRC,
    page_title=PAGE_TITLE,
    description_1=DESCRIPTION_1,
    description_2=DESCRIPTION_search,
)


# ------------------------------------------------------------------------------
# User interface controls
search_input = st.text_input(
    "Search for something...", value=DEFAULT_SEARCH_INPUT
)

with st.sidebar:
    selected_model = st.radio(
        "Model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
        horizontal=True,
    )
    # top_k = st.slider(
    #     "Top N hotels", 
    #     min_value=TOP_K_MIN, 
    #     max_value=TOP_K_MAX, 
    #     value=DEFAULT_TOP_K, 
    #     step=TOP_K_STEP
    # )
    top_k = DEFAULT_TOP_K
    
    # desc_limit = st.slider(
    #     "Description limit (words)", 
    #     min_value=DESC_LIMIT_MIN, 
    #     max_value=DESC_LIMIT_MAX, 
    #     value=DEFAULT_DESC_LIMIT, 
    #     step=DESC_LIMIT_STEP
    # )
    desc_limit = DEFAULT_DESC_LIMIT
# ------------------------------------------------------------------------------
# Text preprocessing
search_input_tokens = fn_clean_tokens(
    text=search_input,
    dict_list=[emoji, teencode, engvie],
    stopword=stopword_vie,
    wrongword=wrongword,
    **CLEAN_TOKENS_CONFIG
)

# ------------------------------------------------------------------------------
# # tạo cột content_wt
# df_hotels['content_wt'] = df_hotels['hotel_description'].apply(
#     lambda x: fn_clean_tokens(
#         text=str(x),
#         dict_list=[emoji, teencode, engvie],
#         stopword=stopword_vie,
#         wrongword=wrongword,
#         remove_number=True,
#         remove_punctuation=True,
#         remove_vie_tone=False,
#         lower=True,
#     )
# )
# df_hotels.to_csv("../input/hotel_info_token.csv", index=False)

# ------------------------------------------------------------------------------
# main program
if search_input and search_input_tokens:
    fn_main_display_search_results(
        df_hotels=df_hotels,
        search_input_tokens=search_input_tokens,
        selected_model=selected_model,
        top_k=top_k,
        gensim_matrix=loaded_matrix_gensim,
        corpora_dictionary=loaded_dictionary,
        tfidf_model=loaded_model_tfidf,
        search_input=search_input,
        desc_limit=desc_limit,
    )

# ------------------------------------------------------------------------------
# Render footer
fn_render_footer(OWNER, PROJECT_INFO, SHOW_FOOTER)
