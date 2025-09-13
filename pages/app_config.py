# Hotel Data Science: Application Configuration Reference
"""
Configuration file containing default values for the hotel recommendation application.
This file centralizes all configurable parameters for easy maintenance and updates.
"""

# ------------------------------------------------------------------------------
# UI Configuration
# ------------------------------------------------------------------------------

# Main page header configuration
IMG_SRC = "https://cdn6.agoda.net/images/kite-js/logo/agoda/color-default.svg"
PAGE_TITLE = "AGODA Hotel Recommendation"

# Page descriptions
DESCRIPTION_1 = "Get tailored hotel suggestions with advanced filtering"
DESCRIPTION_name = "Discover top-rated stays tailored to your unique preferences"
DESCRIPTION_search = "Describe your dream stay—beachfront, city, or luxury—and we'll find your perfect match"
DESCRIPTION_finalreport = "About 'Final report' of the project"
DESCRIPTION_insight = "Discover hotel strengths and improve guest experience through review insights"
DESCRIPTION_als = "Advanced content-based filtering for personalized hotel recommendations"

# ------------------------------------------------------------------------------
# Default Parameters
# ------------------------------------------------------------------------------

# Search parameters
DEFAULT_SEARCH_INPUT = "khách sạn mới, rộng, gần biển, có trẻ em"

# UI Control ranges
TOP_K_MIN = 1
TOP_K_MAX = 10
TOP_K_STEP = 1
DEFAULT_TOP_K = 3

DESC_LIMIT_MIN = 30
DESC_LIMIT_MAX = 500
DESC_LIMIT_STEP = 10
DEFAULT_DESC_LIMIT = 30

# ------------------------------------------------------------------------------
# File Paths
# ------------------------------------------------------------------------------

# Data files
HOTEL_INFO_CSV = "result/01_df_info.csv"
HOTEL_INFO_TOKEN_CSV = "input/hotel_info_token.csv"
HOTEL_COMMENTS_CSV = "result/02_df_comments.csv"
HOTEL_COMMENTS_TOKEN_CSV = "result/13_insight_textmining_comments_classification_token_cleaned.csv"
ALS_RECOMMENDATION_FOR_ALL_USERS = "result/09_ALS_recommendation_for_all_user.csv"

# Report files
FINAL_REPORT = "static/Final_Report.pdf"
FINAL_REPORT_IMAGES_DIR = "static/final_report/"
COL_NUM = 2

# Similarity matrices
GENSIM_MATRIX_CSV = "result/03_gensim_tfidf_df_index_matrix.csv"
COSINE_MATRIX_CSV = "result/06_cosine_tfidf_df_index_matrix.csv"

# Model files
TFIDF_MODEL_PKL = "models/01_tfidf.pkl"
DICTIONARY_DICT = "models/02_dictionary.dict"
GENSIM_MATRIX_INDEX = "models/03_gesim_sparsematrixsimilarity.index"

# Dictionary files
EMOJI_DICT_PATH = "input/dictionaries/emojicon.txt"
ENGVIE_DICT_PATH = "input/dictionaries/english-vnmese.txt"
TEENCODE_DICT_PATH = "input/dictionaries/teencode.txt"
STOPWORD_PATH = "input/dictionaries/vietnamese-stopwords.txt"
WRONGWORD_PATH = "input/dictionaries/wrong-word.txt"

# ------------------------------------------------------------------------------
# Model Configuration
# ------------------------------------------------------------------------------

# Available models
AVAILABLE_MODELS = ["gensim", "cosine"]
DEFAULT_MODEL = "gensim"

# ------------------------------------------------------------------------------
# Display Configuration
# ------------------------------------------------------------------------------

# Grid layout
DEFAULT_GRID_COLS = 3

# Table styling
TABLE_FACT_WIDTH = 18
TABLE_HEADER_STYLE = "background:#f5f6fa;color:#2C3E50;padding:8px;text-align:center;border:1px solid #e1e1e1;"
TABLE_CELL_STYLE = "padding:8px;text-align:center;border:1px solid #e1e1e1;"

# Colors
PRIMARY_COLOR = "#1e3c72"
SECONDARY_COLOR = "#2a5298"
ACCENT_COLOR = "#2C3E50"
ERROR_COLOR = "#c0392b"

# ------------------------------------------------------------------------------
# Insight Analysis Configuration
# ------------------------------------------------------------------------------

# Chart configuration
DEFAULT_FIGSIZE = (10, 5)
DEFAULT_BINS_TIME = 2
DEFAULT_DPI = 150

# Chart colors
CHART_COLORS = {
    "hist": "#4C72B0",
    "kde": "#55A868", 
    "highlight_bin": "#FFB000",
    "edge": "black"
}

# Insight analysis parameters
INSIGHT_DESC_LIMIT = 150

# Rating columns for analysis
RATING_COLS = ["location", "cleanliness", "service", "facilities", "value_for_money", "comfort_and_room_quality"]

# Score classification thresholds
SCORE_CLASSIFY_DICT = {
    "Strength": 8.5,
    "Neutral": 7.5,
}

# Chart types configuration
CHART_TYPES = {
    "radar": "Radar Chart",
    "customer": "Customer Analysis", 
    "wordcloud": "Word Cloud",
    "distribution": "Score Distribution"
}

# Customer analysis chart configuration
CUSTOMER_CHART_COLS = {
    "fn_bar_chart": ["nationality", "group_name", "room_type"],
    "fn_line_timeseries_chart": ["review_month"]
}

# Customer comparison chart configuration
CUSTOMER_COMPARISON_CONFIG = {
    "chart_types": [
        {"type": "bar", "column": "nationality", "title": "Top 10 Nationality"},
        {"type": "bar", "column": "group_name", "title": "Top 10 Group Name"},
        {"type": "bar", "column": "room_type", "title": "Top 10 Room Type"},
        {"type": "line", "column": "review_month", "title": "Review by Month"},
        {"type": "line", "column": "review_year", "title": "Review by Year"}
    ],
    "colors": {
        "selected_hotel": "#4C72B0",
        "all_hotels": "#55A868"
    }
}

# Word cloud configuration
WORDCLOUD_CONFIG = {
    "width": 1000,
    "height": 500,
    "background_color": "white",
    "colormap": "viridis",
    "prefer_horizontal": 1.0,
    "max_words": 50
}

# Word count limit configuration
WORD_COUNT_LIMIT_MIN = 10
WORD_COUNT_LIMIT_MAX = 100
WORD_COUNT_LIMIT_DEFAULT = 20
WORD_COUNT_LIMIT_STEP = 5

# ------------------------------------------------------------------------------
# Text Processing Configuration
# ------------------------------------------------------------------------------

# Token cleaning parameters
CLEAN_TOKENS_CONFIG = {
    "remove_number": True,
    "remove_punctuation": True,
    "remove_vie_tone": False,
    "lower": True
}

# ------------------------------------------------------------------------------
# Owner Information
# ------------------------------------------------------------------------------

# Owner 1 - Phạm Ngọc Trọng
OWNER_1 = {
    "name": "Phạm Ngọc Trọng",
    "position": "Owner",
    "email": "phanlong.trong@gmail.com",
    "phone": "034 981 6784",
    "website": "https://www.facebook.com/TrongPhamDA",
    "image_src": "static/owner.png"
}

# Owner 2 - Trần Đình Hùng
OWNER_2 = {
    "name": "Trần Đình Hùng",
    "position": "Business Domain Advisor", 
    "email": "tdhung.dl@gmail.com",
    "phone": "000 0000 000",
    "website": "https://github.com/trandinhhung",
    "image_src": "static/team_member.png"
}

# Owner 3 - Khuất Phương
OWNER_3 = {
    "name": "Khuất Thùy Phương",
    "position": "Instructor",
    "email": "tubirona@gmail.com",
    "phone": "000 0000 000",
    "website": "https://github.com/khuatphuong",
    "image_src": "static/instructor.png"
}

# Combined owners list
OWNER = [OWNER_1, OWNER_2, OWNER_3]

# Project Information
PROJECT_INFO = {
    "title": "AGODA Hotel Recommendation System",
    "course": "DL07_K306",
    "submission_date": "13/09/2025",
    "university": "Data Science & Machine Learning Certificate"
}

# ------------------------------------------------------------------------------
# Display Configuration
# ------------------------------------------------------------------------------

# Footer display control
SHOW_FOOTER = True

# ------------------------------------------------------------------------------
# Session State Keys
# ------------------------------------------------------------------------------

SESSION_KEYS = {
    "random_hotels": "random_hotels",
    "selected_hotel_id": "selected_hotel_id"
}
