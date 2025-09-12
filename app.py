# ------------------------------------------------------------------------------
# Import libraries
# ------------------------------------------------------------------------------
import streamlit as st

# ------------------------------------------------------------------------------
# Page definitions
# ------------------------------------------------------------------------------
home_page = st.Page("pages/home_page.py", title="Get started", icon=":material/launch:")
content_based = st.Page("pages/content_based.py", title="by hotel name", icon=":material/search:")
content_based_search = st.Page("pages/content_based_search.py", title="by hotel description", icon=":material/search:")
als_page = st.Page("pages/als_page.py", title="by rating review", icon=":material/search:")
insight_page = st.Page("pages/insight_page.py", title="Hotel Insight", icon=":material/lightbulb:")
info_page = st.Page("pages/final_report_page.py", title="Final report", icon=":material/book:")

# ------------------------------------------------------------------------------
# Navigation structure
# ------------------------------------------------------------------------------
pg = st.navigation(
    {
        "Started": [home_page],
        "Suggestions": [content_based, content_based_search, als_page],
        "Insight": [insight_page],
        "About this project": [info_page],
    }
)

# ------------------------------------------------------------------------------
# App configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AGODA Hotel Recommendation",
    page_icon=":material/hotel:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": "mailto:phanlong.trong@gmail.com"},
)

# ------------------------------------------------------------------------------
# Run application
# ------------------------------------------------------------------------------
pg.run()
