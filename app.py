import streamlit as st

# định nghĩa trang
home_page = st.Page("pages/home_page.py", title="Get started", icon=":material/launch:")
content_based = st.Page("pages/content_based.py", title="select a hotel", icon=":material/search:")
content_based_search = st.Page("pages/content_based_search.py", title="describe a hotel", icon=":material/search:")
als_page = st.Page("pages/als_page.py", title="personalize your preferences", icon=":material/search:")
insight_page = st.Page("pages/insight_page.py", title="Hotel Insight", icon=":material/thumb_up:")
info_page = st.Page("pages/info_page.py", title="Final report", icon=":material/book:")

# tạo thanh navigation
pg = st.navigation(
    {"Started" : [home_page],
    "Suggestions" : [content_based, content_based_search, als_page],
    "Insight" : [insight_page],
    "Our team" : [info_page]
    })

# config
st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon=":material/hotel:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:phanlong.trong@gmail.com'
    }
)

# show trang
pg.run()