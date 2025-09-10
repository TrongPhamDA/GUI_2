import streamlit as st
from PIL import Image

# load project main page
st.markdown(
    """
    <style>
    .main-title {
        font-size: 64px;
        font-weight: 900;
        color: #231F54;
        line-height: 1.0;
        margin-bottom: 0px;
        letter-spacing: -2px;
    }
    .subtitle {
        font-size: 32px;
        font-weight: 700;
        color: #231F54;
        margin-top: 0px;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }
    .team {
        font-size: 18px;
        font-weight: 500;
        color: #231F54;
        background-color: #F3F3F3;
        border-radius: 6px;
        padding: 8px 16px;
        display: inline-block;
        margin-bottom: 16px;
    }
    .submitted {
        font-size: 14px;
        font-weight: 400;
        color: #fff;
        background-color: #231F54;
        border-radius: 4px;
        padding: 4px 12px;
        display: inline-block;
        margin-top: 8px;
    }
    .report {
        font-size: 14px;
        font-weight: 600;
        color: #231F54;
        letter-spacing: 1px;
        margin-bottom: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="report">PROJECT_02</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AGODA</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">RECOMMENDATION<br>SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="team">DL07_K306, PHẠM NGỌC TRỌNG, TRẦN ĐÌNH HÙNG</div>', unsafe_allow_html=True)
st.markdown('<div class="submitted">submitted: 13/09/2025</div>', unsafe_allow_html=True)


# load Agoda banner
st.markdown("<br><br>", unsafe_allow_html=True)
st.image("static/agoda_mainpage.jpg", use_container_width =True)