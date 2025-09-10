# Hotel Data Science: Load hotel dataset and similarity matrix for content-based hotel recommendation
import streamlit as st
import pandas as pd

df_hotels = pd.read_csv('result/01_df_info.csv', header=0)
df_hotels['hotel_id'] = df_hotels['hotel_id'].astype(str)
st.session_state.random_hotels = df_hotels

df_matrix_gensim = pd.read_csv('result/03_gensim_tfidf_df_index_matrix.csv', header=None, index_col=False)
df_matrix_cosine = pd.read_csv('result/06_cosine_tfidf_df_index_matrix.csv', header=None, index_col=False)


####################################################
# Display main hotel platform image for user interface context
# st.image("static/agoda_mainpage.jpg", use_container_width=True)
import seaborn as sns
import matplotlib.pyplot as plt

st.markdown("""
<div style="background: #fff; border-radius: 18px; box-shadow: 0 4px 24px rgba(30,60,114,0.10); padding: 2.5rem 2rem 2rem 2rem; margin-bottom: 2.5rem;">
    <div style="display: flex; align-items: center;">
        <div style="flex: 0 0 140px; display: flex; align-items: center; justify-content: center; background: #f5f7fa; border-radius: 12px; height: 120px; margin-right: 2.5rem;">
            <img src="https://cdn6.agoda.net/images/kite-js/logo/agoda/color-default.svg" width="100" style="display: block;">
        </div>
        <div style="flex: 1;">
            <h1 style="color: #1e3c72; margin-bottom: 0.7rem; font-size: 2.1rem; font-weight: 700; letter-spacing: 0.5px;">
                AGODA Hotel Recommendation System
            </h1>
            <p style="color: #2a5298; font-size: 1.15rem; margin-bottom: 0.7rem; font-weight: 500;">
                Get personalized hotel suggestions powered by advanced content-based filtering.<br>
                Discover top-rated stays tailored to your unique preferences.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
####################################################

# Generate hotel selection options for user dropdown
hotel_options = [(row['hotel_name'], row['hotel_id']) for _, row in st.session_state.random_hotels.iterrows()]

# User interface: hotel selection and number of recommendations
# st.subheader("Hotel Search")
selected_hotel = st.selectbox("Try it now!", options=hotel_options, format_func=lambda x: x[0])
with st.sidebar:
    selected_model = st.radio("Select similarity model", options=["gensim", "cosine"], index=0, horizontal=True)
    top_k = st.slider("Select top N similar hotels", min_value=1, max_value=10, value=3, step=1)
    desc_limit = st.slider("Description limit (words)", min_value=50, max_value=500, value=100, step=50)

# Model used
df_matrix = df_matrix_gensim if selected_model == "gensim" else df_matrix_cosine

# Recommendation engine: retrieve top-k similar hotels based on content similarity matrix
def get_recommendations(df: pd.DataFrame, hotel_id, matrix: pd.DataFrame, top_k: int) -> pd.DataFrame:
    idx = df[df['hotel_id'] == hotel_id].index[0]
    sims = matrix.iloc[idx].copy().drop(idx, errors='ignore')
    top_idx = sims.nlargest(top_k).index
    return df.iloc[top_idx].assign(similarity=sims.loc[top_idx].values).sort_values('similarity', ascending=False).drop(columns='similarity')

# Display recommended hotels in a grid layout for enhanced user experience
def display_recommended_hotels(recommended_hotels: pd.DataFrame, cols: int):
    for i in range(0, len(recommended_hotels), cols):
        col_objs = st.columns(cols)
        for j, col in enumerate(col_objs):
            if i + j < len(recommended_hotels):
                hotel = recommended_hotels.iloc[i + j]
                with col:
                    st.write(hotel['hotel_name'])
                    st.expander("Hotel Description")

# Store selected hotel state for session continuity
if 'selected_hotel_id' not in st.session_state:
    st.session_state.selected_hotel_id = None
st.session_state.selected_hotel_id = selected_hotel[1]

# Display detailed hotel information including business metrics and guest ratings
def display_hotel_info(hotel_info: pd.Series):
    st.write('###', hotel_info['hotel_name'])
    with st.container():
        st.markdown(f"**Address:** {hotel_info['hotel_address']}")
        cols_info = st.columns(6)
        cols_info[0].metric("Total Score", hotel_info['total_score'])
        cols_info[1].metric("Rank", hotel_info['hotel_rank'])
        cols_info[2].metric("Comments Count", f"{hotel_info['comments_count']:,}".replace(",", "."))
        st.markdown("##### Ratings")
        cols = st.columns(6)
        cols[0].metric("Location", hotel_info['location'])
        cols[1].metric("Cleanliness", hotel_info['cleanliness'])
        cols[2].metric("Service", hotel_info['service'])
        cols[3].metric("Facilities", hotel_info['facilities'])
        cols[4].metric("Value for Money", hotel_info['value_for_money'])
        cols[5].metric("Comfort & Room Quality", hotel_info['comfort_and_room_quality'])
        with st.expander("Hotel Description", expanded=True):
            st.write(' '.join(str(hotel_info['hotel_description']).split()[:desc_limit]) + "...")

# Display comparison table for selected and recommended hotels based on key business criteria
def display_comparison_table(selected_hotel: pd.Series, recommended_hotels: pd.DataFrame, criteria: list):
    columns = ['Fact', selected_hotel['hotel_name']] + recommended_hotels['hotel_name'].tolist()
    data = []
    for crit in criteria:
        row = [crit.replace('_', ' ').title()]
        row.append(selected_hotel.get(crit, ""))
        for _, hotel in recommended_hotels.iterrows():
            row.append(hotel.get(crit, ""))
        data.append(row)
    st.table(pd.DataFrame(data, columns=columns))

# Display recommendations section with professional comparison table for hotel analytics
def display_recommendations_section(hotel_info: pd.Series, recommendations: pd.DataFrame):
    st.markdown("<h4 style='color:#2C3E50;font-weight:700;'>Compare Hotels Side by Side</h4>", unsafe_allow_html=True)
    if not recommendations.empty:
        criteria = [
            'hotel_address', 'hotel_rank', 'comments_count', 'total_score',
            'location', 'cleanliness', 'service', 'facilities', 'value_for_money', 'comfort_and_room_quality', 
            'hotel_description',
        ]
        names = [hotel_info['hotel_name']] + recommendations['hotel_name'].tolist()
        num_hotels = len(names)
        col_fact_width = 18
        col_other_width = round((100 - col_fact_width) / num_hotels, 2)
        th_style = f"background:#f5f6fa;color:#2C3E50;padding:8px;text-align:center;border:1px solid #e1e1e1;"
        td_style = f"padding:8px;text-align:center;border:1px solid #e1e1e1;"
        st.markdown(f"""
        <style>
        .compare-table th, .compare-table td {{border:1px solid #e1e1e1;padding:8px;text-align:center;}}
        .compare-table th {{background:#f5f6fa;color:#2C3E50;}}
        .compare-table tr:nth-child(even) {{background:#f9f9f9;}}
        </style>
        """, unsafe_allow_html=True)
        table = f"<table class='compare-table' style='width:100%;border-collapse:collapse;'><tr>"
        table += f"<th style='{th_style}width:{col_fact_width}%;text-align:left;'>Fact</th>"
        for n in names:
            table += f"<th style='{th_style}width:{col_other_width}%;'>{n}</th>"
        table += "</tr>"
        for c in criteria:
            table += f"<tr><td style='font-weight:600;text-align:left;width:{col_fact_width}%;{td_style}'>{c.replace('_',' ').title()}</td>"
            if c == 'hotel_description':
                desc_main = ' '.join(str(hotel_info.get(c, '')).split()[:desc_limit]) + "..."
                table += f"<td style='max-width:350px;word-break:break-word;text-align:left;vertical-align:top;width:{col_other_width}%;{td_style};text-align:left;vertical-align:top;'>{desc_main}</td>"
                for _, h in recommendations.iterrows():
                    desc_rec = ' '.join(str(h.get(c, '')).split()[:desc_limit]) + "..."
                    table += f"<td style='max-width:350px;word-break:break-word;text-align:left;vertical-align:top;width:{col_other_width}%;{td_style};text-align:left;vertical-align:top;'>{desc_rec}</td>"
            elif c in ['hotel_rank', 'total_score']:
                value_main = hotel_info.get(c, '')
                table += f"<td style='font-size:1.5em;font-weight:bold;width:{col_other_width}%;{td_style}'>{value_main}</td>"
                for _, h in recommendations.iterrows():
                    value_rec = h.get(c, '')
                    table += f"<td style='font-size:1.5em;font-weight:bold;width:{col_other_width}%;{td_style}'>{value_rec}</td>"
            else:
                table += f"<td style='width:{col_other_width}%;{td_style}'>{hotel_info.get(c,'')}</td>"
                for _, h in recommendations.iterrows():
                    table += f"<td style='width:{col_other_width}%;{td_style}'>{h.get(c,'')}</td>"
            table += "</tr>"
        table += "</table>"
        st.markdown(table, unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#c0392b;'>No recommended hotels found.</div>", unsafe_allow_html=True)

# Main function: display selected hotel information and content-based recommendations
def main_display_selected_hotel(selected_hotel_id, df_hotels, df_matrix, top_k):
    selected_hotel_df = df_hotels[df_hotels['hotel_id'] == selected_hotel_id]
    if not selected_hotel_df.empty:
        hotel_info = selected_hotel_df.iloc[0]
        display_hotel_info(hotel_info)
        recommendations = get_recommendations(df=df_hotels, hotel_id=selected_hotel_id, matrix=df_matrix, top_k=top_k)
        display_recommendations_section(hotel_info, recommendations)
    else:
        st.write(f"No hotel found with ID: {selected_hotel_id}")

# Execute display logic when a hotel is selected by the user
if st.session_state.selected_hotel_id is not None:
    main_display_selected_hotel(
        selected_hotel_id=st.session_state.selected_hotel_id,
        df_hotels=df_hotels,
        df_matrix=df_matrix,
        top_k=top_k
    )