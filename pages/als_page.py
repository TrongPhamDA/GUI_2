# ------------------------------------------------------------------------------
# Import libraries
import streamlit as st
import pandas as pd
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from myfunctions import *
from app_config import *

# ------------------------------------------------------------------------------
# Load hotel dataset
df_als_recs = pd.read_csv(ALS_RECOMMENDATION_FOR_ALL_USERS, header=0)
df_hotels = pd.read_csv(HOTEL_INFO_CSV, header=0)

# ------------------------------------------------------------------------------
# Display main hotel platform image
# st.image("static/agoda_mainpage.jpg", use_container_width=True)

# Render main page header
fn_render_mainpage_header(
    img_src=IMG_SRC,
    page_title=PAGE_TITLE,
    description_1=DESCRIPTION_1,
    description_2=DESCRIPTION_3,
)


# ------------------------------------------------------------------------------
# Preprocess data
df_reviewer = df_als_recs[['reviewer_name', 'hotel_id']].merge(
    df_hotels[['hotel_id']], 
    on='hotel_id', 
    how='inner'
).drop_duplicates()

df_reviewer_counts = df_reviewer.groupby('reviewer_name').size().reset_index(name='count')
df_reviewer_counts = df_reviewer_counts.sort_values(['count', 'reviewer_name'], ascending=[False, True])

# ------------------------------------------------------------------------------
# User interface controls
with st.sidebar:
    hotel_count_option = st.radio(
        "Select users with minimum available hotels:",
        options=[3, 2, 1],
        format_func=lambda x: f"{x} hotels",
        index=0
    )
    
    filtered_users = df_reviewer_counts[df_reviewer_counts['count'] == hotel_count_option]
    st.info(f"Found {len(filtered_users)} users with = {hotel_count_option} available hotels")
    
    if not filtered_users.empty:
        selected_user = st.selectbox(
            "Select Reviewer:",
            options=filtered_users['reviewer_name'].tolist(),
            index=0,
            format_func=lambda x: f"{x} ({filtered_users[filtered_users['reviewer_name']==x]['count'].iloc[0]} hotels)"
        )
    else:
        selected_user = None
        st.warning(f"No reviewers found with ≥{hotel_count_option} available hotels")
    
    # Description limit control
    desc_limit = st.slider(
        "Description limit (words)", 
        min_value=DESC_LIMIT_MIN, 
        max_value=DESC_LIMIT_MAX, 
        value=DEFAULT_DESC_LIMIT, 
        step=DESC_LIMIT_STEP
    )

# ------------------------------------------------------------------------------
# Main program - Display ALS recommendations
if selected_user:
    user_recommendations = df_als_recs[df_als_recs['reviewer_name'] == selected_user].sort_values('score', ascending=False)
    user_available_hotels = df_reviewer[df_reviewer['reviewer_name'] == selected_user]['hotel_id'].tolist()
    user_recommendations = user_recommendations[user_recommendations['hotel_id'].isin(user_available_hotels)].head(3)
    
    if not user_recommendations.empty:
        st.markdown(f"### Customer: {selected_user}")
        st.markdown(f"**Top {len(user_recommendations)} available recommended hotels:**")
        st.markdown("---")
        
        recommendation_hotel_ids = user_recommendations['hotel_id'].tolist()
        recommendations = df_hotels[df_hotels['hotel_id'].isin(recommendation_hotel_ids)].copy()
        
        score_dict = dict(zip(user_recommendations['hotel_id'], user_recommendations['score']))
        recommendations['als_score'] = recommendations['hotel_id'].map(score_dict)
        recommendations = recommendations.set_index('hotel_id').reindex(recommendation_hotel_ids).reset_index()
        
        for idx, (_, hotel) in enumerate(recommendations.iterrows()):
            st.markdown(f"**{idx+1}. {hotel['hotel_name']}** (Score: {hotel['als_score']:.2f})")
        
        st.markdown("---")
        
        if len(recommendations) > 1:
            main_hotel = recommendations.iloc[0]
            other_recommendations = recommendations.iloc[1:]
            
            fn_display_recommendations_section(
                main_hotel=main_hotel,
                recommendations=other_recommendations,
                top_k=len(recommendations),
                desc_limit=desc_limit
            )
        else:
            st.markdown(f"#### About this hotel:")
            fn_display_hotel_info(recommendations.iloc[0], desc_limit)
    else:
        st.warning("No recommendations found for this user.")

# ------------------------------------------------------------------------------
# Render footer
fn_render_footer(OWNER, PROJECT_INFO, SHOW_FOOTER)
