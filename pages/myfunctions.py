# Hotel Data Science Functions
import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from wordcloud import WordCloud
from collections import Counter
import sys
import ast

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app_config import *


# ------------------------------------------------------------------------------
# Text Processing Functions
# ------------------------------------------------------------------------------

def fn_clean_tokens(
    text,
    dict_list=None,
    stopword=None,
    wrongword=None,
    remove_number=False,
    remove_punctuation=True,
    remove_vie_tone=False,
    lower=True,
):
    """
    Clean hotel review text tokens for NLP analysis
    
    Args:
        text: Input text or list of texts
        dict_list: Dictionary list for text replacement
        stopword: Stopwords to remove
        wrongword: Wrong words to remove
        remove_number: Remove numbers flag
        remove_punctuation: Remove punctuation flag
        remove_vie_tone: Remove Vietnamese tones flag
        lower: Convert to lowercase flag
    
    Returns:
        Cleaned tokens list
    """
    def remove_accents(text):
        text = unicodedata.normalize("NFD", text)
        text = text.encode("ascii", "ignore").decode("utf-8")
        return str(text)

    punctuations = set(string.punctuation)

    def _process_token(tok):
        if dict_list:
            for d in dict_list:
                if tok in d:
                    tok = d[tok]
        if remove_number and re.fullmatch(r"[0-9]+(\.[0-9]+)?", tok):
            return None
        if remove_punctuation:
            tok = "".join([c for c in tok if c not in punctuations])
            if not tok.strip():
                return None
        if lower:
            tok = tok.lower()
        if wrongword and tok in wrongword:
            return None
        if remove_vie_tone:
            tok = remove_accents(tok)
        if tok.strip():
            return tok
        return None

    def is_empty_or_nan(x):
        if x is None:
            return True
        if isinstance(x, float) and np.isnan(x):
            return True
        if isinstance(x, str) and str(x).strip() == "":
            return True
        return False

    if is_empty_or_nan(text) or (
        isinstance(text, list) and all(is_empty_or_nan(t) for t in text)
    ):
        return [] if isinstance(text, list) else []
    if isinstance(text, list):
        texts = text
    else:
        texts = [text]
    results = []
    for t in texts:
        if is_empty_or_nan(t):
            results.append([])
            continue
        tokens = str(t).split()
        cleaned = []
        for tok in tokens:
            res = _process_token(tok)
            if res is not None:
                cleaned.append(res)
        if stopword:
            cleaned = [tok for tok in cleaned if tok not in stopword]
        results.append(cleaned)
    if isinstance(text, list):
        return results
    else:
        return results[0]


def fn_read_txt(path):
    """
    Read text file lines
    
    Args:
        path: Text file path
    
    Returns:
        Lines list
    """
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def fn_read_dict(path):
    """
    Read dictionary file with tab-separated pairs
    
    Args:
        path: Dictionary file path
    
    Returns:
        Key-value dictionary
    """
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if "\t" in line:
                eng, vie = line.strip().split("\t", 1)
                d[eng.strip()] = vie.strip()
    return d


# ------------------------------------------------------------------------------
# Hotel Recommendation Functions
# ------------------------------------------------------------------------------

def fn_get_recommendations_page1(
    df: pd.DataFrame, hotel_id, matrix: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    """
    Get top-k similar hotels using content similarity matrix
    
    Args:
        df: Hotel information DataFrame
        hotel_id: Selected hotel ID
        matrix: Similarity matrix
        top_k: Number of recommendations
    
    Returns:
        Top-k similar hotels DataFrame
    """
    idx = df[df["hotel_id"] == hotel_id].index[0]
    sims = matrix.iloc[idx].copy().drop(idx, errors="ignore")
    top_idx = sims.nlargest(top_k).index
    return (
        df.iloc[top_idx]
        .assign(similarity=sims.loc[top_idx].values)
        .sort_values("similarity", ascending=False)
        .drop(columns="similarity")
    )


def fn_get_recommendations_gensim(
    gensim_matrix,
    corpora_dictionary,
    tfidf_model,
    df_hotel_info,
    search_input_tokens,
    top_k: int,
) -> pd.DataFrame:
    """
    Get hotel recommendations using Gensim TF-IDF model
    
    Args:
        gensim_matrix: Gensim similarity matrix
        corpora_dictionary: Gensim dictionary
        tfidf_model: TF-IDF model
        df_hotel_info: Hotel information DataFrame
        search_input_tokens: Tokenized search input
        top_k: Number of recommendations
    
    Returns:
        Recommended hotels DataFrame
    """
    sim = gensim_matrix[tfidf_model[corpora_dictionary.doc2bow(search_input_tokens)]]
    recommend = pd.DataFrame({"id": range(len(sim)), "sim": sim}).nlargest(
        n=top_k, columns="sim"
    )
    top_indices = recommend["id"].to_list()
    return df_hotel_info.loc[df_hotel_info.index[top_indices]]


def fn_get_recommendations_cosine(
    df_hotel_info, search_input, top_k: int
) -> pd.DataFrame:
    """
    Get hotel recommendations using cosine similarity
    
    Args:
        df_hotel_info: Hotel information DataFrame
        search_input: Search query text
        top_k: Number of recommendations
    
    Returns:
        Recommended hotels DataFrame
    """
    vectorizer = TfidfVectorizer(analyzer="word", stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(
        df_hotel_info["content_wt"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x)
        )
    )
    search_vec = vectorizer.transform([search_input])
    cosine_sim_search = cosine_similarity(search_vec, tfidf_matrix).flatten()
    recommend_cosine = pd.DataFrame(
        {"id": range(len(cosine_sim_search)), "sim": cosine_sim_search}
    ).nlargest(n=top_k, columns="sim")
    top_indices = recommend_cosine["id"].to_list()
    return df_hotel_info.loc[df_hotel_info.index[top_indices]]


def fn_get_recommendations_search(
    selected_model,
    gensim_matrix,
    corpora_dictionary,
    tfidf_model,
    df_hotel_info,
    search_input_tokens,
    search_input,
    top_k: int,
) -> pd.DataFrame:
    """
    Get hotel recommendations based on search input using selected model
    
    Args:
        selected_model: Model type ('gensim' or 'cosine')
        gensim_matrix: Gensim similarity matrix
        corpora_dictionary: Gensim dictionary
        tfidf_model: TF-IDF model
        df_hotel_info: Hotel information DataFrame
        search_input_tokens: Tokenized search input
        search_input: Original search input text
        top_k: Number of recommendations
    
    Returns:
        Recommended hotels DataFrame
    """
    if selected_model == "gensim":
        return fn_get_recommendations_gensim(
            gensim_matrix=gensim_matrix,
            corpora_dictionary=corpora_dictionary,
            tfidf_model=tfidf_model,
            search_input_tokens=search_input_tokens,
            df_hotel_info=df_hotel_info,
            top_k=top_k,
        )
    elif selected_model == "cosine":
        return fn_get_recommendations_cosine(
            search_input=search_input,
            df_hotel_info=df_hotel_info,
            top_k=top_k,
        )


# ------------------------------------------------------------------------------
# Render main page header
# ------------------------------------------------------------------------------
def fn_render_mainpage_header(img_src, page_title, description_1, description_2):
    st.markdown(
        f"""
    <div style="background: #fff; border-radius: 18px; box-shadow: 0 4px 24px rgba(30,60,114,0.10); padding: 2.5rem 2rem 2rem 2rem; margin-bottom: 2.5rem;">
        <div style="display: flex; align-items: center;">
            <div style="flex: 0 0 140px; display: flex; align-items: center; justify-content: center; background: #f5f7fa; border-radius: 12px; height: 120px; margin-right: 2.5rem;">
                <img src="{img_src}" width="100" style="display: block;">
            </div>
            <div style="flex: 1;">
                <h1 style="color: #1e3c72; margin-bottom: 0.7rem; font-size: 2.1rem; font-weight: 700; letter-spacing: 0.5px;">
                    {page_title}
                </h1>
                <p style="color: #2a5298; font-size: 1.15rem; margin-bottom: 0.7rem; font-weight: 500;">
                    {description_1}<br>
                    <strong>{description_2}</strong>
                </p>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def fn_render_footer(owner_list, project_info, show_footer=True):
    """
    Render website footer with owner information and project details
    
    Args:
        owner_list: List of owner dictionaries
        project_info: Dictionary with project information
        show_footer: Boolean to control footer display
    """
    if not show_footer:
        return

    # Footer Bottom Section
    st.markdown("---")
    st.markdown(f"© 2025 {project_info['title']}")
    
    # Create columns for owners
    owner_columns = st.columns(len(owner_list))
    
    # Render each owner using loop
    for idx, owner in enumerate(owner_list):
        with owner_columns[idx]:
            # Create two columns for each owner: left for image, right for info
            col_img, col_info = st.columns([1, 2])
            
            with col_img:
                # Display owner image
                st.image(owner['image_src'], caption="")
            
            with col_info:
                # Display owner information
                st.markdown(
                    f"""
                    <div style="padding: 0.5rem;">
                        <div style="font-weight: 600; color: #1e3c72; margin-bottom: 0.3rem; font-size: 1rem;">{owner['name']}</div>
                        <div style="font-style: italic; color: #6c757d; margin-bottom: 0.4rem; font-size: 0.85rem;">{owner['position']}</div>
                        <div style="color: #495057; font-size: 0.8rem; margin-bottom: 0.2rem;">{owner['email']}</div>
                        <div style="color: #495057; font-size: 0.8rem; margin-bottom: 0.2rem;">{owner['phone']}</div>
                        <a href="{owner['website']}" style="color: #007bff; text-decoration: none; font-size: 0.8rem;">GitHub Profile</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    st.write("---")


# ------------------------------------------------------------------------------
# PDF Display Functions
# ------------------------------------------------------------------------------

def fn_display_report(pdf_path, images_dir, col_num = 2):
    """Display PDF report with download option and images viewer"""
    with open(pdf_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()
    
    st.download_button("📄 Download Final Report (PDF)", pdf_bytes, "Final_Report.pdf", "application/pdf")
    st.markdown("---")
    
    st.markdown("#### Slides")
    images_per_row = col_num
    
    # Lấy danh sách tất cả file ảnh trong thư mục và sắp xếp theo thứ tự a-z
    image_files = sorted(glob.glob(f"{images_dir}*.png"))
    
    for idx, image_path in enumerate(image_files):
        if idx % images_per_row == 0:
            cols = st.columns(images_per_row)
        
        with cols[idx % images_per_row]:
            try:
                # Lấy tên file từ đường dẫn
                filename = os.path.basename(image_path)
                st.image(image_path, use_container_width=True, caption=filename)
            except Exception as e:
                st.error(f"Không thể hiển thị ảnh {filename}: {str(e)}")


# ------------------------------------------------------------------------------
# Hotel Display Functions
# ------------------------------------------------------------------------------

def fn_display_recommended_hotels(recommended_hotels: pd.DataFrame, cols: int):
    """
    Display recommended hotels in grid layout
    
    Args:
        recommended_hotels: Recommended hotels DataFrame
        cols: Number of grid columns
    """
    for i in range(0, len(recommended_hotels), cols):
        col_objs = st.columns(cols)
        for j, col in enumerate(col_objs):
            if i + j < len(recommended_hotels):
                hotel = recommended_hotels.iloc[i + j]
                with col:
                    st.write(hotel["hotel_name"])
                    st.expander("Hotel Description")


def fn_rank_star(rank):
    if pd.isna(rank) or str(rank).strip() == "...":
        return " "
    full_stars = int(rank)
    half_star = 1 if (rank - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    star_html = (
        "<span style='color: #FFD700; font-size: 2.0rem;'>"
        + "★" * full_stars
        + ("½" if half_star else "")
        + "<span style='color: #e1e1e1;'>" + "★" * empty_stars + "</span>"
        + "</span>"
        + f" <span style='font-size:1.1rem; color:#444;'>({rank:.1f})</span>"
    )
    return star_html


def fn_display_hotel_info(hotel_info: pd.Series, desc_limit: int = 100):
    """
    Display detailed hotel information including business metrics and guest ratings
    
    Args:
        hotel_info: Series containing hotel information
        desc_limit: Maximum number of words in description
    """
    st.write(f"<h4 style='font-size: 2.2rem; font-weight: 700; color: #1e3c72; margin-bottom: 1rem;'>{hotel_info['hotel_name']}</h2>", unsafe_allow_html=True)
    with st.container():
        st.markdown(f"**Address:** {hotel_info['hotel_address']}")
        cols_info = st.columns(6)
        cols_info[0].markdown(f"**Total Score**  \n<span style='font-size:2rem; font-weight:700; color:#1e3c72'>{hotel_info['total_score'] if pd.notna(hotel_info['total_score']) else '...'}</span>", unsafe_allow_html=True)
        cols_info[1].markdown(f"**Rank**  \n{fn_rank_star(hotel_info['hotel_rank'])}", unsafe_allow_html=True)
        cols_info[2].metric(
            "Comments Count", f"{hotel_info['comments_count']:,}".replace(",", ".") if pd.notna(hotel_info["comments_count"]) else "..."
        )
        st.markdown("##### Ratings")
        cols = st.columns(6)
        cols[0].metric("Location", hotel_info["location"] if pd.notna(hotel_info["location"]) else "...")
        cols[1].metric("Cleanliness", hotel_info["cleanliness"] if pd.notna(hotel_info["cleanliness"]) else "...")
        cols[2].metric("Service", hotel_info["service"] if pd.notna(hotel_info["service"]) else "...")
        cols[3].metric("Facilities", hotel_info["facilities"] if pd.notna(hotel_info["facilities"]) else "...")
        cols[4].metric("Value for Money", hotel_info["value_for_money"] if pd.notna(hotel_info["value_for_money"]) else "...")
        cols[5].metric("Comfort & Room Quality", hotel_info["comfort_and_room_quality"] if pd.notna(hotel_info["comfort_and_room_quality"]) else "...")
        with st.expander("Hotel description: Show Details", expanded=False):
            st.write(
                " ".join(str(hotel_info["hotel_description"]).split()[:desc_limit])
                + "..."
            )


def fn_display_comparison_table(
    main_hotel: pd.Series, recommended_hotels: pd.DataFrame, criteria: list
):
    """
    Display comparison table for selected and recommended hotels
    
    Args:
        main_hotel: Series with main hotel information
        recommended_hotels: DataFrame with recommended hotels
        criteria: List of criteria to compare
    """
    columns = ["Fact", main_hotel["hotel_name"]] + recommended_hotels[
        "hotel_name"
    ].tolist()
    data = []
    for crit in criteria:
        row = [crit.replace("_", " ").title()]
        row.append(main_hotel.get(crit, ""))
        for _, hotel in recommended_hotels.iterrows():
            row.append(hotel.get(crit, ""))
        data.append(row)
    st.table(pd.DataFrame(data, columns=columns))


def fn_display_recommendations_section(
    main_hotel: pd.Series, recommendations: pd.DataFrame, top_k: int, desc_limit: int = 100
):
    """
    Display recommendations section with professional comparison table
    
    Args:
        main_hotel: Series with main hotel information
        recommendations: DataFrame with recommended hotels
        top_k: Number of recommendations
        desc_limit: Maximum number of words in description
    """
    st.markdown(
        f"<h4 style='color:#2C3E50;font-weight:700;'>Compare top {top_k} hotels for you</h4>",
        unsafe_allow_html=True,
    )
    if not recommendations.empty:
        criteria = [
            "hotel_address",
            "hotel_rank",
            "comments_count",
            "total_score",
            "location",
            "cleanliness",
            "service",
            "facilities",
            "value_for_money",
            "comfort_and_room_quality",
            "hotel_description",
        ]
        
        main_hotel_clean = main_hotel.copy()
        recommendations_clean = recommendations.copy()
        
        for c in criteria:
            if c in main_hotel_clean.index and pd.isna(main_hotel_clean[c]):
                main_hotel_clean[c] = "..."
            if c in recommendations_clean.columns:
                recommendations_clean[c] = recommendations_clean[c].fillna("...")
        
        names = [main_hotel_clean["hotel_name"]] + recommendations_clean["hotel_name"].tolist()
        num_hotels = len(names)
        col_fact_width = 18
        col_other_width = round((100 - col_fact_width) / num_hotels, 2)
        th_style = "background:#f5f6fa;color:#2C3E50;padding:8px;text-align:center;border:1px solid #e1e1e1;"
        td_style = "padding:8px;text-align:center;border:1px solid #e1e1e1;"
        st.markdown(
            """
        <style>
        .compare-table th, .compare-table td {border:1px solid #e1e1e1;padding:8px;text-align:center;}
        .compare-table th {background:#f5f6fa;color:#2C3E50;}
        .compare-table tr:nth-child(even) {background:#f9f9f9;}
        </style>
        """,
            unsafe_allow_html=True,
        )
        table = "<table class='compare-table' style='width:100%;border-collapse:collapse;'><tr>"
        table += (
            f"<th style='{th_style}width:{col_fact_width}%;text-align:left;'>Fact</th>"
        )
        for n in names:
            table += f"<th style='{th_style}width:{col_other_width}%;'>{n}</th>"
        table += "</tr>"
        for c in criteria:
            table += f"<tr><td style='font-weight:600;text-align:left;width:{col_fact_width}%;{td_style}'>{c.replace('_',' ').title()}</td>"
            if c == "hotel_description":
                # Sử dụng expander cho từng khách sạn, chỉ show desc_rec_short khi nhấn Show details
                desc_main_full = str(main_hotel_clean.get(c, ""))
                desc_main_short = " ".join(desc_main_full.split()[:desc_limit]) + ("..." if len(desc_main_full.split()) > desc_limit else "")
                table += f"<td style='max-width:350px;word-break:break-word;text-align:left;vertical-align:top;width:{col_other_width}%;{td_style};text-align:left;vertical-align:top;'>"
                table += f"<details><summary style='cursor:pointer;'>Show details</summary><div style='margin-top:8px;text-align:left;'>{desc_main_short}</div></details></td>"
                for _, h in recommendations_clean.iterrows():
                    desc_rec_full = str(h.get(c, ""))
                    desc_rec_short = " ".join(desc_rec_full.split()[:desc_limit]) + ("..." if len(desc_rec_full.split()) > desc_limit else "")
                    table += f"<td style='max-width:350px;word-break:break-word;text-align:left;vertical-align:top;width:{col_other_width}%;{td_style};text-align:left;vertical-align:top;'>"
                    table += f"<details><summary style='cursor:pointer;'>Show details</summary><div style='margin-top:8px;text-align:left;'>{desc_rec_short}</div></details></td>"
            elif c == "hotel_rank":
                value_main = main_hotel_clean.get(c, "")
                table += f"<td style='font-size:2.0em;font-weight:bold;width:{col_other_width}%;{td_style}'>{fn_rank_star(value_main)}</td>"
                for _, h in recommendations_clean.iterrows():
                    value_rec = h.get(c, "")
                    table += f"<td style='font-size:2.0em;font-weight:bold;width:{col_other_width}%;{td_style}'>{fn_rank_star(value_rec)}</td>"
            elif c == "total_score":
                value_main = main_hotel_clean.get(c, "")
                table += f"<td style='font-size:2.0em;font-weight:bold;width:{col_other_width}%;{td_style}'>{value_main}</td>"
                for _, h in recommendations_clean.iterrows():
                    value_rec = h.get(c, "")
                    table += f"<td style='font-size:2.0em;font-weight:bold;width:{col_other_width}%;{td_style}'>{value_rec}</td>"
            else:
                table += f"<td style='width:{col_other_width}%;{td_style}'>{main_hotel_clean.get(c,'')}</td>"
                for _, h in recommendations_clean.iterrows():
                    table += f"<td style='width:{col_other_width}%;{td_style}'>{h.get(c,'')}</td>"
            table += "</tr>"
        table += "</table>"
        st.markdown(table, unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='color:#c0392b;'>No recommended hotels found.</div>",
            unsafe_allow_html=True,
        )


def fn_main_display_selected_hotel(selected_hotel_id, df_hotels, df_matrix, top_k, desc_limit=100):
    """
    Main function to display selected hotel information and content-based recommendations
    
    Args:
        selected_hotel_id: ID of the selected hotel
        df_hotels: DataFrame with hotel information
        df_matrix: Similarity matrix
        top_k: Number of recommendations
        desc_limit: Maximum number of words in description
    """
    selected_hotel_df = df_hotels[df_hotels["hotel_id"] == selected_hotel_id]
    if not selected_hotel_df.empty:
        hotel_info = selected_hotel_df.iloc[0]
        st.write("####", "About this hotel:")
        fn_display_hotel_info(hotel_info, desc_limit)
        recommendations = fn_get_recommendations_page1(
            df=df_hotels, hotel_id=selected_hotel_id, matrix=df_matrix, top_k=top_k
        )
        fn_display_recommendations_section(hotel_info, recommendations, top_k, desc_limit)
    else:
        st.write(f"No hotel found with ID: {selected_hotel_id}")


def fn_main_display_search_results(df_hotels, search_input_tokens, selected_model, top_k, 
                                 gensim_matrix, corpora_dictionary, tfidf_model, 
                                 search_input, desc_limit=100):
    """
    Main function to display search results
    
    Args:
        df_hotels: DataFrame with hotel information
        search_input_tokens: Tokenized search input
        selected_model: Model type ('gensim' or 'cosine')
        top_k: Number of recommendations
        gensim_matrix: Gensim similarity matrix
        corpora_dictionary: Gensim dictionary
        tfidf_model: TF-IDF model
        search_input: Original search input text
        desc_limit: Maximum number of words in description
    """
    recommendations = fn_get_recommendations_search(
        selected_model=selected_model,
        gensim_matrix=gensim_matrix,
        corpora_dictionary=corpora_dictionary,
        tfidf_model=tfidf_model,
        search_input_tokens=search_input_tokens,
        search_input=search_input,
        df_hotel_info=df_hotels,
        top_k=top_k,
    )
    if not recommendations.empty:
        main_hotel = recommendations.iloc[0]
        st.write("####", "Best place for you:")
        fn_display_hotel_info(main_hotel, desc_limit)
        fn_display_recommendations_section(main_hotel, recommendations.iloc[1:], top_k, desc_limit)
    else:
        st.write("No hotels found matching your search.")


# ------------------------------------------------------------------------------
# Hotel Insight Analysis Functions
# ------------------------------------------------------------------------------

def fn_safe_get_row(df, key_col, key_val):
    """
    Safely get row from DataFrame by key column and value
    
    Args:
        df: DataFrame to search
        key_col: Column name to search
        key_val: Value to search for
    
    Returns:
        Series or None if not found
    """
    sub = df[df[key_col] == key_val]
    return sub.iloc[0] if len(sub) else None


def fn_get_hotel_overview(df_info, hotel_id):
    """
    Get hotel overview information including calculated average score
    
    Args:
        df_info: DataFrame with hotel information
        hotel_id: Hotel ID to get overview for
    
    Returns:
        Dictionary with hotel overview data
    """
    r = fn_safe_get_row(df_info, "hotel_id", hotel_id)
    if r is None:
        return None
    
    # Calculate average score if total_score is missing
    rating_cols = [
        "total_score",
        "location",
        "cleanliness", 
        "service",
        "facilities",
        "value_for_money",
        "comfort_and_room_quality",
    ]
    available = [c for c in rating_cols if c in r.index]
    total_score = r["total_score"] if pd.notna(r.get("total_score", np.nan)) else np.nan
    
    if pd.isna(total_score) and available:
        vals = [r[c] for c in available if pd.notna(r[c])]
        total_score = np.mean(vals) if vals else np.nan
    
    out = {
        "hotel_id": r["hotel_id"],
        "hotel_name": r.get("hotel_name", ""),
        "hotel_address": r.get("hotel_address", ""),
        "hotel_rank": r.get("hotel_rank", np.nan),
        "avg_score": total_score,
    }
    return out


def fn_chart_score_distribution(
    df,
    score_col,
    hotel_id_col,
    selected_hotel_id,
    color_dict={
        "hist": "#4C72B0",
        "kde": "#55A868", 
        "highlight_bin": "#FFB000",
        "edge": "black"
    },
    figsize=(10, 6),
    bins_time: int = 1,
    dpi=150,
    image_name: str = ''
):
    """
    Create score distribution chart with highlighted selected hotel
    
    Args:
        df: DataFrame with hotel data
        score_col: Column name for scores
        hotel_id_col: Column name for hotel IDs
        selected_hotel_id: ID of selected hotel to highlight
        color_dict: Color scheme for chart elements
        figsize: Figure size tuple
        bins_time: Number of bins per unit score
        dpi: DPI for saved image
        image_name: Name for saved image file
    """
    plt.figure(figsize=figsize)
    scores = df[score_col].dropna().values
    min_score, max_score = np.nanmin(scores), np.nanmax(scores)
    n_bins = max(int((max_score - min_score) * bins_time), 1)
    
    selected_row = df[df[hotel_id_col] == selected_hotel_id]
    selected_score = selected_row[score_col].values[0] if not selected_row.empty else None
    selected_hotel_name = selected_row["hotel_name"].values[0] if ("hotel_name" in df.columns and not selected_row.empty) else "Selected Hotel"
    
    bins = np.linspace(min_score, max_score, n_bins + 1)
    hist, bin_edges = np.histogram(scores, bins=bins)
    bin_idx = np.digitize(selected_score, bin_edges) - 1 if selected_score is not None else None
    
    if bin_idx == len(hist):
        bin_idx -= 1
    
    bar_colors = [color_dict["hist"]] * len(hist)
    if bin_idx is not None and 0 <= bin_idx < len(hist):
        bar_colors[bin_idx] = color_dict["highlight_bin"]
    
    for i in range(len(hist)):
        plt.bar(
            (bin_edges[i] + bin_edges[i+1]) / 2,
            hist[i],
            width=bin_edges[i+1] - bin_edges[i],
            color=bar_colors[i],
            edgecolor=color_dict["edge"],
            alpha=0.85,
            align="center"
        )
    
    ax = plt.gca()
    kde = sns.kdeplot(
        scores,
        bw_adjust=1,
        color=color_dict["kde"],
        linewidth=2,
        fill=False,
        ax=ax,
        label="KDE"
    )
    
    kde_y = kde.get_lines()[-1].get_ydata()
    if len(kde_y) > 0 and hist.sum() > 0 and kde_y.max() > 0:
        scale = hist.max() / kde_y.max()
        kde.get_lines()[-1].set_ydata(kde_y * scale)
    
    legend_handles = [
        Patch(facecolor=color_dict["hist"], edgecolor=color_dict["edge"], label=f"Other Hotels - {score_col}: {scores.mean():.1f}"),
        Patch(facecolor=color_dict["highlight_bin"], edgecolor=color_dict["edge"], label=f"{selected_hotel_name} - {score_col}: {str(selected_score)}")
    ]
    
    plt.legend(handles=legend_handles, fontsize=10, loc="upper left", bbox_to_anchor=(0, 1), borderaxespad=1.0)
    plt.title(f"Distribution of Hotel '{score_col}'", fontsize=16)
    plt.xlabel(score_col, fontsize=13)
    plt.ylabel("Number of Hotels", fontsize=13)
    
    plt.tight_layout()
    return plt.gcf()


def fn_display_hotel_insights(selected_hotel_id, df_hotels,figsize=DEFAULT_FIGSIZE, bins_time=2, 
                            rating_cols=None, score_classify_dict=None, show_radar=True, 
                            show_customer=True, show_wordcloud=True, word_count_limit=20, 
                            df_comments=None, df_comments_token=None):
    """
    Display comprehensive hotel insights with multiple chart types
    
    Args:
        selected_hotel_id: ID of selected hotel
        df_hotels: DataFrame with hotel information
        figsize: Figure size for charts
        bins_time: Number of bins per unit score
        rating_cols: List of rating columns for analysis
        score_classify_dict: Dictionary with classification thresholds
        show_radar: Whether to show radar chart
        show_customer: Whether to show customer analysis
        show_wordcloud: Whether to show word cloud
        word_count_limit: Number of words for word cloud
        df_comments: DataFrame with comment data (optional)
    """
    # Default parameters
    if rating_cols is None:
        rating_cols = ["location", "cleanliness", "service", "facilities", "value_for_money", "comfort_and_room_quality"]
    if score_classify_dict is None:
        score_classify_dict = {"Strength": 8.5, "Neutral": 7.5}
    
    # Get hotel overview data for all hotels
    hotels_overview = [fn_get_hotel_overview(df_hotels, hotel_id) for hotel_id in df_hotels["hotel_id"]]
    hotels_overview = [hotel for hotel in hotels_overview if hotel is not None and pd.notna(hotel["avg_score"])]
    
    if not hotels_overview:
        st.error("No hotel data available for analysis")
        return
    
    df_overview = pd.DataFrame(hotels_overview)
    
    # Display selected hotel information
    selected_hotel = df_hotels[df_hotels["hotel_id"] == selected_hotel_id]
    
    # Display hotel overview
    if not selected_hotel.empty:
        hotel_info = selected_hotel.iloc[0]
        st.markdown("### Hotel Analysis Overview")
        fn_display_hotel_info(hotel_info, desc_limit=150)
    
    # Create tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs(["📈Score Distribution", "🎯Strengths & Weaknesses", "🛒Customer Analysis", "💬Text Mining"])
    
    with tab1:
        st.markdown("#### Score Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Average Score Distribution")
            fig1 = fn_chart_score_distribution(
                df=df_overview,
                score_col="avg_score",
                hotel_id_col="hotel_id", 
                selected_hotel_id=selected_hotel_id,
                figsize=figsize,
                bins_time=bins_time,
                image_name="avg_score_distribution"
            )
            st.pyplot(fig1)
            plt.close()
        
        with col2:
            st.markdown("##### Hotel Rank Distribution")
            df_rank = df_overview[df_overview["hotel_rank"].notna()]
            if not df_rank.empty:
                fig2 = fn_chart_score_distribution(
                    df=df_rank,
                    score_col="hotel_rank",
                    hotel_id_col="hotel_id",
                    selected_hotel_id=selected_hotel_id,
                    figsize=figsize,
                    bins_time=bins_time,
                    image_name="hotel_rank_distribution"
                )
                st.pyplot(fig2)
                plt.close()
            else:
                st.info("No hotel rank data available")

        # Display insights summary with color coding
        selected_overview = next((h for h in hotels_overview if h["hotel_id"] == selected_hotel_id), None)
        
        if selected_overview:
            avg_score = selected_overview["avg_score"]
            rank = selected_overview["hotel_rank"]
            
            # Calculate market averages for comparison
            all_scores = df_overview["avg_score"].dropna()
            market_avg_score = all_scores.mean() if not all_scores.empty else 0
            
            # Calculate market average rank if rank data available
            all_ranks = df_overview["hotel_rank"].dropna()
            market_avg_rank = all_ranks.mean() if not all_ranks.empty else 0
            
            # Display detailed statistics
            st.markdown("### Market Position Analysis")
            
            if selected_overview and not df_overview.empty:
                avg_score = selected_overview["avg_score"]
                all_scores = df_overview["avg_score"].dropna()
                
                if not all_scores.empty and pd.notna(avg_score):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Market Average",
                            f"{all_scores.mean():.1f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Market Median", 
                            f"{all_scores.median():.1f}"
                        )
                    
                    with col3:
                        fn_display_insights_with_colors("Your score", avg_score, market_avg_score)
                    
                    with col4:
                        fn_display_insights_with_colors("Your rank", rank, market_avg_rank, "{:.0f}")

    with tab2:
        if show_radar:
            st.markdown("#### Strengths & Weaknesses Analysis")
            
            # Get strengths and weaknesses analysis
            strengths_analysis = fn_get_strengths_weaknesses_analysis(
                selected_hotel_id, df_hotels, rating_cols, score_classify_dict
            )
            
            if not strengths_analysis.empty:
                # Create 2-column layout for radar chart and analysis table
                col_radar, col_table = st.columns(2)
                
                with col_radar:
                    # Display radar chart
                    selected_hotel_name = selected_hotel.iloc[0]["hotel_name"] if not selected_hotel.empty else "Selected Hotel"
                    fig_radar = fn_chart_radar(
                        df=strengths_analysis,
                        selected_hotel_name=selected_hotel_name,
                        figsize=(figsize[1]/2, figsize[1]/2),  # Square for radar chart
                        image_name="strengths_weaknesses_radar"
                    )
                    st.pyplot(fig_radar)
                    plt.close()
                
                with col_table:
                    # Display detailed analysis table
                    display_df = strengths_analysis.copy()
                    display_df['diff_formatted'] = display_df['diff'].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "Missing")
                    display_df['classification_color'] = display_df['classification'].map({
                        'Strength': '🟢', 'Neutral': '🟡', 'Weakness': '🔴', None: '⚪'
                    })
                    
                    st.dataframe(
                        display_df[['attr', 'selected_hotel', 'all_mean', 'diff_formatted', 'classification_color']].rename(columns={
                            'attr': 'Attribute',
                            'selected_hotel': 'Your Hotel',
                            'all_mean': 'Market Average',
                            'diff_formatted': 'Difference',
                            'classification_color': 'Classification'
                        }),
                        use_container_width=True
                    )
                
                # Summary insights table
                summary_data = []
                for classification in ['Strength', 'Neutral', 'Weakness', 'Missing']:
                    if classification == 'Missing':
                        items = strengths_analysis[strengths_analysis['classification'].isna()]
                    else:
                        items = strengths_analysis[strengths_analysis['classification'] == classification]
                    
                    if not items.empty:
                        summary_data.append({
                            'Classification': classification,
                            'Count': len(items),
                            'Attributes': ', '.join(items['attr'].tolist())
                        })
                    else:
                        summary_data.append({
                            'Classification': classification,
                            'Count': 0,
                            'Attributes': 'None'
                        })
                
                # Display as table with color coding
                summary_df = pd.DataFrame(summary_data)
                summary_df['Color'] = summary_df['Classification'].map({
                    'Strength': '🟢', 'Neutral': '🟡', 'Weakness': '🔴', 'Missing': '⚪'
                })
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write("") # bỏ qua để set layout cho bảng bên dưới
                with col2:
                    st.markdown("##### Summary Insights")
                    st.dataframe(
                        summary_df[['Color', 'Classification', 'Count', 'Attributes']],
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.warning("No rating data available for strengths & weaknesses analysis")
    
    with tab3:
        if show_customer:
            st.markdown("#### Customer Analysis")
            
            if df_comments is not None and not df_comments.empty:
                # Get selected hotel name
                selected_hotel_name = df_hotels[df_hotels['hotel_id'] == selected_hotel_id]['hotel_name'].iloc[0] if not df_hotels[df_hotels['hotel_id'] == selected_hotel_id].empty else "Selected Hotel"
                
                # Get comparison data
                comparison_data = fn_get_customer_comparison_data(selected_hotel_id, df_comments)
                selected_data = comparison_data['selected_hotel']
                all_data = comparison_data['all_hotels']
                
                # Get colors from config
                selected_color = CUSTOMER_COMPARISON_CONFIG["colors"]["selected_hotel"]
                all_color = CUSTOMER_COMPARISON_CONFIG["colors"]["all_hotels"]
                
                # Create charts for each configuration in table format
                for i, config in enumerate(CUSTOMER_COMPARISON_CONFIG["chart_types"]):
                    col = config["column"]
                    chart_type = config["type"]
                    title = config["title"]
                    
                    # Check if column exists in data
                    if col not in selected_data.columns or col not in all_data.columns:
                        continue
                    
                    # Create two columns for each row
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Selected hotel chart
                        fig_selected, ax_selected = plt.subplots(figsize=figsize)
                        if chart_type == "bar":
                            fn_bar_chart(ax_selected, selected_data, col, selected_color, f"{selected_hotel_name}\n{title}")
                        else:
                            fn_line_timeseries_chart(ax_selected, selected_data, col, selected_color, f"{selected_hotel_name}\n{title}")
                        st.pyplot(fig_selected)
                        plt.close()
                    
                    with col2:
                        # All hotels chart
                        fig_all, ax_all = plt.subplots(figsize=figsize)
                        if chart_type == "bar":
                            fn_bar_chart(ax_all, all_data, col, all_color, f"All Hotels\n{title}")
                        else:
                            fn_line_timeseries_chart(ax_all, all_data, col, all_color, f"All Hotels\n{title}")
                        st.pyplot(fig_all)
                        plt.close()
            else:
                st.info("Customer analysis requires review data. This feature will be available when review data is loaded.")
    
    with tab4:
        if show_wordcloud:
            st.markdown("#### Text Mining Analysis")
            
            # Get text mining data
            text_data = fn_get_text_mining_data(selected_hotel_id, df_comments=df_comments_token, topN=word_count_limit)
            
            if not text_data['hotel_positive'].empty or not text_data['hotel_negative'].empty or not text_data['all_positive'].empty or not text_data['all_negative'].empty:
                
                # Create 2x2 layout for wordclouds
                st.markdown("##### WordCloud Comparison: Selected Hotel vs All Hotels")
                
                # Row 1: Positive Keywords
                # st.markdown("**Positive Keywords**")
                col1, col2 = st.columns(2)
                
                with col1:
                    # st.markdown("**Selected Hotel**")
                    if not text_data['hotel_positive'].empty:
                        fig_pos_hotel = fn_chart_wordcloud(
                            df=text_data['hotel_positive'],
                            title="Selected Hotel - Positive",
                            figsize=(figsize[0], figsize[1]),
                            image_name="positive_wordcloud_hotel"
                        )
                        st.pyplot(fig_pos_hotel)
                        plt.close()
                    else:
                        st.info("No positive keywords found for selected hotel")
                
                with col2:
                    # st.markdown("**All Hotels**")
                    if not text_data['all_positive'].empty:
                        fig_pos_all = fn_chart_wordcloud(
                            df=text_data['all_positive'],
                            title="All Hotels - Positive",
                            figsize=(figsize[0], figsize[1]),
                            image_name="positive_wordcloud_all"
                        )
                        st.pyplot(fig_pos_all)
                        plt.close()
                    else:
                        st.info("No positive keywords found for all hotels")
                
                # Row 2: Negative Keywords
                # st.markdown("**Negative Keywords**")
                col3, col4 = st.columns(2)
                
                with col3:
                    # st.markdown("**Selected Hotel**")
                    if not text_data['hotel_negative'].empty:
                        fig_neg_hotel = fn_chart_wordcloud(
                            df=text_data['hotel_negative'],
                            title="Selected Hotel - Negative",
                            figsize=(figsize[0], figsize[1]),
                            image_name="negative_wordcloud_hotel"
                        )
                        st.pyplot(fig_neg_hotel)
                        plt.close()
                    else:
                        st.info("No negative keywords found for selected hotel")
                
                with col4:
                    # st.markdown("**All Hotels**")
                    if not text_data['all_negative'].empty:
                        fig_neg_all = fn_chart_wordcloud(
                            df=text_data['all_negative'],
                            title="All Hotels - Negative",
                            figsize=(figsize[0], figsize[1]),
                            image_name="negative_wordcloud_all"
                        )
                        st.pyplot(fig_neg_all)
                        plt.close()
                    else:
                        st.info("No negative keywords found for all hotels")
                
            else:
                st.info("Text mining analysis requires review text data. This feature will be available when review data is loaded.")
    



def fn_score_classify(score, score_classify_dict):
    """
    Classify score into Strength, Neutral, or Weakness categories
    
    Args:
        score: Score value to classify
        score_classify_dict: Dictionary with classification thresholds
    
    Returns:
        Classification string or None
    """
    if pd.isna(score):
        return None
    if score >= score_classify_dict["Strength"]:
        return "Strength"
    elif score >= score_classify_dict["Neutral"]:
        return "Neutral"
    else:
        return "Weakness"


def fn_chart_radar(
    df, 
    selected_hotel_name='', 
    image_name='', 
    figsize=(10, 8), 
    dpi=150, 
    color_dict={
        "all_mean": "#4C72B0",
        "kde": "#55A868",
        "highlight_bin": "#FFB000",
        "edge": "black"
    }
):
    """
    Create radar chart for strengths & weaknesses analysis
    
    Args:
        df: DataFrame with attributes, selected_hotel, and all_mean columns
        selected_hotel_name: Name of selected hotel
        image_name: Name for saved image
        figsize: Figure size tuple
        dpi: DPI for saved image
        color_dict: Color scheme for chart elements
    
    Returns:
        matplotlib figure object
    """
    attributes = df['attr'].tolist()
    values = df['selected_hotel'].tolist() + [df['selected_hotel'].tolist()[0]]
    means = df['all_mean'].tolist() + [df['all_mean'].tolist()[0]]
    angles = np.linspace(0, 2 * np.pi, len(attributes), endpoint=False).tolist() + [0]
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True), dpi=dpi)
    ax.plot(angles, values, label=selected_hotel_name, linewidth=2, color=color_dict["highlight_bin"])
    ax.fill(angles, values, alpha=0.20, color=color_dict["highlight_bin"])
    ax.plot(angles, means, label="Other Hotels", linewidth=2, linestyle='dashed', color=color_dict["all_mean"])
    ax.fill(angles, means, alpha=0.1, color=color_dict["all_mean"])
    ax.set_thetagrids(np.degrees(angles[:-1]), attributes, fontsize=10)
    # title = "Strengths & Weaknesses"
    # ax.set_title(label=title, fontsize=16, fontweight='bold', pad=16)
    ax.set_ylim(0, 10)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    
    plt.tight_layout()
    return fig


def fn_chart_customer_analysis(df, chart_cols_dict, color, figsize, dpi, image_name, selected_hotel_name=''):
    """
    Create customer analysis charts with multiple subplots
    
    Args:
        df: DataFrame with customer data
        chart_cols_dict: Dictionary mapping chart types to column lists
        color: Color for charts
        figsize: Figure size tuple
        dpi: DPI for saved image
        image_name: Name for saved image
        selected_hotel_name: Name of selected hotel
    
    Returns:
        matplotlib figure object
    """
    def fn_bar_chart(ax, df, cat_col, color, title):
        data = df[cat_col].value_counts().sort_values(ascending=False)
        sns.barplot(x=data.values, y=data.index, ax=ax, color=color)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Count")
        ax.set_ylabel(cat_col)

    def fn_line_timeseries_chart(ax, df, time_col, color, title):
        data = df[time_col].value_counts().sort_index()
        try:
            x = pd.to_datetime(data.index)
        except Exception:
            x = data.index
        sns.lineplot(x=x, y=data.values, marker="o", ax=ax, color=color)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(time_col)
        ax.set_ylabel("Count")
        if time_col in ['stay_month', 'review_month']:
            for label in ax.get_xticklabels():
                label.set_rotation(45)

    ncols = 2
    n_items = sum(len(cols) for cols in chart_cols_dict.values())
    nrows = int(np.ceil(n_items / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes = np.array(axes).reshape(nrows, ncols)
    idx = 0
    
    for fn_name, cols in chart_cols_dict.items():
        for col in cols:
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            if fn_name == 'fn_bar_chart':
                nlarge_items = df[col].nunique()
                if nlarge_items < 10:
                    fn_bar_chart(ax, df, col, color, f"'{col}' distribution")
                else:
                    top10_idx = df[col].value_counts().head(10).index
                    df_top10 = df[df[col].isin(top10_idx)]
                    fn_bar_chart(ax, df_top10, col, color, f"Top 10 '{col}'")
            elif fn_name == 'fn_line_timeseries_chart':
                fn_line_timeseries_chart(ax, df, col, color, f"'{col}' distribution")
            idx += 1
    
    for i in range(idx, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis('off')
    
    fig.suptitle(t=f"Customer Analysis: {selected_hotel_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def fn_chart_wordcloud(df, title=None, figsize=(12, 6), dpi=150, image_name=None):
    """
    Create wordcloud chart from word frequency data
    
    Args:
        df: DataFrame with 'word' and 'count' columns
        title: Chart title
        figsize: Figure size tuple
        dpi: DPI for saved image
        image_name: Name for saved image
    
    Returns:
        matplotlib figure object
    """
    freq_dict = pd.Series(df['count'].values, index=df['word']).to_dict()
    wc = WordCloud(width=1000, height=500, background_color='white',
                   colormap='viridis', prefer_horizontal=1.0,
                   font_path=None, max_words=len(df))
    wc.generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    if not title:
        title = "Top Keywords WordCloud"
    ax.set_title(title, fontsize=20, fontweight='bold', color='#222222', loc='center', pad=20)
    plt.tight_layout(pad=2.0)
    return fig


def fn_get_strengths_weaknesses_analysis(selected_hotel_id, df_hotels, rating_cols, score_classify_dict):
    """
    Analyze strengths and weaknesses of selected hotel compared to market average
    
    Args:
        selected_hotel_id: ID of selected hotel
        df_hotels: DataFrame with hotel information
        rating_cols: List of rating columns to analyze
        score_classify_dict: Dictionary with classification thresholds
    
    Returns:
        DataFrame with analysis results
    """
    selected_hotel = df_hotels[df_hotels["hotel_id"] == selected_hotel_id]
    if selected_hotel.empty:
        return pd.DataFrame()
    
    hotel_data = selected_hotel.iloc[0]
    analysis_data = []
    
    for col in rating_cols:
        if col in df_hotels.columns:
            hotel_value = hotel_data.get(col, np.nan)
            market_mean = df_hotels[col].mean()
            diff = hotel_value - market_mean if pd.notna(hotel_value) else np.nan
            classification = fn_score_classify(hotel_value, score_classify_dict) if pd.notna(hotel_value) else None
            
            analysis_data.append({
                'attr': col.replace('_', ' ').title(),
                'selected_hotel': hotel_value if pd.notna(hotel_value) else 0,
                'all_mean': market_mean,
                'diff': diff,
                'classification': classification
            })
    
    return pd.DataFrame(analysis_data)


def fn_topwords(df, col='body_new_clean', topN=20, filter=None):
    """
    Extract top words from text data with optional filtering
    
    Args:
        df: DataFrame with text data
        col: Column name containing tokenized text
        topN: Number of top words to return
        filter: Filter by sentiment ('pos', 'nev', or None)
    
    Returns:
        DataFrame with top words and their counts
    """
    if filter in ['pos', 'nev']:
        df_filtered = df[df['classify'] == filter]
    else:
        df_filtered = df

    all_tokens = []
    for tokens in df_filtered[col]:
        if isinstance(tokens, list):
            all_tokens.extend(tokens)
    
    counter = Counter(all_tokens)
    top_words = counter.most_common(topN)
    df_top = pd.DataFrame(top_words, columns=['word', 'count'])
    total = df_top['count'].sum()
    df_top['pct'] = (df_top['count'] / total * 100).round(1)
    return df_top


def fn_score_stats(df, score_col: str = "score", score_level_col: str = "score_level"):
    """
    Calculate statistics for score distribution by score level
    
    Args:
        df: DataFrame with score data
        score_col: Column name for scores
        score_level_col: Column name for score levels
    
    Returns:
        DataFrame with statistics
    """
    stats = (
        df.groupby(score_level_col)[score_col]
        .agg(
            count="count",
            mean="mean",
            min="min",
            q1=lambda x: x.quantile(0.25),
            q2=lambda x: x.quantile(0.5),
            q3=lambda x: x.quantile(0.75),
            max="max",
        )
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    total_count = stats["count"].sum()
    stats["count_pct"] = (stats["count"] / total_count * 100).round(1)
    return stats[["score_level", "count", "count_pct", "mean", "min", "q1", "q2", "q3", "max"]]


def fn_display_competitive_analysis_table(similar_hotels, selected_hotel_id):
    """
    Display competitive analysis table with show/hide and download functionality
    
    Args:
        similar_hotels: DataFrame with similar hotels
        selected_hotel_id: ID of selected hotel
    """
    if similar_hotels.empty:
        st.info("No hotels found with similar scores")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        show_table = st.button("📊 Show Competitive Analysis Table", key="show_competitive_table")
    
    with col2:
        if show_table:
            # Prepare data for download
            csv_data = similar_hotels[['hotel_name', 'avg_score', 'hotel_rank', 'hotel_address']].copy()
            csv_data = csv_data.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"competitive_analysis_{selected_hotel_id}.csv",
                mime="text/csv"
            )
    
    if show_table:
        st.markdown("#### Competitive Hotels Analysis")
        
        # Create display table
        display_data = similar_hotels[['hotel_name', 'avg_score', 'hotel_rank', 'hotel_address']].copy()
        display_data = display_data.rename(columns={
            'hotel_name': 'Hotel Name',
            'avg_score': 'Average Score',
            'hotel_rank': 'Rank',
            'hotel_address': 'Address'
        })
        
        # Format the table
        display_data['Average Score'] = display_data['Average Score'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        display_data['Rank'] = display_data['Rank'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
        
        st.dataframe(display_data, use_container_width=True, hide_index=True)
        
        # Hide button
        if st.button("🔼 Hide Table", key="hide_competitive_table"):
            st.rerun()


def fn_display_insights_with_colors(metric_name, value, comparison_value, format_str="{:.1f}"):
    """
    Display metric with color coding and arrows based on comparison
    
    Args:
        metric_name: Name of the metric
        value: Current value
        comparison_value: Value to compare against
        format_str: Format string for the value
    """
    if pd.isna(value) or pd.isna(comparison_value):
        st.metric(metric_name, "N/A")
        return
    
    diff = value - comparison_value
    if diff > 0:
        delta_color = "normal"
        delta = f"↗️ Above Market ({format_str.format(diff)})"
    elif diff < 0:
        delta_color = "inverse"
        delta = f"↘️ Below Market ({format_str.format(abs(diff))})"
    else:
        delta_color = "normal"
        delta = "➡️ Equal to Market"
    
    st.metric(
        metric_name,
        format_str.format(value),
        delta=delta,
        delta_color=delta_color
    )


def fn_get_customer_analysis_data(selected_hotel_id, df_comments=None):
    """
    Get customer analysis data for selected hotel
    
    Args:
        selected_hotel_id: ID of selected hotel
        df_comments: DataFrame with comment data (optional)
    
    Returns:
        Dictionary with customer analysis data
    """
    if df_comments is None or df_comments.empty:
        return {
            'nationality': pd.DataFrame(),
            'group_name': pd.DataFrame(),
            'room_type': pd.DataFrame(),
            'review_month': pd.DataFrame(),
            'score_stats': pd.DataFrame()
        }
    
    # Filter comments for selected hotel
    hotel_comments = df_comments[df_comments['hotel_id'] == selected_hotel_id]
    
    if hotel_comments.empty:
        return {
            'nationality': pd.DataFrame(),
            'group_name': pd.DataFrame(),
            'room_type': pd.DataFrame(),
            'review_month': pd.DataFrame(),
            'score_stats': pd.DataFrame()
        }
    
    # Get score statistics
    score_stats = fn_score_stats(hotel_comments, "score", "score_level") if 'score_level' in hotel_comments.columns else pd.DataFrame()
    
    return {
        'nationality': hotel_comments['nationality'].value_counts().head(10) if 'nationality' in hotel_comments.columns else pd.DataFrame(),
        'group_name': hotel_comments['group_name'].value_counts().head(10) if 'group_name' in hotel_comments.columns else pd.DataFrame(),
        'room_type': hotel_comments['room_type'].value_counts().head(10) if 'room_type' in hotel_comments.columns else pd.DataFrame(),
        'review_month': hotel_comments['review_month'].value_counts().sort_index() if 'review_month' in hotel_comments.columns else pd.DataFrame(),
        'score_stats': score_stats
    }


def fn_bar_chart(ax, df, cat_col, color, title):
    """
    Create bar chart for categorical data
    
    Args:
        ax: matplotlib axis object
        df: DataFrame with data
        cat_col: categorical column name
        color: color for bars
        title: chart title
    """
    data = df[cat_col].value_counts().head(10).sort_values(ascending=False)
    if not data.empty:
        sns.barplot(x=data.values, y=data.index, ax=ax, color=color)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Count")
    ax.set_ylabel(cat_col)


def fn_line_timeseries_chart(ax, df, time_col, color, title):
    """
    Create line chart for time series data
    
    Args:
        ax: matplotlib axis object
        df: DataFrame with data
        time_col: time column name
        color: color for line
        title: chart title
    """
    data = df[time_col].value_counts().sort_index()
    if not data.empty:
        # Use data.index directly since it's already sorted integers
        x = data.index
        y = data.values
        
        # Create line plot with markers
        ax.plot(x, y, marker="o", color=color, linewidth=2, markersize=6)
        
        # Force display all x-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels(x)
        
        # Set x-axis labels and rotation for time columns
        if time_col in ['stay_month', 'review_month', 'review_year']:
            ax.tick_params(axis='x', rotation=45)
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(time_col)
    ax.set_ylabel("Count")


def fn_get_customer_comparison_data(selected_hotel_id, df_comments=None):
    """
    Get customer analysis data for both selected hotel and all hotels
    
    Args:
        selected_hotel_id: ID of selected hotel
        df_comments: DataFrame with comment data (optional)
    
    Returns:
        Dictionary with comparison data for selected hotel and all hotels
    """
    if df_comments is None or df_comments.empty:
        return {
            'selected_hotel': pd.DataFrame(),
            'all_hotels': pd.DataFrame()
        }
    
    # Filter comments for selected hotel
    selected_hotel_comments = df_comments[df_comments['hotel_id'] == selected_hotel_id]
    
    return {
        'selected_hotel': selected_hotel_comments,
        'all_hotels': df_comments
    }


def fn_get_text_mining_data(selected_hotel_id, df_comments=None, topN=20):
    """
    Get text mining data for selected hotel
    
    Args:
        selected_hotel_id: ID of selected hotel
        df_comments: DataFrame with comment data (optional)
        topN: Number of top words to extract
    
    Returns:
        Dictionary with text mining data
    """
    if df_comments is None or df_comments.empty:
        return {
            'all_positive': pd.DataFrame(),
            'all_negative': pd.DataFrame(),
            'hotel_positive': pd.DataFrame(),
            'hotel_negative': pd.DataFrame()
        }
    
    # Filter comments for selected hotel
    hotel_comments = df_comments[df_comments['hotel_id'] == selected_hotel_id]
    
    if hotel_comments.empty:
        return {
            'all_positive': pd.DataFrame(),
            'all_negative': pd.DataFrame(),
            'hotel_positive': pd.DataFrame(),
            'hotel_negative': pd.DataFrame()
        }
    
    # Get top words for different categories
    all_positive = fn_topwords(df_comments, 'body_new_clean', topN, 'pos') if 'body_new_clean' in df_comments.columns else pd.DataFrame()
    all_negative = fn_topwords(df_comments, 'body_new_clean', topN, 'nev') if 'body_new_clean' in df_comments.columns else pd.DataFrame()
    hotel_positive = fn_topwords(hotel_comments, 'body_new_clean', topN, 'pos') if 'body_new_clean' in hotel_comments.columns else pd.DataFrame()
    hotel_negative = fn_topwords(hotel_comments, 'body_new_clean', topN, 'nev') if 'body_new_clean' in hotel_comments.columns else pd.DataFrame()
    
    return {
        'all_positive': all_positive,
        'all_negative': all_negative,
        'hotel_positive': hotel_positive,
        'hotel_negative': hotel_negative
    }


# Convert body_new_clean from string to list
def convert_string_to_list(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    return x