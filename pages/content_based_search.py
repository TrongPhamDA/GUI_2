# Hotel Data Science: Load hotel dataset and similarity matrix for content-based hotel recommendation
import streamlit as st
import pandas as pd
import numpy as np
from gensim import corpora
from gensim import similarities
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re
import unicodedata
import string

###########################################
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
    def remove_accents(text):
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore').decode("utf-8")
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
            tok = ''.join([c for c in tok if c not in punctuations])
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
    if is_empty_or_nan(text) or (isinstance(text, list) and all(is_empty_or_nan(t) for t in text)):
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
        # remove stopword ở bước cuối cùng
        if stopword:
            cleaned = [tok for tok in cleaned if tok not in stopword]
        results.append(cleaned)
    if isinstance(text, list):
        return results
    else:
        return results[0]


def fn_read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


def fn_read_dict(path):
    d = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if '\t' in line:
                eng, vie = line.strip().split('\t', 1)
                d[eng.strip()] = vie.strip()
    return d
###########################################
        
df_hotels = pd.read_csv('result/01_df_info.csv', header=0)
df_hotels['hotel_id'] = df_hotels['hotel_id'].astype(str)
st.session_state.random_hotels = df_hotels

df_matrix_gensim = pd.read_csv('result/03_gensim_tfidf_df_index_matrix.csv', header=None, index_col=False)
df_matrix_cosine = pd.read_csv('result/06_cosine_tfidf_df_index_matrix.csv', header=None, index_col=False)

loaded_dictionary = corpora.Dictionary.load('models/02_dictionary.dict')
loaded_model_tfidf = joblib.load('models/01_tfidf.pkl')
# loaded_index_matrix = joblib.load('models/03_gesim_sparsematrixsimilarity.index')

emoji = fn_read_dict(path="input/dictionaries/emojicon.txt")
engvie = fn_read_dict(path="input/dictionaries/english-vnmese.txt")
teencode = fn_read_dict(path="input/dictionaries/teencode.txt")
stopword_vie = fn_read_txt(path="input/dictionaries/vietnamese-stopwords.txt")
wrongword = fn_read_txt(path="input/dictionaries/wrong-word.txt")

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
                Simply describe the hotel experience you’re dreaming of—whether it’s a beachfront escape, a cozy city retreat, or a luxury getaway—and we’ll match you with the perfect place for your next adventure.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
####################################################

# User interface: search input and number of recommendations
search_input = st.text_input("Search for something...")

with st.sidebar:
    selected_model = st.radio("Select similarity model", options=["gensim", "cosine"], index=0, horizontal=True)
    top_k = st.slider("Select top N similar hotels", min_value=1, max_value=10, value=3, step=1)
    desc_limit = st.slider("Description limit (words)", min_value=50, max_value=500, value=100, step=50)

# Xử lý từ khóa tìm kiếm
search_input_tokens = fn_clean_tokens(
    text=search_input,
    dict_list=[emoji, teencode, engvie],
    stopword=stopword_vie,
    wrongword=wrongword,
    remove_number=True,
    remove_punctuation=True,
    remove_vie_tone=False,
    lower=True,
)

# Recommendation engine: retrieve top-k similar hotels based on content similarity matrix
def get_recommendations(df: pd.DataFrame, search_input_tokens, selected_model: str, top_k: int) -> pd.DataFrame:
    if not search_input_tokens or len(search_input_tokens) == 0:
        return pd.DataFrame()
    if selected_model == "gensim":
        # Instead of using MatrixSimilarity on a numpy array (which expects a corpus of sparse vectors),
        # we use the precomputed similarity matrix directly.
        # df_matrix_gensim is a DataFrame where each row is the similarity of a hotel to all others.
        # We treat the search query as a new document, compute its tfidf vector, and then compute similarity
        # with all hotels using the same tfidf model and dictionary.
        # However, since the code previously tried to use MatrixSimilarity on a numpy array (which is not a corpus),
        # we will instead use the precomputed similarity matrix as follows:
        # - Each row in df_matrix_gensim corresponds to a hotel, and each column is the tfidf value for a term.
        # - To get the similarity between the query and all hotels, we need to compute the query tfidf vector,
        #   and then compute cosine similarity between the query vector and all hotel vectors.

        # Prepare the hotel tfidf matrix (each row: hotel, columns: terms)
        hotel_tfidf_matrix = df_matrix_gensim.values  # shape: (num_hotels, num_terms)
        # Compute query tfidf vector (sparse list of (term_id, value))
        query_bow = loaded_dictionary.doc2bow(search_input_tokens)
        query_tfidf = loaded_model_tfidf[query_bow]
        # Convert query_tfidf to dense vector
        num_terms = hotel_tfidf_matrix.shape[1]
        query_vec_dense = np.zeros(num_terms, dtype=np.float32)
        for idx, value in query_tfidf:
            if idx < num_terms:
                query_vec_dense[idx] = value
        # Compute cosine similarity between query_vec_dense and all hotel tfidf vectors
        # Add a small epsilon to denominator to avoid division by zero
        hotel_norms = np.linalg.norm(hotel_tfidf_matrix, axis=1) + 1e-10
        query_norm = np.linalg.norm(query_vec_dense) + 1e-10
        dot_products = hotel_tfidf_matrix @ query_vec_dense
        similarities_arr = dot_products / (hotel_norms * query_norm)
        # Get top_k indices
        top_idx = pd.Series(similarities_arr).nlargest(top_k).index
        df_recommend = df.iloc[top_idx].copy()
        df_recommend["similarity"] = pd.Series(similarities_arr).iloc[top_idx].values
        return df_recommend.sort_values("similarity", ascending=False).reset_index(drop=True)
    if selected_model == "cosine":
        processed_content = df["hotel_description"].apply(
            lambda x: fn_clean_tokens(
                text=str(x),
                dict_list=[emoji, teencode, engvie],
                stopword=stopword_vie,
                wrongword=wrongword,
                remove_number=True,
                remove_punctuation=True,
                remove_vie_tone=False,
                lower=True,
            )
        )
        vectorizer = TfidfVectorizer(analyzer="word", stop_words=stopword_vie)
        tfidf_matrix = vectorizer.fit_transform(processed_content.apply(lambda x: " ".join(x) if isinstance(x, list) else str(x)))
        query_vec = vectorizer.transform([" ".join(search_input_tokens)])
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = pd.Series(cosine_sim).nlargest(top_k).index
        df_recommend = df.iloc[top_indices].copy()
        df_recommend["similarity"] = pd.Series(cosine_sim).iloc[top_indices].values
        return df_recommend.sort_values("similarity", ascending=False).reset_index(drop=True)

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
def display_comparison_table(main_hotel: pd.Series, recommended_hotels: pd.DataFrame, criteria: list):
    columns = ['Fact', main_hotel['hotel_name']] + recommended_hotels['hotel_name'].tolist()
    data = []
    for crit in criteria:
        row = [crit.replace('_', ' ').title()]
        row.append(main_hotel.get(crit, ""))
        for _, hotel in recommended_hotels.iterrows():
            row.append(hotel.get(crit, ""))
        data.append(row)
    st.table(pd.DataFrame(data, columns=columns))

# Display recommendations section with professional comparison table for hotel analytics
def display_recommendations_section(main_hotel: pd.Series, recommendations: pd.DataFrame):
    st.markdown("<h4 style='color:#2C3E50;font-weight:700;'>Compare Hotels Side by Side</h4>", unsafe_allow_html=True)
    if not recommendations.empty:
        criteria = [
            'hotel_address', 'hotel_rank', 'comments_count', 'total_score',
            'location', 'cleanliness', 'service', 'facilities', 'value_for_money', 'comfort_and_room_quality', 
            'hotel_description',
        ]
        names = [main_hotel['hotel_name']] + recommendations['hotel_name'].tolist()
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
                desc_main = ' '.join(str(main_hotel.get(c, '')).split()[:desc_limit]) + "..."
                table += f"<td style='max-width:350px;word-break:break-word;text-align:left;vertical-align:top;width:{col_other_width}%;{td_style};text-align:left;vertical-align:top;'>{desc_main}</td>"
                for _, h in recommendations.iterrows():
                    desc_rec = ' '.join(str(h.get(c, '')).split()[:desc_limit]) + "..."
                    table += f"<td style='max-width:350px;word-break:break-word;text-align:left;vertical-align:top;width:{col_other_width}%;{td_style};text-align:left;vertical-align:top;'>{desc_rec}</td>"
            elif c in ['hotel_rank', 'total_score']:
                value_main = main_hotel.get(c, '')
                table += f"<td style='font-size:1.5em;font-weight:bold;width:{col_other_width}%;{td_style}'>{value_main}</td>"
                for _, h in recommendations.iterrows():
                    value_rec = h.get(c, '')
                    table += f"<td style='font-size:1.5em;font-weight:bold;width:{col_other_width}%;{td_style}'>{value_rec}</td>"
            else:
                table += f"<td style='width:{col_other_width}%;{td_style}'>{main_hotel.get(c,'')}</td>"
                for _, h in recommendations.iterrows():
                    table += f"<td style='width:{col_other_width}%;{td_style}'>{h.get(c,'')}</td>"
            table += "</tr>"
        table += "</table>"
        st.markdown(table, unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#c0392b;'>No recommended hotels found.</div>", unsafe_allow_html=True)

def main_display_search_results(df_hotels, search_input_tokens, selected_model, top_k):
    recommendations = get_recommendations(
        df_hotels,
        search_input_tokens,
        selected_model,
        top_k
    )
    if not recommendations.empty:
        # The most relevant hotel is the first one
        main_hotel = recommendations.iloc[0]
        display_hotel_info(main_hotel)
        # Show comparison with the rest (excluding the main hotel)
        display_recommendations_section(main_hotel, recommendations.iloc[1:])
    else:
        st.write("No hotels found matching your search.")

if search_input and search_input_tokens:
    main_display_search_results(
        df_hotels,
        search_input_tokens,
        selected_model,
        top_k
    )