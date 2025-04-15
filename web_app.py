# File: web_app.py
import streamlit as st
import pandas as pd
import pickle
import os
from gensim import corpora, models, similarities
import numpy as np
import scipy.sparse
import plotly.express as px

# Cấu hình trang
st.set_page_config(page_title="Shoppe Thời Trang Nam", layout="wide")

# Đường dẫn
DATA_DIR = 'D:/Streamlit'

@st.cache_resource
def load_models_and_data():
    with open('surprise_svd_model.pkl', 'rb') as f:
        svd_algo = pickle.load(f)
    dictionary = corpora.Dictionary.load('gensim_dictionary.dict')
    corpus = corpora.MmCorpus('gensim_corpus.mm')
    corpus = list(corpus)
    tfidf = models.TfidfModel.load('gensim_tfidfmodel')
    tfidf_matrix = scipy.sparse.load_npz('sklearn_tfidf_matrix.npz')
    #tfidf_matrix = np.load('sklearn_tfidf_matrix_.npz')
    # Tải ma trận cosine similarity từ file .npz (sparse format)
    cosine_sim_sparse = scipy.sparse.load_npz('sklearn_cosine_sim_sparse.npz')
    #cosine_sim_sparse = np.load('sklearn_cosine_sim_sparse_.npz')
    # Chuyển ma trận thưa thành ma trận mật (dense) để sử dụng
    #cosine_sim = cosine_sim_sparse.toarray()
    cosine_sim = cosine_sim_sparse
    df_product = pd.read_csv('processed_data.csv')
    df_rating = pd.read_csv('processed_ratings.csv')
    return svd_algo, dictionary, corpus, tfidf, tfidf_matrix, cosine_sim, df_product, df_rating

svd_algo, dictionary, corpus, tfidf, tfidf_matrix, cosine_sim, df_product, df_rating = load_models_and_data()

# Khởi tạo giỏ hàng và trạng thái người dùng
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'username' not in st.session_state:
    st.session_state.username = "Khách"
if 'current_product_id' not in st.session_state:
    st.session_state.current_product_id = None
if 'current_user_id' not in st.session_state:
    st.session_state.current_user_id = None
if 'show_product_detail_section1' not in st.session_state:
    st.session_state.show_product_detail_section1 = None
if 'show_product_detail_section2' not in st.session_state:
    st.session_state.show_product_detail_section2 = None
if 'recommendations_tab2' not in st.session_state:
    st.session_state.recommendations_tab2 = None

# Hàm hiển thị thông tin sản phẩm (tái sử dụng)
def display_product_card(row, idx, prefix="", section=""):
    image_url = row.get('image', 'https://via.placeholder.com/150')
    st.image(image_url, width=150)
    st.markdown(f"""
        <div class="product-card">
            <div class="product-name">{row['product_name']}</div>
            <div class="product-price">{int(row['price']):,} VNĐ</div>
            <div class="product-rating">⭐ {row['rating']:.1f}</div>
            <div class="product-category">{row['sub_category']}</div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Xem chi tiết", key=f"{prefix}_detail_{row['product_id']}_{idx}"):
        if section == "section1":
            st.session_state.show_product_detail_section1 = row['product_id']
        elif section == "section2":
            st.session_state.show_product_detail_section2 = row['product_id']
    if st.button("Thêm vào giỏ hàng", key=f"{prefix}_cart_{row['product_id']}_{idx}"):
        add_to_cart(row['product_id'])

# Các hàm gợi ý
@st.cache_data
def recommend_products_svd(user_id, df_product, df_rating, _algo, nums=5, max_price=None, category=None, min_rating=None):
    # Lấy danh sách sản phẩm đã đánh giá bởi user_id
    rated_products = set(df_rating[df_rating['user_id'] == user_id]['product_id'].dropna().tolist())
    
    # Lọc các sản phẩm chưa được đánh giá
    df_unrated = df_product[~df_product['product_id'].isin(rated_products)]
    
    # Áp dụng bộ lọc
    if max_price:
        df_unrated = df_unrated[df_unrated['price'] <= max_price]
    if category:
        df_unrated = df_unrated[df_unrated['sub_category'] == category]
    if min_rating:
        df_unrated = df_unrated[df_unrated['rating'] >= min_rating]
    
    # Kiểm tra nếu không có sản phẩm nào phù hợp
    if len(df_unrated) == 0:
        return pd.DataFrame(columns=['product_id', 'product_name', 'price', 'rating', 'sub_category', 'image'])
    
    # Tối ưu: Giới hạn số lượng sản phẩm cần dự đoán để tăng tốc độ
    if len(df_unrated) > 100:
        df_unrated = df_unrated.sample(n=100, random_state=42)
    
    # Dự đoán điểm số cho các sản phẩm chưa đánh giá
    predictions = [(pid, _algo.predict(user_id, pid).est) for pid in df_unrated['product_id']]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:nums]
    recommended_ids = [pred[0] for pred in predictions]
    
    # Trả về thông tin sản phẩm được gợi ý
    return df_product[df_product['product_id'].isin(recommended_ids)][['product_id', 'product_name', 'price', 'rating', 'sub_category', 'image']].drop_duplicates(subset=['product_id'])

@st.cache_data
def recommend_products_gensim(product_id, df, _corpus, _tfidf, _dictionary, top_n=5, max_price=500000, category=None, min_rating=None):
    if product_id not in df['product_id'].values:
        return pd.DataFrame(columns=['product_id', 'product_name', 'price', 'rating', 'sub_category', 'image'])
    
    idx = df.index[df['product_id'] == product_id].tolist()[0]
    sub_category = df.loc[idx, 'sub_category']
    
    original_product = df[df['product_id'] == product_id]
    df_filtered = df[(df['sub_category'] == sub_category) & (df['price'] <= max_price) & (df['product_id'] != product_id)]
    if category:
        df_filtered = df_filtered[df_filtered['sub_category'] == category]
    if min_rating:
        df_filtered = df_filtered[df_filtered['rating'] >= min_rating]
    df_filtered = df_filtered.reset_index(drop=True)
    
    if len(df_filtered) < top_n:
        df_filtered = df[(df['sub_category'] == sub_category) & (df['product_id'] != product_id)]
        if category:
            df_filtered = df_filtered[df_filtered['sub_category'] == category]
        if min_rating:
            df_filtered = df_filtered[df_filtered['rating'] >= min_rating]
        df_filtered = df_filtered.reset_index(drop=True)
    
    df_filtered = pd.concat([original_product.reset_index(drop=True), df_filtered], ignore_index=True)
    df_filtered['form_score'] = df_filtered['Content_wt'].apply(lambda x: 1 if any(kw in x.lower() for kw in ['form', 'body', 'ôm', 'fit']) else 0)
    
    content_gem_filtered = [x.split() for x in df_filtered['Content_wt']]
    corpus_filtered = [_dictionary.doc2bow(text) for text in content_gem_filtered]
    tfidf_vector = _tfidf[_corpus[idx]]
    tfidf_vectors_filtered = _tfidf[corpus_filtered]
    
    tfidf_vector_dense = np.zeros(len(_dictionary))
    for term_id, value in tfidf_vector:
        tfidf_vector_dense[term_id] = value
    tfidf_matrix_filtered = np.zeros((len(tfidf_vectors_filtered), len(_dictionary)))
    for i, vec in enumerate(tfidf_vectors_filtered):
        for term_id, value in vec:
            tfidf_matrix_filtered[i, term_id] = value
    
    from sklearn.metrics.pairwise import cosine_similarity
    sim_scores = cosine_similarity([tfidf_vector_dense], tfidf_matrix_filtered)[0]
    sim_scores = list(enumerate(sim_scores))
    
    price_norm = (df_filtered['price'].max() - df_filtered['price']) / (df_filtered['price'].max() - df_filtered['price'].min() + 1e-6)
    rating_norm = df_filtered['rating'] / 5.0
    final_scores = [(i, 0.6 * sim + 0.2 * rating_norm[i] + 0.2 * price_norm[i] + 0.1 * df_filtered['form_score'][i]) for i, sim in sim_scores]
    
    final_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
    filtered_idx = df_filtered.index[df_filtered['product_id'] == product_id].tolist()[0]
    final_scores = [score for score in final_scores if score[0] != filtered_idx][:top_n]
    
    product_indices = [i[0] for i in final_scores]
    return df_filtered.iloc[product_indices][['product_id', 'product_name', 'price', 'rating', 'sub_category', 'image']].drop_duplicates(subset=['product_id'])

@st.cache_data
def recommend_products_cosine(product_id, df, _cosine_sim, top_n=5, max_price=500000, category=None, min_rating=None):
    if product_id not in df['product_id'].values:
        return pd.DataFrame(columns=['product_id', 'product_name', 'price', 'rating', 'sub_category', 'image'])
    
    idx = df.index[df['product_id'] == product_id].tolist()[0]
    sub_category = df.loc[idx, 'sub_category']
    
    original_product = df[df['product_id'] == product_id]
    df_filtered = df[(df['sub_category'] == sub_category) & (df['price'] <= max_price) & (df['product_id'] != product_id)]
    if category:
        df_filtered = df_filtered[df_filtered['sub_category'] == category]
    if min_rating:
        df_filtered = df_filtered[df_filtered['rating'] >= min_rating]
    df_filtered = df_filtered.reset_index(drop=True)
    
    if len(df_filtered) < top_n:
        df_filtered = df[(df['sub_category'] == sub_category) & (df['product_id'] != product_id)]
        if category:
            df_filtered = df_filtered[df_filtered['sub_category'] == category]
        if min_rating:
            df_filtered = df_filtered[df_filtered['rating'] >= min_rating]
        df_filtered = df_filtered.reset_index(drop=True)
    
    df_filtered = pd.concat([original_product.reset_index(drop=True), df_filtered], ignore_index=True)
    df_filtered['form_score'] = df_filtered['Content_wt'].apply(lambda x: 1 if any(kw in x.lower() for kw in ['form', 'body', 'ôm', 'fit']) else 0)
    
    filtered_indices = df_filtered.index.map(lambda x: df.index[df['product_id'] == df_filtered.loc[x, 'product_id']].tolist()[0]).tolist()
    sim_scores = _cosine_sim[idx, filtered_indices]
    sim_scores = list(enumerate(sim_scores))
    
    price_norm = (df_filtered['price'].max() - df_filtered['price']) / (df_filtered['price'].max() - df_filtered['price'].min() + 1e-6)
    rating_norm = df_filtered['rating'] / 5.0
    final_scores = [(i, 0.6 * sim + 0.2 * rating_norm[i] + 0.2 * price_norm[i] + 0.1 * df_filtered['form_score'][i]) for i, sim in sim_scores]
    
    final_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
    filtered_idx = df_filtered.index[df_filtered['product_id'] == product_id].tolist()[0]
    final_scores = [score for score in final_scores if score[0] != filtered_idx][:top_n]
    
    product_indices = [i[0] for i in final_scores]
    return df_filtered.iloc[product_indices][['product_id', 'product_name', 'price', 'rating', 'sub_category', 'image']].drop_duplicates(subset=['product_id'])

@st.cache_data
def recommend_products_combined(product_id, df, _corpus, _tfidf, _dictionary, _cosine_sim, top_n=6, max_price=500000, category=None, min_rating=None):
    rec_gensim = recommend_products_gensim(product_id, df, _corpus, _tfidf, _dictionary, top_n=top_n, max_price=max_price, category=category, min_rating=min_rating)
    rec_cosine = recommend_products_cosine(product_id, df, _cosine_sim, top_n=top_n, max_price=max_price, category=category, min_rating=min_rating)
    combined = pd.concat([rec_gensim, rec_cosine]).drop_duplicates(subset=['product_id']).head(top_n)
    return combined

# Hàm thêm sản phẩm vào giỏ hàng
def add_to_cart(product_id):
    product = df_product[df_product['product_id'] == product_id].iloc[0]
    st.session_state.cart.append({
        'product_id': product_id,
        'product_name': product['product_name'],
        'price': product['price'],
        'image': product['image']
    })
    st.success(f"Đã thêm '{product['product_name']}' vào giỏ hàng!")

# CSS tùy chỉnh
st.markdown("""
    <style>
    .header {
        background: linear-gradient(90deg, #FF5722, #FF8A65);
        padding: 15px;
        color: white;
        text-align: center;
        font-size: 28px;
        font-family: 'Arial', sans-serif;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 22px;
        color: #FF5722;
        font-family: 'Arial', sans-serif;
        margin-top: 20px;
        border-bottom: 2px solid #FF5722;
        padding-bottom: 5px;
    }
    .section-title {
        font-size: 26px;
        color: #FF5722;
        font-family: 'Arial', sans-serif;
        margin-top: 30px;
        margin-bottom: 20px;
        border-left: 5px solid #FF5722;
        padding-left: 10px;
    }
    .product-card {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        padding: 15px;
        margin: 10px;
        width: 220px;
        display: inline-block;
        vertical-align: top;
        text-align: center;
        font-family: 'Arial', sans-serif;
        transition: transform 0.2s;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
    }
    .product-name {
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 8px;
        color: #333;
        height: 40px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .product-price {
        color: #FF5722;
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .product-rating {
        color: #FFD700;
        font-size: 12px;
        margin-bottom: 5px;
    }
    .product-category {
        font-size: 12px;
        color: #666;
        margin-bottom: 5px;
    }
    .filter-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .id-list {
        background-color: #e0f7fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .nav-button {
        font-size: 24px;
        cursor: pointer;
        margin: 0 10px;
    }
    .detail-box {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .intro-box {
        background-color: #fff3e0;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .search-bar {
        margin-bottom: 20px;
        padding: 10px;
        border-radius: 5px;
        border: 2px solid #FF5722;
        width: 100%;
        font-size: 16px;
    }
    .user-info {
        background-color: #e0f7fa;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar: Nhập tên người dùng và bộ lọc
st.sidebar.header("Thông tin người dùng")
username_input = st.sidebar.text_input("Nhập tên của bạn:", value=st.session_state.username, placeholder="Nhập tên của bạn (ví dụ: Nguyễn Văn A)")
if username_input:
    st.session_state.username = username_input
else:
    st.session_state.username = "Khách"  # Đặt lại thành "Khách" nếu không nhập

st.sidebar.header("Bộ lọc")
max_price = st.sidebar.slider("Giá tối đa (VNĐ)", min_value=0, max_value=1000000, value=500000, step=1000)
min_rating = st.sidebar.slider("Đánh giá tối thiểu (sao)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)

# Hiển thị giỏ hàng
st.sidebar.header("Giỏ hàng")
if st.session_state.cart:
    for item in st.session_state.cart:
        st.sidebar.image(item['image'], width=50)
        st.sidebar.write(f"{item['product_name']} - {int(item['price']):,} VNĐ")
    if st.sidebar.button("Xóa giỏ hàng"):
        st.session_state.cart = []
        st.rerun()
else:
    st.sidebar.write("Giỏ hàng trống.")

# Header
st.markdown(f'<div class="header">Chào mừng {st.session_state.username} đến với Shoppe Thời Trang Nam</div>', unsafe_allow_html=True)

# Phần giới thiệu (mục riêng)
with st.expander("Giới thiệu"):
    st.markdown('<div class="intro-box">', unsafe_allow_html=True)
    st.markdown("""
        <h3>Đồ án: Hệ thống gợi ý sản phẩm thời trang nam</h3>
        <p><strong>Họ tên:</strong> Nguyễn Phạm Duy & Phạm Mạch Lam Phương</p>
        <p><strong>Giáo viên hướng dẫn:</strong> Phuong Khuat Thuy</p>
    """, unsafe_allow_html=True)
    st.markdown("""
        <h3>Về Shoppe Thời Trang Nam</h3>
        <p>Chào mừng bạn đến với Shoppe Thời Trang Nam - nơi mang đến những sản phẩm thời trang nam chất lượng cao, phong cách và hiện đại. Chúng tôi cam kết cung cấp trải nghiệm mua sắm trực tuyến chuyên nghiệp, tiện lợi với các sản phẩm được gợi ý thông minh dựa trên sở thích của bạn.</p>
        <p><strong>Sứ mệnh:</strong> Mang đến phong cách thời trang nam đẳng cấp, phù hợp với mọi cá tính.</p>
        <p><strong>Giá trị cốt lõi:</strong> Chất lượng - Phong cách - Trải nghiệm khách hàng.</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Tabs cho các thuật toán
tab1, tab2 = st.tabs(["Gợi ý sản phẩm tương tự (Product ID)", "Gợi ý bằng SVD (User ID)"])

# Tab 1: Gợi ý sản phẩm tương tự (Dựa trên Product ID)
with tab1:
    # Bộ lọc Danh mục cho Tab 1 (góc phải)
    col1, col2 = st.columns([4, 1])
    with col2:
        category_options_tab1 = ['Tất cả'] + sorted(df_product['sub_category'].unique().tolist())
        selected_category_tab1 = st.selectbox("Danh mục", category_options_tab1, key="category_tab1")
        category_tab1 = None if selected_category_tab1 == 'Tất cả' else selected_category_tab1

    # Hiển thị danh sách Product ID hợp lệ
    st.subheader("Danh sách Product ID hợp lệ")
    valid_product_ids = df_product['product_id'].unique()
    st.markdown('<div class="id-list">', unsafe_allow_html=True)
    st.write(f"**Product ID hợp lệ (Gensim & Cosine):** {', '.join(map(str, valid_product_ids[:10]))} {'...' if len(valid_product_ids) > 10 else ''} (Tổng: {len(valid_product_ids)})")
    st.markdown('</div>', unsafe_allow_html=True)

    # Nhập Product ID (ô nhập ban đầu)
    if not st.session_state.current_product_id:
        product_id_input = st.text_input("Nhập Product ID để gợi ý:", value="1947", key="product_id_input")
        product_id = int(product_id_input) if product_id_input.isdigit() else None

        if product_id and product_id in valid_product_ids:
            st.session_state.current_product_id = product_id
            st.rerun()
        elif product_id:
            st.write("Vui lòng nhập Product ID hợp lệ (xem danh sách Product ID hợp lệ ở trên).")

    # Hiển thị gợi ý nếu đã chọn Product ID
    if st.session_state.current_product_id:
        product_id = st.session_state.current_product_id

        # Thanh tìm kiếm Product ID
        product_search = st.text_input("Tìm kiếm Product ID:", key="product_search", help="Nhập Product ID để tìm nhanh")
        if product_search and product_search.isdigit():
            product_id_search = int(product_search)
            if product_id_search in valid_product_ids:
                st.session_state.current_product_id = product_id_search
                st.session_state.show_product_detail_section1 = None  # Reset chi tiết sản phẩm khi tìm kiếm mới
                st.rerun()
            else:
                st.write("Product ID không hợp lệ. Vui lòng chọn Product ID từ danh sách hợp lệ ở trên.")

        # Kiểm tra xem Product ID có tồn tại trong df_product không
        product_info_df = df_product[df_product['product_id'] == product_id]
        if not product_info_df.empty:
            product_info = product_info_df.iloc[0]

            # Hiển thị sản phẩm gốc
            st.markdown(f'<div class="subheader">Sản phẩm gốc (Product ID: {product_id})</div>', unsafe_allow_html=True)
            display_product_card(product_info, 0, prefix="original", section="section1")

            # Gợi ý sản phẩm tương tự (kết hợp Gensim và Cosine Similarity, áp dụng bộ lọc danh mục của Tab 1)
            st.markdown('<div class="subheader">Sản phẩm tương tự</div>', unsafe_allow_html=True)
            recommendations_combined = recommend_products_combined(product_id, df_product, corpus, tfidf, dictionary, cosine_sim, top_n=6, max_price=max_price, category=category_tab1, min_rating=min_rating)
            if not recommendations_combined.empty:
                cols = st.columns(3)
                for idx, (_, row) in enumerate(recommendations_combined.iterrows()):
                    with cols[idx % 3]:
                        display_product_card(row, idx, prefix="combined", section="section1")
            else:
                st.write("Không tìm thấy sản phẩm tương tự phù hợp với bộ lọc. Có thể danh mục được chọn không khớp với danh mục của sản phẩm gốc.")
        else:
            st.write(f"Không tìm thấy thông tin cho Product ID {product_id}. Vui lòng chọn Product ID khác.")

        # Hiển thị chi tiết sản phẩm (chỉ trong Mục 1)
        if st.session_state.show_product_detail_section1:
            detail_product_df = df_product[df_product['product_id'] == st.session_state.show_product_detail_section1]
            if not detail_product_df.empty:
                detail_product = detail_product_df.iloc[0]
                st.markdown('<div class="subheader">Chi tiết sản phẩm</div>', unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="detail-box">
                        <img src="{detail_product['image']}" width="200" style="float: left; margin-right: 20px;">
                        <h3>{detail_product['product_name']}</h3>
                        <p><strong>Giá:</strong> {int(detail_product['price']):,} VNĐ</p>
                        <p><strong>Đánh giá:</strong> ⭐ {detail_product['rating']:.1f}</p>
                        <p><strong>Danh mục:</strong> {detail_product['sub_category']}</p>
                        <p><strong>Mô tả:</strong> {detail_product.get('description', 'Không có mô tả')}</p>
                    </div>
                """, unsafe_allow_html=True)
                if st.button("Đóng chi tiết", key="close_detail_section1"):
                    st.session_state.show_product_detail_section1 = None
                    st.rerun()
            else:
                st.write("Không tìm thấy thông tin chi tiết cho sản phẩm này.")

        # Điều hướng Product ID với khả năng chọn nhanh
        current_idx = list(valid_product_ids).index(product_id)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if current_idx > 0:
                # Lọc Product ID phù hợp (cùng danh mục, giá tương tự)
                similar_products_prev = df_product[
                    (df_product['product_id'].isin(valid_product_ids[:current_idx])) &
                    (df_product['sub_category'] == product_info['sub_category']) &
                    (df_product['price'].between(product_info['price'] * 0.8, product_info['price'] * 1.2))
                ]['product_id'].tolist()
                if similar_products_prev:
                    selected_prev = st.selectbox("Chọn Product ID trước:", similar_products_prev, key="prev_product_select")
                    if st.button("⬅ Product ID trước", key="prev_product"):
                        st.session_state.current_product_id = selected_prev
                        st.session_state.show_product_detail_section1 = None
                        st.rerun()
        with col2:
            if st.button("Quay lại", key="back_product"):
                st.session_state.current_product_id = None
                st.session_state.show_product_detail_section1 = None
                st.rerun()
        with col3:
            if current_idx < len(valid_product_ids) - 1:
                # Lọc Product ID phù hợp (cùng danh mục, giá tương tự)
                similar_products_next = df_product[
                    (df_product['product_id'].isin(valid_product_ids[current_idx + 1:])) &
                    (df_product['sub_category'] == product_info['sub_category']) &
                    (df_product['price'].between(product_info['price'] * 0.8, product_info['price'] * 1.2))
                ]['product_id'].tolist()
                if similar_products_next:
                    selected_next = st.selectbox("Chọn Product ID tiếp theo:", similar_products_next, key="next_product_select")
                    if st.button("Product ID tiếp theo ➡", key="next_product"):
                        st.session_state.current_product_id = selected_next
                        st.session_state.show_product_detail_section1 = None
                        st.rerun()

# Tab 2: Gợi ý bằng SVD (Dựa trên User ID)
with tab2:
    # Bộ lọc Danh mục cho Tab 2 (góc phải)
    col1, col2 = st.columns([4, 1])
    with col2:
        category_options_tab2 = ['Tất cả'] + sorted(df_product['sub_category'].unique().tolist())
        selected_category_tab2 = st.selectbox("Danh mục", category_options_tab2, key="category_tab2")
        category_tab2 = None if selected_category_tab2 == 'Tất cả' else selected_category_tab2

    # Hiển thị danh sách User ID hợp lệ
    st.subheader("Danh sách User ID hợp lệ")
    valid_user_ids = df_rating['user_id'].unique()
    if len(valid_user_ids) == 0:
        st.error("Không có dữ liệu User ID để gợi ý. Vui lòng kiểm tra file processed_ratings.csv.")
    else:
        st.markdown('<div class="id-list">', unsafe_allow_html=True)
        st.write(f"**User ID hợp lệ (SVD):** {', '.join(map(str, valid_user_ids[:10]))} {'...' if len(valid_user_ids) > 10 else ''} (Tổng: {len(valid_user_ids)})")
        st.markdown('</div>', unsafe_allow_html=True)

        # Nhập User ID
        if not st.session_state.current_user_id:
            user_id_input = st.text_input("Nhập User ID để gợi ý (SVD):", value="144905", key="user_id_input")
            user_id = int(user_id_input) if user_id_input.isdigit() else None

            if user_id and user_id in valid_user_ids:
                st.session_state.current_user_id = user_id
                st.session_state.recommendations_tab2 = None  # Reset gợi ý khi chọn User ID mới
                st.rerun()
            elif user_id:
                st.write("Vui lòng nhập User ID hợp lệ (xem danh sách User ID hợp lệ ở trên).")

        # Hiển thị gợi ý nếu đã chọn User ID
        if st.session_state.current_user_id:
            user_id = st.session_state.current_user_id
            # Lấy tên người dùng từ df_rating
            user_info = df_rating[df_rating['user_id'] == user_id]
            if not user_info.empty and 'user' in user_info.columns:
                # Lấy tên người dùng (lấy giá trị đầu tiên nếu user_id có nhiều dòng)
                username_from_data = user_info['user'].iloc[0]
            else:
                username_from_data = "Không xác định"

            # Hiển thị thông tin người dùng (User ID và Username từ dữ liệu)
            st.markdown(f'<div class="subheader">Kết quả cho User ID: {user_id}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="user-info">Tên người dùng: {username_from_data}</div>', unsafe_allow_html=True)
            # Hiển thị username từ sidebar (nếu có)
            if st.session_state.username != "Khách":
                st.markdown(f'<div class="user-info">Tên khách hàng: {st.session_state.username}</div>', unsafe_allow_html=True)

            # Thanh tìm kiếm User ID
            user_search = st.text_input("Tìm kiếm User ID:", key="user_search", help="Nhập User ID để tìm nhanh")
            if user_search and user_search.isdigit():
                user_id_search = int(user_search)
                if user_id_search in valid_user_ids:
                    st.session_state.current_user_id = user_id_search
                    st.session_state.show_product_detail_section2 = None
                    st.session_state.recommendations_tab2 = None  # Reset gợi ý khi tìm kiếm mới
                    st.rerun()
                else:
                    st.write("User ID không hợp lệ. Vui lòng chọn User ID từ danh sách hợp lệ ở trên).")

            # Hiển thị số lượng sản phẩm đã đánh giá
            num_rated = len(df_rating[df_rating['user_id'] == user_id])
            st.write(f"Người dùng này đã đánh giá {num_rated} sản phẩm.")

            # Hiển thị sản phẩm đã đánh giá (nếu có)
            rated_products = df_rating[df_rating['user_id'] == user_id]['product_id'].tolist()
            if rated_products:
                st.markdown('<div class="subheader">Sản phẩm đã đánh giá</div>', unsafe_allow_html=True)
                rated_products_info = df_product[df_product['product_id'].isin(rated_products)]
                if not rated_products_info.empty:
                    cols = st.columns(3)
                    for idx, (_, row) in enumerate(rated_products_info.iterrows()):
                        with cols[idx % 3]:
                            display_product_card(row, idx, prefix="rated", section="section2")
                else:
                    st.write("Không tìm thấy thông tin chi tiết cho các sản phẩm đã đánh giá.")

            # Gợi ý sản phẩm
            st.markdown('<div class="subheader">Sản phẩm được gợi ý cho bạn</div>', unsafe_allow_html=True)
            # Chỉ tính toán gợi ý nếu chưa có hoặc bộ lọc thay đổi
            if (st.session_state.recommendations_tab2 is None or
                'last_max_price_tab2' not in st.session_state or
                st.session_state.last_max_price_tab2 != max_price or
                st.session_state.last_min_rating_tab2 != min_rating or
                st.session_state.last_category_tab2 != category_tab2):
                st.session_state.recommendations_tab2 = recommend_products_svd(
                    user_id, df_product, df_rating, svd_algo, nums=6,
                    max_price=max_price, category=category_tab2, min_rating=min_rating
                )
                st.session_state.last_max_price_tab2 = max_price
                st.session_state.last_min_rating_tab2 = min_rating
                st.session_state.last_category_tab2 = category_tab2

            recommendations = st.session_state.recommendations_tab2
            if not recommendations.empty:
                cols = st.columns(3)
                for idx, (_, row) in enumerate(recommendations.iterrows()):
                    with cols[idx % 3]:
                        display_product_card(row, idx, prefix="svd", section="section2")
            else:
                st.write("Không tìm thấy sản phẩm phù hợp với bộ lọc.")

            # Hiển thị chi tiết sản phẩm (chỉ trong Mục 2)
            if st.session_state.show_product_detail_section2:
                detail_product_df = df_product[df_product['product_id'] == st.session_state.show_product_detail_section2]
                if not detail_product_df.empty:
                    detail_product = detail_product_df.iloc[0]
                    st.markdown('<div class="subheader">Chi tiết sản phẩm</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="detail-box">
                            <img src="{detail_product['image']}" width="200" style="float: left; margin-right: 20px;">
                            <h3>{detail_product['product_name']}</h3>
                            <p><strong>Giá:</strong> {int(detail_product['price']):,} VNĐ</p>
                            <p><strong>Đánh giá:</strong> ⭐ {detail_product['rating']:.1f}</p>
                            <p><strong>Danh mục:</strong> {detail_product['sub_category']}</p>
                            <p><strong>Mô tả:</strong> {detail_product.get('description', 'Không có mô tả')}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    if st.button("Đóng chi tiết", key="close_detail_section2"):
                        st.session_state.show_product_detail_section2 = None
                        st.rerun()
                else:
                    st.write("Không tìm thấy thông tin chi tiết cho sản phẩm này.")

            # Điều hướng User ID
            current_idx = list(valid_user_ids).index(user_id)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if current_idx > 0:
                    if st.button("⬅ User ID trước", key="prev_user"):
                        st.session_state.current_user_id = valid_user_ids[current_idx - 1]
                        st.session_state.show_product_detail_section2 = None
                        st.session_state.recommendations_tab2 = None
                        st.rerun()
            with col2:
                if st.button("Quay lại", key="back_user"):
                    st.session_state.current_user_id = None
                    st.session_state.show_product_detail_section2 = None
                    st.session_state.recommendations_tab2 = None
                    st.rerun()
            with col3:
                if current_idx < len(valid_user_ids) - 1:
                    if st.button("User ID tiếp theo ➡", key="next_user"):
                        st.session_state.current_user_id = valid_user_ids[current_idx + 1]
                        st.session_state.show_product_detail_section2 = None
                        st.session_state.recommendations_tab2 = None
                        st.rerun()

# Thanh tìm kiếm sản phẩm
st.markdown('<div class="section-title">Tìm kiếm sản phẩm</div>', unsafe_allow_html=True)
search_query = st.text_input("Nhập tên sản phẩm hoặc danh mục để tìm kiếm:", key="search_bar", help="Ví dụ: Áo thun, Quần jeans")
if search_query:
    search_results = df_product[
        df_product['product_name'].str.contains(search_query, case=False, na=False) |
        df_product['sub_category'].str.contains(search_query, case=False, na=False)
    ]
    if not search_results.empty:
        st.markdown(f'<div class="subheader">Kết quả tìm kiếm cho "{search_query}"</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, (_, row) in enumerate(search_results.iterrows()):
            with cols[idx % 3]:
                display_product_card(row, idx, prefix="search")
    else:
        st.write("Không tìm thấy sản phẩm phù hợp.")

# Biểu đồ trực quan hóa
st.markdown('<div class="section-title">Thống kê sản phẩm</div>', unsafe_allow_html=True)
fig1 = px.histogram(df_product, x="price", nbins=20, title="Phân bố giá sản phẩm (VNĐ)")
fig1.update_layout(xaxis_title="Giá (VNĐ)", yaxis_title="Số lượng sản phẩm")
st.plotly_chart(fig1, use_container_width=True)

avg_rating_by_category = df_product.groupby('sub_category')['rating'].mean().reset_index()
fig2 = px.bar(avg_rating_by_category, x='sub_category', y='rating', title="Xếp hạng trung bình theo danh mục")
fig2.update_layout(xaxis_title="Danh mục", yaxis_title="Xếp hạng trung bình (sao)")
st.plotly_chart(fig2, use_container_width=True)

product_count_by_category = df_product['sub_category'].value_counts().reset_index()
product_count_by_category.columns = ['sub_category', 'count']
fig3 = px.pie(product_count_by_category, names='sub_category', values='count', title="Số lượng sản phẩm theo danh mục")
st.plotly_chart(fig3, use_container_width=True)
# "C:\Program Files\Python311\python.exe" -m streamlit run "C:\Users\npd20\Downloads\Streamlit\web_app.py"



