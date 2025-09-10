# Agoda Hotel Recommendation & Insights

## Giới thiệu
Ứng dụng Streamlit cho phép:
- Đề xuất top-3 khách sạn tương tự dựa trên `hotel_id`, `keyword`, hoặc `reviewer_name`.
- Cung cấp insight cho chủ khách sạn (biểu đồ, thống kê từ dữ liệu đánh giá).

## Tính năng
- **Recommendation Models**:
  - Gensim
  - Cosine Similarity
  - ALS (Collaborative Filtering)
- **Hotel Insights**:
  - Thống kê điểm số
  - Biểu đồ phân tích comment
  - Xu hướng rating

## Cấu trúc thư mục
- `app.py` – file chạy Streamlit chính
- `notebooks/` – code gốc để tham khảo
- `data/` – dữ liệu CSV và dictionary
- `models/` – mô hình và ma trận đã train
- `src/` – code Python tách riêng: recommender, insights, utils
- `images/` – hình ảnh minh hoạ dashboard

## Cách chạy
```bash
# Cài môi trường
pip install -r requirements.txt

# Chạy ứng dụng
streamlit run app.pym
c