# Project 2: Recommendation System

## Team Members
1. Trần Đình Hùng  
2. Phạm Ngọc Trọng  

## Executive Summary
- **Target**: Sử dụng data về Khách sạn + các review về khách sạn trên Agoda. Từ đó xây dựng các mô hình recommendation system & cung cấp insight cho Chủ khách sạn.  
- **Data**: Có dữ liệu giao dịch & sản phẩm (`hotel_info.csv`, `hotel_comments.csv`).  
- **Các yêu cầu (phương pháp thực hiện)**:  
  1. Content-based filtering: sử dụng gensim + TF-IDF  
  2. Content-based filtering: sử dụng cosine-similarity  
  3. Collaborative filtering: sử dụng ALS model (spark)  
  4. Cung cấp insight cho Chủ khách sạn  

## Folder Structure
```
Project_2_RecommenderSystem/
├── Final_Report.pptx
│     Báo cáo tổng kết kết quả và insight chính
│
├── topic_RecommendationSystem_26082025.pdf
│     Lý thuyết nền tảng + Mô tả đề tài (Business problem) + Hướng dẫn thực hiện
│
├── source_code/
│   ├── project02_Agoda_1_Recommendation_models.ipynb
│   └── project02_Agoda_2_Hotel_Insight.ipynb
│
├── data_input/
│   ├── Data_Agoda_raw/
│   │   ├── hotel_comments.csv
│   │   └── hotel_info.csv
│   │
│   └── files/
│       ├── emojicon.txt
│       ├── english-vnmese.txt
│       ├── teencode.txt
│       ├── vietnamese-stopwords.txt
│       └── wrong-word.txt
│
├── images/
│     Biểu đồ minh họa trong quá trình phân tích
│
├── data_output/
│     Các file .csv kết quả sau xử lý & tổng hợp
│
└── models/
      Các model sau khi train sẽ được lưu lại
```

## Reading Order
1. `topic_RecommendationSystem_26082025.pdf`  
2. `Final_Report.pptx`
