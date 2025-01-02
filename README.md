1. Cài Python 3.12:
lên website https://www.python.org/downloads/release/python-3120/ và chọn file thích hợp với hệ thống để tải xuống
Mở file .exe vừa tải về để cài python vào hệ thống

2. Pull github repo:
Mở terminal, di chuyển địa chỉ vào 1 folder bất kì
Gõ lệnh: git clone https://github.com/LongMystic/Project-3
Mở folder Project-3 bằng IDE hoặc Editor bất kì

3. Cài thư viện cần thiết
chạy lệnh pip install -r requirements.txt

4. Chạy ứng dụng
Trước hết cần thêm dữ liệu xe vào file OrderLists và kết hợp với file WhCapacities:
python3 gen_and_merge_data.py
Khởi tạo model để dự đoán:
python3 create_model.py
Cuối cùng chỉ cần chạy file st-app.py:
streamlit run st-app.py
App được chạy ở localhost:8501 


