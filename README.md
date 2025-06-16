# 🎤 Tìm hiểu và cài đặt thử nghiệm phương pháp định danh giọng nói dùng học máy

## 📌 Giới thiệu dự án

Đây là đồ án tốt nghiệp triển khai và đánh giá các mô hình học sâu hiện đại cho bài toán **định danh người nói** dựa trên ảnh **Mel Spectrogram** chuyển đổi từ tín hiệu âm thanh. Mục tiêu là xác định đúng người nói tương ứng với từng đoạn âm thanh, đặc biệt áp dụng cho các giọng đọc của biên tập viên thời sự tiếng Việt.

Điểm nổi bật của dự án là mô hình **Multi-Backbone Fusion** kết hợp:
- **VGG16** đã tiền huấn luyện trên ImageNet
- **Mạng CNN tự xây dựng** tối ưu riêng cho ảnh Mel Spectrogram

Cách tiếp cận kết hợp này giúp khai thác đồng thời cả đặc trưng tổng quát và đặc trưng chuyên biệt của dữ liệu.

---

## 🧠 Động lực

Bài toán định danh giọng nói có ứng dụng trong:
- Gán nhãn tự động trong hệ thống lưu trữ nội dung truyền hình
- Xác thực truy cập, bảo mật sinh trắc học bằng giọng
- Phân tích nội dung phát thanh, kiểm chứng nguồn phát

Các phương pháp truyền thống thường yếu trong môi trường có nhiễu và biến động, do đó học sâu là một hướng tiếp cận hiệu quả hơn.

---

## 🏗️ Kiến trúc mô hình

Mô hình **Multi-Backbone Fusion** gồm các bước chính:

1. **Input**: Ảnh Mel Spectrogram 128×128 RGB
2. **Nhánh 1**: VGG16 (loại bỏ phần fully-connected gốc, dùng làm feature extractor)
3. **Nhánh 2**: Mạng CNN tự xây dựng gồm nhiều block có BatchNorm, SpatialDropout
4. **Fusion**: Ghép (concatenate) hai vector đặc trưng từ hai nhánh
5. **Head Classifier**: Các lớp Dense xen kẽ Dropout & BatchNorm
6. **Output**: Phân loại người nói với Softmax

---

## 📊 Các tiêu chí đánh giá

- **Accuracy (Độ chính xác tổng thể)**
- **Precision / Recall / F1-score**
- **Ma trận nhầm lẫn (Confusion Matrix)**

> Mô hình kết hợp CNN tự xây và VGG16 đạt kết quả vượt trội trên tập dữ liệu, với các chỉ số đều trên **94%**.

---

## 🧪 Thử nghiệm

Các mô hình được huấn luyện và so sánh:
- CNN (Cơ bản)
- VGG16 (Pre-trained)
- MobileNetV2
- ViT (Vision Transformer)
- CNN-LSTM
- ✅ **Multi-Backbone Fusion CNN–VGG16** (Hiệu quả nhất)

---

## 🗂️ Dữ liệu sử dụng

- 10 biên tập viên thời sự tiếng Việt
- Mỗi người: 20–25 phút giọng đọc
- Cắt thành đoạn 3 giây → chuyển thành ảnh Mel Spectrogram
- Chia tập train / validation hợp lý

---

## 🧰 Công nghệ sử dụng

- **Python**
- **Keras / TensorFlow / PyTorch**
- **Librosa, OpenCV** (xử lý âm thanh và ảnh)
- **Streamlit** (ứng dụng demo)
- **Google Colab / Local GPU**

---

## 🚀 Hướng dẫn chạy mô hình

```bash
# Clone repository
git clone https://github.com/ndhuy3000ride/Project_Final.git
cd Project_Final

# (Tùy chọn) Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # hoặc .\venv\Scripts\activate trên Windows

# Cài thư viện
pip install -r requirements.txt

# Huấn luyện mô hình (thay bằng file cụ thể nếu khác)
python train_model.py

# Chạy demo Streamlit nếu có
streamlit run app.py
```

---

## 📚 Tài liệu tham khảo

- VGG16 – Simonyan & Zisserman (2014)
- Lý thuyết mạng CNN – Stanford CS230
- Librosa – Thư viện xử lý âm thanh Python
- Kỹ thuật Mel Spectrogram

---

## 👨‍🎓 Tác giả

**Nguyễn Đức Huy**  
Sinh viên ngành Kỹ thuật Máy tính  
Đại học Bách Khoa Hà Nội  
GVHD: **PGS.TS. Lã Thế Vinh**

---

> ✨ Mọi góp ý, phản hồi xin vui lòng liên hệ qua GitHub hoặc email cá nhân.
