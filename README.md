# 🗣️ Voice Classification using Deep Learning

Dự án này thực hiện **phân loại giọng nói** sử dụng các mô hình học sâu (Deep Learning), hỗ trợ nhận dạng người nói dựa trên đặc trưng âm thanh. Dữ liệu đầu vào là **file âm thanh**, được xử lý khử nhiễu, sau đó chuyển sang ảnh **Mel Spectrogram** để đưa vào mô hình học sâu.

## 🚀 Tính năng nổi bật

- Hỗ trợ nhiều mô hình mạnh mẽ: `CNN`, `VGG16`, `VGG32`, `ResNet`,...
- Khử nhiễu đầu vào bằng kỹ thuật xử lý tín hiệu trước khi trích xuất đặc trưng.
- Chuyển đổi âm thanh sang ảnh Mel Spectrogram.
- Huấn luyện và đánh giá mô hình phân loại giọng nói.
- Hỗ trợ kiểm thử với file âm thanh đầu vào thực tế.

---

## 🧱 Kiến trúc tổng quan

1. **Preprocessing**:  
   - Tải dữ liệu âm thanh đầu vào (WAV, MP3,...).
   - Khử nhiễu tín hiệu bằng kỹ thuật như spectral gating, noise reduction,...
   - Chia nhỏ thành các đoạn 3s (nếu cần).
   - Chuyển đổi âm thanh thành ảnh Mel Spectrogram (dùng Librosa).
   
2. **Training Models**:  
   - Huấn luyện các mô hình học sâu: CNN custom, VGG16, VGG32, ResNet,...
   - Đầu vào là ảnh Mel Spectrogram.
   - Sử dụng CrossEntropy Loss, Adam Optimizer,...
   - Kiểm thử với tập validation/test để đánh giá độ chính xác.

3. **Inference**:  
   - Dự đoán người nói từ một file âm thanh đầu vào.

---

## 🧰 Thư viện sử dụng

```bash
pip install -r requirements.txt
```

## 📁 Cấu trúc thư mục

```
├── data/                     # Dữ liệu âm thanh
├── spectrograms/            # Ảnh Mel Spectrogram sau xử lý
├── models/                  # Thư mục lưu mô hình
├── preprocessing.py         # Xử lý khử nhiễu + tạo Mel Spectrogram
├── train.py                 # File huấn luyện
├── test.py                  # Kiểm thử mô hình
├── inference.py             # Dự đoán file âm thanh đầu vào
├── model_cnn.py             # Kiến trúc CNN cơ bản
├── model_vgg.py             # Kiến trúc VGG16, VGG32
├── model_resnet.py          # Kiến trúc ResNet
└── README.md
```

## 🧪 Huấn luyện mô hình

```
python train.py --model cnn --epochs 30
python train.py --model vgg16 --epochs 50
python train.py --model resnet --epochs 40
```

## 📊 Kết quả

| Mô hình   | Accuracy |
|----------|----------|
| CNN      | 87.5%    |
| VGG16    | 91.3%    |
| ResNet18 | 92.7%    |


## ✅ TODO

- Huấn luyện mô hình CNN, VGG, ResNet

- Xử lý khử nhiễu âm thanh

- Chuyển âm thanh thành ảnh

- Triển khai giao diện Web/App (streamlit, flask,...)

- Tăng cường dữ liệu giọng nói (augmentation)

