# 🌍 English AI - Ứng dụng Học Tiếng Anh Thông Minh

Một ứng dụng học tiếng Anh tích hợp AI giúp người dùng cải thiện kỹ năng nghe, nói, đọc, viết và từ vựng thông qua các bài tập tương tác.

## ✨ Các Chức Năng Chính

### 1. **Speaking Module** (Luyện Phát Âm & Nghe Hiểu)
- 🎙️ **Nhận dạng giọng nói**: Sử dụng Whisper AI để chuyển đổi giọng nói thành text
- 🔊 **Phân tích phát âm**: Đánh giá độ chính xác phát âm tiếng Anh
- 💬 **Đáp thoại tương tác**: Thực hành đối thoại với AI assistant
- **Công nghệ**: Whisper, Wav2Vec2 (Facebook)

### 2. **Essay Scoring Module** (Chấm Điểm Bài Viết)
- ✍️ **Tự động chấm điểm bài viết**: Đánh giá chất lượng bài viết tiếng Anh
- 📊 **Phân tích chi tiết**: Có 6 tiêu chí đánh giá
- 🤖 **Mô hình học sâu**: Sử dụng transformer (BERT-based) để phân tích
- **Công nghệ**: PyTorch, Transformers, AutoModel

### 3. **Reading Module** (Luyện Đọc Hiểu)
- 📖 **Bài đọc tiếng Anh**: Cung cấp các bài viết để luyện đọc
- ❓ **Câu hỏi trắc nghiệm**: Kiểm tra hiểu biết sau khi đọc
- 🧠 **RAG (Retrieval-Augmented Generation)**: Sử dụng FAISS để lấy thông tin liên quan
- **Công nghệ**: HuggingFace Embeddings, FAISS

### 4. **Vocabulary Module** (Luyện Từ Vựng)
- 📚 **Danh sách từ vựng CEFR-J**: Từ vựng theo tiêu chuẩn quốc tế
- 🔄 **Hệ thống ghi nhớ**: Sử dụng FSRS (Free Spaced Repetition System)
- 📈 **Theo dõi tiến độ**: Ghi lại tiến độ học tập
- **Dữ liệu**: CEFR-J Wordlist Ver1.6

### 5. **AI Chat Agent** (Trợ Lý Hỏi Đáp)
- 💬 **Chat đối thoại**: Trò chuyện tự do với AI
- 🌐 **Dịch thuật**: Hỗ trợ dịch từ/câu tiếng Anh sang tiếng Việt
- 📝 **Lưu lịch sử**: Ghi nhớ lịch sử hội thoại
- **Công nghệ**: Google Generative AI, Deep Translator

### 6. **Reading Comprehension** (Bài Tập Đọc Hiểu Thực Tế)
- 🛒 **Bài tập kịch bản thực tế**: Tình huống mua sắm tại cửa hàng tiện lợi
- 💰 **Hiểu giá cả & thông tin sản phẩm**: Thực hành đọc hiểu thực tế
- 🎯 **Phản hồi theo ngữ cảnh**: Trả lời câu hỏi dựa trên dữ liệu sản phẩm

## 🏗️ Kiến Trúc Dự Án

```
English AI/
├── main.py                      # Ứng dụng chính (Gradio UI)
├── agent.py                     # Chat manager và xử lý hội thoại
├── speaking_module.py           # Xử lý giọng nói & Whisper AI
├── essay_scoring_module.py      # Model chấm điểm bài viết
├── reading_module.py            # Bài đọc hiểu & RAG
├── vocabulary_module.py         # Luyện từ vựng & FSRS
├── selection_listener.js        # Frontend (JavaScript)
├── style.css                    # Kiểu dáng giao diện
├── reset_database.py            # Công cụ reset cơ sở dữ liệu
├── datasets/                    # Dữ liệu (CEFR-J wordlist)
├── models/                      # Lưu trữ mô hình AI
├── huggingface_cache/          # Cache HuggingFace models
└── README.md                    # File này
```

## 🚀 Yêu Cầu & Cài Đặt

### Yêu Cầu Hệ Thống
- Python 3.8+
- pip (trình quản lý package)
- Kết nối internet (để tải mô hình AI)

### Các Thư Viện Chính
```
gradio              # Giao diện web tương tác
transformers        # Mô hình NLP (BERT, etc.)
torch               # Framework học sâu
whisper             # Nhận dạng giọng nói (OpenAI)
google-generativeai  # Google Gemini AI
langchain           # Framework xử lý ngôn ngữ
faiss-cpu           # Vector search (Meta)
nltk                # Xử lý ngôn ngữ tự nhiên
deep-translator     # Dịch thuật Google
fsrs                # Spaced Repetition System
```

### Cài Đặt

1. Clone repository:
```bash
git clone https://github.com/ChuuniP/English-AI.git
cd English-AI
```

2. Tạo môi trường ảo (tùy chọn nhưng khuyến cáo):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Cài đặt các thư viện:
```bash
pip install -r requirements.txt
```

4. Tạo file `.env` để cấu hình:
```
GOOGLE_API_KEY=your_api_key_here
APP_TEMP_DIR=./temp
APP_HF_CACHE_DIR=./huggingface_cache
```

5. Chạy ứng dụng:
```bash
python main.py
```

Ứng dụng sẽ chạy tại: `http://localhost:7860`

## 📊 Dữ Liệu & Cơ Sở Dữ Liệu

- **SQLite Database**: `user_study_data.db` - Lưu trữ tiến độ người dùng
- **CEFR-J Wordlist**: Danh sách 3,000+ từ vựng theo chuẩn CEFR-J
- **Mô hình AI**: Tự động tải từ HuggingFace hub

## 🔧 Công Nghệ Sử Dụng

| Tính Năng | Công Nghệ |
|-----------|-----------|
| Nhận dạng giọng nói | Whisper (OpenAI) |
| Mô hình ngôn ngữ | BERT, Transformers |
| Giao diện web | Gradio |
| Học sâu | PyTorch |
| Vector search | FAISS |
| Ghi nhớ | Spaced Repetition (FSRS) |
| AI Chat | Google Generative AI |
| Dịch thuật | Google Translate |

## 📚 Tiêu Chí CEFR-J

Dự án này sử dụng danh sách từ vựng CEFR-J (Common European Framework of Reference for Languages - Japanese version), bao gồm các mức độ:
- **A1-A2**: Sơ cấp
- **B1-B2**: Trung cấp
- **C1-C2**: Nâng cao

## 💾 Quản Lý Cơ Sở Dữ Liệu

Để reset cơ sở dữ liệu người dùng:
```bash
python reset_database.py
```

## 🎯 Tính Năng Trong Tương Lai

- [ ] Hỗ trợ đa ngôn ngữ
- [ ] Bài kiểm tra TOEIC/IELTS
- [ ] Học từ vựng theo chủ đề
- [ ] Phân tích lỗi ngữ pháp chi tiết
- [ ] Gamification (điểm, huy hiệu)
- [ ] Ứng dụng di động

## 📝 Ghi Chú

- Lần đầu chạy ứng dụng sẽ tải các mô hình AI (mất thời gian)
- Yêu cầu kết nối internet để sử dụng các tính năng AI
- Có thể cấu hình các đường dẫn cache để tối ưu hóa

## 📄 Giấy Phép

Dự án này được phát triển cho mục đích giáo dục.

## 👤 Tác Giả

- **ChuuniP** - https://github.com/ChuuniP

## 📞 Liên Hệ & Hỗ Trợ

Nếu bạn gặp vấn đề hoặc có đề xuất, vui lòng mở issue trên GitHub.

---

**Chúc bạn học tiếng Anh vui vẻ! 🎓**
