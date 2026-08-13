# 🌍 English AI - Ứng dụng Học Tiếng Anh Thông Minh

Một ứng dụng học tiếng Anh tích hợp AI giúp người dùng cải thiện toàn diện các kỹ năng nghe, nói, đọc, viết và từ vựng thông qua giao diện Gradio tương tác và các công nghệ học máy hiện đại.

## ✨ Các Chức Năng Chính

### 1. **Listening Module** (Luyện Nghe Hiểu) 🆕
- 🎥 **Hỗ trợ video & audio**: Luyện nghe qua video (MP4) và các tệp âm thanh ngắn (MP3) tích hợp sẵn trong ứng dụng.
- 🎙️ **Nhận dạng & Transcribe**: Tự động chuyển đổi nội dung nghe thành văn bản (transcript) sử dụng Whisper AI.
- ❓ **Tự động tạo câu hỏi hiểu bài**: AI (Gemini) tự động phân tích transcript để đặt 3 câu hỏi trắc nghiệm/tự luận phù hợp ngữ cảnh.
- 💯 **Chấm điểm & Feedback chi tiết**: Chấm câu trả lời của người dùng trên thang điểm 10 và đưa ra lời giải thích chi tiết cho từng câu.
- **Công nghệ**: Whisper, Google GenAI (Gemini 2.5 Flash), Edge-TTS.

### 2. **Speaking Module** (Luyện Phát Âm & Nghe Hiểu)
- 🎙️ **Nhận dạng giọng nói**: Ghi âm trực tiếp và chuyển đổi giọng nói thành văn bản bằng Whisper AI.
- 🔊 **Phân tích phát âm**: Sử dụng mô hình học sâu Wav2Vec2 của Facebook để đối chiếu giọng đọc và đánh giá độ chính xác của từng từ.
- 💬 **Đáp thoại tương tác (Conversation)**: Trò chuyện và thực hành đối thoại theo ngữ cảnh với trợ lý ảo AI.
- 📖 **Luyện đọc theo câu (Sentence/Topic)**: Luyện nói theo các chủ đề hoặc câu mẫu có sẵn trong database.
- **Công nghệ**: Whisper, Wav2Vec2 (Facebook), Google GenAI, LangChain, FAISS.

### 3. **Writing Module** (Chấm Điểm & Luyện Viết Essay)
- ✍️ **Chấm điểm tự động**: Đánh giá chi tiết các bài viết luận (Essay) theo chuẩn quốc tế.
- 📊 **6 tiêu chí đánh giá**: Phân tích chi tiết theo: Cohesion (Sự liên kết), Syntax (Cú pháp), Vocabulary (Từ vựng), Phraseology (Cách dùng cụm từ), Grammar (Ngữ pháp), Conventions (Quy chuẩn viết).
- 🤖 **Mô hình Fine-tune BERT**: Sử dụng mô hình `microsoft/deberta-v3-base` đã được tinh chỉnh nhằm đưa ra điểm số chính xác và đáng tin cậy.
- 📜 **Lịch sử viết bài**: Tự động lưu trữ lịch sử viết luận, điểm số và nhận xét chi tiết vào database để người dùng theo dõi tiến độ.
- **Công nghệ**: PyTorch, Transformers, AutoModel.

### 4. **Reading Module** (Luyện Đọc Hiểu & Tra Từ)
- 📖 **Bài đọc phân cấp**: Cung cấp các văn bản luyện đọc được phân loại theo cấp độ CEFR (A1-C2).
- ❓ **Hỏi đáp kiểm tra**: Đánh giá mức độ hiểu bài đọc qua các câu hỏi đi kèm.
- 🔍 **Tra từ & lưu từ vựng thông minh**: Người học có thể bôi đen/chọn trực tiếp từ hoặc câu trong bài đọc để dịch nghĩa, tra phát âm và lưu trực tiếp vào danh sách từ vựng để ôn tập.
- **Công nghệ**: Google Translate, SpaCy, Gradio.

### 5. **Vocabulary Module** (Luyện Từ Vựng)
- 📚 **Từ điển CEFR-J**: Danh sách hơn 3,000+ từ vựng chuẩn hóa theo khung tham chiếu CEFR-J.
- 🔄 **Thuật toán FSRS**: Sử dụng thuật toán ghi nhớ giãn cách tối ưu FSRS (Free Spaced Repetition System) giúp tối ưu hóa thời gian và tần suất ôn tập từ mới.
- 📈 **Thống kê tiến độ**: Theo dõi trạng thái ghi nhớ của từng từ (due date, difficulty, stability).
- **Công nghệ**: FSRS, SQLite.

### 6. **AI Chat Agent** (Trợ Lý Trò Chuyện Tự Do)
- 💬 **Hỏi đáp đa năng**: Trò chuyện, giải đáp thắc mắc về ngữ pháp, từ vựng mọi lúc.
- 🌐 **Dịch thuật nhanh**: Hỗ trợ dịch song ngữ Anh - Việt.
- **Công nghệ**: Google GenAI Client.

---

## 🏗️ Kiến Trúc Thư Mục Dự Án

Cấu trúc mã nguồn của dự án đã được sắp xếp gọn gàng vào các thư mục chức năng:

```
English AI/
├── code/                         # Thư mục chứa mã nguồn chính của ứng dụng
│   ├── main.py                   # Ứng dụng chính điều phối giao diện Gradio
│   ├── agent.py                  # Quản lý phiên hội thoại AI Chat Agent
│   ├── speaking_module.py        # Module luyện phát âm (Whisper & Wav2Vec2)
│   ├── listening_module.py       # Module luyện nghe qua video & audio
│   ├── reading_module.py         # Module đọc hiểu & tương tác bài đọc
│   ├── writting_module.py        # Module chấm điểm & quản lý lịch sử viết luận
│   ├── vocabulary_module.py      # Module từ vựng & thuật toán FSRS
│   ├── selection_listener.js     # Code JavaScript hỗ trợ sự kiện bôi đen/chọn từ
│   └── style_demo.css            # File cấu hình CSS cho giao diện Gradio
├── database/                     # Thư mục quản lý cơ sở dữ liệu
│   ├── database_manager.py       # Khởi tạo schema và các hàm CRUD SQLite
│   ├── add_data.py               # Script seed dữ liệu mẫu (Writing prompts, v.v.)
│   └── app_data.db               # File database SQLite chính
├── datasets/                     # Thư mục chứa dữ liệu tĩnh
│   ├── CEFR-J Wordlist Ver1.6.xlsx # File từ điển CEFR-J gốc
│   ├── Videos/                   # Video luyện nghe
│   └── short_audios/             # Các tệp audio ngắn luyện nghe
├── models/                       # Thư mục lưu trữ checkpoint của mô hình AI local
├── huggingface_cache/            # Thư mục lưu cache các mô hình tải từ HuggingFace
├── temp/                         # Thư mục chứa file tạm thời (file ghi âm, cache audio)
├── .env                          # File cấu hình biến môi trường và API key
├── .gitignore                    # Cấu hình bỏ qua các tệp không cần thiết khi git push
└── README.md                     # Tài liệu hướng dẫn sử dụng này
```

---

## 🚀 Yêu Cầu Hệ Thống & Cài Đặt

### Yêu Cầu Hệ Thống
- **Python 3.10+** (Khuyến nghị để tương thích tốt nhất với Gradio và PyTorch)
- Kết nối internet để tải mô hình học sâu ban đầu và kết nối API Gemini.

### Các Thư Viện Chính
Ứng dụng sử dụng các gói thư viện Python chính dưới đây:
```bash
gradio                  # Tạo giao diện Web UI tương tác nhanh
google-genai            # Google GenAI SDK mới nhất hỗ trợ mô hình Gemini 2.5
transformers            # Load và inference mô hình Deep Learning (BERT, Deberta)
torch                   # Backend tính toán học sâu cho các mô hình AI
openai-whisper          # Nhận diện giọng nói và chuyển thành văn bản
deep-translator         # Dịch thuật ngôn ngữ miễn phí
spacy                   # Xử lý ngôn ngữ tự nhiên (NLP)
fsrs                    # Triển khai thuật toán Spaced Repetition
rich                    # Định dạng log và hiển thị dữ liệu đẹp trên console
edge-tts                # Text-to-Speech chất lượng cao từ Microsoft Edge
nest-asyncio            # Quản lý luồng chạy bất đồng bộ trong Gradio
pandas openpyxl         # Đọc dữ liệu từ file Excel từ điển
```

### Hướng Dẫn Cài Đặt

1. **Tải mã nguồn về máy**:
   ```bash
   git clone https://github.com/ChuuniP/English-AI.git
   cd English-AI
   ```

2. **Khởi tạo môi trường ảo (Khuyên dùng)**:
   ```bash
   python -m venv venv
   # Kích hoạt trên Windows:
   venv\Scripts\activate
   # Kích hoạt trên macOS/Linux:
   source venv/bin/activate
   ```

3. **Cài đặt các gói thư viện cần thiết**:
   Tạo hoặc cài đặt các thư viện trực tiếp:
   ```bash
   pip install gradio google-genai transformers torch openai-whisper deep-translator spacy fsrs rich edge-tts nest-asyncio pandas openpyxl soundfile librosa
   ```
   Tải mô hình tiếng Anh nhỏ cho thư viện SpaCy:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Cấu hình biến môi trường**:
   Tạo file `.env` tại thư mục gốc của dự án với nội dung:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   APP_TEMP_DIR=./temp
   APP_HF_CACHE_DIR=./huggingface_cache
   ```

5. **Khởi chạy ứng dụng**:
   Chạy file chính từ thư mục gốc của dự án bằng lệnh:
   ```bash
   python code/main.py
   ```
   Sau khi khởi động thành công, mở trình duyệt và truy cập: `http://localhost:7860`

---

## 💾 Quản Lý & Xem Cơ Sở Dữ Liệu (Database)

Dữ liệu học tập và tiến độ của người học được lưu trữ tại `database/app_data.db`.

- **Xem cấu trúc và dữ liệu hiện có trong Database**:
  Bạn có thể chạy trực tiếp script manager để hiển thị nhanh bảng biểu dữ liệu trực quan trên Console nhờ thư viện `rich`:
  ```bash
  python database/database_manager.py
  ```
- **Nạp dữ liệu mẫu cho bài viết (Writing Prompts)**:
  Để nạp các đề bài luyện viết mẫu vào Database, bạn có thể chạy:
  ```bash
  python database/add_data.py
  ```

---

## 📝 Lưu Ý Khi Sử Dụng
- **Tải mô hình ban đầu**: Ở lần đầu chạy ứng dụng hoặc khi truy cập các tính năng như Speak (Whisper, Wav2Vec2) hay Write (Deberta), hệ thống sẽ mất vài phút để tự động tải các file weights mô hình từ HuggingFace về thư mục `huggingface_cache/`. Các lần chạy tiếp theo sẽ diễn ra ngay lập tức.
- **API Key**: Luyện nghe và Chat agent yêu cầu kết nối mạng ổn định cùng một API Key Gemini hợp lệ được thiết lập trong file `.env`.
