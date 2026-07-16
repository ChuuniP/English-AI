import ast
import gradio as gr
import time
from nltk.corpus import wordnet
import sqlite3
import datetime
from fsrs import Card

from vocabulary_module import pipeline_load

try:
    import eng_to_ipa as ipa
    _IPA_AVAILABLE = True
except ImportError:
    ipa = None
    _IPA_AVAILABLE = False
    print("⚠️ Chưa cài đặt thư viện 'eng_to_ipa'. Chạy: pip install eng_to_ipa để bật tính năng phiên âm IPA.")


def render_html_reading_zone(text_content, font_family, font_size, bg_color, text_color):
    # Ánh xạ từ nhãn Tiếng Việt sang CSS thật
    font_map = {
        "Serif (Có chân)": "'Source Serif 4', Georgia, serif",
        "Sans-serif (Không chân)": "'Inter', sans-serif",
        "Monospace (Đơn cách)": "'IBM Plex Mono', monospace"
    }
    bg_map = {
        "Giấy cổ (Mặc định)": "#FAF8F3",
        "Trắng tinh": "#FFFFFF",
        "Chế độ tối (Dark mode)": "#1E1E1E",
        "Xanh mint dịu mắt": "#E6F2ED"
    }
    text_map = {
        "Xanh mực (Mặc định)": "#1B2A45",
        "Xám đậm": "#2D3748",
        "Trắng sáng (Cho nền tối)": "#E2E8F0",
        "Xanh đại dương": "#1A365D"
    }

    real_font = font_map.get(font_family, "'Source Serif 4', Georgia, serif")
    real_bg = bg_map.get(bg_color, "#FAF8F3")
    real_text = text_map.get(text_color, "#1B2A45")

    # Tạo ID ngẫu nhiên hoặc cố định để khoanh vùng CSS, tránh bị file style.css đè
    custom_zone_id = "reading-zone-active"

    # Định nghĩa thẻ style động - Đây là chìa khóa để ép màu chữ hoạt động
    dynamic_css = f"""
    <style>
        #{custom_zone_id} {{
            background-color: {real_bg} !important;
            border: 1px solid #D9D2C0;
            border-radius: 6px;
            height: 450px;
            overflow-y: auto;
            padding: 20px;
        }}
        #{custom_zone_id} p, 
        #{custom_zone_id} .prose p,
        #{custom_zone_id} * {{
            color: {real_text} !important;
            font-family: {real_font} !important;
            font-size: {font_size} !important;
            line-height: 1.8 !important;
            margin-top: 0px !important;
            margin-bottom: 16px !important;
            text-align: justify !important;
            background: transparent !important;
        }}
    </style>
    """

    # Nếu không có text, trả về khung trống chuẩn style đã chọn
    if not text_content or not text_content.strip():
        default_html = f"""
        {dynamic_css}
        <div id="{custom_zone_id}">
            <p style="font-style: italic;">Vui lòng tải lên một file .txt để bắt đầu đọc...</p>
        </div>
        """
        return gr.update(value=default_html)

    paragraphs = text_content.split("\n")
    html_body = ""
    for p in paragraphs:
        clean_p = p.strip()
        if clean_p:
            html_body += f"<p>{clean_p}</p>"

    # Kết hợp mã CSS động bọc ngoài vùng hiển thị văn bản
    full_html = f"""
    {dynamic_css}
    <div id="{custom_zone_id}">
        {html_body}
    </div>
    """

    return gr.update(value=full_html)

def handle_file_upload(file_obj, font_family, font_size, bg_color, text_color):
    """Hàm xử lý khi người dùng upload file mới"""
    if file_obj is None:
        return "", render_html_reading_zone("", font_family, font_size, bg_color, text_color)

    try:
        with open(file_obj.name, "r", encoding="utf-8") as f:
            raw_content = f.read().strip()

        if not raw_content:
            return "", render_html_reading_zone("", font_family, font_size, bg_color, text_color)

        if raw_content.startswith("[") and raw_content.endswith("]"):
            try:
                word_list = ast.literal_eval(raw_content)
                if isinstance(word_list, list):
                    content = " ".join(str(w) for w in word_list)
                else:
                    content = raw_content
            except Exception:
                content = raw_content
        else:
            content = raw_content

        return content, render_html_reading_zone(content, font_family, font_size, bg_color, text_color)

    except Exception as e:
        error_html = f"<div style='color: #AE4A3B; padding: 20px;'>❌ Lỗi khi đọc file: {str(e)}</div>"
        return "", gr.update(value=error_html)


def get_phonetic_ipa(word):
    """Trả về phiên âm IPA của một từ tiếng Anh, dùng thư viện eng_to_ipa (offline).
    Trả về chuỗi rỗng nếu không tra được hoặc thư viện chưa được cài đặt."""
    if not word or not str(word).strip():
        return ""

    clean_word = str(word).strip().strip(".,!?\"'()[]{}*:;").lower()
    if not clean_word:
        return ""

    if not _IPA_AVAILABLE:
        return ""

    try:
        result = ipa.convert(clean_word)
        # eng_to_ipa bọc từ không tra được trong dấu * (vd: "*unknownword*")
        if not result or result.strip("*") == clean_word:
            return ""
        return result.strip()
    except Exception:
        return ""

def get_base_word(lemmatizer, word):
    clean_word = word.strip().strip(".,!?\"'()[]{}*:;").lower()
    if not clean_word:
        return ""

    base_v = lemmatizer.lemmatize(clean_word, pos=wordnet.VERB)
    if base_v != clean_word:
        return base_v

    base_n = lemmatizer.lemmatize(clean_word, pos=wordnet.NOUN)
    return base_n


def translate_and_get_cefr_with_excel(CEFR_DICT, translator, lemmatizer, words, max_retries=3, delay=0.5):
    # Luôn trả về 3 giá trị: nghĩa, cấp độ CEFR, phiên âm IPA
    if not words or not str(words).strip():
        return "", "N/A", ""

    word_str = str(words).strip()
    meaning = word_str

    for attempt in range(max_retries):
        try:
            meaning = translator.translate(word_str)
            if meaning and meaning.strip():  # Kiểm tra nếu kết quả trả về hợp lệ
                break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Lỗi dịch sau {max_retries} lần thử: {e}")
                meaning = f"[Lỗi dịch] {word_str}"
            else:
                time.sleep(delay)

    word_to_check = word_str.split()[0] if " " in word_str else word_str
    base_word = get_base_word(lemmatizer, word_to_check)

    cefr_level = "N/A"
    if CEFR_DICT:
        cefr_level = CEFR_DICT.get(base_word.strip().lower(), "N/A")
        if cefr_level == "N/A":
            clean_word = word_to_check.strip().strip(".,!?\"'()[]{}*:;").lower()
            cefr_level = CEFR_DICT.get(clean_word, "N/A")

    phonetic_ipa = get_phonetic_ipa(base_word if base_word else word_to_check)

    return meaning.strip(), cefr_level.strip(), phonetic_ipa


def add_new_word_to_db(DB_PATH, CEFR_DICT, lemmatizer, word, cefr_j, meaning, phonetic=""):
    error_vocab_updates = (*[gr.update()] * 8, gr.update(), gr.update(), gr.update(), gr.update())

    if not word or not str(word).strip():
        return (*error_vocab_updates, "⚠️ Vui lòng chọn hoặc nhập từ hợp lệ!")

    lemma_word = get_base_word(lemmatizer, word)
    if not lemma_word:
        return (*error_vocab_updates, "⚠️ Không tìm thấy từ gốc hợp lệ!")

    clean_word = lemma_word.capitalize() if word.strip()[0].isupper() else lemma_word

    # Chuẩn hóa loại bỏ khoảng trắng giả lập (ví dụ người dùng nhập toàn dấu cách)
    clean_meaning = str(meaning).strip() if meaning else ""
    clean_cefr = str(cefr_j).strip() if cefr_j else "N/A"

    if not clean_meaning:
        clean_meaning = "Chưa rõ nghĩa"

    if not clean_cefr or clean_cefr == "":
        clean_cefr = "N/A"

    clean_phonetic = str(phonetic).strip() if phonetic else ""
    if not clean_phonetic:
        # Nếu giao diện chưa gửi phiên âm, thử tra lại theo từ gốc
        clean_phonetic = get_phonetic_ipa(clean_word)

    lookup_key = clean_word.lower()
    if not CEFR_DICT or lookup_key not in CEFR_DICT:
        # Nếu từ điển không có, tự động gán N/A thay vì chặn không cho người dùng học từ đó
        if clean_cefr == "N/A":
            clean_cefr = "N/A"

    # Điền giá trị tự động từ từ điển nếu giao diện gửi lên bị thiếu
    if clean_cefr == "N/A" and CEFR_DICT and lookup_key in CEFR_DICT:
        clean_cefr = CEFR_DICT[lookup_key]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT word FROM user_vocabulary WHERE word = ?", (clean_word,))
        if cursor.fetchone():
            conn.close()
            return (*pipeline_load(DB_PATH), f"⚠️ Từ gốc **'{clean_word}'** đã tồn tại trong danh sách FSRS!")

        now = datetime.datetime.now(datetime.timezone.utc)
        card = Card()

        cursor.execute("""
            INSERT INTO user_vocabulary (
                word, cefr_j, meaning, phonetic, state, stability, difficulty, 
                elapsed_days, scheduled_days, last_review, due, added_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            clean_word,
            clean_cefr,
            clean_meaning,
            clean_phonetic,
            card.state.value,
            card.stability,
            card.difficulty,
            0, 0, None,
            card.due.isoformat(),
            now.isoformat()
        ))
        conn.commit()
        msg = f"✅ Đã thêm từ gốc thành công: **'{clean_word}'** vào FSRS!"
    except Exception as e:
        return (*error_vocab_updates, f"❌ Lỗi hệ thống khi lưu: {str(e)}")
    finally:
        conn.close()

    return (*pipeline_load(DB_PATH), msg)