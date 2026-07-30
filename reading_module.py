import ast
import datetime
import gradio as gr
from fsrs import Card

from vocabulary_module import (
    pipeline_load,
    process_vocabulary_info,
    get_base_word,
    get_word_phonetic
)
# Import DatabaseManager chuẩn theo cấu trúc dự án
from database.database_manager import DatabaseManager

db = DatabaseManager()


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

    custom_zone_id = "reading-zone-active"

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


def translate_and_get_cefr_with_excel(CEFR_DICT, translator, lemmatizer, words):
    """Ủy quyền toàn bộ xử lý dịch/CEFR/Phonetic sang vocabulary_module."""
    if not words or not str(words).strip():
        return "", "N/A", ""

    return process_vocabulary_info(CEFR_DICT, translator, lemmatizer, words)


def add_new_word_to_db(CEFR_DICT, lemmatizer, word, cefr_j, meaning, phonetic=""):
    error_vocab_updates = tuple([gr.update()] * 12)

    if not word or not str(word).strip():
        return (*error_vocab_updates, gr.update(), "⚠️ Vui lòng chọn hoặc nhập từ hợp lệ!")

    lemma_word = get_base_word(lemmatizer, word)
    if not lemma_word:
        return (*error_vocab_updates, gr.update(), "⚠️ Không tìm thấy từ gốc hợp lệ!")

    clean_word = lemma_word.capitalize() if word.strip()[0].isupper() else lemma_word

    clean_meaning = str(meaning).strip() if meaning else "Chưa rõ nghĩa"
    clean_cefr = str(cefr_j).strip() if cefr_j else "N/A"

    if not clean_cefr:
        clean_cefr = "N/A"

    clean_phonetic = str(phonetic).strip() if phonetic else get_word_phonetic(clean_word)

    lookup_key = clean_word.lower()
    if clean_cefr == "N/A" and CEFR_DICT and lookup_key in CEFR_DICT:
        clean_cefr = CEFR_DICT[lookup_key]

    try:
        # Kiểm tra sự tồn tại của từ bằng phương thức có sẵn trong DatabaseManager
        if db.get_user_vocabulary_by_word(clean_word):
            return (*pipeline_load(), gr.update(visible=False), f"⚠️ Từ gốc **'{clean_word}'** đã tồn tại trong danh sách FSRS!")

        now = datetime.datetime.now(datetime.timezone.utc)
        card = Card()

        # Lưu từ vựng thông qua upsert_user_vocabulary của DatabaseManager
        db.upsert_user_vocabulary(
            word=clean_word,
            cefr_j=clean_cefr,
            meaning=clean_meaning,
            phonetic=clean_phonetic,
            state=card.state.value,
            stability=card.stability,
            difficulty=card.difficulty,
            elapsed_days=0,
            scheduled_days=0,
            last_review=None,
            due=card.due.isoformat(),
            added_at=now.isoformat()
        )
        msg = f"✅ Đã thêm từ gốc thành công: **'{clean_word}'** vào FSRS!"
    except Exception as e:
        return (*error_vocab_updates, gr.update(), f"❌ Lỗi hệ thống khi lưu: {str(e)}")

    # Reset lại dữ liệu pipeline và ẩn hàng nút FSRS
    return (*pipeline_load(), gr.update(visible=False), msg)


def add_new_word_to_db_no_ui_update(CEFR_DICT, lemmatizer, word, cefr_j, meaning, phonetic=""):
    """
    Thêm từ vào DB nhưng không cập nhật UI Vocabulary.
    Dùng khi thêm từ từ tab Reading - UI sẽ reload khi user click vào Vocabulary tab.
    """
    empty_vocab_updates = tuple([gr.update()] * 12)

    if not word or not str(word).strip():
        return (*empty_vocab_updates, gr.update(), "⚠️ Vui lòng chọn hoặc nhập từ hợp lệ!")

    lemma_word = get_base_word(lemmatizer, word)
    if not lemma_word:
        return (*empty_vocab_updates, gr.update(), "⚠️ Không tìm thấy từ gốc hợp lệ!")

    clean_word = lemma_word.capitalize() if word.strip()[0].isupper() else lemma_word

    clean_meaning = str(meaning).strip() if meaning else "Chưa rõ nghĩa"
    clean_cefr = str(cefr_j).strip() if cefr_j else "N/A"

    if not clean_cefr:
        clean_cefr = "N/A"

    clean_phonetic = str(phonetic).strip() if phonetic else get_word_phonetic(clean_word)

    lookup_key = clean_word.lower()
    if clean_cefr == "N/A" and CEFR_DICT and lookup_key in CEFR_DICT:
        clean_cefr = CEFR_DICT[lookup_key]

    try:
        if db.get_user_vocabulary_by_word(clean_word):
            return (*empty_vocab_updates, gr.update(visible=False), f"⚠️ Từ gốc **'{clean_word}'** đã tồn tại trong danh sách FSRS!")

        now = datetime.datetime.now(datetime.timezone.utc)
        card = Card()

        db.upsert_user_vocabulary(
            word=clean_word,
            cefr_j=clean_cefr,
            meaning=clean_meaning,
            phonetic=clean_phonetic,
            state=card.state.value,
            stability=card.stability,
            difficulty=card.difficulty,
            elapsed_days=0,
            scheduled_days=0,
            last_review=None,
            due=card.due.isoformat(),
            added_at=now.isoformat()
        )
        msg = f"✅ Đã thêm từ gốc thành công: **'{clean_word}'** vào FSRS!"
    except Exception as e:
        return (*empty_vocab_updates, gr.update(), f"❌ Lỗi hệ thống khi lưu: {str(e)}")

    return (*empty_vocab_updates, gr.update(visible=False), msg)
