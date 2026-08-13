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

# ------------------------------------------

def load_reading_list(db):
    """Trả về dữ liệu cho gr.Dataframe: mỗi dòng là [id, title, cefr]."""
    passages = db.list_all_reading_passages()
    rows = [
        [p["id"], p["title"] or f"Reading {p['id']}", p["cefr"] or "-"]
        for p in passages
    ]
    return rows


def load_reading_list_for_ui(db):
    rows = load_reading_list(db)
    return rows, rows


def back_to_reading_list(db):
    rows = load_reading_list(db)
    return (
        gr.update(visible=True),  # rp_list_panel
        gr.update(visible=False),  # rp_detail_panel
        gr.update(visible=False),  # rp_result_panel
        rows,  # rp_list_df
        rows,  # rp_list_state
    )


# ---------- Mở 1 bài đọc ----------

def _render_question(idx: int, questions: list, answers: list):
    """Dựng nội dung hiển thị cho câu hỏi thứ idx (0-based)."""
    total = len(questions)
    q = questions[idx]

    question_md = f"**Câu {idx + 1}/{total}:** {q['question']}"
    progress_md = f"Câu {idx + 1}/{total}"
    choices = q["options"]
    radio_value = answers[idx]  # None nếu chưa chọn

    prev_interactive = idx > 0
    next_visible = idx < total - 1
    submit_visible = idx == total - 1

    return question_md, choices, radio_value, progress_md, prev_interactive, next_visible, submit_visible


def open_reading_passage(evt: gr.SelectData, table_data, db):
    """Xử lý khi người dùng click vào 1 dòng trong bảng danh sách bài đọc."""
    row_index = evt.index[0]
    passage_id = int(table_data[row_index][0])

    data = db.get_reading_passage(passage_id)
    questions = data["questions"] or []
    answers = [None] * len(questions)

    title_md = f"## {data['title'] or f'Reading {passage_id}'}\n**CEFR:** {data['cefr'] or 'N/A'}"
    passage_md = data["passage"]

    (question_md, choices, radio_value, progress_md,
     prev_interactive, next_visible, submit_visible) = _render_question(0, questions, answers)

    return (
        gr.update(visible=False),  # rp_list_panel
        gr.update(visible=True),  # rp_detail_panel
        gr.update(visible=False),  # rp_result_panel
        title_md,  # rp_title_md
        passage_md,  # rp_passage_md
        questions,  # rp_questions_state
        0,  # rp_current_index_state
        answers,  # rp_answers_state
        question_md,  # rp_question_md
        gr.update(choices=choices, value=radio_value),  # rp_options_radio
        progress_md,  # rp_progress_md
        gr.update(interactive=prev_interactive),  # rp_prev_btn
        gr.update(visible=next_visible),  # rp_next_btn
        gr.update(visible=submit_visible),  # rp_submit_btn
        gr.update(visible=True),  # rp_question_panel
        gr.update(visible=False, value=""),  # rp_warning_md
    )


# ---------- Điều hướng câu hỏi ----------

def go_to_question(direction: int, current_index: int, questions: list, answers: list, selected_answer):
    """direction: +1 (Next) hoặc -1 (Previous). Lưu đáp án hiện tại trước khi chuyển câu."""
    answers = list(answers)
    answers[current_index] = selected_answer

    new_index = current_index + direction
    new_index = max(0, min(new_index, len(questions) - 1))

    (question_md, choices, radio_value, progress_md,
     prev_interactive, next_visible, submit_visible) = _render_question(new_index, questions, answers)

    return (
        new_index,  # rp_current_index_state
        answers,  # rp_answers_state
        question_md,  # rp_question_md
        gr.update(choices=choices, value=radio_value),  # rp_options_radio
        progress_md,  # rp_progress_md
        gr.update(interactive=prev_interactive),  # rp_prev_btn
        gr.update(visible=next_visible),  # rp_next_btn
        gr.update(visible=submit_visible),  # rp_submit_btn
    )


# ---------- Nộp bài ----------

def submit_reading_quiz(current_index: int, questions: list, answers: list, selected_answer):
    """Lưu đáp án câu cuối, kiểm tra đã trả lời đủ chưa, và chấm điểm nếu đủ."""
    answers = list(answers)
    answers[current_index] = selected_answer

    total = len(questions)
    # Luôn ép lại đúng trạng thái 3 nút điều hướng theo current_index, thay vì
    # để nguyên giá trị cũ (giá trị cũ có thể sai lệch, gây ra lỗi hiện nút
    # "Next" dù đang ở câu cuối cùng).
    prev_interactive = current_index > 0
    next_visible = current_index < total - 1
    submit_visible = current_index == total - 1

    unanswered = [i + 1 for i, a in enumerate(answers) if a is None]
    if unanswered:
        warning = (
                "⚠️ Bạn chưa trả lời câu: " + ", ".join(str(i) for i in unanswered) +
                ". Vui lòng hoàn thành tất cả câu hỏi trước khi nộp bài."
        )
        return (
            answers,  # rp_answers_state
            gr.update(visible=True),  # rp_question_panel (vẫn giữ)
            gr.update(visible=False),  # rp_result_panel
            "",  # rp_result_md
            gr.update(visible=True, value=warning),  # rp_warning_md
            gr.update(interactive=prev_interactive),  # rp_prev_btn
            gr.update(visible=next_visible),  # rp_next_btn
            gr.update(visible=submit_visible),  # rp_submit_btn
        )

    score = sum(1 for a, q in zip(answers, questions) if a == q["answer"])

    lines = [f"## 📊 Kết quả: {score}/{total} câu đúng\n"]
    for i, (q, a) in enumerate(zip(questions, answers), start=1):
        correct = q["answer"]
        is_correct = (a == correct)
        icon = "✅" if is_correct else "❌"
        lines.append(f"**Câu {i}: {q['question']}** {icon}")
        for opt in q["options"]:
            if opt == correct:
                lines.append(f"- ✅ {opt}  *(Đáp án đúng)*")
            elif opt == a:
                lines.append(f"- ❌ {opt}  *(Bạn đã chọn)*")
            else:
                lines.append(f"-   {opt}")
        lines.append("")

    result_md = "\n".join(lines)

    return (
        answers,  # rp_answers_state
        gr.update(visible=False),  # rp_question_panel
        gr.update(visible=True),  # rp_result_panel
        result_md,  # rp_result_md
        gr.update(visible=False, value=""),  # rp_warning_md
        gr.update(interactive=prev_interactive),  # rp_prev_btn
        gr.update(visible=next_visible),  # rp_next_btn
        gr.update(visible=submit_visible),  # rp_submit_btn
    )


def retry_reading_quiz(questions: list):
    """Làm lại bài đọc hiện tại: reset đáp án + quay về câu 1."""
    answers = [None] * len(questions)

    (question_md, choices, radio_value, progress_md,
     prev_interactive, next_visible, submit_visible) = _render_question(0, questions, answers)

    return (
        0,  # rp_current_index_state
        answers,  # rp_answers_state
        question_md,  # rp_question_md
        gr.update(choices=choices, value=radio_value),  # rp_options_radio
        progress_md,  # rp_progress_md
        gr.update(interactive=prev_interactive),  # rp_prev_btn
        gr.update(visible=next_visible),  # rp_next_btn
        gr.update(visible=submit_visible),  # rp_submit_btn
        gr.update(visible=True),  # rp_question_panel
        gr.update(visible=False),  # rp_result_panel
        gr.update(visible=False, value=""),  # rp_warning_md
    )