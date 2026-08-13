import datetime
from fsrs import Card, State, Rating
import gradio as gr
import time
from nltk.corpus import wordnet

# Import DatabaseManager chuẩn theo cấu trúc dự án
from database.database_manager import DatabaseManager

db = DatabaseManager()

try:
    import eng_to_ipa as ipa
    _IPA_AVAILABLE = True
except ImportError:
    ipa = None
    _IPA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper nội bộ (mới thêm để sửa lỗi)
# ---------------------------------------------------------------------------
def _parse_iso_datetime_utc(value):
    """
    Parse chuỗi ISO datetime an toàn về dạng aware UTC.
    Sửa lỗi: nếu chuỗi lưu trong DB không có tzinfo (naive), so sánh trực
    tiếp với datetime.now(timezone.utc) sẽ raise TypeError. Hàm này luôn
    trả về datetime có tzinfo UTC, hoặc None nếu value rỗng/không hợp lệ.
    """
    if not value:
        return None
    if isinstance(value, datetime.datetime):
        dt = value
    else:
        dt = datetime.datetime.fromisoformat(value)

    if dt.tzinfo is None:
        # Dữ liệu cũ không có offset -> coi như đã là UTC (giả định hợp lý
        # vì mọi timestamp mới đều được lưu bằng now(timezone.utc).isoformat())
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    else:
        dt = dt.astimezone(datetime.timezone.utc)
    return dt


def load_review_session():
    """Tải dữ liệu từ vựng cần ôn tập bằng DatabaseManager."""
    total_vocab = db.get_total_vocabulary_count()

    now = datetime.datetime.now(datetime.timezone.utc)
    today = now.date()

    rows = db.get_all_user_vocabulary()
    due_list = []
    completed_count = 0

    for row in rows:
        # Giữ tương thích tuple index theo cấu trúc SQLite cũ:
        # 0: word, 1: cefr_j, 2: meaning, 3: phonetic, 4: state, 5: stability, 6: difficulty,
        # 7: elapsed_days, 8: scheduled_days, 9: last_review, 10: due
        state_val = int(row[4])
        due_time = _parse_iso_datetime_utc(row[10])
        last_review_str = row[9]

        if state_val == 0 or due_time is None or due_time <= now:
            due_list.append(row)
        elif last_review_str:
            last_review_dt = _parse_iso_datetime_utc(last_review_str)
            if last_review_dt is not None and last_review_dt.date() == today:
                completed_count += 1

    total_due = len(due_list) + completed_count
    if not due_list:
        return (total_due, total_vocab, completed_count, 0, "All Words Have Done!", "N/A", "N/A", "", [],
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                gr.update(visible=False))

    current_word_data = due_list[0]
    return (total_due, total_vocab, completed_count, completed_count + 1,
            current_word_data[0], current_word_data[1], current_word_data[2], current_word_data[3],
            due_list, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False))


def _build_cefr_html(cefr, meaning, phonetic, word):
    safe_word_for_js = str(word).replace("\\", "\\\\").replace("'", "\\'")
    phonetic_html = f"<div style='font-size: 18px; color: #6b7280; font-style: italic; margin-top: 8px;'>/{phonetic}/</div>" if phonetic else ""
    pronounce_btn = f"""
    <button type="button" class="btn-pronounce"
        onclick="try {{ const u = new SpeechSynthesisUtterance('{safe_word_for_js}'); u.lang = 'en-US'; window.speechSynthesis.cancel(); window.speechSynthesis.speak(u); }} catch(e) {{ console.error(e); }}"
        style="margin-top: 8px; padding: 4px 12px; border-radius: 6px; border: 1px solid #2f73d8; background: transparent; color: #2f73d8; cursor: pointer; font-size: 14px;">
        🔊 Phát âm
    </button>
    """
    return f"""
    <div style='font-size: 18px; color: #2f73d8; font-weight: bold;'>Cefr: {cefr}</div>
    <div style='font-size: 18px; color: #10b981; font-weight: bold; margin-top: 8px;'>Nghĩa: {meaning}</div>
    {phonetic_html}
    {pronounce_btn}
    """


def show_answer_action(due_list_state):
    if due_list_state:
        word, cefr, meaning, phonetic, *_ = due_list_state[0]
        cefr_html = _build_cefr_html(cefr, meaning, phonetic, word)
        print(cefr_html)
    else:
        cefr_html = ""
        print("Không có dữ liệu, lỗi chỗ này")

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(value=cefr_html, visible=True)
    )


def show_answer_toggle_visibility():
    """
    Bước 1: chỉ đổi visibility của area_question / area_answer / fsrs_buttons_row.
    Tách riêng khỏi việc set value cho cefr_display để tránh bug Gradio:
    khi 1 gr.HTML đang visible=False mà đổi visible=True và value cùng lúc
    trong 1 lần update, đôi khi frontend chỉ toggle hiển thị mà KHÔNG
    re-render nội dung HTML ở lần đầu (phải bấm lần 2 mới thấy).
    """
    return (
        gr.update(visible=False),  # area_question
        gr.update(visible=True),   # area_answer
        gr.update(visible=True),   # fsrs_buttons_row
    )


def show_answer_action_content_only(due_list_state):
    """
    Bước 2: chạy SAU khi cefr_display đã visible=True (nhờ .then()),
    lúc này chỉ cần set value -> component sẽ re-render nội dung đúng
    ngay từ lần bấm đầu tiên.
    """
    if due_list_state:
        word, cefr, meaning, phonetic, *_ = due_list_state[0]
        cefr_html = _build_cefr_html(cefr, meaning, phonetic, word)
        print(cefr_html)
    else:
        cefr_html = ""
        print("Không có dữ liệu, lỗi chỗ này")

    return gr.update(value=cefr_html, visible=True)


def review_word_action(fsrs_app, choice_str, due_list_state):
    if not due_list_state:
        return pipeline_load()

    current_word_data = due_list_state[0]
    word, cefr_j, meaning, phonetic, state, stability, difficulty, elapsed_days, scheduled_days, last_review, due = current_word_data

    rating_map = {
        "Again": Rating.Again,
        "Hard": Rating.Hard,
        "Good": Rating.Good,
        "Easy": Rating.Easy
    }

    # SỬA: choice_str không hợp lệ trước đây sẽ làm KeyError và crash callback.
    # Giờ fallback về "Good" và log cảnh báo thay vì crash toàn bộ UI.
    rating_choice = rating_map.get(choice_str)
    if rating_choice is None:
        print(f"Cảnh báo: rating '{choice_str}' không hợp lệ, dùng mặc định 'Good'.")
        rating_choice = Rating.Good

    card = Card()
    card.state = State(state)
    card.stability = stability
    card.difficulty = difficulty
    card.elapsed_days = elapsed_days
    # SỬA: parse an toàn (tránh crash nếu last_review thiếu tzinfo)
    card.last_review = _parse_iso_datetime_utc(last_review)
    card.scheduled_days = scheduled_days
    card.due = _parse_iso_datetime_utc(due) or datetime.datetime.now(datetime.timezone.utc)

    now = datetime.datetime.now(datetime.timezone.utc)
    if not hasattr(card, "difficulty") or card.difficulty is None or card.difficulty < 1.0:
        card.difficulty = 5.0
    new_card, _ = fsrs_app.review_card(card, rating_choice, now)

    # SỬA: trước đây elapsed_days/scheduled_days luôn bị ghi cứng = 0, làm mất
    # dữ liệu thật mà fsrs vừa tính ra, ảnh hưởng tới độ chính xác lịch ôn tập
    # ở những lần review kế tiếp. Giờ lưu đúng giá trị mới từ new_card.
    db.upsert_user_vocabulary(
        word=word,
        cefr_j=cefr_j,
        meaning=meaning,
        phonetic=phonetic,
        state=new_card.state.value,
        stability=new_card.stability,
        difficulty=new_card.difficulty,
        elapsed_days=new_card.elapsed_days,
        scheduled_days=new_card.scheduled_days,
        last_review=new_card.last_review.isoformat() if new_card.last_review else None,
        due=new_card.due.isoformat(),
        added_at=now.isoformat()
    )

    return pipeline_load()


def render_stats_and_progress(t_due, t_vocab, comp, curr_idx, word, cefr, meaning, phonetic, due_list):
    percent = int((comp / t_due) * 100) if t_due > 0 else 100
    html_due = f'<div class="stat-card"><div class="stat-title">Review Today</div><div class="stat-value">{t_due}</div></div>'
    html_total = f'<div class="stat-card"><div class="stat-title">Total Words</div><div class="stat-value">{t_vocab}</div></div>'
    html_comp = f'<div class="stat-card"><div class="stat-title">Done</div><div class="stat-value">{comp}/{t_due}</div></div>'
    html_bar = f'<div class="custom-progress-bar"><div class="custom-progress-fill" style="width: {percent}%;"></div></div>'

    idx_str = f"<div class='word-index'>Từ số {curr_idx} / {t_due}</div>" if t_due > 0 else "<div class='word-index'>0 / 0</div>"
    main_str = f"<div class='word-main'>{word}</div>"

    cefr_str = gr.update(value="", visible=False)

    return (html_due, html_total, html_comp, f"<p style='text-align: right;'><b>{percent}%</b></p>", html_bar,
            idx_str, main_str, cefr_str, due_list)


def pipeline_load():
    res = load_review_session()
    outputs = render_stats_and_progress(res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8])
    return outputs + (res[9], res[10], res[11], res[12])


def refresh_review_session():
    """
    Dùng riêng cho btn_refresh ("Check again"). pipeline_load() tự nó không cho
    người dùng biết là nút đã thật sự chạy hay chưa: nếu vẫn chưa có từ mới đến
    hạn, kết quả trả về y hệt trạng thái cũ -> màn hình đứng yên, người dùng dễ
    hiểu lầm là nút bị lỗi/không phản hồi. Thêm toast gr.Info() để xác nhận.
    """
    result = pipeline_load()
    due_list = result[8]  # vị trí due_list trong tuple trả về của pipeline_load
    if not due_list:
        gr.Info("Chưa có từ mới nào đến hạn ôn tập. Bạn quay lại sau nhé!")
    return result


def get_base_word(lemmatizer, word: str) -> str:
    """Chuyển từ về dạng nguyên thể (Lemma)."""
    clean_word = word.strip().strip(".,!?\"'()[]{}*:;").lower()
    if not clean_word:
        return ""

    if not lemmatizer:
        return clean_word
    base_n = lemmatizer.lemmatize(clean_word, pos=wordnet.NOUN)
    if base_n != clean_word:
        return base_n

    base_v = lemmatizer.lemmatize(clean_word, pos=wordnet.VERB)
    if base_v != clean_word:
        return base_v

    return clean_word


def get_translation_with_retry(translator, words: str, max_retries: int = 3, delay: float = 0.5) -> str:
    """
    Hàm 1: Dịch nghĩa từ/câu từ Tiếng Anh sang Tiếng Việt có cơ chế retry.
    """
    if not words or not str(words).strip():
        return ""

    word_str = str(words).strip()
    meaning = word_str

    for attempt in range(max_retries):
        try:
            result = translator.translate(word_str)
            # SỬA: translate() có thể trả về None dù không raise exception
            # (vd timeout im lặng ở một số bản deep_translator). Trước đây
            # meaning.strip() ở cuối hàm sẽ crash AttributeError trong
            # trường hợp này. Giờ chỉ nhận kết quả khi nó là chuỗi hợp lệ.
            if result and isinstance(result, str) and result.strip():
                meaning = result
                break
            if attempt < max_retries - 1:
                time.sleep(delay)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Lỗi dịch sau {max_retries} lần thử: {e}")
                meaning = f"[Lỗi dịch] {word_str}"
            else:
                time.sleep(delay)

    return meaning.strip() if isinstance(meaning, str) else word_str


def estimate_cefr_level(CEFR_DICT: dict, target_word: str) -> str:
    """
    Hàm 2: Xác định cấp độ CEFR. Nếu là N/A hoặc không tìm thấy trong từ điển
    thì tự động đoán dựa trên độ dài từ.
    """
    if not target_word or not target_word.strip():
        return "N/A"

    clean_word = target_word.strip().strip(".,!?\"'()[]{}*:;").lower()
    cefr_level = "N/A"

    # Tra cứu trong từ điển CEFR
    if CEFR_DICT:
        cefr_level = CEFR_DICT.get(clean_word, "N/A")

    # Nếu vẫn là N/A thì tự động tính theo độ dài từ
    if cefr_level == "N/A" or not cefr_level:
        length = len(clean_word)
        if length <= 4:
            cefr_level = "A1"
        elif length == 5:
            cefr_level = "A2"
        elif length == 6:
            cefr_level = "B1"
        elif length == 7:
            cefr_level = "B2"
        elif length == 8:
            cefr_level = "C1"
        else:
            cefr_level = "C2"

    return cefr_level.strip()


def get_word_phonetic(word: str) -> str:
    """
    Hàm 3: Lấy phiên âm IPA cho từ vựng (chạy offline bằng eng_to_ipa).
    """
    if not word or not str(word).strip() or not _IPA_AVAILABLE:
        return ""

    clean_word = str(word).strip().strip(".,!?\"'()[]{}*:;").lower()
    if not clean_word:
        return ""

    try:
        result = ipa.convert(clean_word)
        if not result or result.strip("*") == clean_word:
            return ""
        return result.strip()
    except Exception:
        return ""


def process_vocabulary_info(CEFR_DICT, translator, lemmatizer, words: str):
    """
    Hàm tổng hợp gọi 3 hàm thành phần ở trên.
    """
    if not words or not str(words).strip():
        return "", "N/A", ""

    word_str = str(words).strip()
    word_to_check = word_str.split()[0] if " " in word_str else word_str
    base_word = get_base_word(lemmatizer, word_to_check)
    target_word = base_word if base_word else word_to_check

    meaning = get_translation_with_retry(translator, word_str)
    cefr_level = estimate_cefr_level(CEFR_DICT, target_word)
    phonetic_ipa = get_word_phonetic(target_word)

    return meaning, cefr_level, phonetic_ipa