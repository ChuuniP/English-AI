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
        due_time = datetime.datetime.fromisoformat(row[10])
        last_review_str = row[9]

        if state_val == 0 or due_time <= now:
            due_list.append(row)
        elif last_review_str:
            last_review_dt = datetime.datetime.fromisoformat(last_review_str)
            if last_review_dt.astimezone(datetime.timezone.utc).date() == today:
                completed_count += 1

    total_due = len(due_list) + completed_count
    if not due_list:
        return (total_due, total_vocab, completed_count, 0, "All Words Have Done!", "N/A", "N/A", "", [],
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))

    current_word_data = due_list[0]
    return (total_due, total_vocab, completed_count, completed_count + 1,
            current_word_data[0], current_word_data[1], current_word_data[2], current_word_data[3],
            due_list, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))


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
    else:
        cefr_html = ""

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(value=cefr_html, visible=True)
    )


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

    rating_choice = rating_map[choice_str]

    card = Card()
    card.state = State(state)
    card.stability = stability
    card.difficulty = difficulty
    card.elapsed_days = elapsed_days
    if last_review and isinstance(last_review, str):
        card.last_review = datetime.datetime.fromisoformat(last_review)
    else:
        card.last_review = last_review

    card.scheduled_days = scheduled_days
    card.due = datetime.datetime.fromisoformat(due)

    now = datetime.datetime.now(datetime.timezone.utc)
    if not hasattr(card, "difficulty") or card.difficulty is None or card.difficulty < 1.0:
        card.difficulty = 5.0
    new_card, _ = fsrs_app.review_card(card, rating_choice, now)

    # Cập nhật trạng thái từ vựng đã ôn qua DatabaseManager
    db.upsert_user_vocabulary(
        word=word,
        cefr_j=cefr_j,
        meaning=meaning,
        phonetic=phonetic,
        state=new_card.state.value,
        stability=new_card.stability,
        difficulty=new_card.difficulty,
        elapsed_days=0,
        scheduled_days=0,
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
    return outputs + (res[9], res[10], res[11])

def get_base_word(lemmatizer, word: str) -> str:
    """Chuyển từ về dạng nguyên thể (Lemma)."""
    clean_word = word.strip().strip(".,!?\"'()[]{}*:;").lower()
    if not clean_word:
        return ""

    if lemmatizer:
        base_v = lemmatizer.lemmatize(clean_word, pos=wordnet.VERB)
        if base_v != clean_word:
            return base_v
        base_n = lemmatizer.lemmatize(clean_word, pos=wordnet.NOUN)
        return base_n
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
            meaning = translator.translate(word_str)
            if meaning and meaning.strip():
                break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Lỗi dịch sau {max_retries} lần thử: {e}")
                meaning = f"[Lỗi dịch] {word_str}"
            else:
                time.sleep(delay)

    return meaning.strip()


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