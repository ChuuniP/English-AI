import json
import gradio as gr
import random
import asyncio
import nest_asyncio
import edge_tts
import os
import pandas as pd
import re
from pathlib import Path
import glob
from vocabulary_module import process_vocabulary_info
# Đổi sang import DatabaseManager từ file database_manager
from database.database_manager import DatabaseManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_CACHE_DIR = PROJECT_ROOT / "temp" / "audio_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

db = DatabaseManager()

def load_video(VIDEO_PATH):
    return VIDEO_PATH

def lm_transcribe(model_whisper, video_path: str) -> str:
    result = model_whisper.transcribe(video_path, fp16=False)
    return result["text"].strip()

def lm_generate_questions(client, transcript: str):
    prompt = f"""Bạn là giáo viên tiếng Anh đang thiết kế bài tập luyện nghe.
Dựa vào đoạn transcript audio dưới đây, hãy đặt ra ĐÚNG 3 câu hỏi kiểm tra khả năng nghe hiểu của người học.
Câu hỏi nên bao quát nội dung chính, chi tiết cụ thể, và suy luận (không chỉ hỏi sự kiện bề mặt).

Transcript:
\"\"\"{transcript}\"\"\"

Trả lời CHỈ bằng JSON, không thêm chữ nào khác, theo đúng format:
{{
  "questions": [
    {{"id": 1, "question": "..."}},
    {{"id": 2, "question": "..."}},
    {{"id": 3, "question": "..."}}
  ]
}}
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    text = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)["questions"]

def lm_evaluate_answers(client, transcript: str, qa_list: list):
    qa_text = "\n".join(
        [f"Câu {qa['id']}: {qa['question']}\nTrả lời của người học: {qa['answer']}" for qa in qa_list]
    )
    prompt = f"""Bạn là giáo viên chấm bài luyện nghe tiếng Anh.
Dựa vào transcript gốc, hãy đánh giá câu trả lời của người học cho từng câu hỏi.

Transcript gốc:
\"\"\"{transcript}\"\"\"

Câu hỏi và câu trả lời:
{qa_text}

Với mỗi câu, hãy:
- Chấm điểm 0-10 cho độ chính xác/mức độ hiểu
- Giải thích ngắn gọn tại sao đúng/sai
- Đưa ra ý đúng nếu người học trả lời sai/thiếu

Cuối cùng cho một đánh giá tổng quan về mức độ nghe hiểu của người học.

Trả lời CHỈ bằng JSON theo format:
{{
  "per_question": [
    {{"id": 1, "score": 0-10, "feedback": "..."}}
  ],
  "overall_score": 0-10,
  "overall_feedback": "..."
}}
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    text = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)

def render_listening_score(evaluation: dict) -> str:
    """Render bảng điểm cuối bài, tái dùng class .band-seal / .feedback-item có sẵn trong style.css"""
    overall = evaluation.get("overall_score", 0)
    pct = max(0, min(100, overall * 10))

    items_html = ""
    for item in evaluation.get("per_question", []):
        score = item.get("score", 0)
        if score >= 8:
            tag_class, tag_label = "good", "TỐT"
        elif score >= 5:
            tag_class, tag_label = "warn", "KHÁ"
        else:
            tag_class, tag_label = "error", "CẦN CẢI THIỆN"

        items_html += f"""
        <div class="feedback-item">
            <span class="feedback-tag {tag_class}">{tag_label} · {score}/10</span>
            <p style="margin:6px 0 0 0;">Câu {item.get('id')}: {item.get('feedback', '')}</p>
        </div>
        """

    return f"""
    <div style="display:flex; gap:24px; align-items:flex-start; flex-wrap:wrap;">
        <div class="band-seal" style="flex:0 0 140px;">
            <div class="band-ring" style="--pct:{pct};">
                <div class="band-ring-inner">
                    <span class="band-number">{overall}</span>
                </div>
            </div>
            <span class="band-caption">Điểm nghe hiểu / 10</span>
        </div>
        <div class="feedback-card" style="flex:1; min-width:260px; border:none; padding:0;">
            {items_html}
            <div style="border-top:1px solid var(--rule); margin-top:10px; padding-top:10px;">
                <strong>Nhận xét tổng quan:</strong>
                <p style="margin:6px 0 0 0; color:var(--ink-soft);">{evaluation.get('overall_feedback', '')}</p>
            </div>
        </div>
    </div>
    """

def start_listening_quiz(model_whisper, client, VIDEO_PATH):
    """Transcribe video + sinh 3 câu hỏi, hiện câu hỏi đầu tiên."""
    if not client:
        error_html = "<p style='color:var(--clay)'>Vui lòng cấu hình biến môi trường GEMINI_API_KEY.</p>"
        return (
            "", [], 0, [],
            gr.update(visible=True),          # start_quiz_btn (giữ nguyên để thử lại)
            gr.update(visible=False),         # question_group
            gr.update(value=""),              # question_display
            gr.update(value=""),              # answer_input
            gr.update(value=""),              # progress_display
            gr.update(visible=False),         # score_group
            gr.update(value=error_html),      # score_display
            gr.update(value="Câu tiếp theo →"),  # submit_answer_btn
        )

    video_name = os.path.basename(VIDEO_PATH)

    # 1. Kiểm tra cache trong DB trước (theo tên file video)
    cached_transcript = db.get_mp3_transcript(video_name)
    cached_questions = db.get_mp3_questions(video_name) if cached_transcript else None

    if cached_transcript and cached_questions:
        # Đã có sẵn transcript + bộ câu hỏi -> load thẳng, không cần làm lại
        transcript = cached_transcript
        questions = cached_questions
    else:
        # 2. Chưa có (hoặc thiếu 1 phần) -> transcribe (nếu cần) + gọi Gemini sinh câu hỏi
        transcript = cached_transcript or lm_transcribe(model_whisper, VIDEO_PATH)
        questions = lm_generate_questions(client, transcript)

        # 3. Lưu lại vào DB để lần sau dùng luôn, không phải làm lại
        db.save_mp3_transcript_and_questions(video_name, transcript, questions)

    return (
        transcript, questions, 0, [],
        gr.update(visible=False),                                   # ẩn nút bắt đầu
        gr.update(visible=True),                                    # hiện khung câu hỏi
        gr.update(value=f"**{questions[0]['question']}**"),
        gr.update(value=""),
        gr.update(value=f"Câu 1 / {len(questions)}"),
        gr.update(visible=False),                                   # ẩn bảng điểm cũ (nếu có)
        gr.update(value=""),
        gr.update(value="Câu tiếp theo →"),
    )

def submit_listening_answer(client, current_index, answer_text, transcript, questions, answers):
    answers = answers + [{
        "id": questions[current_index]["id"],
        "question": questions[current_index]["question"],
        "answer": (answer_text or "").strip() or "(không trả lời)"
    }]

    # Còn câu tiếp theo -> hiện câu kế tiếp
    if current_index < len(questions) - 1:
        next_index = current_index + 1
        is_last = next_index == len(questions) - 1
        return (
            next_index, answers,
            gr.update(value=f"**{questions[next_index]['question']}**"),
            gr.update(value=""),
            gr.update(value=f"Câu {next_index + 1} / {len(questions)}"),
            gr.update(),                       # question_group vẫn hiện
            gr.update(),                        # score_group vẫn ẩn
            gr.update(),                         # score_display giữ nguyên
            gr.update(value="Nộp bài & Xem điểm" if is_last else "Câu tiếp theo →"),
        )

    # Câu cuối cùng -> chấm điểm và hiện bảng kết quả
    evaluation = lm_evaluate_answers(client, transcript, answers)
    score_html = render_listening_score(evaluation)

    return (
        current_index, answers,
        gr.update(), gr.update(), gr.update(),
        gr.update(visible=False),          # ẩn khung câu hỏi
        gr.update(visible=True),           # hiện bảng điểm
        gr.update(value=score_html),
        gr.update(),
    )

nest_asyncio.apply()

async def _tts_async(text: str, output_path: str):
    communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural", rate="-10%")
    await communicate.save(output_path)


def generate_audio(text: str, output_path: str):
    asyncio.run(_tts_async(text, output_path))
    return output_path


def get_random_words_by_level(cefr_dict: dict, level: str, count: int = 2) -> list:
    """Lấy ngẫu nhiên N từ theo cấp độ CEFR."""
    words = [
        word for word, lvl in cefr_dict.items()
        if str(lvl).strip().upper() == level.strip().upper()
    ]
    if len(words) < count:
        return words
    return random.sample(words, count)


def get_random_words_with_examples_by_level(cefr_level: str, count: int = 2) -> list:
    rows = db.get_all_user_vocabulary()
    candidates = []
    for row in rows:
        # Cấu trúc row theo DatabaseManager: 0: word, 1: cefr_j, ...
        word, cefr_j = row[0], row[1]
        if str(cefr_j).strip().upper() != str(cefr_level).strip().upper():
            continue
        record = db.get_word(word)
        if record and record.get("examples"):
            candidates.append(word)

    if not candidates:
        return []
    if len(candidates) <= count:
        return candidates
    return random.sample(candidates, count)


def generate_sentences_with_gemini(client, words: list) -> list:
    prompt = f"""
    Bạn là một giáo viên tiếng Anh.
    Với danh sách các từ sau: {json.dumps(words)}

    Hãy tạo đúng {len(words)} câu tiếng Anh ngắn gọn, tự nhiên. Mỗi câu phải chứa đúng 1 từ tương ứng trong danh sách.

    Trả về định dạng JSON duy nhất như sau, không kèm bất kỳ markdown nào khác:
    [
      {{"word": "từ_1", "full_sentence": "câu hoàn chỉnh chứa từ_1"}}
    ]
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    json_match = re.search(r'\[.*]', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))

    clean_json = text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_json)


def get_or_generate_sentences(client, selected_words: list, cefr_level: str, CEFR_DICT=None, translator=None,
                              lemmatizer=None) -> list:
    results = {}
    words_to_generate = []

    for word in selected_words:
        record = db.get_word(word)

        if record and record.get("examples"):
            results[word] = {
                "word": word,
                "full_sentence": random.choice(record["examples"])
            }
        else:
            words_to_generate.append(word)

    if words_to_generate:
        generated = generate_sentences_with_gemini(client, words_to_generate)

        for g in generated:
            w_text = g["word"]
            results[w_text] = g

            # Lấy nghĩa và phiên âm IPA tự động từ vocabulary_module
            meaning, calc_cefr, phonetic_ipa = process_vocabulary_info(CEFR_DICT, translator, lemmatizer, w_text)

            # Ưu tiên lấy cefr_level truyền vào nếu calc_cefr bị N/A
            final_cefr = cefr_level if cefr_level and cefr_level != "N/A" else calc_cefr

            # Lưu từ vựng kèm meaning và ipa vào DB
            # LƯU Ý: Đảm bảo phương thức db.add_word (hoặc db.upsert_user_vocabulary) chấp nhận tham số meaning và phonetic/ipa
            word_id = db.add_word(
                word=w_text,
                cefr=final_cefr,
                meaning=meaning,
                phonetic=phonetic_ipa
            )

            db.add_example(word_id, g["full_sentence"])

    return [results[word] for word in selected_words if word in results]

def get_ceft_word(CEFR_PATH):
    df_cefr = pd.read_excel(CEFR_PATH, sheet_name="ALL")

    columns_lower = [str(c).strip().lower() for c in df_cefr.columns]
    word_col_index = columns_lower.index("headword") if "headword" in columns_lower else 0
    level_col_index = columns_lower.index("cefr") if "cefr" in columns_lower else 1

    real_word_col = df_cefr.columns[word_col_index]
    real_level_col = df_cefr.columns[level_col_index]

    cefr_dict = dict(zip(
        df_cefr[real_word_col].astype(str).str.strip().str.lower(),
        df_cefr[real_level_col].astype(str).str.strip()
    ))
    return cefr_dict

def process_start_practice(client, cefr_dict_sample, cefr_level, translator, lemmatizer):
    selected_words = get_random_words_by_level(cefr_dict_sample, cefr_level, count=2)
    print(selected_words)
    if not selected_words:
        return (
            gr.update(visible=True), gr.update(visible=False),
            [], 0, 0, f"Không tìm thấy từ nào thuộc cấp độ {cefr_level}!",
            "", "", None, "", "", gr.update(interactive=False)
        )

    try:
        # Truyền thêm cefr_dict_sample, translator, lemmatizer vào hàm
        generated_data = get_or_generate_sentences(
            client,
            selected_words,
            cefr_level,
            CEFR_DICT=cefr_dict_sample,
            translator=translator,
            lemmatizer=lemmatizer
        )
    except Exception as e:
        print(f"⚠️ Lỗi gọi API sinh câu ({e}). Fallback sang từ có sẵn trong DB cùng CEFR level {cefr_level}.")
        fallback_words = get_random_words_with_examples_by_level(cefr_level, count=len(selected_words))

        if not fallback_words:
            return (
                gr.update(visible=True), gr.update(visible=False),
                [], 0, 0,
                (f"⚠️ Không gọi được dịch vụ tạo câu ví dụ, và hiện chưa có từ nào "
                 f"cấp độ {cefr_level} sẵn có trong dữ liệu để luyện tạm. Vui lòng thử lại sau!"),
                "", "", None, "", "", gr.update(interactive=True)
            )

        try:
            # Các từ fallback đã được lọc là đã có sẵn example trong DB, nên bước
            # này sẽ không cần gọi API sinh câu mới cho bất kỳ từ nào trong đó.
            generated_data = get_or_generate_sentences(
                client,
                fallback_words,
                cefr_level,
                CEFR_DICT=cefr_dict_sample,
                translator=translator,
                lemmatizer=lemmatizer
            )
            selected_words = fallback_words
        except Exception as e2:
            return (
                gr.update(visible=True), gr.update(visible=False),
                [], 0, 0, f"⚠️ Lỗi hệ thống: {str(e2)}",
                "", "", None, "", "", gr.update(interactive=True)
            )

    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parsed_questions = []

    for idx, item in enumerate(generated_data):
        target_word = item["word"]
        full_sentence = item["full_sentence"]
        blank_sentence = re.sub(rf"\b{re.escape(target_word)}\b", "______", full_sentence)
        audio_file = str(AUDIO_CACHE_DIR / f"q_{idx}.mp3")
        generate_audio(full_sentence, audio_file)

        parsed_questions.append({
            "word": target_word,
            "display_sentence": blank_sentence,
            "audio": audio_file
        })

    first_q = parsed_questions[0]
    total = len(parsed_questions)

    return (
        gr.update(visible=False),      # 1. setup_panel
        gr.update(visible=True),       # 2. practice_panel
        parsed_questions,              # 3. state_words_data
        0,                             # 4. state_current_index
        0,                             # 5. state_score
        "",                            # 6. status_output
        f"Câu 1/{total}",              # 7. progress_tracker
        first_q["display_sentence"],   # 8. sentence_display
        first_q["audio"],              # 9. audio_player
        "",                            # 10. user_answer
        "",                            # 11. feedback_display
        gr.update(interactive=True),   # 12. btn_check
    )


def check_answer(user_input, current_index, words_data, score):
    current_word = words_data[current_index]["word"].strip().lower()
    user_word = (user_input or "").strip().lower()

    if user_word == current_word:
        result_msg = "✅ **Chính xác!**"
        new_score = score + 1
    else:
        result_msg = f"❌ **Chưa đúng.** Đáp án đúng là: **{current_word}**"
        new_score = score

    is_last = (current_index == len(words_data) - 1)
    next_label = "🎉 Hoàn thành" if is_last else "Câu tiếp theo ➡️"

    return (
        new_score,
        result_msg,
        gr.update(interactive=False),
        gr.update(visible=True),
        gr.update(value=next_label),
    )


def next_question(current_index, words_data, score):
    next_idx = current_index + 1
    total = len(words_data)
    print(f"[DEBUG] current_index={current_index}, next_idx={next_idx}, total={total}")
    if next_idx < total:
        next_q = words_data[next_idx]
        return (
            next_idx,
            f"Câu {next_idx + 1}/{total}",
            next_q["display_sentence"],
            next_q["audio"],
            "",
            "",
            gr.update(interactive=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            ""
        )
    else:
        final_msg = f"🎉 **Hoàn thành bài luyện nghe!**\n\nKết quả: **{score}/{total}** câu đúng."
        return (
            next_idx,
            "",
            "",
            gr.update(value=None),
            "",
            "",
            gr.update(interactive=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            final_msg
        )

def reset_to_start():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def clear_transcript(AUDIO_PATH, transcript):
    print(transcript)
    print()
    raw_name = Path(AUDIO_PATH).stem
    formatted_name = raw_name.replace("_", " ").lower()
    label_audio = formatted_name.title()

    transcript = re.sub(r"dot\s+com", ".com", transcript, flags=re.IGNORECASE)

    match = re.search(r'([^.!?]*\.com[^.!?]*[.!?])', transcript)

    clean_transcript = transcript  # fallback
    if match:
        com_index_in_transcript = transcript.find('.com', match.start())
        after_com = com_index_in_transcript + 4
        next_dot_index = transcript.find('.', after_com)

        if next_dot_index != -1:
            clean_transcript = transcript[next_dot_index + 1:].strip()
        else:
            clean_transcript = transcript[after_com:].strip()

    print(clean_transcript)
    print()

    clean_transcript = clean_transcript[len(formatted_name) + 1:].strip()
    print(clean_transcript)

    return clean_transcript, label_audio

def extract_important_token(nlp, transcript, max_blanks=30):
    if not transcript.strip():
        return "", []

    doc = nlp(transcript)
    selected_tokens = []
    important_tokens = []
    used_lemmas = set()

    for token in doc:
        if len(selected_tokens) >= max_blanks:
            break

        if "'" in token.text or "’" in token.text:
            continue

        if token.pos_ not in {"NOUN", "VERB", "ADJ", "PROPN"}:
            continue

        if token.is_stop or not token.is_alpha:
            continue

        lemma = token.lemma_.lower()
        if lemma in used_lemmas:
            continue

        if selected_tokens and (token.idx - (selected_tokens[-1].idx + len(selected_tokens[-1].text)) < 3):
            continue

        used_lemmas.add(lemma)
        selected_tokens.append(token)
        important_tokens.append(token.text)

    output = []
    last = 0
    for idx, token in enumerate(selected_tokens, start=1):
        output.append(transcript[last:token.idx])
        output.append(f" ({idx})＿＿＿ ")
        last = token.idx + len(token.text)

    output.append(transcript[last:])
    blanked_text = "".join(output)

    return blanked_text, important_tokens

PAGE_SIZE = 4

def transcribe_short_audio_text(whisper, nlp, AUDIO_PATH_FOLDER, mp3_name):
    AUDIO_PATH = get_path_audio(AUDIO_PATH_FOLDER, mp3_name)

    # Sử dụng db.get_mp3_transcript từ DatabaseManager
    cached_transcript = db.get_mp3_transcript(mp3_name)

    if cached_transcript:
        raw_text = cached_transcript
    else:
        gr.Info("Please wait a bit for listening data")
        result = whisper.transcribe(AUDIO_PATH, fp16=False)
        raw_text = result["text"].strip()
        db.add_mp3_transcript(mp3_name, raw_text)

    transcript, label_audio = clear_transcript(AUDIO_PATH, raw_text)
    blank_text, answers = extract_important_token(nlp, transcript)

    user_answers = [""] * len(answers)

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        blank_text,
        answers,
        user_answers,
        0,
        *show_page(0, answers, user_answers),
        gr.update(value=AUDIO_PATH, label=f"{label_audio}")
    )

def show_page(page, answers, user_answers):
    start = page * PAGE_SIZE
    updates = []

    for i in range(PAGE_SIZE):
        idx = start + i
        if idx < len(answers):
            updates.append(
                gr.update(
                    visible=True,
                    label=f"({idx+1})",
                    value=user_answers[idx]
                )
            )
        else:
            updates.append(
                gr.update(
                    visible=False,
                    value=""
                )
            )

    return updates

def save_current_page(
    page,
    user_answers,
    box1,
    box2,
    box3,
    box4
):
    user_answers = user_answers.copy()
    start = page * PAGE_SIZE
    values = [box1, box2, box3, box4]

    for i, value in enumerate(values):
        idx = start + i
        if idx < len(user_answers):
            user_answers[idx] = value

    return user_answers

def next_page(
    page,
    answers,
    user_answers,
    box1,
    box2,
    box3,
    box4
):
    user_answers = save_current_page(
        page,
        user_answers,
        box1,
        box2,
        box3,
        box4
    )

    max_page = (len(answers)-1)//PAGE_SIZE
    page = min(page+1, max_page)

    return (
        page,
        user_answers,
        *show_page(page, answers, user_answers)
    )

def previous_page(
    page,
    answers,
    user_answers,
    box1,
    box2,
    box3,
    box4
):
    user_answers = save_current_page(
        page,
        user_answers,
        box1,
        box2,
        box3,
        box4
    )

    page = max(page-1, 0)

    return (
        page,
        user_answers,
        *show_page(page, answers, user_answers)
    )

def check_answer_listen_paragraph(
    correct_answers,
    user_answers,
    page,
    box1,
    box2,
    box3,
    box4
):
    user_answers = save_current_page(
        page,
        user_answers,
        box1,
        box2,
        box3,
        box4
    )

    score = 0

    for gt, user in zip(correct_answers, user_answers):
        if gt.lower().strip() == user.lower().strip():
            score += 1

    return (
        f"{score}/{len(correct_answers)}",
        user_answers
    )

def get_mp3_filename(folder_path):
    if not os.path.exists(folder_path):
        return []
    mp3_paths = glob.glob(os.path.join(folder_path, "*.mp3"))
    mp3_names = [os.path.basename(p) for p in mp3_paths]

    return mp3_names

def load_value_ratio_audio(folder_path):
    mp3_names = get_mp3_filename(folder_path)
    return gr.update(choices = mp3_names, value = mp3_names[1])

def get_path_audio(folder_path, mp3_name):
    mp3_path = os.path.join(folder_path, mp3_name)
    return mp3_path