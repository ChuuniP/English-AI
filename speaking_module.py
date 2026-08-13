import io
import re
import numpy as np
import torch
import soundfile as sf
from scipy import signal
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import gradio as gr
from pydub import AudioSegment
import difflib
import tempfile
from gtts import gTTS
import eng_to_ipa as ipa
import librosa

_IPA_AVAILABLE = True

def get_phonetic_ipa(word):

    if not word or not str(word).strip():
        return ""

    clean_word = str(word).strip().strip(".,!?\"'()[]{}*:;").lower()
    if not clean_word:
        return ""

    if not _IPA_AVAILABLE:
        return ""

    try:
        result = ipa.convert(clean_word)
        if not result or result.strip("*") == clean_word:
            return ""
        return result.strip()
    except Exception:
        return ""

# ==========================================================================
# TAB "PRACTICE" (Speaking - hội thoại tự do theo chủ đề, giọng nói 2 chiều)
# ==========================================================================

TOPICS = [
    {"id": "travel", "title": "Du lịch", "sub": "Kể về chuyến đi, xin gợi ý điểm đến"},
    {"id": "work", "title": "Công việc", "sub": "Nói về công việc, đồng nghiệp, deadline"},
    {"id": "hobby", "title": "Sở thích", "sub": "Chia sẻ đam mê, cuối tuần làm gì"},
    {"id": "interview", "title": "Phỏng vấn", "sub": "Luyện trả lời câu hỏi phỏng vấn xin việc"},
    {"id": "news", "title": "Tin tức", "sub": "Bàn luận một chủ đề thời sự nhẹ nhàng"},
    {"id": "debate", "title": "Tranh luận", "sub": "Bảo vệ quan điểm, phản biện AI"},
]
TOPIC_TITLES = {t["id"]: t["title"] for t in TOPICS}

LEVELS = [
    {"id": "beginner", "label": "Mới bắt đầu"},
    {"id": "intermediate", "label": "Trung cấp"},
    {"id": "advanced", "label": "Nâng cao"},
]
LEVEL_NOTES = {
    "beginner": "Use short, simple sentences and common words. Be patient and encouraging.",
    "intermediate": "Use natural, everyday English with moderate vocabulary.",
    "advanced": "Use rich vocabulary and idiomatic, native-level phrasing.",
}

ORB_LABELS = {
    "idle": "Bấm giữ micro để nói",
    "listening": "Đang nghe...",
    "thinking": "AI đang soạn câu trả lời...",
    "speaking": "AI đang nói...",
}

# CSS cho toàn bộ tab "Practice" - nhúng 1 lần duy nhất bằng gr.HTML ở đầu tab.
# Tách riêng khỏi style_demo.css để không ảnh hưởng các tab khác, chỉ áp dụng
# bên trong #practice-tab-wrap.
PRACTICE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500&family=Inter:wght@400;500;600&display=swap');

#practice-tab-wrap {
  background: #FFFFFF;
  border-radius: 16px;
  padding: 20px;
  border: 1px solid #E5E7EB;
  color: #1F2933;
  font-family: 'Inter', sans-serif;
}
#practice-tab-wrap h1, #practice-tab-wrap h2, #practice-tab-wrap h3,
#practice-tab-wrap .sp-title { font-family: 'Fraunces', serif; font-weight: 500; color: #111827; }

#practice-tab-wrap .sp-topic-card {
  background: #F8F9FB !important;
  border: 1px solid #E2E5EA !important;
  color: #1F2933 !important;
  text-align: left !important;
  border-radius: 12px !important;
}
#practice-tab-wrap .sp-topic-card-active {
  background: rgba(232,163,61,0.14) !important;
  border-color: #E8A33D !important;
  color: #7A4A0A !important;
}
#practice-tab-wrap .sp-topic-grid { display: grid !important; grid-template-columns: 1fr 1fr; gap: 10px; }

#practice-tab-wrap #sp-level-radio label { color: #1F2933 !important; }

#practice-tab-wrap #sp-start-btn {
  background: #E8A33D !important;
  color: #2C1B04 !important;
  border: none !important;
  font-weight: 600 !important;
}

#practice-tab-wrap .sp-orb-wrap { text-align: center; padding: 10px 0 4px; }
#practice-tab-wrap .sp-orb {
  position: relative;
  width: 96px; height: 96px;
  margin: 0 auto;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  background: #EEF1F5;
  transition: background 0.3s ease;
}
#practice-tab-wrap .sp-orb-listening { background: #4FA791; box-shadow: 0 0 0 6px rgba(79,167,145,0.16); }
#practice-tab-wrap .sp-orb-speaking { background: #E8A33D; box-shadow: 0 0 0 6px rgba(232,163,61,0.16); }
#practice-tab-wrap .sp-orb-thinking { background: #EEF1F5; }
#practice-tab-wrap .sp-orb-core { display: flex; gap: 4px; align-items: center; height: 26px; }
#practice-tab-wrap .sp-bar {
  width: 4px; height: 14px; border-radius: 2px; background: #9AA5B1; display: inline-block;
}
#practice-tab-wrap .sp-orb-listening .sp-bar,
#practice-tab-wrap .sp-orb-speaking .sp-bar { background: #FFFFFF; animation: sp-bar 0.9s ease-in-out infinite; }
#practice-tab-wrap .sp-orb-thinking .sp-bar { background: #6B7683; animation: sp-think 1s ease-in-out infinite; }
#practice-tab-wrap .sp-bar:nth-child(2) { animation-delay: 0.12s; }
#practice-tab-wrap .sp-bar:nth-child(3) { animation-delay: 0.24s; }
#practice-tab-wrap .sp-bar:nth-child(4) { animation-delay: 0.36s; }
#practice-tab-wrap .sp-bar:nth-child(5) { animation-delay: 0.48s; }
@keyframes sp-bar { 0%, 100% { height: 8px; } 50% { height: 22px; } }
@keyframes sp-think { 0%, 100% { height: 6px; opacity: 0.4; } 50% { height: 12px; opacity: 1; } }
#practice-tab-wrap .sp-orb-caption { font-size: 13px; color: #6B7683; margin-top: 8px; }

#practice-chatbot { background: #FFFFFF !important; border: 1px solid #E5E7EB !important; border-radius: 12px !important; }
#practice-chatbot .message.user, #practice-chatbot [data-testid="user"] {
  background: rgba(79,167,145,0.12) !important; border: 1px solid rgba(79,167,145,0.35) !important; color: #14352C !important;
}
#practice-chatbot .message.bot, #practice-chatbot [data-testid="bot"] {
  background: #F3F4F6 !important; border: 1px solid #E5E7EB !important; color: #1F2933 !important;
}

#practice-tab-wrap .sp-chip { border-radius: 10px; padding: 8px 12px; font-size: 12px; line-height: 1.5; margin-top: 6px; }
#practice-tab-wrap .sp-chip-fix { background: rgba(226,102,92,0.10); border: 1px solid rgba(226,102,92,0.35); color: #7A241C; }
#practice-tab-wrap .sp-chip-ok { background: rgba(79,167,145,0.10); border: 1px solid rgba(79,167,145,0.35); color: #14352C; }

#practice-tab-wrap #sp-mic-audio { background: #F8F9FB !important; border: 1px solid #E5E7EB !important; border-radius: 12px !important; }
#practice-tab-wrap #sp-end-btn { border-color: #D1D5DB !important; color: #6B7683 !important; }
#practice-tab-wrap #sp-ai-audio { display: none !important; }
"""


def render_orb_html(state="idle"):
    """Trả về HTML cho khối 'voice orb' ở giữa màn hình hội thoại.
    Style/keyframes dùng chung được nhúng 1 lần ở đầu tab (xem PRACTICE_CSS),
    hàm này chỉ trả về phần markup đổi theo state."""
    state = state if state in ORB_LABELS else "idle"
    return f"""
<div class="sp-orb-wrap">
  <div class="sp-orb sp-orb-{state}">
    <div class="sp-orb-core">
      <span class="sp-bar"></span><span class="sp-bar"></span><span class="sp-bar"></span>
      <span class="sp-bar"></span><span class="sp-bar"></span>
    </div>
  </div>
  <p class="sp-orb-caption">{ORB_LABELS[state]}</p>
</div>
"""


def render_correction_html(correction_text):
    """Trả về HTML cho ô 'sửa lỗi' hiển thị dưới bong bóng chat của AI."""
    text = (correction_text or "").strip()
    if not text:
        return ""
    if text.lower().startswith("no notable mistakes") or text.lower().startswith("không có lỗi"):
        return f'<div class="sp-chip sp-chip-ok">✓ {text}</div>'
    return f'<div class="sp-chip sp-chip-fix">{text}</div>'


def select_topic(topic_id):
    """Đánh dấu topic vừa bấm là active, bỏ active các topic khác."""
    updates = [gr.update(elem_classes="sp-topic-card sp-topic-card-active" if t["id"] == topic_id
                          else "sp-topic-card") for t in TOPICS]
    return (topic_id, *updates)


def select_custom_topic():
    """Khi người dùng gõ chủ đề riêng, bỏ active hết các topic card có sẵn."""
    updates = [gr.update(elem_classes="sp-topic-card") for _ in TOPICS]
    return ("custom", *updates)


def resolve_topic_title(topic_id, custom_topic):
    if topic_id == "custom":
        return (custom_topic or "").strip()
    return TOPIC_TITLES.get(topic_id, "")


def parse_practice_sections(text):
    """Tách phần [1. CONVERSATION] và [2. CORRECTION] từ output của Gemini."""
    match_chat = re.search(r"\[1\.\s*CONVERSATION\](.*?)(?=\[2\.|\Z)", text, re.DOTALL | re.IGNORECASE)
    match_corr = re.search(r"\[2\.\s*CORRECTION\](.*?)(?=\Z)", text, re.DOTALL | re.IGNORECASE)

    if match_chat:
        reply = match_chat.group(1).strip()
    else:
        # Fallback: không match được header [1. CONVERSATION] (Gemini trả về
        # sai định dạng). Để tránh đọc nhầm phần [2. CORRECTION] (có tiếng
        # Việt) qua TTS tiếng Anh, cắt bỏ mọi thứ từ "[2." trở đi trước khi
        # dùng làm câu thoại.
        reply = re.split(r"\[2\.", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        # Nếu vẫn còn sót header "[1. ...]" ở đầu thì bỏ luôn cho sạch.
        reply = re.sub(r"^\[1\.\s*CONVERSATION\]\s*", "", reply, flags=re.IGNORECASE).strip()

    correction = match_corr.group(1).strip() if match_corr else ""
    return reply, correction


def build_practice_prompt(topic_title, level, history_str, user_text=None):
    level_note = LEVEL_NOTES.get(level, LEVEL_NOTES["intermediate"])
    turn_instruction = (
        f'The user just said: "{user_text}"'
        if user_text
        else "This is the very first message. Greet the learner briefly and open the conversation on the topic below."
    )

    return f"""
You are a warm, patient English conversation partner helping a Vietnamese learner practice speaking on
the topic "{topic_title}". {level_note}

--- CHAT HISTORY ---
{history_str}

{turn_instruction}

--- SYSTEM INSTRUCTIONS ---
Structure your response into EXACTLY 2 sections, labeled exactly as follows:

[1. CONVERSATION]
(Your natural reply, 1-3 sentences, always ending with a follow-up question to keep the learner talking.
English only.)

[2. CORRECTION]
(If the user's last message had a notable grammar or word-choice mistake, write it in this exact shape:
Original: "..." -> Better: "..." followed by one short reason, in English.
If there is no user message yet, or the sentence was already fine, write exactly: No notable mistakes.
Everything you write, in both sections, must be in English only — do not use any Vietnamese.)
"""


def generate_practice_response_stream(client, topic_title, level, chat_session, user_text=None):
    history_str = chat_session.get_history_as_string()
    prompt = build_practice_prompt(topic_title, level, history_str, user_text)

    response_stream = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    full_text = ""
    for chunk in response_stream:
        if chunk.text:
            full_text += chunk.text
            yield full_text

    if user_text:
        chat_session.add_message("User", user_text)
    reply, _ = parse_practice_sections(full_text)
    chat_session.add_message("Assistant", reply)


def start_practice_topic(topic_id, custom_topic, level, chat_session, client):
    topic_title = resolve_topic_title(topic_id, custom_topic)

    if not topic_title:
        return (
            gr.update(), gr.update(), [], "", "",
            None, render_orb_html("idle"), chat_session,
            "⚠️ Chọn một chủ đề trước đã nhé.",
        )

    new_session = chat_session.__class__()

    if not client:
        greeting = (
            f"Let's talk about {topic_title}. "
            "(⚠️ Chưa cấu hình GEMINI_API_KEY nên AI chưa thể trò chuyện thật.)"
        )
        new_session.add_message("Assistant", greeting)
        return (
            gr.update(visible=False), gr.update(visible=True),
            [{"role": "assistant", "content": greeting}],
            topic_title, topic_title, None, render_orb_html("idle"), new_session, "",
        )

    full_raw = ""
    for partial in generate_practice_response_stream(client, topic_title, level, new_session, user_text=None):
        full_raw = partial
    reply, _ = parse_practice_sections(full_raw) if full_raw else (
        f"Let's talk about {topic_title}. What comes to your mind first?", "")

    audio_path = speak_text(reply)

    return (
        gr.update(visible=False), gr.update(visible=True),
        [{"role": "assistant", "content": reply}],
        topic_title, topic_title, audio_path, render_orb_html("speaking"), new_session, "",
    )


def end_practice_session():
    return (
        gr.update(visible=True), gr.update(visible=False), [], "", "",
        None, render_orb_html("idle"),
    )


def process_practice_voice_stream(audio_path, history, chat_session, topic_title, level, client, model_whisper):
    """Xử lý 1 lượt nói của người dùng trong hội thoại luyện nói theo chủ đề.

    Trả về đúng thứ tự output cho: practice_chatbot, correction_html,
    ai_audio_out, orb_html, chat_session_state, audio_input (reset)
    """
    if history is None:
        history = []

    if not audio_path:
        yield history, "", None, render_orb_html("idle"), chat_session, None
        return

    if not client:
        yield history, render_correction_html("⚠️ Vui lòng cấu hình GEMINI_API_KEY."), None, render_orb_html("idle"), chat_session, None
        return

    try:
        with open(audio_path, "rb") as f:
            file_content = f.read()
        user_text = extract_text(file_content, model_whisper)
    except Exception as e:
        yield history, render_correction_html(f"⚠️ Lỗi nhận diện giọng nói: {e}"), None, render_orb_html("idle"), chat_session, None
        return

    if not user_text or not user_text.strip():
        yield history, render_correction_html("⚠️ Không nghe rõ, bạn thử nói lại nhé."), None, render_orb_html("idle"), chat_session, None
        return

    updated_history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": "..."},
    ]
    yield updated_history, "", None, render_orb_html("thinking"), chat_session, None

    full_raw = ""
    try:
        for partial in generate_practice_response_stream(client, topic_title, level, chat_session, user_text):
            full_raw = partial
            reply, correction = parse_practice_sections(partial)
            updated_history[-1] = {"role": "assistant", "content": reply}
            yield updated_history, render_correction_html(correction), None, render_orb_html("thinking"), chat_session, None
    except Exception as e:
        yield updated_history, render_correction_html(f"⚠️ Lỗi khi gọi AI: {e}"), None, render_orb_html("idle"), chat_session, None
        return

    reply, correction = parse_practice_sections(full_raw)
    updated_history[-1] = {"role": "assistant", "content": reply}
    audio_reply_path = speak_text(reply)

    yield updated_history, render_correction_html(correction), audio_reply_path, render_orb_html("speaking"), chat_session, None

def rag_with_faiss(model_name, storage_database):
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    vector_db = FAISS.from_texts(texts=storage_database, embedding=embedding)
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})

    return retriever

def extract_data_wav(wav_content: bytes):
    if not wav_content:
        return None, None
    try:
        audio_bytes = io.BytesIO(wav_content)
        try:
            data, samplerate = sf.read(audio_bytes)
        except Exception:
            # Trình duyệt ghi âm ra webm/opus, soundfile không đọc trực tiếp được
            # -> dùng pydub (ffmpeg) chuyển sang wav trong bộ nhớ rồi đọc lại.
            audio_bytes.seek(0)
            audio_segment = AudioSegment.from_file(audio_bytes)
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            data, samplerate = sf.read(wav_buffer)
    except Exception as e:
        print(f"Error reading wav data: {e}")
        return None, None

    audio_data = data.astype(np.float32)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    target_sr = 16000
    if samplerate != target_sr:
        num_samples = int(len(audio_data) * target_sr / samplerate)
        audio_data = signal.resample(audio_data, num_samples)
    return audio_data, target_sr


def extract_text(wav_content: bytes, model):
    waveform, _ = extract_data_wav(wav_content)
    if waveform is None:
        return "Không thể xử lý file âm thanh."
    try:
        result = model.transcribe(waveform, language="en", fp16=False)
        return result.get("text", "").strip()
    except Exception:
        return "Lỗi trong quá trình nhận diện giọng nói."


def extract_phonemes(wav_content: bytes, wav2vec2_processor, wav2vec2_model, device):
    waveform, sample_rate = extract_data_wav(wav_content)
    if waveform is None or sample_rate is None:
        return "Không thể trích xuất phonemes do file âm thanh không hợp lệ."

    input_audio = waveform.squeeze()
    inputs = wav2vec2_processor(input_audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = wav2vec2_model(input_values=input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_phonemes = wav2vec2_processor.batch_decode(predicted_ids, clean_up_tokenization_spaces=False)[0]
    return predicted_phonemes


def process_random_speaking_words(db_manager):
    """
    Lấy ngẫu nhiên tối đa 5 từ và khởi tạo state ban đầu.
    """
    words_data = db_manager.get_random_words_for_speaking(limit=5)

    if not words_data:
        return (
            "### ⚠️ Chưa có từ vựng nào trong CSDL!", # content_word_ipa_panel
            gr.update(visible=True),                   # ipa_word_speaking_panel
            [],                                        # state_words_data
            0,                                         # state_current_index
            gr.update(interactive=False),             # pre_ipa_speaking_btn
            gr.update(interactive=False)              # next_ipa_speaking_btn
        )

    # Lấy thông tin từ đầu tiên (index 0)
    word_obj = words_data[0]
    total = len(words_data)
    md_content = render_single_word_md(word_obj, 0, total)

    return (
        md_content,
        gr.update(visible=True),
        words_data,
        0,                                 # Bắt đầu từ vị trí 0
        gr.update(interactive=False),     # Previous luôn khoá ở từ đầu tiên
        gr.update(interactive=total > 1)  # Next mở nếu còn từ tiếp theo
    )

def extract_word(text):
    if not text:
        return ""
    match = re.search(r'<h2[^>]*>(.*?)</h2>', text, re.IGNORECASE)
    if match:
        clean_text = match.group(1).strip()
    else:
        clean_text = re.sub(r'<[^>]+>', '', text).strip()
    if not clean_text:
        return ""
    return clean_text

def speak_text(text):

    clean_text = extract_word(text)

    try:
        tts = gTTS(text=clean_text, lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        print(f"Lỗi TTS: {e}")
        return None

def render_single_word_md(word_obj, index, total):
    """Hàm phụ trợ render Markdown cho 1 từ kèm bộ đếm vị trí."""
    word = word_obj.get("word", "")
    phonetic = word_obj.get("phonetic") or "N/A"
    meaning = word_obj.get("meaning") or ""

    return f"""
<div style="text-align: center; padding: 20px; border: 1px solid #e5e7eb; border-radius: 10px;">
    <span style="font-size: 14px; color: #6b7280; font-weight: bold;">[ Từ {index + 1} / {total} ]</span>
    <h2 style="font-size: 32px; margin: 10px 0; color: #111827;">{word}</h2>
    <p style="font-size: 18px; color: #2563eb; font-style: italic;">/{phonetic}/</p>
    <p style="font-size: 16px; color: #4b5563;">{meaning}</p>
</div>
"""

def recognize_phonemes(audio_file, wav2vec2_model, wav2vec2_processor, device):
    """Nhận diện phoneme từ 1 file audio trên đĩa.

    Trước đây dùng sf.read(audio_file) trực tiếp -> lỗi "Format not recognised"
    với file .webm/.opus do trình duyệt ghi âm ra (libsndfile không đọc được).
    Giờ tái sử dụng extract_data_wav() (đã có sẵn fallback qua pydub/ffmpeg và
    tự resample về 16kHz), giống cách extract_text()/extract_phonemes() đang làm.
    """
    with open(audio_file, "rb") as f:
        file_content = f.read()

    speech, sr = extract_data_wav(file_content)
    if speech is None:
        return ""

    if speech.ndim > 1:
        speech = speech.mean(axis=1)

    inputs = wav2vec2_processor(speech, sampling_rate=sr, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = wav2vec2_model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_phonemes = wav2vec2_processor.batch_decode(predicted_ids)[0]

    return predicted_phonemes.strip()

def handle_record_action(wav2vec2_processor, wav2vec2_model, device, audio_path, current_index, words_data, text):
    if not words_data:
        return gr.update(), None

    total = len(words_data)
    current_word_obj = words_data[current_index]

    if not audio_path:
        return gr.update(), None

    # audio_path = r"E:\Antigravity\English AI\datasets\mp3\1.wav"  # (đang hard-code để test)

    try:
        if wav2vec2_processor and wav2vec2_model:
            user_phonemes = recognize_phonemes(
                audio_path, wav2vec2_model, wav2vec2_processor, device
            )
        else:
            user_phonemes = ""

        clean_word = extract_word(text)
        audio_word = speak_text(clean_word)
        word_ipa = recognize_phonemes(
                audio_word, wav2vec2_model, wav2vec2_processor, device
            )

        target_word = current_word_obj.get("word", "")
        target_ipa = current_word_obj.get("phonetic", "") or ""

        print(f"Từ cần đọc: {target_word} | IPA: /{word_ipa}/ | User Phonemes: {user_phonemes}")

        comparison_html, accuracy = compare_phonemes(word_ipa, user_phonemes)
        new_content = render_pronunciation_feedback_md(
            current_word_obj, current_index, total, comparison_html, accuracy
        )

        return new_content, None

    except Exception as e:
        print(f"Lỗi khi xử lý file audio: {e}")
        return gr.update(), None


def navigate_word(direction, current_index, words_data):
    """
    Xử lý khi nhấn Previous (-1) hoặc Next (+1).
    Next/Previous chỉ khoá khi đang ở từ đầu tiên / cuối cùng.
    """
    total = len(words_data)
    new_index = current_index + direction

    # Đảm bảo index nằm trong vùng an toàn
    if new_index < 0 or new_index >= total:
        new_index = current_index

    word_obj = words_data[new_index]
    md_content = render_single_word_md(word_obj, new_index, total)

    return (
        md_content,
        new_index,
        gr.update(interactive=new_index > 0),
        gr.update(interactive=new_index < total - 1)
    )


def normalize_text_for_compare(text):
    """Chuẩn hoá câu gốc (chữ hoa/thường, có dấu câu) về cùng định dạng với
    output của wav2vec2 (toàn chữ HOA, không dấu câu) để so sánh công bằng."""
    if not text:
        return ""
    text = text.upper()
    text = re.sub(r"[^A-Z0-9' ]", " ", text)
    return " ".join(text.split())


def compare_phonemes(reference, hypothesis):
    """So sánh chuỗi tham chiếu (câu/phoneme chuẩn) với chuỗi người dùng đọc.

    So khớp THEO TỪNG TỪ (tách bằng khoảng trắng) thay vì từng ký tự, để khi
    hiển thị các từ tách rời nhau như câu bình thường thay vì dính liền,
    đồng thời tránh việc lệch 1 ký tự làm sai lệch toàn bộ phần còn lại.

    Khác với bản gốc (dùng cho notebook, gọi display(HTML(...)) trực tiếp),
    bản này CHỈ trả về (comparison_html, accuracy) để nơi gọi tự ghép vào
    nội dung Markdown/HTML hiển thị trên Gradio.
    """
    ref_words = (reference or "").split()
    hyp_words = (hypothesis or "").split()

    if not ref_words:
        return "", 0.0

    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    html_parts = []
    correct_count = 0
    total_count = len(ref_words)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_segment = " ".join(ref_words[i1:i2])
        if tag == "equal":
            correct_count += (i2 - i1)
            html_parts.append(
                f'<span style="color:#16a34a;">{ref_segment}</span>'
            )
        else:
            html_parts.append(
                f'<span style="color:#dc2626; text-decoration: underline wavy #dc2626;">{ref_segment or "∅"}</span>'
            )

    accuracy = round(correct_count / total_count * 100, 1) if total_count else 0.0
    comparison_html = (
            '<div style="font-family:monospace; font-size:15px; letter-spacing:0.5px;">'
            + " ".join(html_parts)
            + "</div>"
    )
    return comparison_html, accuracy


def render_pronunciation_feedback_md(word_obj, index, total, comparison_html, accuracy):
    """Ghép phần hiển thị từ hiện tại (giống render_single_word_md) với kết quả
    so khớp phát âm ngay bên dưới — xanh = phát âm khớp chuẩn, đỏ = lệch/thiếu."""
    word = word_obj.get("word", "")
    phonetic = word_obj.get("phonetic") or "N/A"
    meaning = word_obj.get("meaning") or ""

    return f"""
<div style="text-align: center; padding: 16px; border: 1px solid #e5e7eb; border-radius: 10px;">
    <span style="font-size: 12px; color: #6b7280; font-weight: bold;">[ Từ {index + 1} / {total} ]</span>
    <h2 style="font-size: 22px; margin: 6px 0; color: #111827;">{word}</h2>
    <p style="font-size: 14px; color: #2563eb; font-style: italic; margin: 2px 0;">/{phonetic}/</p>
    <p style="font-size: 13px; color: #4b5563; margin: 2px 0;">{meaning}</p>
    <hr style="margin: 10px 0; border-color: #e5e7eb;">
    <p style="font-size: 12px; color: #6b7280;">Độ khớp phát âm: <b>{accuracy}%</b></p>
    {comparison_html}
</div>
"""


def render_single_sentence_md(sentence_obj):
    """Render Markdown cho 1 câu ngẫu nhiên (không còn bộ đếm vị trí kiểu 'Câu 1/5')."""
    sentence = sentence_obj.get("sentence", "")
    meaning = sentence_obj.get("meaning") or ""

    return f"""
<div style="text-align: center; padding: 20px; border: 1px solid #e5e7eb; border-radius: 10px;">
    <h2 style="font-size: 24px; margin: 10px 0; color: #111827; line-height: 1.4;">{sentence}</h2>
    <p style="font-size: 15px; color: #4b5563;">{meaning}</p>
</div>
"""

def process_random_speaking_sentences(db_manager):
    """
    Lấy NGẪU NHIÊN ĐÚNG 1 câu từ CSDL (không lấy sẵn 5 câu rồi hiển thị lần lượt
    kiểu 'Câu 1/5' nữa), tạo audio mẫu (TTS) và khởi tạo state.
    Trả về đúng thứ tự output cho: sentence_text, sentence_sample_audio,
    state_sentences_data, state_current_sentence_index, sentence_feedback_md
    """
    sentences_data = db_manager.get_random_sentences_for_speaking(limit=1)

    if not sentences_data:
        empty_md = "### ⚠️ Chưa có câu nào trong CSDL!"
        return empty_md, None, [], 0, ""

    sentence_obj = sentences_data[0]
    md_content = render_single_sentence_md(sentence_obj)

    # Tạo audio mẫu bằng TTS cho câu vừa lấy
    sample_audio_path = speak_text(sentence_obj.get("sentence", ""))

    # state_sentences_data chỉ chứa 1 câu, current_index luôn = 0
    return md_content, sample_audio_path, sentences_data, 0, ""


def render_sentence_feedback_md(sentence_obj, comparison_html, accuracy):
    """Ghép hiển thị câu hiện tại + kết quả so khớp phát âm bên dưới,
    tương tự render_pronunciation_feedback_md nhưng dành cho câu."""
    sentence = sentence_obj.get("sentence", "")
    meaning = sentence_obj.get("meaning") or ""

    return f"""
<div style="text-align: center; padding: 16px; border: 1px solid #e5e7eb; border-radius: 10px;">
    <p style="font-size: 12px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: #9ca3af; margin: 0 0 8px 0;">Result</p>
    <h2 style="font-size: 20px; margin: 6px 0; color: #111827; line-height: 1.4;">{sentence}</h2>
    <p style="font-size: 13px; color: #4b5563; margin: 2px 0;">{meaning}</p>
    <hr style="margin: 10px 0; border-color: #e5e7eb;">
    <p style="font-size: 12px; color: #6b7280;">Độ khớp phát âm: <b>{accuracy}%</b></p>
    {comparison_html}
</div>
"""


def handle_sentence_record_action(wav2vec2_processor, wav2vec2_model, device,
                                   audio_path, current_index, sentences_data, sentence_html_text):
    if not sentences_data:
        return gr.update(), None

    if not audio_path:
        return gr.update(), None

    current_sentence_obj = sentences_data[current_index]
    # audio_path = r"E:\Antigravity\English AI\datasets\mp3\2.wav"

    try:
        if wav2vec2_processor and wav2vec2_model:
            user_phonemes = recognize_phonemes(
                audio_path, wav2vec2_model, wav2vec2_processor, device
            )
        else:
            user_phonemes = ""

        target_sentence = extract_word(sentence_html_text)
        target_phonemes = normalize_text_for_compare(target_sentence)

        print(f"Câu cần đọc: {target_sentence} | Target (chuẩn hoá): {target_phonemes} | User Phonemes: {user_phonemes}")

        comparison_html, accuracy = compare_phonemes(target_phonemes, user_phonemes)
        new_feedback_html = render_sentence_feedback_md(
            current_sentence_obj, comparison_html, accuracy
        )

        return new_feedback_html, None

    except Exception as e:
        print(f"Lỗi khi xử lý audio câu: {e}")
        return gr.update(), None


# ==========================================================================
# QUẢN LÝ TAB "1 MINUTE" (Speaking - nói tự do theo topic trong 1 phút)
# ==========================================================================
#
# Workflow 4 bước (khớp với các panel trong giao diện):
#   1. Khi vào tab            -> show_1minute_start_panel()
#      Chỉ hiện start_1minute_panel, ẩn hết các panel còn lại.
#   2. Nhấn "▶️ Bắt đầu"       -> start_1minute_practice()
#      Ẩn start_1minute_panel, hiện topic_1minute_panel + btn_1minute_panel,
#      đồng thời lấy sẵn 1 topic ngẫu nhiên để hiển thị.
#   3. Nhấn "🔀 Đổi câu ngẫu nhiên" -> change_1minute_topic()
#      Lấy topic mới, ẩn lại respond_1minute_panel (nếu đang hiện kết quả cũ).
#   4. Ghi âm xong (sentence_audio_recorder có file) -> handle_1minute_record_action()
#      Transcribe + phân tích -> hiện respond_1minute_panel với kết quả.
#
# LƯU Ý QUAN TRỌNG VỀ GIAO DIỆN:
#   Đoạn HTML nút record trong tab "1 minute" đang dùng
#   data-target="hidden-audio-recorder-sentence", trùng với elem_id của
#   gr.File dùng cho tab "Sentence". Cần đổi thành 1 elem_id riêng, ví dụ
#   "hidden-audio-recorder-1minute", để 2 tab không bị ghi đè lẫn nhau khi
#   cùng mở trong 1 phiên làm việc.
# ==========================================================================

def render_topic_1minute_md(topic_obj):
    """Render Markdown/HTML hiển thị 1 topic cho tab '1 minute'
    (không có bộ đếm vị trí kiểu 'Câu 1/5' vì mỗi lần chỉ lấy đúng 1 topic)."""
    topic = topic_obj.get("topic", "")
    level = topic_obj.get("category") or ""

    level_badge = (
        f'<span style="font-size:12px;color:#6b7280;font-weight:bold;">[{level}]</span>'
        if level else ""
    )

    return f"""
<div style="text-align: center; padding: 20px; border: 1px solid #e5e7eb; border-radius: 10px;">
    {level_badge}
    <h2 style="font-size: 24px; margin: 10px 0; color: #111827; line-height: 1.4;">{topic}</h2>
    <p style="font-size: 13px; color: #6b7280;">🎤 Hãy nói khoảng 1 phút về chủ đề trên</p>
</div>
"""


def process_random_speaking_topic(db_manager, category=None):
    """
    Lấy NGẪU NHIÊN đúng 1 topic từ CSDL (bảng 'topics').
    Trả về đúng thứ tự output cho: topic_1minute_text, state_1minute_topic
    """
    topic_obj = db_manager.get_random_topic_for_speaking(category=category)

    if not topic_obj:
        empty_md = "### ⚠️ Chưa có topic nào trong CSDL!"
        return empty_md, {}

    md_content = render_topic_1minute_md(topic_obj)
    return md_content, topic_obj


def show_1minute_start_panel():
    """
    Reset giao diện tab '1 minute' về trạng thái ban đầu: CHỈ hiện
    start_1minute_panel, ẩn topic/btn/respond panel.
    Gọi khi người dùng chuyển sang (select) tab '1 minute'.
    """
    return (
        gr.update(visible=True),   # start_1minute_panel
        gr.update(visible=False),  # topic_1minute_panel
        gr.update(visible=False),  # btn_1minute_panel
        gr.update(visible=False),  # respond_1minute_panel
    )


def start_1minute_practice(db_manager, category=None):
    md_content, topic_obj = process_random_speaking_topic(db_manager, category=category)

    return (
        gr.update(visible=False),  # start_1minute_panel
        gr.update(visible=True),   # topic_1minute_panel
        gr.update(visible=True),   # btn_1minute_panel
        gr.update(visible=False),  # respond_1minute_panel
        md_content,                 # topic_1minute_text
        topic_obj,                  # state_1minute_topic
    )


def change_1minute_topic(db_manager, category=None):
    """
    Xử lý khi nhấn nút "🔀 Đổi câu ngẫu nhiên": lấy topic mới và ẩn lại
    respond_1minute_panel vì kết quả cũ (nếu có) không còn khớp với topic mới.
    """
    md_content, topic_obj = process_random_speaking_topic(db_manager, category=category)
    return (
        md_content,                 # topic_1minute_text
        topic_obj,                  # state_1minute_topic
        gr.update(visible=False),   # respond_1minute_panel
    )


def generate_1minute_feedback(client, topic_text, transcript, phonemes_text=""):
    """
    Gọi AI (Gemini, không stream) để đánh giá bài nói tự do 1 phút của người
    dùng theo topic. Trả về 4 chuỗi Markdown: (summary_md, pron_md, gram_md, vocab_md)
    """
    if not client:
        return "⚠️ Vui lòng cấu hình GEMINI_API_KEY.", "", "", ""

    if not transcript or not transcript.strip():
        return "⚠️ Không nhận diện được nội dung bạn nói, vui lòng thử ghi âm lại.", "", "", ""

    prompt_template = f"""
    You are an expert English speaking examiner.
    The student was asked to speak freely for about 1 minute on the following topic:

    TOPIC: "{topic_text}"

    Below is the transcript of what the student actually said (converted from speech to text):
    TRANSCRIPT: "{transcript}"

    PHONEME DATA (may be noisy, only use if it looks meaningful): "{phonemes_text}"

    Evaluate the student's answer and structure your response into EXACTLY
    4 sections, labeled exactly as follows:

    [1. SUMMARY]
    (2-3 sentences: overall assessment, whether the answer stays on topic and
    is well-organized, plus an overall score out of 10.)

    [2. PRONUNCIATION]
    (Comment on likely pronunciation issues based on the phoneme data. If no
    phoneme data is usable, comment generally on words in the transcript that
    English learners commonly mispronounce.)

    [3. GRAMMAR]
    (Point out grammar mistakes found in the transcript and how to fix them.
    If there are no notable mistakes, say so briefly.)

    [4. VOCABULARY]
    (Comment on vocabulary range/appropriateness, and suggest 2-3 more
    natural or advanced words/phrases the student could use instead.)

    Write your response in Vietnamese, but keep the specific English
    words/phrases you are correcting or suggesting in English.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_template,
        )
        full_text = response.text or ""
    except Exception as e:
        print(f"Lỗi khi gọi AI đánh giá bài nói 1 phút: {e}")
        return f"⚠️ Lỗi khi gọi AI đánh giá: {e}", "", "", ""

    match_summary = re.search(r"\[1\.\s*SUMMARY\](.*?)(?=\[2\.|\Z)", full_text, re.DOTALL | re.IGNORECASE)
    match_pron = re.search(r"\[2\.\s*PRONUNCIATION\](.*?)(?=\[3\.|\Z)", full_text, re.DOTALL | re.IGNORECASE)
    match_gram = re.search(r"\[3\.\s*GRAMMAR\](.*?)(?=\[4\.|\Z)", full_text, re.DOTALL | re.IGNORECASE)
    match_vocab = re.search(r"\[4\.\s*VOCABULARY\](.*?)(?=\Z)", full_text, re.DOTALL | re.IGNORECASE)

    summary_md = match_summary.group(1).strip() if match_summary else full_text.strip()
    pron_md = match_pron.group(1).strip() if match_pron else ""
    gram_md = match_gram.group(1).strip() if match_gram else ""
    vocab_md = match_vocab.group(1).strip() if match_vocab else ""

    return summary_md, pron_md, gram_md, vocab_md


def handle_1minute_record_action(client, model_whisper, wav2vec2_processor, wav2vec2_model, device,
                                  audio_path, topic_obj):
    """
    Xử lý khi người dùng ghi âm xong bài nói 1 phút (sentence_audio_recorder
    nhận được file):
    - Nhận diện văn bản (Whisper) + phoneme (wav2vec2, nếu có) từ audio
    - Gọi AI đánh giá tổng quan / phát âm / ngữ pháp / từ vựng
    - Hiện respond_1minute_panel kèm kết quả, đồng thời reset lại
      sentence_audio_recorder để có thể ghi âm lượt tiếp theo.

    Trả về đúng thứ tự output cho:
    respond_1minute_panel, result_1minute_summary, transcript_1minute_out,
    pron_1minute_out, gram_1minute_out, vocab_1minute_out, sentence_audio_recorder
    """
    if not audio_path:
        return (
            gr.update(),  # giữ nguyên trạng thái respond_1minute_panel
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            None,
        )

    topic_text = (topic_obj or {}).get("topic", "")

    try:
        with open(audio_path, "rb") as f:
            file_content = f.read()

        transcript = extract_text(file_content, model_whisper)
        phonemes_text = extract_phonemes(
            file_content, wav2vec2_processor, wav2vec2_model, device
        ) if wav2vec2_processor and wav2vec2_model else ""

        summary_md, pron_md, gram_md, vocab_md = generate_1minute_feedback(
            client, topic_text, transcript, phonemes_text
        )

        return (
            gr.update(visible=True),   # respond_1minute_panel
            summary_md,                 # result_1minute_summary
            transcript,                 # transcript_1minute_out
            pron_md,                    # pron_1minute_out
            gram_md,                    # gram_1minute_out
            vocab_md,                   # vocab_1minute_out
            None,                       # reset sentence_audio_recorder
        )

    except Exception as e:
        print(f"Lỗi khi xử lý audio bài nói 1 phút: {e}")
        error_md = f"⚠️ Lỗi xử lý: {e}"
        return (
            gr.update(visible=True),
            error_md, "", "", "", "",
            None,
        )