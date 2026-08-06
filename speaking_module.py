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

def handle_voice_stream(audio, history, client, model_whisper, wav2vec2_processor,
                        wav2vec2_model, device, retriever, chat_session):
    yield from process_voice_gradio_stream(
        client,
        model_whisper,
        wav2vec2_processor,
        wav2vec2_model,
        device,
        retriever,
        chat_session,
        audio,
        history
    )

def process_voice_gradio_stream(client, model_whisper, wav2vec2_processor, wav2vec2_model, device, retriever,
                                chat_session, audio_path, current_chat_history):
    if current_chat_history is None:
        current_chat_history = []

    # 1. Kiểm tra audio
    if not audio_path:
        yield current_chat_history, "Vui lòng ghi âm trước.", "", None
        return

    # 2. Kiểm tra API Client
    if not client:
        yield current_chat_history, "Vui lòng cấu hình GEMINI_API_KEY.", "", None
        return

    try:
        with open(audio_path, "rb") as f:
            file_content = f.read()

        # Trích xuất văn bản từ Audio
        transcribed_text = extract_text(file_content, model_whisper)
        phonemes_text = extract_phonemes(file_content, wav2vec2_processor, wav2vec2_model,
                                         device) if wav2vec2_processor else ""
        transcribed_text = "I want to buy 2 loaf of bread"

        # Placeholder câu nói của User & Assistant
        updated_history = current_chat_history + [
            {"role": "user", "content": transcribed_text},
            {"role": "assistant", "content": "..."}
        ]

        # Update UI lượt đầu
        yield updated_history, "Đang phân tích...", "Đang tạo gợi ý...", None

        # 3. Stream câu trả lời từ Gemini
        raw_stream = generate_ai_response_stream(transcribed_text, phonemes_text, retriever, client, chat_session)

        for partial_ai_response in raw_stream:
            match_chat = re.search(r"\[1\.\s*CONVERSATION\](.*?)(?=\[2\.|\Z)", partial_ai_response, re.DOTALL)
            match_fb = re.search(r"\[2\.\s*PRONUNCIATION[^\]]*\](.*?)(?=\[3\.|\Z)", partial_ai_response,
                                 re.DOTALL | re.IGNORECASE)
            match_sg = re.search(r"\[3\.\s*BETTER[^\]]*\](.*?)(?=\Z)", partial_ai_response, re.DOTALL | re.IGNORECASE)

            chat_display = match_chat.group(1).strip() if match_chat else partial_ai_response
            feedback_display = match_fb.group(1).strip() if match_fb else "Đang nhận dữ liệu đánh giá..."
            suggestions_display = match_sg.group(1).strip() if match_sg else "Đang nhận gợi ý..."

            updated_history[-1] = {"role": "assistant", "content": chat_display}

            # BẮT BUỘC: Luôn yield đủ đúng 4 giá trị
            yield updated_history, feedback_display, suggestions_display, None

    except Exception as e:
        yield current_chat_history, f"Lỗi xử lý: {str(e)}", "", None

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


def generate_ai_response_stream(user_speech_text, mispronunciation_output, retriever, client, chat_session):
    # Lấy context RAG nếu retriever tồn tại
    related_context = ""
    if retriever:
        related_context_docs = retriever.invoke(user_speech_text)
        related_context = "\n".join([doc.page_content for doc in related_context_docs])

    current_history = chat_session.get_history_as_string()

    prompt_template = f"""
    You are an expert English Teacher role-playing as a Grocery Store Shopkeeper.
    The user is a customer coming into your store to buy things.

    --- STORE REAL-TIME DATABASE ---
    {related_context}

    --- PHONETICS DATA ---
    The user's speech phonemes: "{mispronunciation_output}"

    --- SYSTEM INSTRUCTIONS ---
    You must structure your response into exactly 3 separate sections labeled as follows:

    [1. CONVERSATION]
    (Write your next natural response as the shopkeeper here. Keep it 1-2 sentences max.)

    [2. PRONUNCIATION & GRAMMAR FEEDBACK]
    (Analyze the user's text and the Arpabet phonemes.)

    [3. BETTER SUGGESTIONS]
    (Provide 1-2 alternative, natural, and advanced ways the user could have phrased their answer.)

    --- CHAT HISTORY ---
    {current_history}
    User (Customer): {user_speech_text}

    --- YOUR RESPONSE ---
    """

    # 1. Gọi Gemini API ở chế độ stream
    response_stream = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=prompt_template,
    )

    full_text = ""
    # 2. Yield liên tục từng chunk văn bản nhận được từ AI
    for chunk in response_stream:
        if chunk.text:
            full_text += chunk.text
            yield full_text

    # Sau khi kết thúc luồng, lưu vào lịch sử hội thoại
    chat_session.add_message("User", user_speech_text)
    chat_session.add_message("Assistant", full_text)

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

    audio_path = r"E:\Antigravity\English AI\datasets\mp3\1.wav"  # (đang hard-code để test)

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
    <h2 style="font-size: 20px; margin: 6px 0; color: #111827; line-height: 1.4;">{sentence}</h2>
    <p style="font-size: 13px; color: #4b5563; margin: 2px 0;">{meaning}</p>
    <hr style="margin: 10px 0; border-color: #e5e7eb;">
    <p style="font-size: 12px; color: #6b7280;">Độ khớp phát âm: <b>{accuracy}%</b></p>
    {comparison_html}
</div>
"""


def handle_sentence_record_action(wav2vec2_processor, wav2vec2_model, device,
                                   audio_path, current_index, sentences_data, sentence_html_text):
    """
    Xử lý khi người dùng ghi âm xong 1 câu:
    - Lấy phoneme người dùng đọc (từ audio_path thật sự ghi âm được)
    - Dùng CÂU GỐC trong CSDL (đã chuẩn hoá) làm chuẩn để so sánh — không còn
      tạo lại audio TTS rồi cho model nghe lại để suy ra "câu chuẩn" nữa, vì
      cách đó khiến câu chuẩn tự dính lỗi nhận diện (vd "got" bị TTS+ASR
      nghe nhầm thành "god") dù người dùng đọc đúng. Audio mẫu TTS cho người
      nghe vẫn được giữ nguyên, chỉ là tạo 1 lần khi lấy câu ngẫu nhiên
      (process_random_speaking_sentences), không tạo lại ở đây.
    - So khớp 2 chuỗi, tính % và render feedback
    """
    if not sentences_data:
        return gr.update(), None

    if not audio_path:
        return gr.update(), None

    current_sentence_obj = sentences_data[current_index]
    audio_path = r"E:\Antigravity\English AI\datasets\mp3\2.wav"

    try:
        if wav2vec2_processor and wav2vec2_model:
            user_phonemes = recognize_phonemes(
                audio_path, wav2vec2_model, wav2vec2_processor, device
            )
        else:
            user_phonemes = ""

        # Lấy lại câu gốc từ HTML đang hiển thị (tránh lệch nếu state đổi)
        # rồi chuẩn hoá về dạng CHỮ HOA, không dấu câu để so cùng "ngôn ngữ"
        # với output của wav2vec2.
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