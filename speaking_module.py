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
    speech, sr = sf.read(audio_file)
    if sr != 16000:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
    if speech.ndim > 1:
        speech = speech.mean(axis=1)

    inputs = wav2vec2_processor(speech, sampling_rate=16000, return_tensors="pt")
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


def compare_phonemes(reference, hypothesis):
    """So sánh chuỗi phoneme chuẩn với chuỗi phoneme người dùng đọc.

    Khác với bản gốc (dùng cho notebook, gọi display(HTML(...)) trực tiếp),
    bản này CHỈ trả về (comparison_html, accuracy) để nơi gọi tự ghép vào
    nội dung Markdown/HTML hiển thị trên Gradio.
    """
    ref_list = list((reference or "").replace(" ", ""))
    hyp_list = list((hypothesis or "").replace(" ", ""))

    if not ref_list:
        return "", 0.0

    matcher = difflib.SequenceMatcher(None, ref_list, hyp_list)
    html_parts = []
    correct_count = 0
    total_count = len(ref_list)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_segment = "".join(ref_list[i1:i2])
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
            + "".join(html_parts)
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