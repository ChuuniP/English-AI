import io
import re
import numpy as np
import torch
import soundfile as sf
from scipy import signal
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def process_voice_gradio(client, model_whisper, wav2vec2_processor, wav2vec2_model, device, retriever, chat_session, audio_path, current_chat_history):
    if not audio_path:
        return current_chat_history, "Vui lòng ghi âm trước."

    if not client:
        return current_chat_history, "Vui lòng cấu hình biến môi trường GEMINI_API_KEY."

    try:
        with open(audio_path, "rb") as f:
            file_content = f.read()

        transcribed_text = extract_text(file_content, model_whisper)
        phonemes_text = extract_phonemes(file_content, wav2vec2_processor, wav2vec2_model, device)
        transcribed_text = "I want to buy 2 coca-colas"
        full_ai_response, reply_text, feedback_text, suggestions_text = generate_ai_response(transcribed_text, phonemes_text, retriever, client, chat_session)

        if current_chat_history is None:
            current_chat_history = []

        current_chat_history.append({"role": "user", "content": transcribed_text})
        current_chat_history.append({"role": "assistant", "content": reply_text})

        return current_chat_history, feedback_text, suggestions_text, None

    except Exception as e:
        return current_chat_history, f"Lỗi xử lý: {str(e)}", "", None

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
        data, samplerate = sf.read(audio_bytes)
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

def generate_ai_response(user_speech_text, mispronunciation_output, retriever, client, chat_session):
    related_context_docs = retriever.invoke(user_speech_text)
    related_context = "\n".join([doc.page_content for doc in related_context_docs])
    current_history = chat_session.get_history_as_string()

    prompt_template = f"""
    You are an expert English Teacher role-playing as a Grocery Store Shopkeeper.
    The user is a customer coming into your store to buy things.

    --- STORE REAL-TIME DATABASE (Use this data to answer accurately about items, prices, and discounts) ---
    {related_context}

    --- PHONETICS DATA ---
    The user's speech phonemes: "{mispronunciation_output}"
    (Tokens with '_err' like 'æ_err' mean they mispronounced that part. If no '_err', pronunciation is good.)

    --- CONVERSATION FLOW GUIDE ---
    As the shopkeeper, follow these steps as the conversation goes on:
    1. Greet and ask what they want.
    2. Ask for the quantity (how many?).
    3. Calculate the total bill price based on the Store Database provided above.
    4. Ask for payment method (Cash or Card).
    5. Say thank you and goodbye.

    --- SYSTEM INSTRUCTIONS ---
    You must structure your response into exactly 3 separate sections labeled as follows:

    [1. CONVERSATION]
    (Write your next natural response as the shopkeeper here. Keep it 1-2 sentences max.)

    [2. PRONUNCIATION & GRAMMAR FEEDBACK]
    (Analyze the user's text and the Arpabet phonemes. Point out the spelling/grammar errors from the user text, and highlight which word was mispronounced based on the '_err' tokens.)

    [3. BETTER SUGGESTIONS]
    (Provide 1-2 alternative, natural, and advanced ways the user could have phrased their answer to sound better or expand their vocabulary.)

    --- CHAT HISTORY ---
    {current_history}
    User (Customer): {user_speech_text}

    --- YOUR RESPONSE ---
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_template,
    )
    ai_reply = response.text or ""

    # Tách phần [1. CONVERSATION] một cách an toàn hơn bằng regex,
    # tránh lỗi khi định dạng trả về không khớp 100% với template.
    match = re.search(r"\[1\.\s*CONVERSATION\](.*?)(?=\[2\.|\Z)", ai_reply, re.DOTALL)
    if match:
        reply_text = match.group(1).strip()
    else:
        reply_text = ai_reply.strip()

    feedback_match = re.search(r"\[2\.\s*PRONUNCIATION[^\]]*\](.*?)(?=\[3\.|\Z)", ai_reply,
                               re.DOTALL | re.IGNORECASE)
    feedback_text = feedback_match.group(1).strip() if feedback_match else "Không tìm thấy dữ liệu đánh giá."

    suggestions_match = re.search(r"\[3\.\s*BETTER[^\]]*\](.*?)(?=\Z)", ai_reply, re.DOTALL | re.IGNORECASE)
    suggestions_text = suggestions_match.group(1).strip() if suggestions_match else "Không tìm thấy câu gợi ý."

    chat_session.add_message("User", user_speech_text)
    chat_session.add_message("Assitant", reply_text)

    return ai_reply, reply_text, feedback_text, suggestions_text