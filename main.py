import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gradio as gr
import os
from dotenv import load_dotenv
import whisper
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from google import genai

import speaking_module as sm
from agent import SimpleChatSessionManager
import essay_scoring_module as esm
import reading_module as rm
from deep_translator import GoogleTranslator
import vocabulary_module as vm
from fsrs import Scheduler
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

# ===========================================================================
TEMP_DIR = os.getenv("APP_TEMP_DIR", os.path.join(os.getcwd(), "temp"))
HF_CACHE_DIR = os.getenv("APP_HF_CACHE_DIR", os.path.join(os.getcwd(), "huggingface_cache"))

for d in (TEMP_DIR, HF_CACHE_DIR):
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        print(f"Cảnh báo: không tạo được thư mục {d}: {e}")

os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR
os.environ["HF_HOME"] = HF_CACHE_DIR

_ = load_dotenv()

# ===========================================================================
storage_database = [
    "Product: Fresh Milk | Price: $3.00 per carton | Stock: 15 left | Discount: 10% off if you buy 2.",
    "Product: White Bread | Price: $2.50 per loaf | Stock: 8 left | Discount: No discount.",
    "Product: Organic Eggs | Price: $4.00 per dozen | Stock: 20 cartons left | Discount: Buy 1 get 1 free today!",
    "Product: Potato Chips | Price: $1.80 per bag | Stock: 50 left | Discount: 20% off for all snacks.",
    "Product: Coca-Cola | Price: $1.20 per can | Stock: 100 left | Discount: No discount.",
    "Payment Methods: We accept Cash, Credit Cards (Visa/Mastercard), and Apple Pay. No American Express.",
]
# ===========================================================================
MODEL_SIZE = "base"
device = "cpu"
model_whisper = whisper.load_model(MODEL_SIZE, device=device)

model_id = "facebook/wav2vec2-base-960h"
wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_id)
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
# wav2vec2_processor = ""
# wav2vec2_model = ""
# ===========================================================================
api_key = os.getenv("GEMINI_API_KEY")
client = None
if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Lỗi khởi tạo Gemini client: {e}")
else:
    print("Cảnh báo: chưa thiết lập biến môi trường GEMINI_API_KEY. App sẽ báo lỗi khi gọi AI.")
# ===========================================================================
model_name="all-MiniLM-L6-v2"

retriever = sm.rag_with_faiss(model_name, storage_database)
# ===========================================================================
chat_session = SimpleChatSessionManager()
initial_greeting = "Hello! Welcome to our store. What would you like to buy today?"
chat_session.add_message("assistant", initial_greeting)
# ===========================================================================
def process_voice_gradio(audio_path, current_chat_history):
    if not audio_path:
        return current_chat_history, "Vui lòng ghi âm trước."

    if not client:
        return current_chat_history, "Vui lòng cấu hình biến môi trường GEMINI_API_KEY."

    try:
        with open(audio_path, "rb") as f:
            file_content = f.read()

        transcribed_text = sm.extract_text(file_content, model_whisper)
        phonemes_text = sm.extract_phonemes(file_content, wav2vec2_processor, wav2vec2_model, device)

        # transcribed_text = "I want to buy 2 coca-cola"

        full_ai_response, reply_text, feedback_text, suggestions_text = sm.generate_ai_response(transcribed_text, phonemes_text, retriever, client, chat_session)

        if current_chat_history is None:
            current_chat_history = []

        current_chat_history.append({"role": "user", "content": transcribed_text})
        current_chat_history.append({"role": "assistant", "content": reply_text})

        return current_chat_history, feedback_text, suggestions_text, None

    except Exception as e:
        return current_chat_history, f"Lỗi xử lý: {str(e)}", "", None
# ===========================================================================
ESSAY_SCORING_COLUMNS = ["Cohesion", "Syntax", "Vocabulary", "Phraseology", "Grammar", "Conventions"]
ESSAY_MODEL_NAME = "microsoft/deberta-v3-base"
ESSAY_SCORE_MIN, ESSAY_SCORE_MAX = 1.0, 5.0
ESSAY_MAX_LEN = 512
ESSAY_MODEL_WEIGHTS_PATH = r"E:\Đồ án môn học\Demo Speaking Module\models\best_model.bin"
# ===========================================================================
essay_scoring_tokenizer, essay_scoring_model = esm.load_model(ESSAY_MODEL_WEIGHTS_PATH, device, ESSAY_MODEL_NAME, ESSAY_SCORING_COLUMNS)
# ===========================================================================
# ---READING---
translator = GoogleTranslator(source="en", target="vi")
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Tự động lấy thư mục gốc của project làm chuẩn
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
DICTIONARY_PATH = os.path.normpath(os.path.join(BASE_DIR, "datasets", "CEFR-J Wordlist Ver1.6.xlsx"))
CEFR_DICT = {}

if os.path.exists(DICTIONARY_PATH):
    try:
        # Đọc file Excel
        df_cefr = pd.read_excel(DICTIONARY_PATH, sheet_name="ALL")

        # Chuẩn hóa toàn bộ tên cột về chữ thường và xóa khoảng trắng thừa để dễ so sánh
        columns_lower = [str(col).strip().lower() for col in df_cefr.columns]

        # 1. Tìm cột chứa từ vựng
        word_col_index = None
        for idx, col in enumerate(columns_lower):
            if col in ['word', 'words', 'headword', 'vocabulary', 'từ']:
                word_col_index = idx
                break

        # 2. Tìm cột chứa trình độ CEFR
        level_col_index = None
        for idx, col in enumerate(columns_lower):
            if col in ['level', 'level_cefr', 'cefr', 'cefr_j', 'grade', 'trình độ']:
                level_col_index = idx
                break

        if word_col_index is None:
            word_col_index = 0
        if level_col_index is None:
            level_col_index = 1 if len(df_cefr.columns) > 1 else 0

        real_word_col = df_cefr.columns[word_col_index]
        real_level_col = df_cefr.columns[level_col_index]

        CEFR_DICT = dict(zip(
            df_cefr[real_word_col].astype(str).str.strip().str.lower(),
            df_cefr[real_level_col].astype(str).str.strip()
        ))

        print(f"🎉 Tải từ điển CEFR thành công từ: {DICTIONARY_PATH}")
        print(f"   -> Đang sử dụng cột từ vựng: '{real_word_col}' và cột trình độ: '{real_level_col}'")

    except Exception as e:
        print(f"❌ Lỗi khi đọc nội dung file Excel CEFR: {e}")
        if 'df_cefr' in locals():
            print(f"   -> Các cột hiện có trong file của bạn là: {list(df_cefr.columns)}")
else:
    print(f"⚠️ Không tìm thấy file từ điển tại đường dẫn: {DICTIONARY_PATH}")

lemmatizer = WordNetLemmatizer()
# ===========================================================================
DB_PATH = "user_study_data.db"
fsrs_app = Scheduler()

vm.init_database(DB_PATH)
# ===========================================================================


with gr.Blocks(title="Demo English") as demo:
    with gr.Tabs():
        with gr.Tab("Speaking"):
            with gr.Row():
                with gr.Column(scale=3):
                    conversation_history = gr.Chatbot(
                        label="Chat History",
                        elem_id="chat-bot",
                        height=450,
                        value=[{"role": "assistant", "content": initial_greeting}]
                    )
                    with gr.Row():
                        audio_input = gr.Audio(
                            label = "HOLD TO SPEAK",
                            sources = ["microphone"],
                            type = "filepath",
                            container = False,
                            min_width = 100,
                            elem_id="hide-dropdown-audio",
                            waveform_options = gr.WaveformOptions(show_recording_waveform=False)
                        )

                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("### PHẢN HỒI CHI TIẾT")

                    feedback_output = gr.Markdown(
                        value="*Chưa có dữ liệu hội thoại. Hãy nói điều gì đó!*"
                    )

                    gr.Markdown("---")

                    suggestions_output = gr.Markdown(
                        value="*Các gợi ý nâng cao sẽ hiển thị tại đây.*"
                    )

            audio_input.stop_recording(
                fn=process_voice_gradio,
                inputs=[audio_input, conversation_history],
                outputs=[conversation_history, feedback_output, suggestions_output, audio_input]
            )
        #################################################################################
        with gr.Tab("Essay Scoring"):
            with gr.Row():
                with gr.Column(scale=1):
                    essay_topic = gr.Textbox(
                        label="Essay Topic / Prompt",
                        placeholder="e.g., Some people think that robots are important for...",
                        lines=2
                    )
                    essay_content = gr.Textbox(
                        label="Your Essay",
                        placeholder="Start typing your essay here...",
                        lines=21,
                        max_length=2500
                    )
                    submit_btn = gr.Button("Submit for Evaluation", variant="primary")

                    with gr.Column(scale=1, variant="panel"):
                        with gr.Group():
                            gr.Markdown("<h3 style='margin: 0; font-size: 16px;'>📊 Evaluation Summary</h3>")

                            with gr.Row(elem_id="score-summary-container"):
                                with gr.Column(scale=1, min_width=110):
                                    total_band_display = gr.HTML(
                                        value="""
                                                <div style="display: flex; justify-content: center; align-items: center; flex-direction: column; padding-top: 10px;">
                                                    <div style="width: 80px; height: 80px; border-radius: 50%; border: 5px solid #e5e7eb; display: flex; justify-content: center; align-items: center; margin-bottom: 5px;">
                                                        <span style="font-size: 20px; font-weight: bold; color: gray;">--</span>
                                                    </div>
                                                    <span style="color: #6b7280; font-size: 11px; font-weight: 600; text-transform: uppercase;">Estimated Band</span>
                                                </div>
                                                """
                                    )

                                with gr.Column(scale=3, min_width=250):
                                    with gr.Row():
                                        score_cohesion = gr.Label(label="Cohesion", value="0/5", min_width=60)
                                        score_syntax = gr.Label(label="Syntax", value="0/5", min_width=60)
                                        score_vocab = gr.Label(label="Vocabulary", value="0/5", min_width=60)
                                    with gr.Row():
                                        score_phraseology = gr.Label(label="Phraseology", value="0/5", min_width=60)
                                        score_grammar = gr.Label(label="Grammar", value="0/5", min_width=60)
                                        score_conventions = gr.Label(label="Conventions", value="0/5", min_width=60)

                        gr.Markdown("<hr style='margin: 10px 0;'/>")

                        with gr.Group():
                            gr.Markdown(
                                "<h3 style='margin: 0 0 5px 0; font-size: 16px;'>💡 AI Feedback & Corrections</h3>")

                            essay_feedback_display = gr.HTML(
                                value="""
                                        <div class="feedback-scroll-container">
                                            <span style='color: gray; font-style: italic;'>Kết quả nhận xét và sửa lỗi chi tiết từ AI sẽ hiển thị tại đây...</span>
                                        </div>
                                        """
                            )

            result_output = gr.Markdown()

            submit_btn.click(
                fn=lambda content: esm.handle_essay_scoring(
                    content,
                    essay_scoring_tokenizer,
                    essay_scoring_model,
                    ESSAY_MAX_LEN,
                    ESSAY_SCORE_MIN,
                    ESSAY_SCORE_MAX,
                    ESSAY_SCORING_COLUMNS,
                    device,
                    client
                ),
                inputs=[essay_content],
                outputs=[total_band_display, score_cohesion, score_syntax, score_vocab,
                         score_phraseology, score_grammar, score_conventions, essay_feedback_display]
            )

    #################################################################################
        with gr.Tab("Reading"):
            cached_reading_text = gr.State(value="")
            with gr.Row():
                with gr.Column(scale=3):
                    file_uploader = gr.File(
                        label="Upload File",
                        file_types=[".txt"],
                        file_count="single"
                    )
                    with gr.Accordion("🛠️ Tùy chỉnh giao diện đọc", open=False):
                        with gr.Row():
                            font_family_opt = gr.Dropdown(
                                choices=["Serif (Có chân)", "Sans-serif (Không chân)", "Monospace (Đơn cách)"],
                                value="Serif (Có chân)",
                                label="Font chữ"
                            )
                            font_size_opt = gr.Dropdown(
                                choices=["14px", "16px", "18px", "20px", "22px"],
                                value="16px",
                                label="Kích thước chữ"
                            )
                        with gr.Row():
                            bg_color_opt = gr.Dropdown(
                                choices=["Giấy cổ (Mặc định)", "Trắng tinh", "Chế độ tối (Dark mode)",
                                         "Xanh mint dịu mắt"],
                                value="Giấy cổ (Mặc định)",
                                label="Màu nền bảo vệ mắt"
                            )
                            text_color_opt = gr.Dropdown(
                                choices=["Xanh mực (Mặc định)", "Xám đậm", "Trắng sáng (Cho nền tối)",
                                         "Xanh đại dương"],
                                value="Xanh mực (Mặc định)",
                                label="Màu chữ"
                            )
                    # Khung hiển thị HTML bài đọc
                    reading_display = gr.HTML(
                        value="<div class='reading-area'><p class='reading-paragraph' style='color: var(--slate); font-style: italic;'>Vui lòng tải lên một file .txt để bắt đầu đọc...</p></div>"
                    )
                with gr.Column(scale=1):
                    gr.Markdown("#### 🔍 Trợ Lý Tra Từ Nhanh")
                    selected_word_txt = gr.Textbox(
                        label="Từ vựng vừa chọn",
                        interactive = False,
                        placeholder="Bôi đen một từ bên khung đọc...",
                        elem_id="selected_word_txt"
                    )

                    hidden_trigger = gr.Textbox(visible=False, elem_id="hidden_trigger_vocab")

                    translated_word = gr.Textbox(
                        label="Word Meaning",
                        interactive = False,
                        lines=3
                    )

                    ceft_word = gr.Textbox(
                        label="CEFT",
                        interactive = False
                    )

                    add_fsrs_btn = gr.Button("➕ Save to FSRS", variant = "primary")

                    save_status_lbl = gr.Markdown(value="", elem_id="save-status-container")

            config_inputs = [file_uploader, font_family_opt, font_size_opt, bg_color_opt, text_color_opt]

            file_uploader.change(
                fn=rm.handle_file_upload,
                inputs=[file_uploader, font_family_opt, font_size_opt, bg_color_opt, text_color_opt],
                outputs=[cached_reading_text, reading_display],
            )

            ui_change_inputs = [cached_reading_text, font_family_opt, font_size_opt, bg_color_opt, text_color_opt]

            # Loại bỏ cặp dấu [] tại outputs của cả 4 dòng dưới đây
            font_family_opt.change(fn=rm.render_html_reading_zone, inputs=ui_change_inputs, outputs=reading_display)
            font_size_opt.change(fn=rm.render_html_reading_zone, inputs=ui_change_inputs, outputs=reading_display)
            bg_color_opt.change(fn=rm.render_html_reading_zone, inputs=ui_change_inputs, outputs=reading_display)
            text_color_opt.change(fn=rm.render_html_reading_zone, inputs=ui_change_inputs, outputs=reading_display)

            hidden_trigger.change(
                fn=lambda x: x,
                inputs=[hidden_trigger],
                outputs=selected_word_txt
            )

            selected_word_txt.change(
                fn= lambda text: rm.translate_and_get_cefr_with_excel(CEFR_DICT, translator,lemmatizer, text),
                inputs=[selected_word_txt],
                outputs=[translated_word, ceft_word]
            )

    #################################################################################
        with gr.Tab("Vocabulary"):
            due_list_state = gr.State(value=[])
            with gr.Row():
                with gr.Column(scale=1):
                    stat_due = gr.HTML()
                with gr.Column(scale=1):
                    stat_total = gr.HTML()
                with gr.Column(scale=1):
                    stat_completed = gr.HTML()

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        gr.Markdown(f"**Progress Session**")
                        progress_txt = gr.Markdown()
                    progress_bar_html = gr.HTML()

            with gr.Row():
                with gr.Column(scale=1):
                    pass

                with gr.Column(scale=4):
                    with gr.Group(elem_classes="flashcard-container"):

                        word_index_display = gr.HTML()
                        word_main_display = gr.HTML()

                        with gr.Column() as area_question:
                            btn_show = gr.Button("Check answer", variant="primary", elem_classes=["custom-btn-show"])

                        with gr.Column(visible=False) as area_answer:
                            cefr_display = gr.HTML()
                            gr.HTML("<div style='height: 15px;'></div>")

                        with gr.Row(visible=False) as fsrs_buttons_row:
                            btn_again = gr.Button("Again", elem_classes=["btn-fsrs", "btn-again"], min_width=250)
                            btn_hard = gr.Button("Hard", elem_classes=["btn-fsrs", "btn-hard"], min_width=250)
                            btn_good = gr.Button("Good", elem_classes=["btn-fsrs", "btn-good"], min_width=250)
                            btn_easy = gr.Button("Easy", elem_classes=["btn-fsrs", "btn-easy"], min_width=250)

                        with gr.Column(visible=False) as area_empty:
                            gr.Markdown("### 🎉 Congratulation! You have done all words today.")
                            btn_refresh = gr.Button("Check again", variant="secondary")

                with gr.Column(scale=1):
                    pass

        btn_show.click(
            fn=vm.show_answer_action,
            inputs=None,
            outputs=[area_question, area_answer, fsrs_buttons_row],
            show_progress="hidden"
        )

        vocab_outputs = [
            stat_due, stat_total, stat_completed, progress_txt, progress_bar_html,
            word_index_display, word_main_display, cefr_display, due_list_state,
            area_question, area_answer, area_empty
        ]

        for btn, rating_name in [(btn_again, "Again"), (btn_hard, "Hard"), (btn_good, "Good"), (btn_easy, "Easy")]:
            btn.click(
                fn=lambda s, r=rating_name: vm.review_word_action(DB_PATH, fsrs_app, r, s),
                inputs=[due_list_state],
                outputs=vocab_outputs,
                show_progress="hidden"
            ).then(
                fn=lambda: gr.update(visible=False),
                inputs=None,
                outputs=fsrs_buttons_row
            )

        btn_refresh.click(
            fn=lambda: vm.pipeline_load(DB_PATH),
            inputs=None,
            outputs=vocab_outputs,
            show_progress="hidden"

        )

        demo.load(
            fn=lambda: vm.pipeline_load(DB_PATH),
            inputs=None,
            outputs=vocab_outputs,
            show_progress="hidden"
        )

        add_fsrs_btn.click(
            fn=lambda w, c, m: rm.add_new_word_to_db(DB_PATH, CEFR_DICT, lemmatizer, w, c, m),
            inputs=[selected_word_txt, ceft_word, translated_word],
            outputs=[*vocab_outputs, save_status_lbl],
            show_progress="hidden"
        )

current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
css_file_path = os.path.join(current_dir, "style.css")

# Kiểm tra file có tồn tại không và đọc nội dung
if os.path.exists(css_file_path):
    with open(css_file_path, "r", encoding="utf-8") as f:
        custom_css_content = f.read()
else:
    custom_css_content = ""
    print(f"Cảnh báo: Không tìm thấy file {css_file_path}")

js_file_path = os.path.join(current_dir, "selection_listener.js")
if os.path.exists(js_file_path):
    with open(js_file_path, "r", encoding="utf-8") as f:
        custom_js_content = f.read()
else:
    custom_js_content = ""
    print(f"Cảnh báo: Không tìm thấy file {js_file_path}")

# Truyền chuỗi nội dung CSS vào tham số css
demo.launch(css=custom_css_content, js=custom_js_content)