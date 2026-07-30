import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gradio as gr
import os
from dotenv import load_dotenv
import whisper
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from google import genai
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

sys.path.append(os.path.join(current_script_dir, "..", "database"))

import speaking_module as sm
from agent import SimpleChatSessionManager
import writting_module as wm
import reading_module as rm
from deep_translator import GoogleTranslator
import vocabulary_module as vm
from fsrs import Scheduler
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import listening_module as lm
import spacy
from functools import partial

# ---------- KHAI BÁO PATH ----------
TEMP_DIR = os.getenv("APP_TEMP_DIR", os.path.join(project_root, "temp"))
HF_CACHE_DIR = os.getenv("APP_HF_CACHE_DIR", os.path.join(project_root, "huggingface_cache"))

for d in (TEMP_DIR, HF_CACHE_DIR):
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        print(f"Cảnh báo: không tạo được thư mục {d}: {e}")

os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR
os.environ["HF_HOME"] = HF_CACHE_DIR

_ = load_dotenv()

# --------- KHAI BÁO DATABASE -----------
from database.database_manager import DatabaseManager
db = DatabaseManager()

# ---------- KHAI BÁO VOCABULARY ----------
fsrs_app = Scheduler()

# ---------- KHAI BÁO LISTENING ----------
VIDEO_PATH = os.path.join(project_root, "datasets", "Videos", "The_benefits_of_doing_nothing.mp4")
AUDIO_PATH = os.path.join(project_root, "datasets", "short_audios", "being_single.mp3")
AUDIO_PATH_FOLDER = os.path.join(project_root, "datasets", "short_audios")
nlp = spacy.load("en_core_web_sm")
# ---------- KHAI BÁO SPEAKING ----------
storage_database = [
    "Product: Fresh Milk | Price: $3.00 per carton | Stock: 15 left | Discount: 10% off if you buy 2.",
    "Product: White Bread | Price: $2.50 per loaf | Stock: 8 left | Discount: No discount.",
    "Product: Organic Eggs | Price: $4.00 per dozen | Stock: 20 cartons left | Discount: Buy 1 get 1 free today!",
    "Product: Potato Chips | Price: $1.80 per bag | Stock: 50 left | Discount: 20% off for all snacks.",
    "Product: Coca-Cola | Price: $1.20 per can | Stock: 100 left | Discount: No discount.",
    "Payment Methods: We accept Cash, Credit Cards (Visa/Mastercard), and Apple Pay. No American Express.",
]

MODEL_SIZE = "base"
device = "cpu"
model_whisper = whisper.load_model(MODEL_SIZE, device=device)

model_id = "facebook/wav2vec2-base-960h"
wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_id)
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)

# wav2vec2_processor = ""
# wav2vec2_model = ""

api_key = os.getenv("GEMINI_API_KEY")
client = None
if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Lỗi khởi tạo Gemini client: {e}")
else:
    print("Cảnh báo: chưa thiết lập biến môi trường GEMINI_API_KEY. App sẽ báo lỗi khi gọi AI.")

chat_session = SimpleChatSessionManager()
initial_greeting = "Hello! Welcome to our store. What would you like to buy today?"
chat_session.add_message("assistant", initial_greeting)

model_name="all-MiniLM-L6-v2"
retriever = sm.rag_with_faiss(model_name, storage_database)
# retriever = ""

# ---------- KHAI BÁO READING ----------
translator = GoogleTranslator(source="en", target="vi")
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Tự động lấy thư mục gốc của project làm chuẩn
DICTIONARY_PATH = os.path.normpath(os.path.join(project_root, "datasets", "CEFR-J Wordlist Ver1.6.xlsx"))
CEFR_DICT = {}

if os.path.exists(DICTIONARY_PATH):
    try:
        df_cefr = pd.read_excel(DICTIONARY_PATH, sheet_name="ALL")
        columns_lower = [str(col).strip().lower() for col in df_cefr.columns]

        word_col_index = None
        for idx, col in enumerate(columns_lower):
            if col in ['word', 'words', 'headword', 'vocabulary', 'từ']:
                word_col_index = idx
                break

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

    except Exception as e:
        print(f"❌ Lỗi khi đọc nội dung file Excel CEFR: {e}")
else:
    print(f"⚠️ Không tìm thấy file từ điển tại đường dẫn: {DICTIONARY_PATH}")

lemmatizer = WordNetLemmatizer()

# ---------- KHAI BÁO WRITTING ----------
ESSAY_SCORING_COLUMNS = ["Cohesion", "Syntax", "Vocabulary", "Phraseology", "Grammar", "Conventions"]
ESSAY_MODEL_NAME = "microsoft/deberta-v3-base"
ESSAY_SCORE_MIN, ESSAY_SCORE_MAX = 1.0, 5.0
ESSAY_MAX_LEN = 512
ESSAY_MODEL_WEIGHTS_PATH = r"E:\Đồ án môn học\Demo Speaking Module\models\best_model.bin"
# essay_scoring_tokenizer, essay_scoring_model = wm.load_model(ESSAY_MODEL_WEIGHTS_PATH, device, ESSAY_MODEL_NAME, ESSAY_SCORING_COLUMNS)
essay_scoring_tokenizer, essay_scoring_model = "", ""
# ---------- GIAO DIỆN ----------
with gr.Blocks() as demo:

    with gr.Row():
        # --- SIDEBAR ---
        with gr.Sidebar():
            gr.Markdown("# **English AI**\n*Mastering Language*")
            gr.HTML(elem_classes="spacer-20")

            nav_ai = gr.Button("🤖 AI Assistant", elem_classes=["sidebar-btn", "sidebar-active-btn"])
            nav_vocab = gr.Button("📖 Vocabulary", elem_classes="sidebar-btn")
            nav_listen = gr.Button("👂 Listening", elem_classes="sidebar-btn")
            nav_speak = gr.Button("🗣️ Speaking", elem_classes="sidebar-btn")
            nav_read = gr.Button("📖 Reading", elem_classes="sidebar-btn")
            nav_write = gr.Button("✍️ Writing", elem_classes="sidebar-btn")

        # --- NỘI DUNG CHÍNH ---
        with gr.Column(scale=4):
            # --- TAB 1: AI ASSISTANT ---
            with gr.Column(visible=True) as view_ai:
                chatbot = gr.Chatbot(value=[
                    {"role": "assistant", "content": "Hi! I'm your English Assistant. How can I help you today?"}],
                                     height=500)
                with gr.Row():
                    gr.Textbox(placeholder="Type message...", show_label=False, container=False, scale=10)
                    gr.Button("➔", variant="primary", scale=1)

            # --- TAB 2: VOCABULARY ---
            with gr.Column(visible=False) as view_vocab:
                gr.HTML("<h3>Flashcard Review</h3>")
                due_list_state = gr.State(value=[])
                with gr.Row(elem_id="vocab-stats-row"):
                    with gr.Column(scale=1, min_width=0):
                        stat_due = gr.HTML()
                    with gr.Column(scale=1, min_width=0):
                        stat_total = gr.HTML()
                    with gr.Column(scale=1, min_width=0):
                        stat_completed = gr.HTML()

                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            gr.Markdown(f"**Progress Session**")
                            progress_txt = gr.Markdown()
                        progress_bar_html = gr.HTML()

                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Group(elem_classes="flashcard-container", elem_id="flashcard-wrap"):
                            word_index_display = gr.HTML()
                            word_main_display = gr.HTML()

                            with gr.Column() as area_question:
                                btn_show = gr.Button("Check answer", variant="primary",
                                                     elem_classes=["custom-btn-show"])

                            with gr.Column(visible=False) as area_answer:
                                cefr_display = gr.HTML(sanitize_html=False)
                                gr.HTML("<div style='height: 15px;'></div>")

                            with gr.Row(visible=False, elem_classes=["fsrs-buttons-row"]) as fsrs_buttons_row:
                                btn_again = gr.Button("Again", elem_classes=["btn-fsrs", "btn-again"], min_width=110)
                                btn_hard = gr.Button("Hard", elem_classes=["btn-fsrs", "btn-hard"], min_width=110)
                                btn_good = gr.Button("Good", elem_classes=["btn-fsrs", "btn-good"], min_width=110)
                                btn_easy = gr.Button("Easy", elem_classes=["btn-fsrs", "btn-easy"], min_width=110)

                            with gr.Column(visible=False) as area_empty:
                                gr.Markdown("### 🎉 Congratulation! You have done all words today.")
                                btn_refresh = gr.Button("Check again", variant="secondary")
                btn_show.click(
                    fn=vm.show_answer_action,
                    inputs=[due_list_state],
                    outputs=[area_question, area_answer, fsrs_buttons_row, cefr_display],
                    show_progress="hidden"
                )

                vocab_outputs = [
                    stat_due, stat_total, stat_completed, progress_txt, progress_bar_html,
                    word_index_display, word_main_display, cefr_display, due_list_state,
                    area_question, area_answer, area_empty
                ]

                for btn, rating_name in [(btn_again, "Again"), (btn_hard, "Hard"), (btn_good, "Good"),
                                         (btn_easy, "Easy")]:
                    btn.click(
                        fn=lambda s, r=rating_name: vm.review_word_action(fsrs_app, r, s),
                        inputs=[due_list_state],
                        outputs=vocab_outputs,
                        show_progress="hidden"
                    ).then(
                        fn=lambda: (gr.update(visible=False), gr.update(visible=False)),
                        inputs=None,
                        outputs=[fsrs_buttons_row, area_answer]
                    )

                btn_refresh.click(
                    fn=lambda: vm.pipeline_load(),
                    inputs=None,
                    outputs=vocab_outputs,
                    show_progress="hidden"
                )

                demo.load(
                    fn=lambda: vm.pipeline_load(),
                    inputs=None,
                    outputs=vocab_outputs,
                    show_progress="hidden"
                )

            # --- TAB 3: LISTENING ---
            with gr.Column(visible=False) as view_listen:
                transcript_state = gr.State("")
                questions_state = gr.State([])
                current_index_state = gr.State(0)
                answers_state = gr.State([])

                state_words_data = gr.State([])
                state_current_index = gr.State(0)
                state_score = gr.State(0)
                cefr_dict_sample = lm.get_ceft_word(DICTIONARY_PATH)

                with gr.Tabs():
                    with gr.Tab("Word"):
                        with gr.Column() as setup_panel:
                            gr.Markdown("### Chọn cấp độ và bắt đầu")
                            ceft_level = gr.Radio(
                                choices = ["A1", "A2", "B1", "B2", "C1"],
                                value = "A1",
                                label = "Choose Ceft"
                            )
                            btn_start = gr.Button("🚀 Bắt đầu luyện nghe", variant="primary")
                            status_output = gr.Markdown("")

                        with gr.Column(visible = False) as practice_panel:
                            progress_tracker = gr.Markdown("### Câu 1/10")
                            audio_player = gr.Audio(label="Phát âm thanh câu ví dụ", interactive=False)

                            gr.Markdown("### Điền từ còn thiếu  vào ô trống ")
                            sentence_display = gr.Markdown(label="Câu ví dụ")

                            user_answer = gr.Textbox(
                                label="Từ còn thiếu trong ô trống:",
                                placeholder="Nhập từ bạn nghe được...",
                            )


                            btn_check = gr.Button("🔍 Check Answer", variant="primary")
                            feedback_display = gr.Markdown("")

                        with gr.Column(visible=False) as next_panel:
                            btn_next = gr.Button(
                                "Câu tiếp theo ➡️", variant="primary"
                            )

                        with gr.Column(visible=False) as result_panel:
                            summary_display = gr.Markdown("")
                            btn_restart = gr.Button("🔄 Luyện tập lượt mới", variant="secondary")

                    btn_start.click(
                        fn=lambda level: lm.process_start_practice(client, cefr_dict_sample, level, translator, lemmatizer),
                        inputs=[ceft_level],
                        outputs=[
                            setup_panel,
                            practice_panel,
                            state_words_data,
                            state_current_index,
                            state_score,
                            status_output,
                            progress_tracker,
                            sentence_display,
                            audio_player,
                            user_answer,
                            feedback_display,
                            btn_check,
                        ]
                    )

                    btn_check.click(
                        fn=lm.check_answer,
                        inputs=[user_answer, state_current_index, state_words_data, state_score],
                        outputs=[state_score, feedback_display, btn_check, next_panel, btn_next]
                    )

                    btn_next.click(
                        fn=lm.next_question,
                        inputs=[state_current_index, state_words_data, state_score],
                        outputs=[
                            state_current_index, progress_tracker, sentence_display, audio_player,
                            user_answer, feedback_display, btn_check, next_panel, practice_panel,
                            result_panel, summary_display
                        ]
                    )

                    btn_restart.click(
                        fn=lm.reset_to_start,
                        inputs=[],
                        outputs=[setup_panel, practice_panel,next_panel, result_panel]
                    )

                    with gr.Tab("Paragraph") as tab_listen:

                        with gr.Column() as select_topic_panel:
                            topic_dropdown = gr.Dropdown(label="Select topic to listen", filterable=True, interactive=True,)
                            topic_selected_btn = gr.Button("Select")

                        with gr.Column(elem_classes=["bordered-row-column"], visible=False) as audio_content_panel:
                            audio_listen_paragraph = gr.Audio(
                                label="Phát âm thanh câu",
                                # value=AUDIO_PATH,
                                interactive=False,
                            )
                            transcript_text_paragraph = gr.Markdown("")

                        PAGE_SIZE = 4

                        answers_state_listen_paragraph = gr.State([])
                        user_answers_state = gr.State([])
                        current_page = gr.State(0)

                        answer_boxes = []

                        with gr.Row(elem_classes=["bordered-row-column"], visible=False) as answer_paragraph_panel:
                            gr.Markdown("### Fill words into blanks")
                            for i in range(PAGE_SIZE):
                                answer_boxes.append(
                                    gr.Textbox(
                                        label=f"({i + 1})",
                                        visible=False
                                    )
                                )
                        with gr.Row(visible=False) as check_btn_panel:
                            score_listen_paragraph = gr.Markdown("")
                            prev_listen_paragraph_btn = gr.Button("⬅ Previous")
                            check_listen_paragraph_btn = gr.Button("Check", elem_id="custom_green_btn")
                            next_listen_paragraph_btn = gr.Button("Next ➡")

                    tab_listen.select(
                        fn=lambda : lm.load_value_ratio_audio(AUDIO_PATH_FOLDER),
                        outputs=[topic_dropdown]
                    )

                    topic_selected_btn.click(
                        fn=lambda x: lm.transcribe_short_audio_text(
                            model_whisper,
                            nlp,
                            AUDIO_PATH_FOLDER,
                            x
                        ),
                        inputs=[topic_dropdown],
                        outputs=[
                            select_topic_panel,
                            audio_content_panel,
                            answer_paragraph_panel,
                            check_btn_panel,
                            transcript_text_paragraph,
                            answers_state_listen_paragraph,
                            user_answers_state,
                            current_page,
                            *answer_boxes,
                            audio_listen_paragraph
                        ]
                    )

                    next_listen_paragraph_btn.click(
                        fn=lm.next_page,
                        inputs=[
                            current_page,
                            answers_state_listen_paragraph,
                            user_answers_state,
                            *answer_boxes
                        ],
                        outputs=[
                            current_page,
                            user_answers_state,
                            *answer_boxes
                        ]
                    )

                    prev_listen_paragraph_btn.click(
                        fn=lm.previous_page,
                        inputs=[
                            current_page,
                            answers_state_listen_paragraph,
                            user_answers_state,
                            *answer_boxes
                        ],
                        outputs=[
                            current_page,
                            user_answers_state,
                            *answer_boxes
                        ]
                    )

                    check_listen_paragraph_btn.click(
                        fn=lm.check_answer_listen_paragraph,
                        inputs=[
                            answers_state_listen_paragraph,
                            user_answers_state,
                            current_page,
                            *answer_boxes
                        ],
                        outputs=[
                            score_listen_paragraph,
                            user_answers_state
                        ]
                    )
                    with gr.Tab("Understanding"):
                        with gr.Row():
                            with gr.Column(elem_id="video-wrap"):
                                gr.Markdown("# <center>Listening Test</center>")

                                video_player = gr.Video(label="Video Player", autoplay=True)
                                play_btn = gr.Button("Phát Video")

                                start_quiz_btn = gr.Button("🎧 Bắt đầu bài luyện nghe", variant="primary")

                                loading_display = gr.Markdown(value="", visible=False)

                                with gr.Group(visible=False, elem_classes="feedback-card") as question_group:
                                    progress_display = gr.Markdown("Câu 1 / 3")
                                    question_display = gr.Markdown("")
                                    answer_input = gr.Textbox(
                                        label="Câu trả lời của bạn",
                                        lines=3,
                                        placeholder="Nhập câu trả lời của bạn..."
                                    )
                                    submit_answer_btn = gr.Button("Câu tiếp theo →", variant="primary")

                                with gr.Group(visible=False) as score_group:
                                    gr.Markdown("### 📊 KẾT QUẢ BÀI LUYỆN NGHE")
                                    score_display = gr.HTML()

                        play_btn.click(
                            fn=lambda: lm.load_video(VIDEO_PATH),
                            outputs=video_player)

                        start_quiz_btn.click(
                            fn=lambda: (
                                gr.update(interactive=False),
                                gr.update(visible=True,
                                          value="⏳ Đang xử lý audio và tạo câu hỏi, vui lòng chờ trong giây lát...")
                            ),
                            inputs=None,
                            outputs=[start_quiz_btn, loading_display]
                        ).then(
                            fn=lambda: lm.start_listening_quiz(model_whisper, client, VIDEO_PATH),
                            inputs=None,
                            outputs=[
                                transcript_state, questions_state, current_index_state, answers_state,
                                start_quiz_btn, question_group, question_display, answer_input,
                                progress_display, score_group, score_display, submit_answer_btn
                            ]
                        ).then(
                            fn=lambda: (gr.update(interactive=True), gr.update(visible=False, value="")),
                            inputs=None,
                            outputs=[start_quiz_btn, loading_display]
                        )

                        submit_answer_btn.click(
                            fn=lambda a1, b1, c1, d1, e1: lm.submit_listening_answer(client, a1, b1, c1, d1, e1),
                            inputs=[current_index_state, answer_input, transcript_state, questions_state, answers_state],
                            outputs=[
                                current_index_state, answers_state,
                                question_display, answer_input, progress_display,
                                question_group, score_group, score_display, submit_answer_btn
                            ]
                        )

            # --- TAB 4: SPEAKING ---
            with gr.Column(visible=False) as view_speak:
                speaking_words_state = gr.State([])
                speaking_current_idx_state = gr.State(0)

                with gr.Tabs():
                    with gr.Tab("IPA"):
                        with gr.Row() as btn_speaking_panel:
                            random_word_btn = gr.Button("Choose Randomly")
                            fsrs_word_btn = gr.Button("Choose From FSRS")

                        with gr.Column(visible=False) as ipa_word_speaking_panel:
                            with gr.Row():
                                gr.Markdown(scale=1)
                                content_word_ipa_panel = gr.Markdown("Something inside but i don't know what it is",
                                                                     scale=4)
                                gr.Markdown(scale=1)

                            gr.HTML("<style>#hidden-audio-recorder{display:none !important;}</style>")
                            audio_recorder = gr.File(
                                visible=True,
                                elem_id="hidden-audio-recorder",
                                type="filepath"
                            )
                            play_word_speaking_btn = gr.Button("🔊", variant="secondary", size="sm", elem_id="play-audio-btn")
                            with gr.Row():
                                audio_player = gr.Audio(autoplay=True, visible=True, elem_id="hidden-audio-player")
                                gr.HTML("<style>#hidden-audio-player{display:none !important;}</style>")

                            # --- BỘ NÚT ĐIỀU HƯỚNG VÀ RECORD TINH GỌN ---
                            with gr.Row(elem_classes=["align-center-row"]):
                                pre_ipa_speaking_btn = gr.Button("Previous", interactive=False, scale=1)

                                record_html = gr.HTML("""
                                <div style="display:flex;justify-content:center;">
                                    <button id="record-btn" class="record-btn">
                                        🎤 Record
                                    </button>
                                </div>
                                """)

                                next_ipa_speaking_btn = gr.Button("Next", interactive=False, scale=1)

                        # 1. Bấm lấy từ ngẫu nhiên
                        random_word_btn.click(
                            fn=lambda: sm.process_random_speaking_words(db),
                            inputs=[],
                            outputs=[
                                content_word_ipa_panel,
                                ipa_word_speaking_panel,
                                speaking_words_state,
                                speaking_current_idx_state,
                                pre_ipa_speaking_btn,
                                next_ipa_speaking_btn
                            ]
                        )
                        play_word_speaking_btn.click(
                            fn=sm.speak_text,
                            inputs=[content_word_ipa_panel],
                            outputs=[audio_player]
                        )


                        # 2. Xử lý khi thay đổi dữ liệu audio (sau khi ghi âm hoàn tất)
                        # Chỉ để log/trích xuất phoneme, không ảnh hưởng tới nút Next/Previous.
                        audio_recorder.change(
                            fn=lambda x, y, r, s: sm.handle_record_action(wav2vec2_processor, wav2vec2_model, device, x, y, r, s),
                            inputs=[
                                audio_recorder,
                                speaking_current_idx_state,
                                speaking_words_state,
                                content_word_ipa_panel
                            ],
                            outputs=[content_word_ipa_panel, audio_recorder]
                        )

                        # 3. Bấm Next
                        next_ipa_speaking_btn.click(
                            fn=lambda idx, words: sm.navigate_word(1, idx, words),
                            inputs=[speaking_current_idx_state, speaking_words_state],
                            outputs=[
                                content_word_ipa_panel,
                                speaking_current_idx_state,
                                pre_ipa_speaking_btn,
                                next_ipa_speaking_btn
                            ]
                        )

                        # 4. Bấm Previous
                        pre_ipa_speaking_btn.click(
                            fn=lambda idx, words: sm.navigate_word(-1, idx, words),
                            inputs=[speaking_current_idx_state, speaking_words_state],
                            outputs=[
                                content_word_ipa_panel,
                                speaking_current_idx_state,
                                pre_ipa_speaking_btn,
                                next_ipa_speaking_btn
                            ]
                        )

                    with gr.Tab("Practice"):
                        with gr.Row():
                            with gr.Column(scale=3):
                                gr.HTML("🎙️ <b>Grocery Store</b>")
                                conversation_history = gr.Chatbot(
                                    label="Chat History",
                                    elem_id="chat-bot",
                                    height=450,
                                    value=[{"role": "assistant", "content": initial_greeting}]
                                )
                                with gr.Row(elem_classes="audio-row"):
                                    # Nút cảnh báo thiết lập scale=0, sử dụng class tip-btn để CSS ép size cứng 100px
                                    gr.Button("❗", scale=0, min_width=100, elem_classes="tip-btn")
                                    # Component Audio chiếm không gian co giãn tự do
                                    audio_input = gr.Audio(
                                        label="HOLD TO SPEAK",
                                        sources=["microphone"],
                                        type="filepath",
                                        container=False,
                                        scale=2,
                                        elem_id="hide-dropdown-audio",
                                        waveform_options=gr.WaveformOptions(show_recording_waveform=False)
                                    )
                                    # Cột trống scale=1 để dành khoảng trống cho Audio co giãn khi click
                                    gr.Markdown("", scale=1)

                            with gr.Column(scale=2):
                                gr.Markdown("#### Detailed Feedback")
                                with gr.Column(elem_classes="feedback-box"):
                                    gr.HTML("🟢 <b>Pronunciation</b>")
                                    feedback_output = gr.Markdown(
                                        value="*Chưa có dữ liệu hội thoại. Hãy nói điều gì đó!*"
                                    )

                                with gr.Column(elem_classes="feedback-box"):
                                    suggestions_output = gr.Markdown(
                                        value="*Các gợi ý nâng cao sẽ hiển thị tại đây.*"
                                    )

                        # Gọi hàm wrapper thay vì lambda
                        voice_stream_handler = partial(
                            sm.handle_voice_stream,
                            client=client,
                            model_whisper=model_whisper,
                            wav2vec2_processor=wav2vec2_processor,
                            wav2vec2_model=wav2vec2_model,
                            device=device,
                            retriever=retriever,
                            chat_session=chat_session
                        )

                        # Gắn handler đã partial vào event Gradio
                        audio_input.stop_recording(
                            fn=voice_stream_handler,
                            inputs=[audio_input, conversation_history],
                            outputs=[conversation_history, feedback_output, suggestions_output, audio_input]
                        )

            # --- TAB 5: READING ---
            cached_reading_text = gr.State(value="")
            with gr.Column(visible=False) as view_read:
                with gr.Row():
                    with gr.Column(scale=3):
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
                        file_uploader = gr.File(
                            label="Upload File",
                            file_types=[".txt"],
                            file_count="single"
                        )
                        reading_display = gr.HTML(
                            value="<div class='reading-area'><p class='reading-paragraph' style='color: var(--slate); font-style: italic;'>Vui lòng tải lên một file .txt để bắt đầu đọc...</p></div>"
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### 🔍 Trợ Lý Tra Từ Nhanh")
                        selected_word_txt = gr.Textbox(
                            label="Từ vựng vừa chọn",
                            interactive=False,
                            placeholder="Bôi đen một từ bên khung đọc...",
                            elem_id="selected_word_txt"
                        )

                        hidden_trigger = gr.Textbox(visible=False, elem_id="hidden_trigger_vocab")

                        translated_word = gr.Textbox(
                            label="Word Meaning",
                            interactive=False,
                            lines=3
                        )

                        ceft_word = gr.Textbox(
                            label="CEFT",
                            interactive=False
                        )

                        phonetic_word = gr.Textbox(
                            label="Phiên âm (IPA)",
                            interactive=False,
                            placeholder="/.../"
                        )

                        add_fsrs_btn = gr.Button("➕ Save to FSRS", variant="primary")
                        save_status_lbl = gr.Markdown(value="", elem_id="save-status-container")

                    config_inputs = [file_uploader, font_family_opt, font_size_opt, bg_color_opt, text_color_opt]

                    file_uploader.change(
                        fn=rm.handle_file_upload,
                        inputs=config_inputs,
                        outputs=[cached_reading_text, reading_display],
                    )

                    ui_change_inputs = [cached_reading_text, font_family_opt, font_size_opt, bg_color_opt, text_color_opt]

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
                        fn=lambda text: rm.translate_and_get_cefr_with_excel(CEFR_DICT, translator, lemmatizer, text),
                        inputs=[selected_word_txt],
                        outputs=[translated_word, ceft_word, phonetic_word]
                    )

                    add_fsrs_btn.click(
                        fn=lambda w, c, m, p: rm.add_new_word_to_db_no_ui_update(CEFR_DICT, lemmatizer, w, c, m, p),
                        inputs=[selected_word_txt, ceft_word, translated_word, phonetic_word],
                        outputs=[*vocab_outputs, fsrs_buttons_row, save_status_lbl],
                        show_progress="hidden"
                    )

            # --- TAB 6: WRITING ---
            with gr.Column(visible=False) as view_write:
                with gr.Row(elem_id="writing-tab-row", equal_height=False):
                    with gr.Column(scale=1):
                        essay_topic = gr.Textbox(
                            label="Essay Topic / Prompt",
                            placeholder="e.g., Some people think that robots are important for...",
                            lines=2
                        )
                        essay_content = gr.Textbox(
                            label="Your Essay",
                            placeholder="Start typing your essay here...",
                            lines=18,
                            max_length=2500
                        )
                        submit_btn = gr.Button("Submit for Evaluation", variant="primary")

                    with gr.Column(scale=1, min_width=320, variant="panel", elem_id="writing-summary-panel"):
                        with gr.Group():
                            gr.Markdown("### 📊 Evaluation Summary")

                            with gr.Row(elem_id="score-summary-container"):
                                with gr.Column(scale=1, min_width=110):
                                    total_band_display = gr.HTML(
                                        value="""
                                                <div class="band-seal">
                                                    <div class="band-ring" style="--pct: 0;">
                                                        <div class="band-ring-inner">
                                                            <span class="band-number">--</span>
                                                        </div>
                                                    </div>
                                                    <span class="band-caption">Estimated Band</span>
                                                </div>
                                                """
                                    )

                                with gr.Column(scale=3, min_width=220):
                                    with gr.Row():
                                        score_cohesion = gr.Label(label="Cohesion", value="0/5", min_width=60)
                                        score_syntax = gr.Label(label="Syntax", value="0/5", min_width=60)
                                        score_vocab = gr.Label(label="Vocabulary", value="0/5", min_width=60)
                                    with gr.Row():
                                        score_phraseology = gr.Label(label="Phraseology", value="0/5",
                                                                     min_width=60)
                                        score_grammar = gr.Label(label="Grammar", value="0/5", min_width=60)
                                        score_conventions = gr.Label(label="Conventions", value="0/5",
                                                                     min_width=60)

                        with gr.Group():
                            gr.Markdown("### 💡 AI Feedback & Corrections")

                            essay_feedback_display = gr.HTML(
                                value="""
                                        <div class="feedback-card feedback-container feedback-empty">
                                            <p class="feedback-empty-text">Kết quả nhận xét và sửa lỗi chi tiết từ AI sẽ hiển thị tại đây...</p>
                                        </div>
                                        """
                            )

                result_output = gr.Markdown()

                submit_btn.click(
                    fn=lambda content: wm.handle_essay_scoring(
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

    # --- ĐIỀU HƯỚNG TAB ---
    # Mỗi tab gồm: (view tương ứng, nút sidebar tương ứng)
    NAV_TABS = {
        "AI Assistant": (view_ai, nav_ai),
        "Vocabulary": (view_vocab, nav_vocab),
        "Listening": (view_listen, nav_listen),
        "Speaking": (view_speak, nav_speak),
        "Reading": (view_read, nav_read),
        "Writing": (view_write, nav_write),
    }

    def switch_tab(tab_name):
        """Bật view tương ứng + tô màu active cho đúng nút sidebar đang chọn."""
        updates = {}
        for name, (view, btn) in NAV_TABS.items():
            is_active = (name == tab_name)
            updates[view] = gr.update(visible=is_active)
            updates[btn] = gr.update(
                elem_classes=["sidebar-btn", "sidebar-active-btn"] if is_active else ["sidebar-btn"]
            )
        return updates

    # Output chung cho mọi nút nav: tất cả view + tất cả nút sidebar
    nav_tab_outputs = [view for view, _ in NAV_TABS.values()] + [btn for _, btn in NAV_TABS.values()]

    nav_ai.click(lambda: switch_tab("AI Assistant"), outputs=nav_tab_outputs)

    nav_vocab.click(lambda: switch_tab("Vocabulary"), outputs=nav_tab_outputs).then(
        fn=lambda: vm.pipeline_load(),
        inputs=None,
        outputs=vocab_outputs,
        show_progress="hidden"
    )

    nav_listen.click(lambda: switch_tab("Listening"), outputs=nav_tab_outputs)
    nav_speak.click(lambda: switch_tab("Speaking"), outputs=nav_tab_outputs)
    nav_read.click(lambda: switch_tab("Reading"), outputs=nav_tab_outputs)
    nav_write.click(lambda: switch_tab("Writing"), outputs=nav_tab_outputs)

# ---------- ĐỌC CUSTOM CSS & JS ----------
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
css_file_path = os.path.join(current_dir, "style_demo.css")

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

demo.queue(default_concurrency_limit=1, max_size=20)
demo.launch(css=custom_css_content, theme=gr.themes.Soft(), js=custom_js_content, allowed_paths=[project_root, r"E:\tmp\audio_cache"])