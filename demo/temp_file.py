import gradio as gr
import os
from dotenv import load_dotenv
import whisper
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from google import genai


import speaking_module as sm
from agent import SimpleChatSessionManager

# ---------- KHAI BÁO ----------
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)

# Thiết lập đường dẫn tuyệt đối chuẩn xác
TEMP_DIR = os.getenv("APP_TEMP_DIR", os.path.join(project_root, "temp"))
HF_CACHE_DIR = os.getenv("APP_HF_CACHE_DIR", os.path.join(project_root, "huggingface_cache"))

for d in (TEMP_DIR, HF_CACHE_DIR):
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        print(f"Cảnh báo: không tạo được thư mục {d}: {e}")

# Thiết lập biến môi trường cho Hugging Face và Temp
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR
os.environ["HF_HOME"] = HF_CACHE_DIR

_ = load_dotenv()

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

api_key = os.getenv("GEMINI_API_KEY")
client = None
if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Lỗi khởi tạo Gemini client: {e}")
else:
    print("Cảnh báo: chưa thiết lập biến môi trường GEMINI_API_KEY. App sẽ báo lỗi khi gọi AI.")

model_name="all-MiniLM-L6-v2"
retriever = sm.rag_with_faiss(model_name, storage_database)

chat_session = SimpleChatSessionManager()
initial_greeting = "Hello! Welcome to our store. What would you like to buy today?"
chat_session.add_message("assistant", initial_greeting)
# ---------- KHAI BÁO

# ---------- GIAO DIỆN ----------

with gr.Blocks() as demo:
    current_tab = gr.State("AI Assistant")

    with gr.Row():
        # --- SIDEBAR ---
        with gr.Sidebar():
            gr.Markdown("# **English AI**\n*Mastering Language*")
            gr.HTML("<div style='height: 20px;'></div>")

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
                with gr.Row():
                    gr.HTML("""
                        <div class="stat-card">
                            <div class="stat-icon" style="background: #eff6ff; color: #3b82f6;">📅</div>
                            <div><p style="color: gray; font-size: 12px; margin:0;">Review Today</p><b>12</b></div>
                        </div>
                    """)
                    gr.HTML("""
                        <div class="stat-card">
                            <div class="stat-icon" style="background: #f0fdf4; color: #22c55e;">📚</div>
                            <div><p style="color: gray; font-size: 12px; margin:0;">Total Words</p><b>450</b></div>
                        </div>
                    """)
                    gr.HTML("""
                        <div class="stat-card" style="flex-grow:2;">
                            <div style="width:100%"><p style="color: gray; font-size: 12px; margin:0;">Completed</p>
                            <div style="display:flex; align-items:center; gap:10px;">
                                <div class="progress-bar-container"><div class="progress-fill" style="width: 85%;"></div></div>
                                <b>85%</b>
                            </div></div>
                        </div>
                    """)

                with gr.HTML(elem_classes="flashcard"):
                    gr.HTML("""
                        <p style="color: #64748b; font-size: 14px; text-transform: uppercase;">Vocabulary Review</p>
                        <h1>ELOQUENT</h1>
                        <p style="color: #94a3b8;">Click to reveal answer</p>
                    """)

                with gr.Row():
                    gr.HTML("<div style='flex-grow: 1;'></div>")
                    gr.Button("Check answer 👁️", variant="primary", size="lg")
                    gr.HTML("<div style='flex-grow: 1;'></div>")
                gr.Markdown("<center>Press **Space** to Flip</center>")

            # --- TAB 4: SPEAKING ---
            with gr.Column(visible=False) as view_speak:
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.HTML("🎙️ <b>Ordering coffee</b>")
                        conversation_history = gr.Chatbot(
                            label="Chat History",
                            elem_id="chat-bot",
                            height=450,
                            value=[{"role": "assistant", "content": initial_greeting}]
                        )
                        with gr.Row():
                            gr.Button("Hướng dẫn", scale=1)
                            gr.Button("🎙️ Hold to speak", variant="primary", scale=3)

                    with gr.Column(scale=2):
                        gr.Markdown("#### Detailed Feedback")
                        with gr.Column(elem_classes="feedback-box"):
                            gr.HTML("🟢 <b>Pronunciation</b>")
                            gr.Markdown("*Excellent clarity on vowel sounds.*")

                        with gr.Column(elem_classes="feedback-box"):
                            gr.HTML("🟠 <b>Grammar</b>")
                            gr.Markdown("*Try using 'May I have...' for a more natural interaction.*")

                        with gr.Column(elem_classes="feedback-box"):
                            gr.Markdown(
                                "💡 **Suggestions**\n- Polite request: 'Can I get a...'\n- Inquiry: 'What kind of syrups...?'")

            # --- TAB 6: WRITING ---
            with gr.Column(visible=False) as view_write:
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### **English Pro** \nDaily Task: **Writing Workshop**")

                        # Sửa lỗi style: dùng elem_classes="feedback-box-gray" thay vì dùng style inline
                        with gr.Column(elem_classes="feedback-box-gray"):
                            gr.Markdown("**PROMPT OF THE DAY**\nDiscuss the pros and cons of remote work.")

                        essay_input = gr.Textbox(
                            placeholder="Start writing your essay here...",
                            lines=15, show_label=False
                        )
                        with gr.Row():
                            gr.Button("Save Draft")
                            gr.Button("Submit for Review", variant="primary")

                    with gr.Column(scale=2):
                        gr.HTML("""
                            <div style="text-align:center; background:white; padding:20px; border-radius:15px; border:1px solid #e2e8f0;">
                                <svg width="100" height="100" viewBox="0 0 100 100">
                                    <circle cx="50" cy="50" r="40" stroke="#f1f5f9" stroke-width="8" fill="none" />
                                    <circle cx="50" cy="50" r="40" stroke="#22c55e" stroke-width="8" fill="none" stroke-dasharray="210 251" />
                                    <text x="50" y="55" text-anchor="middle" font-size="20" font-weight="bold" fill="#1e293b">8.5</text>
                                </svg>
                                <p style="font-weight:bold; margin-top:10px;">Excellent Progress!</p>
                            </div>
                        """)

                        gr.HTML("<div style='height:15px;'></div>")

                        with gr.Row():
                            gr.HTML(
                                "<div class='feedback-box' style='font-size:12px; width:100%'>Cohesion: <b>8.0</b></div>")
                            gr.HTML(
                                "<div class='feedback-box' style='font-size:12px; width:100%'>Lexical: <b>8.2</b></div>")

                        # Sửa lỗi style: dùng elem_classes="feedback-box-blue" thay vì dùng style inline
                        with gr.Column(elem_classes="feedback-box-blue"):
                            gr.Markdown(
                                "✨ **Key Suggestion**\nEnhance sentence variety by using more subordinating conjunctions.")


    # --- LOGIC ĐIỀU HƯỚNG ---
    def switch_tab(tab_name):
        return {
            view_ai: gr.update(visible=(tab_name == "AI Assistant")),
            view_vocab: gr.update(visible=(tab_name == "Vocabulary")),
            view_speak: gr.update(visible=(tab_name == "Speaking")),
            view_write: gr.update(visible=(tab_name == "Writing")),
        }


    nav_ai.click(lambda: "AI Assistant", None, current_tab).then(switch_tab, current_tab,
                                                                 [view_ai, view_vocab, view_speak, view_write])
    nav_vocab.click(lambda: "Vocabulary", None, current_tab).then(switch_tab, current_tab,
                                                                  [view_ai, view_vocab, view_speak, view_write])
    nav_speak.click(lambda: "Speaking", None, current_tab).then(switch_tab, current_tab,
                                                                [view_ai, view_vocab, view_speak, view_write])
    nav_write.click(lambda: "Writing", None, current_tab).then(switch_tab, current_tab,
                                                               [view_ai, view_vocab, view_speak, view_write])

current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
css_file_path = os.path.join(current_dir, "style.css")

if os.path.exists(css_file_path):
    with open(css_file_path, "r", encoding="utf-8") as f:
        custom_css_content = f.read()
else:
    custom_css_content = ""
    print(f"Cảnh báo: Không tìm thấy file {css_file_path}")


demo.launch(css=custom_css_content, theme=gr.themes.Soft())