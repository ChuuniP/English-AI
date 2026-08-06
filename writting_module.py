import json
import random
import datetime
import html
import traceback
from html.parser import HTMLParser
import torch
import torch.nn as nn
import gradio as gr
from transformers import AutoTokenizer, AutoModel


class EssayScoringModel(nn.Module):
    def __init__(self, model_name, num_labels: int = 6, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)

        if hasattr(self.backbone.config, "hidden_size"):
            hidden = self.backbone.config.hidden_size
        else:
            hidden = getattr(self.backbone.config, "attribute_map", {}).get("hidden_size", 768)

        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
            nn.Sigmoid(),
        )
        self.num_labels = num_labels

    def _mean_pool(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, SCORE_MIN, SCORE_MAX):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)
        logits = self.regressor(self.dropout(pooled))
        scores = logits * (SCORE_MAX - SCORE_MIN) + SCORE_MIN
        return scores

def load_model(weights_path, device, MODEL_NAME, SCORE_COLUMNS):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EssayScoringModel(model_name=MODEL_NAME, num_labels=len(SCORE_COLUMNS))
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Load model thành công!")
    return tokenizer, model

@torch.no_grad()
def score_essay(text: str, tokenizer, model, MAX_LEN, SCORE_MIN, SCORE_MAX, SCORE_COLUMNS, device) -> dict:
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    preds = model(input_ids, attention_mask, SCORE_MIN, SCORE_MAX)
    preds = preds.squeeze(0).cpu().numpy()

    preds = preds.clip(SCORE_MIN, SCORE_MAX)

    result = {col: round(float(score), 2) for col, score in zip(SCORE_COLUMNS, preds)}
    result["Average"] = round(sum(result.values()) / len(SCORE_COLUMNS), 2)
    return result


def check_essay_topic_relevance(topic, content, client) -> dict:
    """
    Dùng AI (Gemini) để kiểm tra xem nội dung bài luận (content) có thực sự
    bám sát đề bài (topic) hay không.

    Trả về dict dạng {"is_relevant": bool, "reason": str}.
    Nếu không có client, hoặc thiếu topic/content, hoặc gọi API lỗi -> mặc định
    coi là hợp lệ (is_relevant=True) để không chặn oan người dùng.
    """
    if not client:
        return {"is_relevant": True, "reason": ""}
    if not topic or not topic.strip() or not content or not content.strip():
        return {"is_relevant": True, "reason": ""}

    prompt = f"""
    You are an English writing examiner (IELTS-style). Determine whether the essay content below
    is actually written in response to the given topic/prompt, or if it is off-topic / unrelated.

    Topic/Prompt:
    \"\"\"{topic}\"\"\"

    Essay Content:
    \"\"\"{content}\"\"\"

    Respond STRICTLY with a single JSON object and nothing else (no markdown, no code fences),
    in exactly this format:
    {{"is_relevant": true or false, "reason": "brief explanation in Vietnamese"}}
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        return {
            "is_relevant": bool(data.get("is_relevant", True)),
            "reason": data.get("reason", ""),
        }
    except Exception as e:
        # Nếu không parse được / lỗi gọi API thì fallback coi như hợp lệ
        return {"is_relevant": True, "reason": f"Không kiểm tra được độ liên quan tới đề bài: {str(e)}"}


def handle_essay_scoring(topic, content, tokenizer, model, max_len, score_min, score_max, columns, device, client):

    zero_score = "0.00/5"

    # 1. Kiểm tra xem bài viết có bám sát đề bài (topic) hay không trước khi chấm điểm
    relevance = check_essay_topic_relevance(topic, content, client)
    if not relevance["is_relevant"]:
        warning_html = f"""
        <div class="band-seal">
            <div class="band-ring" style="--pct: 0;">
                <div class="band-ring-inner">
                    <span class="band-number">0.00</span>
                </div>
            </div>
            <span class="band-caption">Off-topic</span>
        </div>
        """
        off_topic_feedback = f"""
        <div style="color:#b45309; background:#fffbeb; border:1px solid #fcd34d; padding:12px; border-radius:8px;">
            ⚠️ Bài viết có vẻ <b>không bám sát đề bài (topic)</b> đã cho nên hệ thống chưa thể chấm điểm.<br>
            {relevance.get('reason', '')}
        </div>
        """
        return (
            warning_html,
            zero_score, zero_score, zero_score, zero_score, zero_score, zero_score,
            off_topic_feedback
        )

    # 2. Nếu bám sát đề bài thì chấm điểm và lấy feedback như bình thường
    scores = score_essay(content, tokenizer, model, max_len, score_min, score_max, columns, device)
    ai_feedback_html = generate_essay_feedback(content, client)
    if not scores:
        no_score_feedback = "<div style='color: #ef4444; padding: 10px;'>❌ Không thể chấm điểm bài luận.</div>"
        return (
            "0/5",
            zero_score, zero_score, zero_score, zero_score, zero_score, zero_score,
            no_score_feedback
        )

    cohesion = f"{scores.get('Cohesion', 0):.2f}/5"
    syntax = f"{scores.get('Syntax', 0):.2f}/5"
    vocabulary = f"{scores.get('Vocabulary', 0):.2f}/5"
    phraseology = f"{scores.get('Phraseology', 0):.2f}/5"
    grammar = f"{scores.get('Grammar', 0):.2f}/5"
    conventions = f"{scores.get('Conventions', 0):.2f}/5"

    average_score = f"{scores.get('Average', 0):.2f}"

    pct_value = float(average_score) / 5 * 100
    band_html = f"""
    <div class="band-seal">
        <div class="band-ring" style="--pct: {pct_value};">
            <div class="band-ring-inner">
                <span class="band-number">{average_score}</span>
            </div>
        </div>
        <span class="band-caption">Estimated Band</span>
    </div>
    """

    return (
        band_html,
        cohesion,
        syntax,
        vocabulary,
        phraseology,
        grammar,
        conventions,
        ai_feedback_html
    )

def generate_essay_feedback(content, client):
    if not client:
        return "<div style='color: #ef4444; padding: 10px;'>⚠️ Chưa cấu hình GEMINI_API_KEY.</div>"
    if not content or not content.strip():
        return "<div style='color: #6b7280; padding: 10px;'>Vui lòng nhập nội dung bài luận.</div>"

    prompt = f"""
    You are an expert English examiner. Analyze the following essay and provide detailed error corrections, strengths/weaknesses, and advice to improve the score.
    Format your entire response STRICTLY in clean HTML tags only. DO NOT wrap response in markdown code blocks like ```html.

    Use these structures:
    1. <h4 style='color: #1e3a8a; margin: 5px 0;'>🔍 1. Detailed Error Correction</h4>
       [For each error, list Original, Corrected, and brief Vietnamese Explanation]
    2. <h4 style='color: #1e3a8a; margin: 5px 0;'>📊 2. Overall Review (Vietnamese)</h4>
    3. <h4 style='color: #1e3a8a; margin: 5px 0;'>🚀 3. How to Improve (Vietnamese)</h4>

    Essay Content:
    \"\"\"{content}\"\"\"
    """

    try:
        # Gọi Gemini API sử dụng cấu trúc client mới (google-genai)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"<div style='color: #ef4444; padding: 10px;'>❌ Lỗi hệ thống khi tải AI Feedback: {str(e)}</div>"


# ============================================================================
# TAB "WRITTING" — Chọn đề (ngân hàng đề trong DB) -> Viết bài -> Chấm bằng Gemini
# ============================================================================

DEFAULT_MIN_WORDS = 150
DEFAULT_TASK_TYPE_FALLBACK = "General"


# ---------------------------------------------------------------------------
# Helpers định dạng hiển thị
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(text.strip().split())


def _format_wordcount(text: str, min_words: int) -> str:
    count = _word_count(text)
    min_words = min_words or DEFAULT_MIN_WORDS
    icon = "✅" if count >= min_words else "📝"
    return f"{icon} {count} / {min_words} từ"


def _format_timer(seconds: int) -> str:
    seconds = max(0, int(seconds or 0))
    m, s = divmod(seconds, 60)
    return f"⏱️ {m:02d}:{s:02d}"


def _format_badge(prompt: dict) -> str:
    return f"`{prompt.get('difficulty') or '—'}` · `{prompt.get('task_type') or '—'}`"


def _format_prompt_card(prompt: dict) -> str:
    return f"### 📝 Đề bài\n\n{prompt['question_text']}"


def _format_prompt_choice_label(index: int, prompt: dict) -> str:
    """Nhãn hiển thị cho từng lựa chọn trong danh sách 'đề tài' (Panel 1)."""
    text = (prompt.get("question_text") or "").strip()
    if len(text) > 90:
        text = text[:87] + "..."
    return f"Đề {index}: {text}"


# ---------------------------------------------------------------------------
# PANEL 1 — Chọn đề
# ---------------------------------------------------------------------------

def load_writing_filters(db):
    """Khi mở tab Writing: chỉ Độ khó khả dụng; Chủ đề & Dạng bài bị khoá và trống
    cho tới khi người dùng chọn Độ khó."""
    return (
        gr.update(choices=[], value=None, interactive=False),   # writing_topic_dropdown
        gr.update(choices=[], value=None, interactive=False),   # writing_task_type_dropdown
        gr.update(choices=[], value=None, visible=False),       # writing_prompt_radio
        None,                                                     # writing_candidate_prompt_state
    )


def on_difficulty_change(db, difficulty):
    """Chọn/đổi Độ khó -> mở khoá Chủ đề (chỉ hiện chủ đề tồn tại ở độ khó này),
    đồng thời reset Dạng bài + danh sách đề tài về trạng thái ban đầu (khoá lại,
    chọn lại từ đầu) — kể cả khi trước đó người dùng đã chọn xong cả 3 mục."""
    if not difficulty:
        return (
            gr.update(choices=[], value=None, interactive=False),
            gr.update(choices=[], value=None, interactive=False),
            gr.update(choices=[], value=None, visible=False),
            None,
        )

    topic_choices = db.list_writing_topic_categories_by_difficulty(difficulty)
    return (
        gr.update(choices=topic_choices, value=None, interactive=True),   # writing_topic_dropdown
        gr.update(choices=[], value=None, interactive=False),             # writing_task_type_dropdown
        gr.update(choices=[], value=None, visible=False),                 # writing_prompt_radio
        None,                                                               # writing_candidate_prompt_state
    )


def on_topic_change(db, difficulty, topic_category):
    """Chọn Chủ đề -> mở khoá Dạng bài (chỉ hiện dạng bài tồn tại với Độ khó +
    Chủ đề đã chọn), reset danh sách đề tài."""
    if not topic_category:
        return (
            gr.update(choices=[], value=None, interactive=False),
            gr.update(choices=[], value=None, visible=False),
            None,
        )

    task_type_choices = db.list_writing_task_types_by_filters(difficulty=difficulty, topic_category=topic_category)
    return (
        gr.update(choices=task_type_choices, value=None, interactive=True),  # writing_task_type_dropdown
        gr.update(choices=[], value=None, visible=False),                     # writing_prompt_radio
        None,                                                                   # writing_candidate_prompt_state
    )


def on_task_type_change(db, difficulty, topic_category, task_type):
    """Đã chọn đủ cả 3 (Độ khó / Chủ đề / Dạng bài) -> hiển thị TẤT CẢ đề tài
    khớp để người dùng bấm chọn; mặc định chọn sẵn đề tài đầu tiên."""
    if not task_type:
        return gr.update(choices=[], value=None, visible=False), None

    prompts = db.list_all_writing_prompts(
        difficulty=difficulty, topic_category=topic_category, task_type=task_type
    )
    if not prompts:
        gr.Warning("Không tìm thấy đề bài nào phù hợp với lựa chọn này.")
        return gr.update(choices=[], value=None, visible=False), None

    choices = [(_format_prompt_choice_label(i, p), p["id"]) for i, p in enumerate(prompts, start=1)]
    first_prompt = prompts[0]

    return (
        gr.update(choices=choices, value=first_prompt["id"], visible=True),  # writing_prompt_radio
        first_prompt,                                                        # writing_candidate_prompt_state
    )


def on_prompt_select(db, selected_id):
    """Người dùng bấm chọn 1 đề tài cụ thể trong danh sách đang hiển thị.

    Lấy thẳng từ DB theo id thay vì dựa vào writing_prompt_options_state, vì
    việc gr.update() set value cho writing_prompt_radio (ở on_task_type_change)
    cũng tự kích hoạt sự kiện change() của chính nó — nếu handler này đọc lại
    state cũ (chưa kịp cập nhật) sẽ vô tình ghi đè mất đề đã chọn đúng.
    """
    if selected_id is None:
        return None
    return db.get_writing_prompt(selected_id)


def start_writing(prompt: dict | None):
    """Nút '▶️ Bắt đầu viết': chuyển từ màn chọn đề (Panel 1) sang màn viết bài (Panel 2)."""
    if not prompt:
        gr.Warning("Vui lòng chọn hoặc Random một đề bài trước khi bắt đầu viết!")
        return (
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), None, 0, gr.update(active=False),
        )

    min_words = prompt.get("min_words") or DEFAULT_MIN_WORDS
    background = prompt.get("background_info") or "*(Đề này không có thông tin nền bổ sung)*"

    return (
        gr.update(visible=False),                  # start_writing_panel
        gr.update(visible=True),                   # content_writing_panel
        gr.update(visible=False),                  # respond_writing_panel
        _format_badge(prompt),                      # writing_badge_md
        _format_timer(0),                            # writing_timer_md
        _format_wordcount("", min_words),            # writing_wordcount_md
        _format_prompt_card(prompt),                 # writing_prompt_md
        background,                                    # writing_background_info_md
        "",                                              # writing_textbox (reset nội dung)
        prompt,                                           # writing_current_prompt_state
        0,                                                  # writing_elapsed_state
        gr.update(active=True),                              # writing_timer_tick (bật đếm giờ)
    )


# ---------------------------------------------------------------------------
# PANEL 2 — Màn viết bài
# ---------------------------------------------------------------------------

def on_writing_text_change(text, prompt):
    """Cập nhật đếm từ mỗi khi người dùng gõ bài."""
    min_words = (prompt or {}).get("min_words") or DEFAULT_MIN_WORDS
    return _format_wordcount(text, min_words)


def tick_writing_timer(elapsed_seconds):
    """Được gr.Timer gọi mỗi giây khi đang ở màn viết bài."""
    elapsed_seconds = (elapsed_seconds or 0) + 1
    return _format_timer(elapsed_seconds), elapsed_seconds


def save_writing_draft(text):
    """Nút '💾 Lưu nháp' — thông báo nhanh cho người dùng (không cần rời màn hình)."""
    if not text or not text.strip():
        gr.Warning("Chưa có nội dung để lưu nháp!")
    else:
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        gr.Info(f"💾 Đã lưu nháp lúc {now_str} ({_word_count(text)} từ).")


def change_writing_topic():
    """Nút '🔀 Đổi đề khác' — quay lại Panel 1 để chọn đề mới."""
    return (
        gr.update(visible=True),    # start_writing_panel
        gr.update(visible=False),   # content_writing_panel
        gr.update(visible=False),   # respond_writing_panel
        gr.update(active=False),    # writing_timer_tick (tắt đếm giờ)
    )


# ---------------------------------------------------------------------------
# Sanitize HTML trước khi render bằng gr.HTML
# ---------------------------------------------------------------------------
# gr.HTML KHÔNG tự sanitize (khác gr.Markdown mặc định sanitize_html=True), nên
# giá trị được nhét thẳng vào DOM. Nếu bài luận của học viên hoặc HTML do Gemini
# tự sinh chứa ký tự đặc biệt (<, >, &, ") không được escape đúng cách, cấu trúc
# HTML của cả trang có thể bị vỡ -> màn hình trắng. Hai hàm dưới đây xử lý 2
# trường hợp khác nhau:
#   - escape_essay_text(): dùng cho text thuần (nhánh lạc đề) -> escape toàn bộ.
#   - sanitize_annotated_essay_html(): dùng cho HTML do Gemini sinh (có chủ đích
#     giữ lại thẻ <span class="writing-error" title="..."> để highlight lỗi)
#     -> chỉ cho phép thẻ <span> với thuộc tính class/title, escape mọi thứ khác,
#     tự đóng thẻ <span> nếu AI quên đóng.

_ALLOWED_ANNOTATION_TAGS = {"span"}
_ALLOWED_ANNOTATION_ATTRS = {"class", "title"}


def _safe_str(value) -> str:
    """Ép về string an toàn để nhét vào component Gradio (Markdown/HTML), phòng
    trường hợp Gemini trả sai kiểu dữ liệu (null, số, list...) thay vì string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def escape_essay_text(text) -> str:
    """Escape text thuần (không phải HTML) trước khi nhét vào gr.HTML.
    Chấp nhận cả None / kiểu không phải str (vd Gemini trả sai kiểu dữ liệu)."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return html.escape(text)


class _AnnotatedEssaySanitizer(HTMLParser):
    """Parser chỉ giữ lại thẻ <span> (dùng để highlight lỗi chính tả/ngữ pháp).
    Mọi thẻ khác bị escape thành text hiển thị thay vì được render; thẻ <span>
    bị thiếu đóng (do AI trả HTML không hoàn chỉnh) sẽ được tự đóng ở cuối."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._out = []
        self._open_spans = 0

    def handle_starttag(self, tag, attrs):
        if tag in _ALLOWED_ANNOTATION_TAGS:
            safe_attrs = " ".join(
                f'{name}="{html.escape(value or "", quote=True)}"'
                for name, value in attrs
                if name in _ALLOWED_ANNOTATION_ATTRS
            )
            self._out.append(f"<{tag} {safe_attrs}>" if safe_attrs else f"<{tag}>")
            self._open_spans += 1
        else:
            # Thẻ lạ, không nằm trong allowlist -> escape để hiển thị dạng text an toàn
            self._out.append(html.escape(self.get_starttag_text() or ""))

    def handle_startendtag(self, tag, attrs):
        # Thẻ tự đóng (vd <br/>) -> không có nhu cầu trong annotated essay, escape luôn
        self._out.append(html.escape(self.get_starttag_text() or ""))

    def handle_endtag(self, tag):
        if tag in _ALLOWED_ANNOTATION_TAGS and self._open_spans > 0:
            self._out.append(f"</{tag}>")
            self._open_spans -= 1
        # Thẻ đóng lạ hoặc thừa (không khớp thẻ mở nào còn lại) -> bỏ qua, không render

    def handle_data(self, data):
        self._out.append(html.escape(data))

    def get_sanitized(self) -> str:
        # Tự đóng mọi <span> chưa được đóng để không làm vỡ phần DOM phía sau
        return "".join(self._out) + ("</span>" * self._open_spans)


def sanitize_annotated_essay_html(raw_html) -> str:
    """Làm sạch HTML do Gemini sinh ra cho phần 'bài viết đã đánh dấu lỗi'
    trước khi render bằng gr.HTML. Chỉ cho phép thẻ <span class=.. title=..>,
    escape mọi thứ khác, tự đóng thẻ thiếu.

    Chấp nhận cả None hoặc kiểu không phải str (Gemini đôi khi trả sai schema,
    vd trả null hoặc trả 1 list/dict thay vì string) -> luôn trả về string an toàn,
    không bao giờ raise exception ra ngoài."""
    if raw_html is None:
        return ""
    if not isinstance(raw_html, str):
        # Không đúng kiểu string như kỳ vọng -> không cố parse như HTML, escape thẳng
        return html.escape(str(raw_html))
    if not raw_html.strip():
        return ""
    try:
        parser = _AnnotatedEssaySanitizer()
        parser.feed(raw_html)
        parser.close()
        return parser.get_sanitized()
    except Exception:
        # Parser lỗi vì bất kỳ lý do gì -> fallback an toàn tuyệt đối: escape toàn bộ
        return html.escape(raw_html)


# ---------------------------------------------------------------------------
# Chấm điểm bằng Gemini (thay cho model local, dùng thang IELTS 4 tiêu chí)
# ---------------------------------------------------------------------------

def _build_writing_grading_prompt(question_text, background_info, task_type, difficulty, essay_text):
    background_part = ""
    if background_info:
        background_part = f'\nBackground information given to the student:\n"""{background_info}"""\n'

    return f"""
You are a strict, experienced IELTS Writing examiner. Grade the essay below using the 4 official
IELTS Writing band criteria, each scored from 0.0 to 9.0 (one decimal place, in 0.5 increments):
- task_response: Task Response / Task Achievement
- coherence: Coherence & Cohesion
- lexical: Lexical Resource
- grammar: Grammatical Range & Accuracy

Writing task type: {task_type or DEFAULT_TASK_TYPE_FALLBACK}
Difficulty level: {difficulty or "Intermediate"}

Prompt/Question given to the student:
\"\"\"{question_text}\"\"\"
{background_part}
Student's essay:
\"\"\"{essay_text}\"\"\"

Respond STRICTLY with a single JSON object and nothing else (no markdown, no code fences, no text
outside the JSON), in EXACTLY this schema:
{{
  "task_response": <float 0-9>,
  "coherence": <float 0-9>,
  "lexical": <float 0-9>,
  "grammar": <float 0-9>,
  "overall": <float 0-9, average of the 4 scores rounded to the nearest 0.5>,
  "annotated_essay_html": "<the student's essay reproduced as HTML, wrapping every grammar/vocab/spelling error in a <span class=\\"writing-error\\" title=\\"brief explanation in Vietnamese\\">...</span> tag around the erroneous text>",
  "feedback_html": "<detailed feedback written in Vietnamese as clean HTML using <h4> section headers and <ul><li> bullet points, covering strengths, weaknesses per criterion, and concrete advice to improve>",
  "model_essay": "<a concise band-9 model essay in English answering the same prompt>"
}}
"""


def call_gemini_writing_score(question_text, background_info, task_type, difficulty, essay_text, client) -> dict:
    """Gọi Gemini để chấm bài Writing theo 4 tiêu chí IELTS. Trả về dict kết quả hoặc {"error": ...}."""
    if not client:
        return {"error": "⚠️ Chưa cấu hình GEMINI_API_KEY nên không thể chấm bài."}

    prompt = _build_writing_grading_prompt(question_text, background_info, task_type, difficulty, essay_text)

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        for key in ("task_response", "coherence", "lexical", "grammar", "overall"):
            try:
                data[key] = round(float(data.get(key, 0)), 1)
            except (TypeError, ValueError):
                data[key] = 0.0

        return data
    except Exception as e:
        return {"error": f"❌ Lỗi khi gọi Gemini để chấm điểm: {str(e)}"}


def submit_writing(essay_text, prompt: dict | None, client):
    no_change = (
        gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(),
    )

    if not prompt:
        gr.Warning("Không tìm thấy đề bài hiện tại, vui lòng chọn lại đề.")
        yield no_change
        return

    if not essay_text or not essay_text.strip():
        gr.Warning("Vui lòng viết bài trước khi nộp!")
        yield no_change
        return

    gr.Info("⏳ Bài của bạn đang được chấm, vui lòng chờ trong giây lát...")

    # Yield tạm ngay lập tức: chỉ tắt timer, mọi thứ khác giữ nguyên (gr.update() = không đổi).
    # Mục đích duy nhất là chặn writing_timer_tick tick tiếp trong lúc chờ Gemini phía dưới.
    yield (
        gr.update(), gr.update(), gr.update(active=False),
        gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(),
    )

    try:
        question_text = prompt.get("question_text", "")
        background_info = prompt.get("background_info")
        task_type = prompt.get("task_type")
        difficulty = prompt.get("difficulty")

        relevance = check_essay_topic_relevance(question_text, essay_text, client)

        if not relevance["is_relevant"]:
            gr.Warning("⚠️ Bài viết có vẻ chưa bám sát đề bài đã chọn.")
            result = {
                "task_response": 0.0, "coherence": 0.0, "lexical": 0.0, "grammar": 0.0, "overall": 0.0,
                "annotated_essay_html": escape_essay_text(essay_text),
                "feedback_html": (
                    f"<h4 style='color:#b45309;'>⚠️ Off-topic</h4>"
                    f"<p>Bài viết có vẻ <b>không bám sát đề bài</b> đã cho nên chưa thể chấm điểm chi tiết.</p>"
                    f"<p>{escape_essay_text(relevance.get('reason', ''))}</p>"
                ),
                "model_essay": "",
            }
        else:
            result = call_gemini_writing_score(question_text, background_info, task_type, difficulty, essay_text, client)
            if "error" in result:
                gr.Warning(result["error"])
                yield no_change
                return

        result["_meta"] = {
            "question_text": question_text,
            "difficulty": difficulty,
            "task_type": task_type,
            "essay_content": essay_text,
            "prompt_id": prompt.get("id"),
        }

        overall = result.get("overall", 0)
        overall_md = f"## {overall}\n*Điểm tổng*"

        annotated_html = sanitize_annotated_essay_html(result.get("annotated_essay_html", ""))

        yield (
            gr.update(visible=False),                                                     # content_writing_panel
            gr.update(visible=True),                                                      # respond_writing_panel
            gr.update(active=False),                                                      # writing_timer_tick (dừng đếm giờ)
            overall_md,                                                                   # result_overall_score_md
            f"{result.get('task_response', 0)}/9",                                       # score_task_response
            f"{result.get('coherence', 0)}/9",                                           # score_coherence
            f"{result.get('lexical', 0)}/9",                                             # score_lexical
            f"{result.get('grammar', 0)}/9",                                             # score_grammar_writing
            f"<div class='writing-annotated-essay'>{annotated_html}</div>",              # result_annotated_essay_html (đã sanitize)
            _safe_str(result.get("feedback_html", "")),                                  # result_feedback_md
            result,                                                                       # writing_last_result_state
        )
    except Exception:

        print("=" * 30, "LỖI TRONG submit_writing", "=" * 30)
        traceback.print_exc()
        print("=" * 80)
        gr.Warning("❌ Có lỗi xảy ra khi chấm bài. Vui lòng thử lại (xem console server để biết chi tiết lỗi).")
        yield no_change


# ---------------------------------------------------------------------------
# PANEL 3 — Kết quả
# ---------------------------------------------------------------------------

def retry_writing(prompt: dict | None):
    """Nút '🔁 Viết lại đề này' — quay lại Panel 2, giữ nguyên đề, xoá nội dung đã viết."""
    min_words = (prompt or {}).get("min_words") or DEFAULT_MIN_WORDS
    return (
        gr.update(visible=True),                    # content_writing_panel
        gr.update(visible=False),                   # respond_writing_panel
        "",                                            # writing_textbox (reset)
        _format_timer(0),                              # writing_timer_md
        _format_wordcount("", min_words),                # writing_wordcount_md
        0,                                                 # writing_elapsed_state
        gr.update(active=True),                             # writing_timer_tick (bật lại đếm giờ)
    )


def next_writing(db, prompt: dict | None):
    """Nút '➡️ Đề tiếp theo' — random đề mới (ưu tiên cùng độ khó/chủ đề), vào thẳng màn viết bài."""
    difficulty = (prompt or {}).get("difficulty")
    topic_category = (prompt or {}).get("topic_category")

    new_prompt = db.get_random_writing_prompt(difficulty=difficulty, topic_category=topic_category)
    if not new_prompt:
        new_prompt = db.get_random_writing_prompt()

    if not new_prompt:
        gr.Warning("Không tìm thấy đề Writing nào khác trong ngân hàng đề.")
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )

    min_words = new_prompt.get("min_words") or DEFAULT_MIN_WORDS
    background = new_prompt.get("background_info") or "*(Đề này không có thông tin nền bổ sung)*"

    return (
        gr.update(visible=True),                    # content_writing_panel
        gr.update(visible=False),                   # respond_writing_panel
        _format_badge(new_prompt),                    # writing_badge_md
        _format_timer(0),                               # writing_timer_md
        _format_wordcount("", min_words),                 # writing_wordcount_md
        _format_prompt_card(new_prompt),                    # writing_prompt_md
        background,                                           # writing_background_info_md
        "",                                                      # writing_textbox (reset)
        new_prompt,                                                # writing_current_prompt_state
        0,                                                           # writing_elapsed_state
    )


def save_writing_history(db, result: dict | None):
    """Nút '📌 Lưu vào lịch sử' — lưu kết quả bài vừa chấm vào bảng writing_history."""
    if not result or "_meta" not in result:
        gr.Warning("Chưa có kết quả nào để lưu. Hãy nộp bài trước.")
        return

    meta = result["_meta"]
    try:
        db.add_writing_history(
            question_text=meta.get("question_text", ""),
            essay_content=meta.get("essay_content", ""),
            prompt_id=meta.get("prompt_id"),
            difficulty=meta.get("difficulty"),
            task_type=meta.get("task_type"),
            overall_score=result.get("overall"),
            task_response_score=result.get("task_response"),
            coherence_score=result.get("coherence"),
            lexical_score=result.get("lexical"),
            grammar_score=result.get("grammar"),
            feedback=result.get("feedback_html"),
            annotated_essay=result.get("annotated_essay_html"),
            model_essay=result.get("model_essay"),
        )
        gr.Info("📌 Đã lưu bài viết vào lịch sử!")
    except Exception as e:
        gr.Warning(f"❌ Lỗi khi lưu vào lịch sử: {str(e)}")