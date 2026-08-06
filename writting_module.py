import json
import torch
import torch.nn as nn
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