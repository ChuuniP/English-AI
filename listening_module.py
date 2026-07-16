import json
import gradio as gr

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

    transcript = lm_transcribe(model_whisper, VIDEO_PATH)
    questions = lm_generate_questions(client, transcript)

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