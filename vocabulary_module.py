import sqlite3
import datetime
from fsrs import Card, State, Rating
import gradio as gr

def init_database(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_vocabulary (
                word TEXT PRIMARY KEY,
                cefr_j TEXT,
                meaning TEXT,
                state INTEGER,
                stability REAL,
                difficulty REAL,
                elapsed_days INTEGER,
                scheduled_days INTEGER,
                last_review TEXT,
                due TEXT,
                added_at TEXT
            )
        """)
    conn.commit()
    conn.close()

def load_review_session(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM user_vocabulary")
    total_vocab = cursor.fetchone()[0]

    now = datetime.datetime.now(datetime.timezone.utc)
    cursor.execute("SELECT word, cefr_j, meaning, state, stability, difficulty, elapsed_days, scheduled_days, last_review, due FROM user_vocabulary")
    rows = cursor.fetchall()
    conn.close()
    due_list = []
    completed_count = 0

    for row in rows:
        state_val = int(row[3])
        due_time = datetime.datetime.fromisoformat(row[9])

        if state_val == 0 or due_time <= now:
            due_list.append(row)
        else:
            completed_count += 1

    total_due = len(due_list) + completed_count
    if not due_list:
        return (total_due, total_vocab, completed_count, 0, "All Words Have Done!", "N/A", "N/A", [],
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))

    current_word_data = due_list[0]
    return (total_due, total_vocab, completed_count, completed_count + 1,
            current_word_data[0], current_word_data[1], current_word_data[2],
            due_list, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))

def show_answer_action():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

def review_word_action(DB_PATH, fsrs_app, choice_str, due_list_state):
    if not due_list_state:
        return pipeline_load(DB_PATH)
    current_word_data = due_list_state[0]
    word, cefr_j, meaning, state, stability, difficulty, elapsed_days, scheduled_days, last_review, due = current_word_data

    rating_map = {
        "Again": Rating.Again,
        "Hard": Rating.Hard,
        "Good": Rating.Good,
        "Easy": Rating.Easy
    }

    rating_choice =rating_map[choice_str]

    card = Card()
    card.state = State(state)
    card.stability = stability
    card.difficulty = difficulty
    card.elapsed_days = elapsed_days
    if last_review and isinstance(last_review, str):
        card.last_review = datetime.datetime.fromisoformat(last_review)
    else:
        card.last_review = last_review

    card.scheduled_days = scheduled_days
    card.due = datetime.datetime.fromisoformat(due)

    now =datetime.datetime.now(datetime.timezone.utc)
    new_card, _ = fsrs_app.review_card(card, rating_choice, now)

    conn =sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
            INSERT INTO user_vocabulary 
            (word, cefr_j, meaning, state, stability, difficulty, elapsed_days, scheduled_days, last_review, due, added_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(word) DO UPDATE SET
                state=excluded.state,
                stability=excluded.stability,
                difficulty=excluded.difficulty,
                last_review=excluded.last_review,
                due=excluded.due
        """, (word, cefr_j, meaning, new_card.state.value, new_card.stability, new_card.difficulty,
              0, 0, new_card.last_review.isoformat() if new_card.last_review else None, new_card.due.isoformat(),
              now.isoformat()))
    conn.commit()
    conn.close()

    return pipeline_load(DB_PATH)


def render_stats_and_progress(t_due, t_vocab, comp, curr_idx, word, cefr, meaning, due_list):
    percent = int((comp / t_due) * 100) if t_due > 0 else 100
    html_due = f'<div class="stat-card"><div class="stat-title">Review Today</div><div class="stat-value">{t_due}</div></div>'
    html_total = f'<div class="stat-card"><div class="stat-title">Total Words</div><div class="stat-value">{t_vocab}</div></div>'
    html_comp = f'<div class="stat-card"><div class="stat-title">Done</div><div class="stat-value">{comp}/{t_due}</div></div>'
    html_bar = f'<div class="custom-progress-bar"><div class="custom-progress-fill" style="width: {percent}%;"></div></div>'

    idx_str = f"<div class='word-index'>Từ số {curr_idx} / {t_due}</div>" if t_due > 0 else "<div class='word-index'>0 / 0</div>"
    main_str = f"<div class='word-main'>{word}</div>"

    cefr_str = f"""
    <div style='font-size: 18px; color: #2f73d8; font-weight: bold;'>Cefr: {cefr}</div>
    <div style='font-size: 18px; color: #10b981; font-weight: bold; margin-top: 8px;'>Nghĩa: {meaning}</div>
    """

    return (html_due, html_total, html_comp, f"<p style='text-align: right;'><b>{percent}%</b></p>", html_bar,
            idx_str, main_str, cefr_str, due_list)

def pipeline_load(DB_PATH):
    res = load_review_session(DB_PATH)
    outputs = render_stats_and_progress(res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7])
    return outputs + (res[8], res[9], res[10])