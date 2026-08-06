import os
import json
import sqlite3
import datetime

# --- SCHEMA DEFINITIONS (ĐÃ TỐI ƯU HÓA) ---

SCHEMA = """
-- 1. Bảng lưu trữ từ vựng gốc
CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL UNIQUE,
    cefr TEXT CHECK (cefr IN ('A1','A2','B1','B2','C1','C2')),
    meaning TEXT,
    phonetic TEXT
);

-- 2. Bảng lưu trữ câu ví dụ
CREATE TABLE IF NOT EXISTS examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id INTEGER NOT NULL,
    sentence TEXT NOT NULL,
    FOREIGN KEY (word_id) REFERENCES words(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_examples_word_id ON examples(word_id);

-- 3. Bảng tiến độ học FSRS (liên kết khóa ngoại word_id sang bảng words)
CREATE TABLE IF NOT EXISTS user_vocabulary (
    word_id INTEGER PRIMARY KEY,
    state INTEGER NOT NULL DEFAULT 0,
    stability REAL NOT NULL DEFAULT 0.0,
    difficulty REAL NOT NULL DEFAULT 0.0,
    elapsed_days INTEGER NOT NULL DEFAULT 0,
    scheduled_days INTEGER NOT NULL DEFAULT 0,
    last_review TEXT,
    due TEXT NOT NULL,
    added_at TEXT NOT NULL,
    FOREIGN KEY (word_id) REFERENCES words(id) ON DELETE CASCADE
);

-- 4. Bảng lưu trữ transcript file nghe + bộ câu hỏi hiểu bài (Understanding tab)
CREATE TABLE IF NOT EXISTS mp3_transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mp3_name TEXT NOT NULL UNIQUE,
    transcript TEXT,
    questions_json TEXT
);

-- 5. Bảng lưu trữ câu độc lập dùng cho luyện nói (Speaking - tab Sentence)
CREATE TABLE IF NOT EXISTS sentences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sentence TEXT NOT NULL UNIQUE,
    meaning TEXT
);

-- 6. Bảng lưu trữ topic dùng cho luyện nói (Speaking - tab 1 minute)
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL UNIQUE,
    category TEXT
);
CREATE INDEX IF NOT EXISTS idx_topics_category ON topics(category);

-- 7. Bảng lưu trữ bài đọc (Reading) kèm bộ câu hỏi hiểu bài
CREATE TABLE IF NOT EXISTS reading_passages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    cefr TEXT CHECK (cefr IN ('A1','A2','B1','B2','C1','C2')),
    passage TEXT NOT NULL,
    questions_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_reading_passages_cefr ON reading_passages(cefr);

-- 8. Table storing writing practice prompts
CREATE TABLE IF NOT EXISTS writing_prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type TEXT,
    topic_category TEXT,
    difficulty TEXT CHECK (difficulty IN ('Beginner','Intermediate','Advanced')),
    question_text TEXT NOT NULL,
    background_info TEXT,
    min_words INTEGER,
    suggested_time_minutes INTEGER,
    tags TEXT
);
CREATE INDEX IF NOT EXISTS idx_writing_prompts_difficulty ON writing_prompts(difficulty);
CREATE INDEX IF NOT EXISTS idx_writing_prompts_topic_category ON writing_prompts(topic_category);

-- 9. Bảng lưu lịch sử các bài Writing đã chấm (tab Writting)
CREATE TABLE IF NOT EXISTS writing_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id INTEGER,
    question_text TEXT NOT NULL,
    difficulty TEXT,
    task_type TEXT,
    essay_content TEXT NOT NULL,
    overall_score REAL,
    task_response_score REAL,
    coherence_score REAL,
    lexical_score REAL,
    grammar_score REAL,
    feedback TEXT,
    annotated_essay TEXT,
    model_essay TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (prompt_id) REFERENCES writing_prompts(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_writing_history_created_at ON writing_history(created_at);
"""

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(_CURRENT_DIR, "app_data.db")


class DatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self):
        """Khởi tạo toàn bộ cấu trúc bảng."""
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._migrate_schema()

    def _migrate_schema(self):
        """Bổ sung cột còn thiếu cho các DB đã tồn tại từ trước (an toàn khi chạy lại)."""
        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info(mp3_transcripts)")
        existing_cols = {row["name"] for row in cur.fetchall()}
        if "questions_json" not in existing_cols:
            cur.execute("ALTER TABLE mp3_transcripts ADD COLUMN questions_json TEXT")
            self.conn.commit()

    # --- QUẢN LÝ TỪ VỰNG & VÍ DỤ ---

    def add_word(self, word: str, cefr: str = None, meaning: str = None, phonetic: str = None) -> int:
        """Thêm mới 1 từ vựng. Trả về word_id (trả về id cũ nếu từ đã tồn tại)."""
        if not word or not word.strip():
            raise ValueError("Từ vựng không được để trống!")

        clean_word = word.strip()
        cur = self.conn.cursor()

        # 1. Tìm xem từ đã tồn tại trong DB chưa (không phân biệt hoa/thường)
        cur.execute("SELECT id FROM words WHERE LOWER(word) = LOWER(?)", (clean_word,))
        row = cur.fetchone()

        if row:
            word_id = row["id"]
            # Cập nhật bổ sung thông tin nếu trước đó chưa có
            cur.execute("""
                UPDATE words 
                SET cefr = COALESCE(cefr, ?),
                    meaning = COALESCE(meaning, ?),
                    phonetic = COALESCE(phonetic, ?)
                WHERE id = ?
            """, (cefr, meaning, phonetic, word_id))
            self.conn.commit()
            return word_id

        # 2. Nếu chưa có thì thêm mới
        cur.execute(
            "INSERT INTO words (word, cefr, meaning, phonetic) VALUES (?, ?, ?, ?)",
            (clean_word, cefr, meaning, phonetic)
        )
        self.conn.commit()
        return cur.lastrowid

    def add_example(self, word_id: int, sentence: str) -> int:
        """Thêm 1 câu ví dụ cho từ vựng theo word_id."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO examples (word_id, sentence) VALUES (?, ?)",
            (word_id, sentence)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_word(self, word: str) -> dict | None:
        """Truy xuất thông tin 1 từ vựng kèm danh sách câu ví dụ."""
        cur = self.conn.cursor()
        cur.execute("SELECT id, word, cefr, meaning, phonetic FROM words WHERE word = ?", (word,))
        row = cur.fetchone()
        if not row:
            return None

        cur.execute("SELECT sentence FROM examples WHERE word_id = ?", (row["id"],))
        examples = [r["sentence"] for r in cur.fetchall()]

        return {
            "id": row["id"],
            "word": row["word"],
            "cefr": row["cefr"],
            "meaning": row["meaning"],
            "phonetic": row["phonetic"],
            "examples": examples
        }

    def list_all_words(self) -> list[dict]:
        """Lấy toàn bộ từ vựng trong hệ thống kèm ví dụ."""
        cur = self.conn.cursor()
        cur.execute("SELECT id, word, cefr, meaning, phonetic FROM words")
        rows = cur.fetchall()

        results = []
        for row in rows:
            cur.execute("SELECT sentence FROM examples WHERE word_id = ?", (row["id"],))
            examples = [r["sentence"] for r in cur.fetchall()]
            results.append({
                "id": row["id"],
                "word": row["word"],
                "cefr": row["cefr"],
                "meaning": row["meaning"],
                "phonetic": row["phonetic"],
                "examples": examples
            })
        return results

    # --- QUẢN LÝ TIẾN ĐỘ THẺ ÔN TẬP (FSRS) ---

    def get_total_vocabulary_count(self) -> int:
        """Đếm tổng số từ vựng trong bảng user_vocabulary."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM user_vocabulary")
        res = cur.fetchone()
        return res[0] if res else 0

    def get_all_user_vocabulary(self):
        """
        Lấy danh sách tất cả từ vựng của user kèm thông tin chi tiết từ bảng words.
        """
        cur = self.conn.cursor()
        query = """
            SELECT 
                w.word,
                w.cefr AS cefr_j,
                w.meaning,
                w.phonetic,
                v.state,
                v.stability,
                v.difficulty,
                v.elapsed_days,
                v.scheduled_days,
                v.last_review,
                v.due
            FROM user_vocabulary v
            JOIN words w ON v.word_id = w.id
        """
        cur.execute(query)
        return cur.fetchall()

    def upsert_user_vocabulary(
            self,
            word: str,
            cefr_j: str = "N/A",
            meaning: str = "",
            phonetic: str = "",
            state: int = 0,
            stability: float = 0.0,
            difficulty: float = 0.0,
            elapsed_days: int = 0,
            scheduled_days: int = 0,
            last_review: str = None,
            due: str = None,
            added_at: str = None
    ):
        """Thêm mới hoặc cập nhật từ vựng kèm tiến độ FSRS mặc định."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        due_val = due if due else now_iso
        added_at_val = added_at if added_at else now_iso

        # 1. Đảm bảo từ vựng tồn tại trong bảng words
        word_id = self.add_word(word=word, cefr=cefr_j, meaning=meaning, phonetic=phonetic)

        # 2. Cập nhật tiến độ FSRS trong user_vocabulary
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO user_vocabulary 
            (word_id, state, stability, difficulty, elapsed_days, scheduled_days, last_review, due, added_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(word_id) DO UPDATE SET
                state=excluded.state,
                stability=excluded.stability,
                difficulty=excluded.difficulty,
                elapsed_days=excluded.elapsed_days,
                scheduled_days=excluded.scheduled_days,
                last_review=excluded.last_review,
                due=excluded.due
        """, (
            word_id,
            state if state is not None else 0,
            stability if stability is not None else 0.0,
            difficulty if difficulty is not None else 0.0,
            elapsed_days if elapsed_days is not None else 0,
            scheduled_days if scheduled_days is not None else 0,
            last_review,
            due_val,
            added_at_val
        ))
        self.conn.commit()

    def load_review_session(self) -> tuple:
        """Tải phiên ôn tập FSRS ngày hôm nay (JOIN bảng user_vocabulary với words)."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) AS count FROM user_vocabulary")
        total_vocab = cur.fetchone()["count"]

        now = datetime.datetime.now(datetime.timezone.utc)
        today = now.date()

        query = """
            SELECT 
                w.word, w.cefr AS cefr_j, w.meaning, w.phonetic,
                uv.state, uv.stability, uv.difficulty, 
                uv.elapsed_days, uv.scheduled_days, uv.last_review, uv.due
            FROM user_vocabulary uv
            JOIN words w ON uv.word_id = w.id
        """
        cur.execute(query)
        rows = cur.fetchall()

        due_list = []
        completed_count = 0

        for row in rows:
            state_val = int(row["state"])
            due_time = datetime.datetime.fromisoformat(row["due"])
            last_review_str = row["last_review"]

            if state_val == 0 or due_time <= now:
                due_list.append(row)
            elif last_review_str:
                last_review_dt = datetime.datetime.fromisoformat(last_review_str)
                if last_review_dt.astimezone(datetime.timezone.utc).date() == today:
                    completed_count += 1

        total_due = len(due_list) + completed_count
        return total_due, total_vocab, completed_count, due_list

    def update_user_vocab_review(self, word_id: int, new_card, now: datetime.datetime):
        """Cập nhật kết quả đánh giá FSRS cho 1 thẻ từ vựng."""
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO user_vocabulary 
            (word_id, state, stability, difficulty, elapsed_days, scheduled_days, last_review, due, added_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(word_id) DO UPDATE SET
                state=excluded.state,
                stability=excluded.stability,
                difficulty=excluded.difficulty,
                elapsed_days=0,
                scheduled_days=0,
                last_review=excluded.last_review,
                due=excluded.due
        """, (
            word_id,
            new_card.state.value,
            new_card.stability,
            new_card.difficulty,
            0, 0,
            new_card.last_review.isoformat() if new_card.last_review else None,
            new_card.due.isoformat(),
            now.isoformat()
        ))
        self.conn.commit()

    # --- QUẢN LÝ ----

    def get_user_vocabulary_by_word(self, word: str):
        """Tìm kiếm một từ trong bảng user_vocabulary thông qua bảng words."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT w.word, v.* 
            FROM user_vocabulary v
            JOIN words w ON v.word_id = w.id
            WHERE w.word = ?
        """, (word,))
        return cur.fetchone()

    # --- QUẢN LÝ MP3 TRANSCRIPT ---

    def add_mp3_transcript(self, mp3_name: str, transcript: str) -> int:
        """Lưu hoặc ghi đè transcript cho file MP3."""
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO mp3_transcripts (mp3_name, transcript)
            VALUES (?, ?)
            ON CONFLICT(mp3_name) DO UPDATE SET transcript = excluded.transcript
        """, (mp3_name, transcript))
        self.conn.commit()
        cur.execute("SELECT id FROM mp3_transcripts WHERE mp3_name = ?", (mp3_name,))
        return cur.fetchone()["id"]

    def get_mp3_transcript(self, mp3_name: str) -> str | None:
        """Truy xuất transcript của 1 file MP3."""
        cur = self.conn.cursor()
        cur.execute("SELECT transcript FROM mp3_transcripts WHERE mp3_name = ?", (mp3_name,))
        row = cur.fetchone()
        return row["transcript"] if row else None

    def get_mp3_questions(self, mp3_name: str) -> list | None:
        """Truy xuất bộ câu hỏi (Understanding) đã lưu cho 1 file MP3/video, nếu có."""
        cur = self.conn.cursor()
        cur.execute("SELECT questions_json FROM mp3_transcripts WHERE mp3_name = ?", (mp3_name,))
        row = cur.fetchone()
        if not row or not row["questions_json"]:
            return None
        try:
            return json.loads(row["questions_json"])
        except (TypeError, ValueError):
            return None

    def save_mp3_transcript_and_questions(self, mp3_name: str, transcript: str, questions: list) -> int:
        """Lưu (hoặc ghi đè) transcript + bộ câu hỏi cho 1 file MP3/video trong 1 lần ghi."""
        questions_json = json.dumps(questions, ensure_ascii=False)
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO mp3_transcripts (mp3_name, transcript, questions_json)
            VALUES (?, ?, ?)
            ON CONFLICT(mp3_name) DO UPDATE SET
                transcript = excluded.transcript,
                questions_json = excluded.questions_json
        """, (mp3_name, transcript, questions_json))
        self.conn.commit()
        cur.execute("SELECT id FROM mp3_transcripts WHERE mp3_name = ?", (mp3_name,))
        return cur.fetchone()["id"]

    def close(self):
        """Đóng kết nối cơ sở dữ liệu."""
        self.conn.close()

    def get_random_words_for_speaking(self, limit: int = 5) -> list[dict]:
        """Lấy ngẫu nhiên tối đa 'limit' từ vựng từ bảng words."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT word, phonetic, meaning 
            FROM words 
            ORDER BY RANDOM() 
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        return [dict(row) for row in rows]

# --- QUẢN LÝ CÂU LUYỆN NÓI (Speaking - tab Sentence) ---

    def add_sentence(self, sentence: str, meaning: str = None) -> int:
        """Thêm mới 1 câu luyện nói. Trả về sentence_id (trả về id cũ nếu câu đã tồn tại)."""
        if not sentence or not sentence.strip():
            raise ValueError("Câu không được để trống!")

        clean_sentence = sentence.strip()
        cur = self.conn.cursor()

        # 1. Kiểm tra câu đã tồn tại chưa (không phân biệt hoa/thường)
        cur.execute("SELECT id FROM sentences WHERE LOWER(sentence) = LOWER(?)", (clean_sentence,))
        row = cur.fetchone()

        if row:
            sentence_id = row["id"]
            cur.execute("""
                UPDATE sentences
                SET meaning = COALESCE(meaning, ?)
                WHERE id = ?
            """, (meaning, sentence_id))
            self.conn.commit()
            return sentence_id

        # 2. Nếu chưa có thì thêm mới
        cur.execute(
            "INSERT INTO sentences (sentence, meaning) VALUES (?, ?)",
            (clean_sentence, meaning)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_sentence(self, sentence: str) -> dict | None:
        """Truy xuất thông tin 1 câu theo nội dung câu."""
        cur = self.conn.cursor()
        cur.execute("SELECT id, sentence, meaning FROM sentences WHERE sentence = ?", (sentence,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list_all_sentences(self) -> list[dict]:
        """Lấy toàn bộ câu luyện nói trong hệ thống."""
        cur = self.conn.cursor()
        cur.execute("SELECT id, sentence, meaning FROM sentences")
        return [dict(row) for row in cur.fetchall()]

    def delete_sentence(self, sentence_id: int):
        """Xoá 1 câu luyện nói theo id."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM sentences WHERE id = ?", (sentence_id,))
        self.conn.commit()

    def get_random_sentences_for_speaking(self, limit: int = 5) -> list[dict]:
        """Lấy ngẫu nhiên tối đa 'limit' câu từ bảng sentences."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT sentence, meaning
            FROM sentences
            ORDER BY RANDOM()
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        return [dict(row) for row in rows]

# --- QUẢN LÝ TOPIC (Speaking - tab 1 minute) ---

    def add_topic(self, topic: str, category: str = None) -> int:
        """Thêm mới 1 topic. Trả về topic_id (trả về id cũ nếu topic đã tồn tại)."""
        if not topic or not topic.strip():
            raise ValueError("Topic không được để trống!")

        clean_topic = topic.strip()
        cur = self.conn.cursor()

        # 1. Kiểm tra topic đã tồn tại chưa (không phân biệt hoa/thường)
        cur.execute("SELECT id FROM topics WHERE LOWER(topic) = LOWER(?)", (clean_topic,))
        row = cur.fetchone()

        if row:
            topic_id = row["id"]
            cur.execute("""
                UPDATE topics
                SET category = COALESCE(category, ?)
                WHERE id = ?
            """, (category, topic_id))
            self.conn.commit()
            return topic_id

        # 2. Nếu chưa có thì thêm mới
        cur.execute(
            "INSERT INTO topics (topic, category) VALUES (?, ?)",
            (clean_topic, category)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_topic(self, topic: str) -> dict | None:
        """Truy xuất thông tin 1 topic theo nội dung topic."""
        cur = self.conn.cursor()
        cur.execute("SELECT id, topic, category FROM topics WHERE topic = ?", (topic,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list_all_topics(self, category: str = None) -> list[dict]:
        """Lấy toàn bộ topic trong hệ thống, có thể lọc theo category."""
        cur = self.conn.cursor()
        if category:
            cur.execute("SELECT id, topic, category FROM topics WHERE category = ?", (category,))
        else:
            cur.execute("SELECT id, topic, category FROM topics")
        return [dict(row) for row in cur.fetchall()]

    def list_all_topic_categories(self) -> list[str]:
        """Lấy danh sách các phân loại (category) hiện có, không trùng lặp."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT DISTINCT category FROM topics
            WHERE category IS NOT NULL AND category != ''
            ORDER BY category
        """)
        return [row["category"] for row in cur.fetchall()]

    def update_topic(self, topic_id: int, topic: str = None, category: str = None):
        """Cập nhật nội dung topic và/hoặc category theo id."""
        cur = self.conn.cursor()
        if topic is not None:
            cur.execute("UPDATE topics SET topic = ? WHERE id = ?", (topic.strip(), topic_id))
        if category is not None:
            cur.execute("UPDATE topics SET category = ? WHERE id = ?", (category, topic_id))
        self.conn.commit()

    def delete_topic(self, topic_id: int):
        """Xoá 1 topic theo id."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM topics WHERE id = ?", (topic_id,))
        self.conn.commit()

    def get_random_topic_for_speaking(self, category: str = None) -> dict | None:
        """Lấy ngẫu nhiên 1 topic, có thể lọc theo category (dùng cho tab 1 minute)."""
        cur = self.conn.cursor()
        if category:
            cur.execute("""
                SELECT topic, category FROM topics
                WHERE category = ?
                ORDER BY RANDOM() LIMIT 1
            """, (category,))
        else:
            cur.execute("""
                SELECT topic, category FROM topics
                ORDER BY RANDOM() LIMIT 1
            """)
        row = cur.fetchone()
        return dict(row) if row else None

    def get_random_topics_for_speaking(self, limit: int = 5, category: str = None) -> list[dict]:
        """Lấy ngẫu nhiên tối đa 'limit' topic từ bảng topics, có thể lọc theo category."""
        cur = self.conn.cursor()
        if category:
            cur.execute("""
                SELECT topic, category
                FROM topics
                WHERE category = ?
                ORDER BY RANDOM()
                LIMIT ?
            """, (category, limit))
        else:
            cur.execute("""
                SELECT topic, category
                FROM topics
                ORDER BY RANDOM()
                LIMIT ?
            """, (limit,))
        rows = cur.fetchall()
        return [dict(row) for row in rows]

    # --- QUẢN LÝ BÀI ĐỌC (Reading) ---

    def add_reading_passage(self, title: str = None, cefr: str = None, passage: str = "", questions: list = None) -> int:
        """Thêm mới 1 bài đọc kèm bộ câu hỏi (nếu có). Trả về passage_id."""
        if not passage or not passage.strip():
            raise ValueError("Nội dung bài đọc không được để trống!")

        questions_json = json.dumps(questions, ensure_ascii=False) if questions is not None else None
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO reading_passages (title, cefr, passage, questions_json) VALUES (?, ?, ?, ?)",
            (title, cefr, passage.strip(), questions_json)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_reading_passage(self, passage_id: int) -> dict | None:
        """Truy xuất thông tin 1 bài đọc (kèm câu hỏi đã parse) theo id."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, title, cefr, passage, questions_json FROM reading_passages WHERE id = ?",
            (passage_id,)
        )
        row = cur.fetchone()
        if not row:
            return None

        questions = None
        if row["questions_json"]:
            try:
                questions = json.loads(row["questions_json"])
            except (TypeError, ValueError):
                questions = None

        return {
            "id": row["id"],
            "title": row["title"],
            "cefr": row["cefr"],
            "passage": row["passage"],
            "questions": questions
        }

    def get_reading_questions(self, passage_id: int) -> list | None:
        """Truy xuất riêng bộ câu hỏi của 1 bài đọc, nếu có."""
        cur = self.conn.cursor()
        cur.execute("SELECT questions_json FROM reading_passages WHERE id = ?", (passage_id,))
        row = cur.fetchone()
        if not row or not row["questions_json"]:
            return None
        try:
            return json.loads(row["questions_json"])
        except (TypeError, ValueError):
            return None

    def list_all_reading_passages(self, cefr: str = None) -> list[dict]:
        """Lấy toàn bộ bài đọc trong hệ thống, có thể lọc theo cấp độ CEFR."""
        cur = self.conn.cursor()
        if cefr:
            cur.execute(
                "SELECT id, title, cefr, passage, questions_json FROM reading_passages WHERE cefr = ?",
                (cefr,)
            )
        else:
            cur.execute("SELECT id, title, cefr, passage, questions_json FROM reading_passages")
        rows = cur.fetchall()

        results = []
        for row in rows:
            questions = None
            if row["questions_json"]:
                try:
                    questions = json.loads(row["questions_json"])
                except (TypeError, ValueError):
                    questions = None
            results.append({
                "id": row["id"],
                "title": row["title"],
                "cefr": row["cefr"],
                "passage": row["passage"],
                "questions": questions
            })
        return results

    def update_reading_passage(self, passage_id: int, title: str = None, cefr: str = None,
                                passage: str = None, questions: list = None):
        """Cập nhật 1 bài đọc theo id. Chỉ cập nhật những trường được truyền vào."""
        cur = self.conn.cursor()
        if title is not None:
            cur.execute("UPDATE reading_passages SET title = ? WHERE id = ?", (title, passage_id))
        if cefr is not None:
            cur.execute("UPDATE reading_passages SET cefr = ? WHERE id = ?", (cefr, passage_id))
        if passage is not None:
            cur.execute("UPDATE reading_passages SET passage = ? WHERE id = ?", (passage.strip(), passage_id))
        if questions is not None:
            questions_json = json.dumps(questions, ensure_ascii=False)
            cur.execute("UPDATE reading_passages SET questions_json = ? WHERE id = ?", (questions_json, passage_id))
        self.conn.commit()

    def delete_reading_passage(self, passage_id: int):
        """Xoá 1 bài đọc theo id."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM reading_passages WHERE id = ?", (passage_id,))
        self.conn.commit()

    def get_random_reading_passage(self, cefr: str = None) -> dict | None:
        """Lấy ngẫu nhiên 1 bài đọc, có thể lọc theo cấp độ CEFR."""
        cur = self.conn.cursor()
        if cefr:
            cur.execute("""
                SELECT id, title, cefr, passage, questions_json FROM reading_passages
                WHERE cefr = ?
                ORDER BY RANDOM() LIMIT 1
            """, (cefr,))
        else:
            cur.execute("""
                SELECT id, title, cefr, passage, questions_json FROM reading_passages
                ORDER BY RANDOM() LIMIT 1
            """)
        row = cur.fetchone()
        if not row:
            return None

        questions = None
        if row["questions_json"]:
            try:
                questions = json.loads(row["questions_json"])
            except (TypeError, ValueError):
                questions = None

        return {
            "id": row["id"],
            "title": row["title"],
            "cefr": row["cefr"],
            "passage": row["passage"],
            "questions": questions
        }

    # --- WRITING PROMPTS MANAGEMENT ---

    def add_writing_prompt(
            self,
            question_text: str,
            task_type: str = None,
            topic_category: str = None,
            difficulty: str = None,
            background_info: str = None,
            min_words: int = None,
            suggested_time_minutes: int = None,
            tags: str = None
    ) -> int:
        """Add a new writing prompt. Returns the prompt_id."""
        if not question_text or not question_text.strip():
            raise ValueError("Question text cannot be empty!")

        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO writing_prompts
            (task_type, topic_category, difficulty, question_text, background_info, min_words, suggested_time_minutes, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task_type,
            topic_category,
            difficulty,
            question_text.strip(),
            background_info,
            min_words,
            suggested_time_minutes,
            tags
        ))
        self.conn.commit()
        return cur.lastrowid

    def get_writing_prompt(self, prompt_id: int) -> dict | None:
        """Retrieve a single writing prompt by id."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT id, task_type, topic_category, difficulty, question_text,
                   background_info, min_words, suggested_time_minutes, tags
            FROM writing_prompts WHERE id = ?
        """, (prompt_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list_all_writing_prompts(self, difficulty: str = None, topic_category: str = None, task_type: str = None) -> list[dict]:
        """Retrieve all writing prompts, optionally filtered by difficulty, topic_category and/or task_type."""
        cur = self.conn.cursor()
        query = """
            SELECT id, task_type, topic_category, difficulty, question_text,
                   background_info, min_words, suggested_time_minutes, tags
            FROM writing_prompts
        """
        conditions = []
        params = []
        if difficulty:
            conditions.append("difficulty = ?")
            params.append(difficulty)
        if topic_category:
            conditions.append("topic_category = ?")
            params.append(topic_category)
        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]

    def list_all_writing_topic_categories(self) -> list[str]:
        """Retrieve the distinct list of topic_category values in the writing_prompts table."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT DISTINCT topic_category FROM writing_prompts
            WHERE topic_category IS NOT NULL AND topic_category != ''
            ORDER BY topic_category
        """)
        return [row["topic_category"] for row in cur.fetchall()]

    def list_writing_topic_categories_by_difficulty(self, difficulty: str = None) -> list[str]:
        """Retrieve the distinct topic_category values available for a given difficulty."""
        cur = self.conn.cursor()
        query = """
            SELECT DISTINCT topic_category FROM writing_prompts
            WHERE topic_category IS NOT NULL AND topic_category != ''
        """
        params = []
        if difficulty:
            query += " AND difficulty = ?"
            params.append(difficulty)
        query += " ORDER BY topic_category"
        cur.execute(query, params)
        return [row["topic_category"] for row in cur.fetchall()]

    def list_writing_task_types_by_filters(self, difficulty: str = None, topic_category: str = None) -> list[str]:
        """Retrieve the distinct task_type values available for a given difficulty + topic_category."""
        cur = self.conn.cursor()
        query = """
            SELECT DISTINCT task_type FROM writing_prompts
            WHERE task_type IS NOT NULL AND task_type != ''
        """
        conditions = []
        params = []
        if difficulty:
            conditions.append("difficulty = ?")
            params.append(difficulty)
        if topic_category:
            conditions.append("topic_category = ?")
            params.append(topic_category)
        if conditions:
            query += " AND " + " AND ".join(conditions)
        query += " ORDER BY task_type"
        cur.execute(query, params)
        return [row["task_type"] for row in cur.fetchall()]

    def update_writing_prompt(
            self,
            prompt_id: int,
            task_type: str = None,
            topic_category: str = None,
            difficulty: str = None,
            question_text: str = None,
            background_info: str = None,
            min_words: int = None,
            suggested_time_minutes: int = None,
            tags: str = None
    ):
        """Update a writing prompt by id. Only the provided fields are updated."""
        cur = self.conn.cursor()
        if task_type is not None:
            cur.execute("UPDATE writing_prompts SET task_type = ? WHERE id = ?", (task_type, prompt_id))
        if topic_category is not None:
            cur.execute("UPDATE writing_prompts SET topic_category = ? WHERE id = ?", (topic_category, prompt_id))
        if difficulty is not None:
            cur.execute("UPDATE writing_prompts SET difficulty = ? WHERE id = ?", (difficulty, prompt_id))
        if question_text is not None:
            cur.execute("UPDATE writing_prompts SET question_text = ? WHERE id = ?", (question_text.strip(), prompt_id))
        if background_info is not None:
            cur.execute("UPDATE writing_prompts SET background_info = ? WHERE id = ?", (background_info, prompt_id))
        if min_words is not None:
            cur.execute("UPDATE writing_prompts SET min_words = ? WHERE id = ?", (min_words, prompt_id))
        if suggested_time_minutes is not None:
            cur.execute("UPDATE writing_prompts SET suggested_time_minutes = ? WHERE id = ?", (suggested_time_minutes, prompt_id))
        if tags is not None:
            cur.execute("UPDATE writing_prompts SET tags = ? WHERE id = ?", (tags, prompt_id))
        self.conn.commit()

    def delete_writing_prompt(self, prompt_id: int):
        """Delete a writing prompt by id."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM writing_prompts WHERE id = ?", (prompt_id,))
        self.conn.commit()

    def list_all_writing_task_types(self) -> list[str]:
        """Retrieve the distinct list of task_type values in the writing_prompts table."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT DISTINCT task_type FROM writing_prompts
            WHERE task_type IS NOT NULL AND task_type != ''
            ORDER BY task_type
        """)
        return [row["task_type"] for row in cur.fetchall()]

    def get_random_writing_prompt(self, difficulty: str = None, topic_category: str = None) -> dict | None:
        """Retrieve one random writing prompt, optionally filtered by difficulty and/or topic_category."""
        cur = self.conn.cursor()
        query = """
            SELECT id, task_type, topic_category, difficulty, question_text,
                   background_info, min_words, suggested_time_minutes, tags
            FROM writing_prompts
        """
        conditions = []
        params = []
        if difficulty:
            conditions.append("difficulty = ?")
            params.append(difficulty)
        if topic_category:
            conditions.append("topic_category = ?")
            params.append(topic_category)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY RANDOM() LIMIT 1"

        cur.execute(query, params)
        row = cur.fetchone()
        return dict(row) if row else None

    # --- WRITING HISTORY MANAGEMENT ---

    def add_writing_history(
            self,
            question_text: str,
            essay_content: str,
            prompt_id: int = None,
            difficulty: str = None,
            task_type: str = None,
            overall_score: float = None,
            task_response_score: float = None,
            coherence_score: float = None,
            lexical_score: float = None,
            grammar_score: float = None,
            feedback: str = None,
            annotated_essay: str = None,
            model_essay: str = None,
    ) -> int:
        """Lưu 1 bản ghi kết quả bài Writing đã chấm vào lịch sử. Trả về history_id."""
        if not question_text or not essay_content:
            raise ValueError("question_text và essay_content không được để trống!")

        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO writing_history (
                prompt_id, question_text, difficulty, task_type, essay_content,
                overall_score, task_response_score, coherence_score, lexical_score, grammar_score,
                feedback, annotated_essay, model_essay, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prompt_id, question_text, difficulty, task_type, essay_content,
            overall_score, task_response_score, coherence_score, lexical_score, grammar_score,
            feedback, annotated_essay, model_essay,
            datetime.datetime.now().isoformat(timespec="seconds"),
        ))
        self.conn.commit()
        return cur.lastrowid

    def list_writing_history(self, limit: int = 50) -> list[dict]:
        """Lấy danh sách các bài Writing đã lưu vào lịch sử, mới nhất trước."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT id, prompt_id, question_text, difficulty, task_type, essay_content,
                   overall_score, task_response_score, coherence_score, lexical_score, grammar_score,
                   feedback, annotated_essay, model_essay, created_at
            FROM writing_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cur.fetchall()]

    def get_writing_history_entry(self, history_id: int) -> dict | None:
        """Lấy 1 bản ghi lịch sử Writing theo id."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT id, prompt_id, question_text, difficulty, task_type, essay_content,
                   overall_score, task_response_score, coherence_score, lexical_score, grammar_score,
                   feedback, annotated_essay, model_essay, created_at
            FROM writing_history WHERE id = ?
        """, (history_id,))
        row = cur.fetchone()
        return dict(row) if row else None


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()
    db = DatabaseManager()

    console.print("\n[bold cyan]================ BAR CHART / DATABASE SUMMARY ================ [/bold cyan]\n")

    cur = db.conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row["name"] if isinstance(row, sqlite3.Row) else row[0] for row in cur.fetchall()]

    for table_name in tables:
        cur.execute(f"PRAGMA table_info({table_name});")
        columns_info = cur.fetchall()
        column_names = [col["name"] if isinstance(col, sqlite3.Row) else col[1] for col in columns_info]

        grid_table = Table(
            title=f"📋 BẢNG DỮ LIỆU: [bold green]{table_name.upper()}[/bold green]",
            header_style="bold magenta",
            border_style="cyan",
            expand=True
        )

        for col_name in column_names:
            grid_table.add_column(col_name, style="white", overflow="fold")

        cur.execute(f"SELECT * FROM {table_name}")
        rows = cur.fetchall()

        if rows:
            for row in rows:
                row_values = []
                for col_name in column_names:
                    val = row[col_name] if isinstance(row, sqlite3.Row) else row[column_names.index(col_name)]
                    val_str = str(val) if val is not None else "-"

                    if table_name == "mp3_transcripts" and col_name == "transcript" and val:
                        words = val_str.split()
                        if len(words) > 50:
                            val_str = " ".join(words[:50]) + "..."

                    row_values.append(val_str)

                grid_table.add_row(*row_values)
        else:
            empty_row = ["(Trống)" if idx == 0 else "" for idx in range(len(column_names))]
            grid_table.add_row(*empty_row)

        console.print(grid_table)
        console.print("\n" + "─" * 60 + "\n")

    db.close()