import os
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

-- 4. Bảng lưu trữ transcript file nghe
CREATE TABLE IF NOT EXISTS mp3_transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mp3_name TEXT NOT NULL UNIQUE,
    transcript TEXT
);
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