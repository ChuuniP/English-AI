"""
Script thêm 10 topic mẫu vào bảng 'topics' trong database.
Cột 'category' dùng để lưu ĐỘ KHÓ của topic (Easy / Medium / Hard).
Chạy: python seed_topics.py
"""

from database_manager import DatabaseManager

# Danh sách 10 topic mẫu: (nội dung topic, độ khó)
# Độ khó gồm 3 mức: "Easy", "Medium", "Hard"
TOPICS_TO_ADD = [
    ("Talk about your favorite hobby", "Easy"),
    ("Describe your favorite food", "Easy"),
    ("What do you usually do on weekends?", "Easy"),
    ("Describe your hometown", "Medium"),
    ("Talk about a memorable trip you took", "Medium"),
    ("Describe your best friend", "Medium"),
    ("Talk about a book or movie you enjoyed", "Medium"),
    ("What is your dream job?", "Hard"),
    ("What are your plans for the future?", "Hard"),
    ("Talk about an important lesson you learned in life", "Hard"),
]


def main():
    db = DatabaseManager()
    for topic, level in TOPICS_TO_ADD:
        topic_id = db.add_topic(topic, level)
        print(f"[OK] id={topic_id} | level={level} | topic={topic}")
    db.close()
    print(f"\nĐã thêm/kiểm tra xong {len(TOPICS_TO_ADD)} dòng vào bảng topics.")


if __name__ == "__main__":
    main()