import os
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(current_script_dir, "..", "database"))
db_folder = os.path.join(current_script_dir, "..", "database")
print("Đường dẫn đang thêm vào sys.path:", os.path.abspath(db_folder))
print("Thư mục có tồn tại không?", os.path.exists(db_folder))
print("File vocab_db.py có trong đó không?", os.path.exists(os.path.join(db_folder, "vocab_db.py")))