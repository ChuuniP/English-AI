import sqlite3
import os
from vocabulary_module import init_database

DB_PATH = "user_study_data.db"


def reset_database():
    print("=" * 50)
    print("🔄 HỆ THỐNG KHỞI TẠO LẠI DATABASE FSRS")
    print("=" * 50)

    # Bước 1: Xóa dữ liệu cũ bằng cách drop table hoặc xóa hẳn file .db
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # Xóa bảng dữ liệu từ vựng nếu tồn tại
            cursor.execute("DROP TABLE IF EXISTS user_vocabulary")
            conn.commit()
            conn.close()
            print(f"🗑️  Đã xóa sạch bảng dữ liệu cũ trong '{DB_PATH}'.")

        except Exception as e:
            print(f"❌ Lỗi khi xóa bảng dữ liệu: {e}")
            print("Đang thử phương án xóa trực tiếp file database...")
            try:
                os.remove(DB_PATH)
                print(f"🗑️  Đã xóa trực tiếp file '{DB_PATH}'.")
            except Exception as ex:
                print(f"❌ Không thể xóa file database: {ex}")
                return
    else:
        print(f"ℹ️  Không tìm thấy file '{DB_PATH}'. Sẽ tiến hành tạo mới hoàn toàn.")

    # Bước 2: Gọi hàm khởi tạo từ vocabulary_module để làm sạch và tạo cấu trúc mới
    try:
        init_database(DB_PATH)
        print(f"✨ Đã khởi tạo lại cấu trúc database trống thành công tại '{DB_PATH}'!")
        print("💡 Bây giờ bạn đã có thể khởi chạy lại file main.py với database sạch.")
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo cấu trúc bảng mới: {e}")

    print("=" * 50)


if __name__ == "__main__":
    # Xác nhận trước khi xóa tránh người dùng bấm nhầm
    confirm = input("⚠️ CẢNH BÁO: Hành động này sẽ xóa TOÀN BỘ từ vựng đã lưu! Bạn có chắc chắn? (y/n): ")
    if confirm.strip().lower() == 'y':
        reset_database()
    else:
        print("❌ Đã hủy thao tác xóa database.")