import base64
import os
import sys

def audio_to_base64(file_path, output_path=None):
    """
    Chuyển file ghi âm (.m4v, .mp3, .wav, ...) thành chuỗi base64.
    """
    try:
        # Lấy đường dẫn tuyệt đối
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Không tìm thấy file: {abs_path}")

        # Đọc file và mã hóa
        with open(abs_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        # Nếu có yêu cầu lưu ra file .txt
        if output_path:
            abs_out = os.path.abspath(output_path)
            with open(abs_out, "w", encoding="utf-8") as out:
                out.write(encoded)
            print(f"✅ Đã lưu chuỗi Base64 vào: {abs_out}")

        return encoded
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return None


if __name__ == "__main__":
    # Cho phép truyền file qua command line
    if len(sys.argv) < 2:
        print("⚠️  Cách dùng: python base64.py <file_audio> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None

    base64_str = audio_to_base64(input_file, output_file)

    if base64_str:
        print("✅ Mã hóa thành công!")
        print("Chuỗi Base64 (rút gọn):", base64_str[:100], "...")
