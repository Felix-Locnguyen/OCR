"""
CẤU HÌNH CHUNG CHO CẢ DỰ ÁN
============================
File này chứa "thông số" (đường dẫn, tên model, ngưỡng...) để khi muốn
chỉnh gì bạn chỉ cần sửa MỘT chỗ thay vì đi tìm trong nhiều file.

Đây là kỹ thuật gọi là "centralized configuration" — rất nên làm vì:
  - Dễ tìm, dễ sửa
  - Tránh sửa nhầm (bug phổ biến: sửa file này quên file kia)
  - Khi chia sẻ project, người khác biết ngay đang dùng tham số gì
"""

# pathlib.Path: class thay cho cách nối chuỗi os.path.join('/a', 'b')
# Path giúp đường dẫn hoạt động đúng trên Windows, Linux, macOS mà không cần sửa.
from pathlib import Path

# ---------------------------------------------------------------------------
# ĐƯỜNG DẪN
# ---------------------------------------------------------------------------
# __file__  = đường dẫn file config.py này đang chạy
# .resolve() = biến thành đường dẫn ABSOLUTE (ví dụ: nếu đang chạy từ thư mục
#              khác, nó vẫn trỏ đúng vào chỗ thật, không bị nhầm)
# .parents[1] = nhấc lên 1 cấp thư mục cha
#
# Cấu trúc thực tế:
#   OCR/
#     ocr/            <- parents[0] (thư mục chứa config.py)
#       config.py
#     README.md       <- parents[1] = ROOT (thư mục gốc dự án)
#     Data/
#
# Nhờ vậy bạn chạy file từ BẤT KỲ thư mục nào cũng không bị lỗi đường dẫn.
ROOT = Path(__file__).resolve().parents[1]

# Dữ liệu của dự án
DATA = ROOT / "Data" / "vietnamese_original"
TRAIN_IMAGES = DATA / "train_images"           # 1200 ảnh để train (không dùng ở đây)
TEST_IMAGES = DATA / "test_image"              # 300 ảnh CÓ label -> để ĐÁNH GIÁ
UNSEEN_IMAGES = DATA / "unseen_test_images"    # 500 ảnh CHƯA có label -> chạy thử
LABELS = DATA / "labels"                       # các file gt_1.txt ... gt_2000.txt

# Nơi lưu ảnh kết quả + file JSON
RESULTS = ROOT / "Data" / "results"

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
# Tên model detection của PaddleOCR.
#   - "_det"   : chỉ bước TÌM KHUNG chữ (detection)
#   - "_mobile": bản nhỏ, chạy nhanh (bản "_server" to hơn, chính xác hơn)
# PaddleOCR sẽ tự tải file model về máy trong lần chạy đầu tiên (cần mạng).
DET_MODEL = "PP-OCRv5_mobile_det"

# Model nhận diện chữ của VietOCR. Có 2 lựa chọn:
#   - "vgg_transformer" : chính xác hơn, nhưng chậm hơn (~86ms/ảnh)
#   - "vgg_seq2seq"     : nhanh hơn (~12ms/ảnh), kém chính xác hơn đôi chút
# Thử đổi sang "vgg_seq2seq" để cảm nhận sự khác biệt về tốc độ.
VIETOCR_CONFIG = "vgg_transformer"

# Thiết bị chạy: "cuda:0" = GPU thứ 0. Đổi thành "cpu" nếu máy không có GPU.
VIETOCR_DEVICE = "cuda:0"

# ---------------------------------------------------------------------------
# NGƯỠNG
# ---------------------------------------------------------------------------
# IoU (Intersection over Union) = độ chồng lấn giữa 2 khung chữ.
#   IoU = 1.0 -> 2 khung trùng nhau hoàn toàn
#   IoU = 0.0 -> không hề chồng lấn
# Khi đánh giá, ta coi dự đoán "ĐÚNG" nếu IoU >= 0.5 với ground truth.
IOU_THRESHOLD = 0.5
