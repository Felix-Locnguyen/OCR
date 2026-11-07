"""
SMOKE TEST — kiểm tra nhanh pipeline trên 1 ảnh thật
=====================================================
"Smoke test" (kiếm tra khói) = chạy thử sơ bộ xem mọi thứ có HOẠT ĐỘNG
hay không, giống như cắm điện xem có khói bốc lên không.

Chạy khi:
  - Vừa cài đặt xong (xem model có tải được không)
  - Sửa code xong (xem có gì hỏng không)

  venv_paddle/bin/python smoke_test.py
"""

import sys
import time
from pathlib import Path

# Thêm thư mục chứa file này vào đường dẫn tìm kiếm (giống run_ocr.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from ocr.config import RESULTS, TEST_IMAGES
from ocr.pipeline import OcrPipeline, draw_results

# --- Chọn 1 ảnh có label để dễ đối chiếu ---
img_path = TEST_IMAGES / "im1201.jpg"
image = cv2.imread(str(img_path))
print("Ảnh:", img_path.name, "kích thước:", image.shape)
# shape = (cao, rộng, kênh màu) -> ví dụ (600, 800, 3)

# --- Bước 1: nạp model (đo thời gian tải) ---
t = time.time()
pipeline = OcrPipeline()
print(f"Khởi tạo 2 model: {time.time()-t:.1f}s\n")

# --- Bước 2: chạy OCR (đo thời gian dự đoán) ---
t = time.time()
results = pipeline.run(image)
print(f"\nOCR: {time.time()-t:.2f}s -> {len(results)} khung")

print("\n--- DỰ ĐOÁN ---")
for r in results:
    print(f"  {r.text!r}")
#   !r = dùng repr(): in trong nháy đơn, hiện cả khoảng trắng ẩn
#        (dễ phát hiện lỗi thừa/thiếu khoảng trắng)

# --- Bước 3: lưu ảnh kết quả để nhìn bằng mắt ---
RESULTS.mkdir(parents=True, exist_ok=True)
out = RESULTS / "smoke_im1201.jpg"
cv2.imwrite(str(out), draw_results(image, results))
print("\nẢnh kết quả:", out)

# --- Bước 4: đối chiếu với ĐÁP ÁN ĐÚNG (ground truth) ---
from ocr.evaluate import read_labels

print("\n--- GROUND TRUTH (đáp án đúng) ---")
for _, txt in read_labels(img_path.name):
    print(f"  {txt!r}")
# So sánh 2 bảng trên để tự đánh giá model đọc đúng/sai ở đâu.
