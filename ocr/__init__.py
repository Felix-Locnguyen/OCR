"""
GÓI OCR — PaddleOCR (tìm khung) + VietOCR (đọc chữ)
====================================================
"Package" = thư mục chứa file .py có file __init__.py bên trong.
Khi có __init__.py, Python nhận diện "ocr" là 1 gói và cho import như:

    from ocr.pipeline import OcrPipeline

File __init__.py thường rất ngắn: chỉ khai báo những gì người dùng
CẦN dùng từ ngoài (như "cửa ngõ" của gói).
"""

# Import từ các file con để người dùng không phải nhớ đường dẫn chi tiết.
# Thay vì:  from ocr.pipeline import OcrPipeline, OcrResult, draw_results
# thì chỉ cần:  from ocr import OcrPipeline
from .pipeline import OcrPipeline, OcrResult, draw_results

# __all__: danh sách CHÍNH THỨC những tên được export.
#   from ocr import *  -> chỉ import các tên trong list này.
#   Không bắt buộc, nhưng giúp người đọc biết "gói này cung cấp gì".
__all__ = ["OcrPipeline", "OcrResult", "draw_results"]
