"""
BƯỚC 3 — GHÉP 2 BƯỚC LẠI THÀNH PIPELINE
=========================================
Pipeline = "đường ống": dữ liệu chảy qua nhiều công đoạn nối tiếp nhau.

  Ảnh gốc ──► [Detector: tìm khung] ──► [cắt crop] ──► [Recognizer: đọc chữ] ──► kết quả

File này làm 2 việc:
  1. OcrPipeline    : điều phối 2 bước trên
  2. draw_results   : vẽ khung + chữ lên ảnh để nhìn trực quan
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Detector
from .recognizer import Recognizer


@dataclass
class OcrResult:
    """Kết quả OCR cho MỘT khung chữ.

    @dataclass (data class) tự动生成 __init__, __repr__ cho bạn, nên mình chỉ
    cần khai báo TRƯỜNG là xong. Nếu viết thủ công sẽ dài dòng hơn nhiều.

    Thay vì trả 2 giá trị rời rạc (box, text) rồi lo bị nhầm chỗ,
    ta gộp vào 1 đối tượng cho gọn và dễ đọc code hơn.
    """
    box: np.ndarray      # 4 điểm (x, y), shape (4, 2)
    text: str            # chữ đọc được từ khung này

    @property
    def center(self):
        """Toạ độ tâm của khung.

        @property biến phương thức thành "thuộc tính" — dùng r.center
        chứ không phải r.center() cho tự nhiên hơn.

        box.mean(axis=0) = lấy trung bình theo TRỤC 0 (cột):
          box = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
          mean(axis=0) = [(x1+x2+x3+x4)/4, (y1+y2+y3+y4)/4] = tâm
        """
        return self.box.mean(axis=0)


class OcrPipeline:
    """Điều phối toàn bộ: ảnh vào -> list kết quả text ra."""

    def __init__(self, use_batch=True):
        """
        Args:
            use_batch: True = đọc chữ theo lô (nhanh),
                       False = đọc từng ảnh một (dễ debug, chậm hơn)
        """
        # Khởi tạo 2 model. Mỗi lần chạy chương trình chỉ làm việc này 1 lần,
        # sau đó tái sử dụng cho nhiều ảnh.
        print("Đang tải model PaddleOCR (detection)...")
        self.detector = Detector()

        print("Đang tải model VietOCR (recognition)...")
        self.recognizer = Recognizer()

        self.use_batch = use_batch

    def run(self, image_bgr) -> list[OcrResult]:
        """Chạy 2 bước lên 1 ảnh.

        Returns:
            list[OcrResult], đã sắp xếp theo thứ tự đọc (trên xuống, trái sang phải)
        """
        # --- BƯỚC 1: tìm các khung chữ ---
        boxes = self.detector.detect(image_bgr)
        if not boxes:
            return []              # ảnh không có chữ nào

        # Cắt mỗi khung ra ảnh riêng
        crops = [self.detector.crop(image_bgr, b) for b in boxes]

        if self.use_batch:
            # VẤN ĐỀ: có thể có crop bị lỗi (trả về None) -> không đọc được.
            # Nhưng mình PHẢI giữ đúng vị trí của nó trong list,
            # nếu không thì text sẽ bị NỐI SAI với khung tương ứng!
            #
            # Giải pháp: ghi lại index (vị trí) gốc của crop còn tốt,
            #           chỉ đưa crop tốt đi đọc, rồi ghép lại theo index.
            valid = [(i, c) for i, c in enumerate(crops) if c is not None]
            texts = self.recognizer.read_batch([c for _, c in valid])

            # zip gộp index với text tương ứng -> khôi phục đúng khung
            result = [OcrResult(boxes[i], t) for (i, _), t in zip(valid, texts)]
        else:
            # Đường đơn giản: bỏ qua crop lỗi, chạy từng cái
            result = [
                OcrResult(b, self.recognizer.read(c))
                for b, c in zip(boxes, crops)
                if c is not None
            ]

        # SẮP XẾP LẠI theo thứ tự đọc:
        #   round(y / 30) = gom vào "hàng" (cứ 30px 1 hàng)
        #   sau đó theo x (trái sang phải)
        return sorted(
            result,
            key=lambda r: (round(r.center[1] / 30), r.center[0]),
        )


def draw_results(image_bgr, results: list[OcrResult]) -> np.ndarray:
    """Vẽ khung chữ (xanh lá) + text (đỏ) lên ảnh để dễ kiểm tra.

    Đây là bước "visualization" — rất hữu ích khi muốn xem model
    tìm sai chỗ nào, đọc sai chữ nào.

    Returns:
        ẢNH MỚI (bản copy), ảnh gốc không bị thay đổi.
    """
    # .copy() để không ghi đè lên ảnh gốc (phòng khi cần dùng lại)
    canvas = image_bgr.copy()

    for r in results:
        # polylines: vẽ đường gấp khúc đóng kín (khung tứ giác)
        #   [r.box.astype(np.int32)] : np cần 2D array -> bọc trong []
        #   True : đường ĐÓNG kín (nối điểm cuối về điểm đầu)
        #   (0,255,0) : màu XANH LÁ (OpenCV dùng BGR, không phải RGB)
        #   2 : độ dày nét vẽ (pixel)
        cv2.polylines(canvas, [r.box.astype(np.int32)], True, (0,255,0), 2)

        # Đặt chữ phía TRÊN khung một chút:
        #   y nhỏ nhất của khung, trừ 4px
        x = int(r.box[:, 0].min())          # x nhỏ nhất (cột 0 = toạ độ x)
        y = int(r.box[:, 1].min()) - 4      # y nhỏ nhất (cột 1 = toạ độ y)

        # cv2.putText: viết chữ lên ảnh
        #   max(y, 12) : nếu y < 12 thì chữ sẽ bị cắt mất khỏi ảnh
        #               -> dùng max để giữ nó luôn nằm trong khung nhìn thấy
        cv2.putText(
            canvas, r.text, (x, max(y, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,   # kiểu chữ
            0.5,                        # cỡ chữ
            (0, 0, 255),                # màu ĐỎ (thứ tự BGR)
            1,                          # độ dày
            cv2.LINE_AA,                # làm mịn nét chữ
        )

    return canvas
