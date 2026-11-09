"""
BƯỚC 1 — TÌM KHUNG CHỮ (Text Detection)
=========================================
Nhiệm vụ: đưa vào 1 ảnh, trả về các KHUNG (box) bao quanh vùng có chữ.

  Ảnh ──► PaddleOCR ──► [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] cho mỗi vùng chữ
                         (4 điểm tạo thành 1 tứ giác, có thể nghiêng)

Sau đó ta CẮT từng khung ra thành ảnh riêng để đưa sang bước đọc chữ.
"""

import cv2
import numpy as np

# TextDetection: lớp detection của PaddleOCR 3.x
# (Khác với lớp PaddleOCR đầy đủ — lớp đó làm cả detection lẫn recognition,
#  nhưng ở đây mình chỉ cần detection nên dùng lớp con cho nhẹ, chạy nhanh hơn)
from paddleocr import TextDetection

from .config import DET_MODEL

# Dấu "." phía trước nghĩa là "import từ package hiện tại"
# -> mình đang ở trong package ocr/, nên ".config" = ocr/config.py


class Detector:
    """Bọc model PaddleOCR thành class gọn để dễ dùng ở nơi khác."""

    def __init__(self, model_name=DET_MODEL, device="gpu"):
        """Khởi tạo: tải model về RAM (mất vài giây, chỉ làm 1 lần).

        Args:
            model_name: tên model, xem ở config.py
            device: "gpu" dùng card đồ họa, "cpu" dùng bộ xử lý thường
        """
        # Lần đầu chạy, PaddleOCR tự tải file model về máy (cần có mạng)
        self.model = TextDetection(model_name=model_name, device=device)

    def detect(self, image_bgr) -> list[np.ndarray]:
        """Tìm các khung chữ trong ảnh.

        Args:
            image_bgr: ảnh dạng numpy array (OpenCV đọc ảnh theo thứ tự BGR,
                       nghĩa là kênh xanh dương - xanh lá - đỏ)

        Returns:
            Danh sách các mảng numpy shape (4, 2) — 4 điểm (x, y) mỗi khung.
            (Thứ tự đọc sẽ được sắp ở OcrPipeline.run — chỉ cần 1 chỗ.)
        """
        # predict() trả về LIST kết quả (mỗi phần tử cho 1 ảnh đầu vào).
        # Vì mình chỉ truyền 1 ảnh nên lấy phần tử [0].
        result = self.model.predict(image_bgr)[0]

        # "dt_polys" = detection text polygons: các tứ giác tìm được.
        # dtype=int32 vì tọa độ pixel phải là số nguyên.
        boxes = np.array(result["dt_polys"], dtype=np.int32)

        # Không có chữ nào trong ảnh -> trả về list rỗng
        return list(boxes)

    @staticmethod
    def crop(image_bgr, box) -> np.ndarray:
        """Cắt 1 khung chữ ra ảnh chữ nhật riêng.

        Tại sao cần làm việc này?
          Khung tìm được có thể NGHIÊNG (chữ trong ảnh chụp nghiêng).
          Nếu chỉ cắt hình chữ nhật bình thường thì chữ sẽ bị méo.
          -> Giải pháp: dùng Perspective Transform, "duỗi thẳng" khung nghiêng
             thành ảnh chữ nhật ngay ngắn.

        Args:
            image_bgr: ảnh gốc
            box: 4 điểm của khung, shape (4, 2)

        Returns:
            Ảnh đã cắt (đuôi theo chiều rộng x chiều cao), hoặc None nếu khung quá nhỏ.
        """
        box = np.array(box, dtype=np.float32)

        # Tính kích thước ảnh đầu ra = chiều dài cạnh dài nhất của tứ giác.
        # np.linalg.norm(a - b) = khoảng cách từ điểm a đến điểm b.
        #
        # Giả sử thứ tự 4 điểm là:
        #   0 ── 1        box[0]-box[1]: cạnh trên
        #   │    │        box[3]-box[2]: cạnh dưới
        #   3 ── 2        box[0]-box[3]: cạnh trái
        #                 box[1]-box[2]: cạnh phải
        width = int(max(np.linalg.norm(box[0] - box[1]),   # cạnh trên
                        np.linalg.norm(box[3] - box[2])))  # cạnh dưới
        height = int(max(np.linalg.norm(box[0] - box[3]),  # cạnh trái
                         np.linalg.norm(box[1] - box[2]))) # cạnh phải

        # Khung quá nhỏ (1-2 pixel) thì vô nghĩa, bỏ qua
        if width < 2 or height < 2:
            return None

        # "Ảnh đích" = hình chữ nhật width x height mà ta muốn khung nghiêng
        # được duỗi vào. Bắt đầu tại (0,0), cùng chiều với box.
        dst = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]],
            dtype=np.float32,
        )

        # Tính MA TRẬN biến đổi: tìm phép chiếu biến đổi tuyến tính hình học
        # sao cho 4 điểm của box rơi đúng vào 4 điểm của dst.
        matrix = cv2.getPerspectiveTransform(box, dst)

        # Áp phép biến đổi lên cả ảnh -> vùng chữ được "duỗi thẳng" ra
        return cv2.warpPerspective(image_bgr, matrix, (width, height))
