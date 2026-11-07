"""
BƯỚC 2 — NHẬN DIỆN CHỮ (Text Recognition)
===========================================
Nhiệm vụ: đưa vào 1 ảnh khung chữ (đã cắt ở bước 1), trả về CHUỖI KÝ TỰ.

  Ảnh "CÀ PHÊ" ──► VietOCR ──► "CÀ PHÊ"

VietOCR là model tiếng Việt pretrained (đã được huấn luyện sẵn trên 10 triệu ảnh),
nên mình KHÔNG cần train lại — chỉ cần gọi là dùng.
"""

import cv2

# PIL (Python Imaging Library) — thư viện xử lý ảnh khác của Python.
#   OpenCV  : đọc ảnh bằng cv2, thứ tự kênh BGR
#   PIL     : dùng cho ảnh bitmap, thứ tự kênh RGB
# VietOCR viết bằng PIL nên mình phải chuyển sang PIL trước khi đưa vào.
from PIL import Image

# Cfg: lớp đọc file cấu hình (hyperparameter) của VietOCR
# Predictor: lớp chạy dự đoán — nhận ảnh vào, trả text ra
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

from .config import VIETOCR_CONFIG, VIETOCR_DEVICE


class Recognizer:
    """Bọc VietOCR thành class gọn."""

    def __init__(self, config_name=VIETOCR_CONFIG, device=VIETOCR_DEVICE):
        """Khởi tạo: tải config + trọng số (weights) của model.

        Args:
            config_name: tên cấu hình ("vgg_transformer" hoặc "vgg_seq2seq")
            device: "cuda:0" (GPU) hoặc "cpu"

        Lưu ý: lần đầu chạy sẽ tải weights từ vocr.vn (cần mạng,
        khoảng vài trăm MB). Lần sau dùng lại file đã tải.
        """
        # Tải file cấu hình từ vocr.vn — chứa các thông số như:
        #   - kiến trúc model (backbone VGG19, lớp Transformer...)
        #   - kích thước ảnh đầu vào
        #   - đường dẫn file trọng số
        self.config = Cfg.load_config_from_name(config_name)

        # Chỉ định nơi chạy model. VietOCR đọc giá trị này để đưa tensor lên GPU.
        self.config["device"] = device

        # Tải trọng số (weights) đã train sẵn rồi nạp vào model.
        # Predictor tự xử lý: đọc config -> dựng model -> load weights.
        self.predictor = Predictor(self.config)

    def read(self, image_bgr) -> str:
        """Đọc 1 ảnh khung chữ -> chuỗi văn bản.

        Args:
            image_bgr: ảnh khung chữ (do Detector.crop trả về), dạng BGR

        Returns:
            Chuỗi tiếng Việt, đã bỏ khoảng trắng thừa ở 2 đầu
        """
        # Chỗ này phòng hờ: ảnh bị lỗi/cắt rỗng thì trả về chuỗi rỗng
        # thay vì để chương trình sập.
        if image_bgr is None or image_bgr.size == 0:
            return ""

        # ĐỔI THỨ TỰ KÊNH MÀU:
        #   OpenCV lưu ảnh thứ tự BGR (Blue-Green-Red)
        #   PIL/most model mong đợi RGB (Red-Green-Blue)
        #   Nếu KHÔNG đổi thì màu bị đảo -> model nhận diện kém đi nhiều.
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Image.fromarray(): chuyển numpy array -> đối tượng PIL Image
        # predictor.predict(): chạy model, trả về chuỗi ký tự
        text = self.predictor.predict(Image.fromarray(rgb))

        # .strip() bỏ khoảng trắng đầu/cuối (VD: "  hello  " -> "hello")
        return text.strip()

    def read_batch(self, images: list) -> list[str]:
        """Đọc NHIỀU ảnh một lúc — nhanh hơn nhiều so gọi lẻ từng ảnh.

        Tại sao nhanh hơn?
          Khi gọi lẻ, mỗi ảnh chạy 1 lần -> GPU hoạt động không hết công suất.
          Khi gộp batch, GPU xử lý nhiều ảnh CÙNG LÚC (tận dụng song song).

        Args:
            images: list ảnh BGR

        Returns:
            List text, THỨ TỰ GIỐNG hệt list input (rất quan trọng để nối
            kết quả đúng với khung chữ tương ứng)
        """
        # Lọc bỏ ảnh lỗi, đồng thời đổi sang PIL
        pil_images = [
            Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            for im in images
            if im is not None and im.size > 0
        ]
        if not pil_images:
            return []

        # predict_batch(): xử lý cả list, trả về list text cùng thứ tự input
        texts = self.predictor.predict_batch(pil_images)
        return [str(t).strip() for t in texts]
