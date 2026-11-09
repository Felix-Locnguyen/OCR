"""
GIAO DIỆN GRADIO CHO DỰ ÁN OCR
================================
Đây là "lớp vỏ" đồ họa (GUI) bọc quanh pipeline OCR có sẵn.

BỐ CỤC GRID 2x2:
  ┌───────────────────────┬───────────────────────┐
  │ Ô 1: ẢNH ĐẦU VÀO      │ Ô 2: ẢNH NHẬN DIỆN    │
  │ (chọn xong tự chạy)   │ (vẽ khung chữ)        │
  ├───────────────────────┼───────────────────────┤
  │ Ô 3: KÝ TỰ NHẬN DIỆN  │ Ô 4: THÔNG TIN + LƯU  │
  │ (text đọc được)       │ [Lưu ảnh] [Lưu JSON]  │
  └───────────────────────┴───────────────────────┘

NGUYÊN TẮC QUAN TRỌNG:
  app.py CHỈ làm nhiệm vụ hiển thị + gọi hàm.
  Toàn bộ logic OCR nằm ở package `ocr/` (Detector, Recognizer, OcrPipeline).
  -> Nhờ vậy giao diện và dòng lệnh LUÔN cho kết quả giống nhau,
     và bạn chỉ có 1 nơi để sửa khi bug.

CÁCH CHẠY:
  venv_paddle/bin/python app.py
  Sau đó mở trình duyệt tại http://127.0.0.1:7860
"""

import json 
import os
import sys
import time 
from pathlib import Path

import cv2
import gradio as gr

# Thêm thư mục chứa file này vào đường dẫn tìm kiếm để import được package ocr
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr.config import RESULTS
from ocr.pipeline import OcrPipeline, draw_results

# ---------------------------------------------------------------------------
# 0. ZEROGPU — decorator @spaces.GPU
# ---------------------------------------------------------------------------
# Space đang chạy hardware zero-a10g (ZeroGPU). Quy tắc của Hugging Face:
# Space ZeroGPU PHẢI có ít nhất MỘT hàm gắn @spaces.GPU, nếu không app sẽ bị
# terminate lúc startup với lỗi:
#   "No @spaces.GPU function detected during startup"
# Thật ra code mình vẫn chạy CPU (config.py ép CPU khi có env SPACE) —
# decorator chỉ để thỏa mãn quy tắc của platform.
#
# Máy local KHÔNG cài thư viện `spaces` -> fallback: nếu import thất bại thì
# dùng decorator "vô hại" (trả về nguyên hàm) để app vẫn chạy local được.
try:
    import spaces
except Exception as _e:
    print(f"[WARN] Không import được thư viện spaces ({_e}) -> dùng fallback.")
    class spaces:                    # shim khi chạy local (không có ZeroGPU)
        @staticmethod
        def GPU(fn=None, **kwargs):
            if fn is None:           # dùng dạng @spaces.GPU(...) hoặc @spaces.GPU
                return lambda f: f
            return fn


# ---------------------------------------------------------------------------
# 1. MODEL — nạp MỘT LẦN DUY NHẤT
# ---------------------------------------------------------------------------
# Đây là điểm dễ sai nhất của ứng dụng OCR trên web.
#
#   SAI:  def chay(img):  pipeline = OcrPipeline()   <- mỗi lần upload ảnh
#                          ^ tải lại 2 model (3-80 giây) -> app chậm khủng khiếp
#
#   ĐÚNG: tạo MỘT instance dùng chung cho mọi request.

_pipeline: OcrPipeline | None = None   # None = chưa tải


def get_pipeline() -> OcrPipeline:
    """Trả về instance model duy nhất."""
    global _pipeline
    if _pipeline is None:
        print("Đang tải model (có thể mất khoảng 5-80s)")
        _pipeline = OcrPipeline()
        print("Model đã sẵn sàng")
    return _pipeline


# ---------------------------------------------------------------------------
# 2. HÀM XỬ LÝ CHÍNH — gọi tự động khi người dùng đưa ảnh vào
# ---------------------------------------------------------------------------
@spaces.GPU             # ZeroGPU: bắt buộc có @spaces.GPU (xem mục 0 ở trên)
def run_ocr(image_path: str | None):
    """
    Nhận ảnh vào -> chạy pipeline -> trả về 5 giá trị cho 5 component.

    Args:
        image_path: đường dẫn file ảnh Gradio nhận được
                    (Gradio tự lưu file upload vào thư mục tạm rồi truyền path)

    Returns:
        (ảnh kết quả, thông tin, text, nút_lưu_ảnh, nút_lưu_json)
        THỨ TỰ này phải khớp hệt với outputs=[...] ở dưới cùng file.
    """

    # --- Kiểm tra đầu vào ---
    # image_path là None khi người dùng chưa chọn ảnh nào
    if not image_path:
        # gr.Warning hiện popup cảnh báo trên giao diện
        # (không ném exception -> giao diện không bị sập)
        gr.Warning("Vui lòng chọn ảnh trước!")
        return (None, "**Chưa có ảnh**", "",
                gr.DownloadButton(value=None, interactive=False),
                gr.DownloadButton(value=None, interactive=False))

    image = cv2.imread(image_path)
    if image is None:
        gr.Warning(f"Không đọc được file ảnh: {image_path}")
        return (None, "**Lỗi đọc ảnh**", "",
                gr.DownloadButton(value=None, interactive=False),
                gr.DownloadButton(value=None, interactive=False))

    h, w = image.shape[:2]           # shape = (cao, rộng, kênh) -> lấy 2 đầu

    # --- Chạy OCR và đo thời gian ---
    pipeline = get_pipeline()
    start = time.time()
    results = pipeline.run(image)
    elapsed = time.time() - start

    # --- Ô 2: ảnh đã vẽ khung ---
    # draw_results() là hàm CÓ SẴN trong ocr/pipeline.py
    # -> không viết lại, đảm bảo kết quả giống hệt khi chạy dòng lệnh
    annotated = draw_results(image, results)

    # Lưu ảnh kết quả để nút "Lưu ảnh nhận diện" tải về
    out_dir = RESULTS / "ui"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_dir / f"result_{Path(image_path).name}"
    cv2.imwrite(str(out_img_path), annotated)

    # --- Ô 3: text ---
    # Mỗi dòng = 1 khung chữ. join với "\n" để hiển thị nhiều dòng
    text = "\n".join(r.text for r in results) if results else "(không tìm thấy chữ)"

    # --- Thông tin kèm theo (hiển thị ở Ô 4) ---
    # Markdown: 2 khoảng trắng + \n = xuống dòng trong Markdown
    info = (
        f"**Kích thước ảnh:** {w}×{h} px  \n"
        f"**Số khung chữ:** {len(results)}  \n"
        f"**Thời gian xử lý:** {elapsed:.2f} giây"
    )

    # --- File JSON để nút "Lưu JSON" tải về ---
    # r.box là numpy array -> JSON không đọc được numpy -> phải .tolist()
    data = {
        "image": Path(image_path).name,
        "num_boxes": len(results),
        "results": [{"box": r.box.tolist(), "text": r.text} for r in results],
    }
    json_path = out_dir / f"result_{Path(image_path).stem}.json"
    # .stem = tên file KHÔNG đuôi ("im1201.jpg" -> "im1201")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    # ensure_ascii=False: giữ nguyên tiếng Việt thay vì escape \u1e00...

    # Trả về theo THỨ TỰ khai báo trong outputs=[...] ở dưới.
    # DownloadButton: gán value=đường_dẫn_file và bật interactive=True
    return (
        annotated,
        info,
        text,
        gr.DownloadButton(value=str(out_img_path), interactive=True),
        gr.DownloadButton(value=str(json_path), interactive=True),
    )


# ---------------------------------------------------------------------------
# 3. XÂY DỰNG GIAO DIỆN — gr.Blocks (GRID 2x2)
# ---------------------------------------------------------------------------
# Grid = 2 hàng gr.Row, mỗi hàng 2 gr.Column.
#   equal_height=True : 2 ô cùng hàng cao bằng nhau
#   scale=1           : 2 cột chia đều chiều rộng
#   min_width=0       : không bị rớt xuống dòng khi cửa sổ hẹp
#   variant="panel"   : nền + viền dạng card
with gr.Blocks(title="OCR tiếng Việt") as demo:

    # --- TIÊU ĐỀ ---
    gr.Markdown(
        "# OCR tiếng Việt\n"
        "**PaddleOCR** tìm khung chữ → **VietOCR** đọc chữ tiếng Việt"
    )

    # ================= HÀNG 1: ẢNH VÀO | ẢNH NHẬN DIỆN =================
    with gr.Row(equal_height=True):

        # ---- Ô 1: ẢNH ĐẦU VÀO ----
        with gr.Column(scale=1, min_width=0, variant="panel"):
            gr.Markdown("###  Ảnh đầu vào")
            input_image = gr.Image(
                type="filepath",       # trả về ĐƯỜNG DẪN file (không phải numpy)
                label="Kéo thả hoặc bấm để chọn ảnh (chọn xong tự chạy OCR)",
                height=380,
            )

        # ---- Ô 2: ẢNH NHẬN DIỆN ----
        with gr.Column(scale=1, min_width=0, variant="panel"):
            gr.Markdown("### 🔍 Ảnh nhận diện")
            output_image = gr.Image(
                type="numpy",          # trả về mảng numpy để hiển thị trực tiếp
                label="Ảnh có khung chữ",
                height=380,
            )

    # ================= HÀNG 2: KÝ TỰ | THÔNG TIN + NÚT LƯU =================
    with gr.Row(equal_height=True):

        # ---- Ô 3: KÝ TỰ NHẬN DIỆN ĐƯỢC ----
        with gr.Column(scale=1, min_width=0, variant="panel"):
            gr.Markdown("### Ký tự nhận diện được")
            output_text = gr.Textbox(
                label="Kết quả",
                lines=10,              # số dòng hiển thị
                max_lines=20,
                placeholder="Chữ đọc được sẽ hiển thị ở đây...",
                buttons=["copy"],      # nút Copy (Gradio 6)
            )

        # ---- Ô 4: THÔNG TIN + NÚT LƯU ----
        with gr.Column(scale=1, min_width=0, variant="panel"):
            gr.Markdown("### Thông tin & Lưu kết quả")
            info_box = gr.Markdown("Chưa chạy")   # info: số khung, thời gian

            # DownloadButton: bấm là tải file về máy.
            # interactive=False: bị mờ cho tới khi OCR xong.
            with gr.Row():
                save_image_btn = gr.DownloadButton(
                    "Lưu ảnh nhận diện",
                    interactive=False,
                )
                save_json_btn = gr.DownloadButton(
                    "Lưu JSON",
                    interactive=False,
                )

    # -----------------------------------------------------------------------
    # 4. NỐI SỰ KIỆN — chạy OCR NGAY khi có ảnh được đưa vào
    # -----------------------------------------------------------------------
    # .upload : kích hoạt khi người dùng kéo thả / chọn file mới.
    #           (khác .change: .change còn chạy cả khi xóa ảnh -> hiện cảnh báo thừa)
    #   inputs : component ĐƯA GIÁ TRỊ vào hàm
    #   outputs: component NHẬN KẾT QUẢ (thứ tự phải khớp return của hàm)
    input_image.upload(
        fn=run_ocr,
        inputs=input_image,
        outputs=[output_image, info_box, output_text,
                 save_image_btn, save_json_btn],
    )


# ---------------------------------------------------------------------------
# 5. KHỞI CHẠY SERVER
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Preload model NGAY khi khởi động thay vì chờ lần upload đầu tiên.
    #   -> Người dùng mở trang là app đã sẵn sàng, không phải chờ tải model.
    get_pipeline()

    demo.launch(
        server_name="0.0.0.0",   # 0.0.0.0 = nghe TẤT CẢ mạng (vào từ máy khác
                                 #   trong cùng wifi được; "127.0.0.1" thì chỉ
                                 #   máy này truy cập được)
        # HF Space cấp sẵn PORT riêng; chạy máy bạn thì dùng 7860 (mặc định)
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,             # True sẽ tạo link công khai *.gradio.live
    )