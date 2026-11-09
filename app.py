"""
GIAO DIỆN GRADIO CHO DỰ ÁN OCR
================================
Đây là "lớp vỏ" đồ họa (GUI) bọc quanh pipeline OCR có sẵn.

CẤU TRÚC GIAO DIỆN — 3 CARD:
  Card 1 (INPUT)     : chọn / kéo thả ảnh vào
  Card 2 (DETECTION) : ảnh đã được vẽ khung chữ
  Card 3 (TEXT)      : danh sách chữ model đọc được

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
import sys
import time 
from pathlib import Path

import cv2
import gradio as gr

# Thêm thư mục chứa file nầy vào đường dẫn tìm kiếm để import được package ocr
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr.config import RESULTS, TEST_IMAGES
from ocr.pipeline import OcrPipeline, draw_results


# 1. MODEL — nạp MỘT LẦN DUY NHẤT
# Đây là điểm dễ sai nhất của ứng dụng OCR trên web.
#
#   SAI:  def chay(img):  pipeline = OcrPipeline()   <- mỗi lần bấm nút
#                          ^ tải lại 2 model (3-80 giây) -> app chậm khủng khiếp
#
#   ĐÚNG: tạo MỘT instance dùng chung cho mọi request.


_pipeline: OcrPipeline | None = None   # None = chưa tải
def get_pipeline() -> OcrPipeline:
    """Trả về instace model duy nhất"""
    global _pipeline
    if _pipeline is None:
        print("Đang tải model (có thể mất khoản 5-80s)")
        _pipeline = OcrPipeline()
        print("Model đã sẵn sàng")
    return _pipeline


# 2. HÀM XỬ LÝ CHÍNH — gọi khi bấm nút "Chạy OCR"
def run_ocr(image_path: str | None):
    """
    Nhận ảnh vào -> chạy pipeline -> trả về 5 giá trị cho 5 component.

    Args:
        image_path: đường dẫn file ảnh Gradio nhận được
                    (Gradio tự lưu file upload vào thư mục tạm rồi truyền path)

    Returns:
        (ảnh kết quả, thông tin, text, file_ảnh_tải, file_json_tải)
        THỨ TỰ này phải khớp hệt với outputs=[...] ở dưới cùng file.
    """

    # --- Kiểm tra đầu vào ---
    # image_path là None khi người dùng chưa chọn ảnh nào
    if not image_path:
        # gr.Warning hiện popup cảnh báo trên giao diện
        # (không ném exception -> giao diện không bị sập)
        gr.Warning("Vui lòng chọn ảnh trước!")
        return None, "**Chưa có ảnh**", "", None, None

    image = cv2.imread(image_path)
    if image is None:
        gr.Warning(f"Không đọc được file ảnh: {image_path}")
        return None, "**Lỗi đọc ảnh**", "", None, None

    h, w = image.shape[:2]           # shape = (cao, rộng, kênh) -> lấy 2 đầu


    # chạy OCR và Đo thời gian 
    pipeline = get_pipeline()
    start = time.time()
    results = pipeline.run(image)
    elapsed = time.time() - start

        # --- CARD 2: ảnh đã vẽ khung ---
    # draw_results() là hàm CÓ SẴN trong ocr/pipeline.py
    # -> không viết lại, đảm bảo kết quả giống hệt khi chạy dòng lệnh
    annotated = draw_results(image, results)

    # Lưu ảnh kết quả để người dùng tải về
    out_dir = RESULTS / "ui"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_dir / f"result_{Path(image_path).name}"
    cv2.imwrite(str(out_img_path), annotated)


    # --- CARD 3: text ---
    # Mỗi dòng = 1 khung chữ. join với "\n" để hiển thị nhiều dòng
    text = "\n".join(r.text for r in results) if results else "(không tìm thấy chữ)"

    # --- Thông tin kèm theo (hiển thị ở Card 2) ---
    # Markdown: 2 khoảng trắng + \n = xuống dòng trong Markdown
    info = (
        f"**Kích thước ảnh:** {w}×{h} px  \n"
        f"**Số khung chữ:** {len(results)}  \n"
        f"**Thời gian xử lý:** {elapsed:.2f} giây"
    )

     # --- File JSON để tải về ---
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

    # Trả về theo THỨ TỰ khai báo trong outputs=[...] ở dưới
    return annotated, info, text, str(out_img_path), str(json_path)





# ---------------------------------------------------------------------------
# 3. ẢNH VÍ DỤ — cho người dùng thử ngay không cần có ảnh
# ---------------------------------------------------------------------------
def list_examples() -> list:
    """Lấy 4 ảnh đầu tiên trong thư mục test làm ví dụ."""
    if not TEST_IMAGES.exists():
        return []
    imgs = sorted(TEST_IMAGES.glob("*.jpg"))
    # Gradio cần list các [đường_dẫn, nhãn_hiển_thế]
    return [[str(p), p.name] for p in imgs[:4]]


# ---------------------------------------------------------------------------
# 4. XÂY DỰNG GIAO DIỆN — gr.Blocks
# ---------------------------------------------------------------------------
# gr.Blocks cho phép layout TỰ DO (hàng, cột) thay vì xếp dọc
# như gr.Interface. Phù hợp khi cần 3 card cạnh nhau.
#
# with gr.Blocks() as demo:  = "context manager"
#   mọi component tạo bên trong sẽ tự động NẰM trong demo
with gr.Blocks(title="OCR tiếng Việt") as demo:

    # --- TIÊU ĐỀ ---
    gr.Markdown(
        "# 📄 OCR tiếng Việt\n"
        "**PaddleOCR** tìm khung chữ → **VietOCR** đọc chữ tiếng Việt"
    )

    # --- HÀNG CHỨA 3 CARD ---
    with gr.Row():                     # Row = xếp CẠNH NHAU (ngang)

        # ================= CARD 1: INPUT =================
        with gr.Column():              # Column = 1 cột trong hàng
            gr.Markdown("### 1️⃣ INPUT — Chọn ảnh")
            input_image = gr.Image(
                type="filepath",       # trả về ĐƯỜNG DẪN file (không phải numpy)
                label="Kéo thả hoặc bấm để chọn ảnh",
                height=260,
            )

            # gr.Examples: bảng ảnh mẫu. Khi TRUYỀN fn thì bấm 1 ảnh
            # là CHẠY LUÔN (không cần bấm nút "Chạy OCR" riêng).
            examples = list_examples()
            if examples:
                gr.Examples(
                    examples=examples,          # [[đường_dẫn, nhãn], ...]
                    inputs=input_image,         # component nhận giá trị
                    fn=run_ocr,                 # hàm chạy khi bấm ảnh mẫu
                    outputs=[None, None, None, None, None],
                    #         ^ 5 None = không cập nhật output nào khi chạy
                    #           (vì Gradio tự vẽ ảnh mẫu, ta không ép)
                    label="📸 Ảnh ví dụ",
                )

        # ================= CARD 2: DETECTION =================
        with gr.Column():
            gr.Markdown("### 2️⃣ DETECTION — Ảnh có khung chữ")
            output_image = gr.Image(
                type="numpy",          # trả về mảng numpy để hiển thị trực tiếp
                label="Kết quả",
                height=260,
            )
            info_box = gr.Markdown("Chưa chạy")   # info: số khung, thời gian

        # ================= CARD 3: TEXT =================
        with gr.Column():
            gr.Markdown("### 3️⃣ TEXT — Chữ đọc được")
            output_text = gr.Textbox(
                label="Kết quả",
                lines=11,              # số dòng hiển thị
                max_lines=20,
                placeholder="Chữ đọc được sẽ hiển thị ở đây...",
                buttons=["copy"],   # nút Copy (Gradio 6: thay cho show_copy_button)
            )

            with gr.Row():             # 2 nút tải xếp ngang
                download_image = gr.File(
                    label="⬇️ Tải ảnh kết quả", interactive=False)
                # interactive=False: chỉ hiển thị khi CÓ file (bấm nút disabled)
                download_json = gr.File(
                    label="⬇️ Tải JSON", interactive=False)

    # --- NÚT CHẠY ---
    run_btn = gr.Button("🚀 Chạy OCR", variant="primary", size="lg")
    gr.Markdown("*Nhấn nút trên để bắt đầu*")

    # ---------------------------------------------------------------------------
    # 5. NỐI NÚT VỚI HÀM — event binding
    # ---------------------------------------------------------------------------
    # .click(fn=..., inputs=..., outputs=...)
    #   fn     : hàm Python gọi khi bấm nút
    #   inputs : component ĐƯA GIÁ TRỊ vào hàm
    #   outputs: component NHẬN KẾT QUẢ (thứ tự phải khớp return của hàm)
    run_btn.click(
        fn=run_ocr,
        inputs=input_image,
        # 5 giá trị return của run_ocr() -> 5 component nhận
        outputs=[output_image, info_box, output_text,
                 download_image, download_json],
    )


# ---------------------------------------------------------------------------
# 6. KHỞI CHẠY SERVER
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Preload model NGAY khi khởi động thay vì chờ lần bấm đầu tiên.
    #   -> Người dùng mở trang là app đã sẵn sàng, không phải chờ tải model.
    get_pipeline()

    demo.launch(
        server_name="0.0.0.0",   # 0.0.0.0 = nghe TẤT CẢ mạng (vào từ máy khác
                                 #   trong cùng wifi được; "127.0.0.1" thì chỉ
                                 #   máy này truy cập được)
        server_port=7860,        # cổng mặc định của Gradio
        share=False,             # True sẽ tạo link công khai *.gradio.live
        inbrowser=True,          # tự mở trình duyệt
    )