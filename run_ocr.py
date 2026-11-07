"""
ĐIỂM VÀO CHƯƠNG TRÌNH (Entry Point)
====================================
Đây là file bạn GỌI ĐỂ CHẠY. Nó chỉ "điều khiển": đọc lệnh dòng lệnh,
gọi các hàm trong package ocr/, rồi in kết quả.

CÁCH DÙNG:
  python run_ocr.py anh.jpg           # đọc 1 ảnh
  python run_ocr.py --test            # chạy 300 ảnh test + đánh giá
  python run_ocr.py --eval            # tính lại đánh giá từ JSON (không chạy model)
  python run_ocr.py --unseen          # chạy 500 ảnh không có label
"""

# argparse: thư viện có sẵn của Python để xử lý LỆNH DÒNG LỆNG (CLI).
#   Nó tự tạo menu --help, bắt lỗi khi bạn gõ thiếu/thừa tham số.
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Tại sao cần dòng này?
#   Khi chạy "python run_ocr.py", Python chỉ thêm thư mục CHỨA file vào
#   sys.path. Mà package "ocr" nằm CÙNG cấp nên không tìm thấy -> phải thêm
#   đường dẫn thủ công.
#   Path(__file__).resolve().parent = thư mục chứa run_ocr.py này
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr.config import RESULTS, TEST_IMAGES, UNSEEN_IMAGES
from ocr.pipeline import OcrPipeline, draw_results


def run_folder(pipeline: OcrPipeline, folder: Path, save: bool = True):
    """Chạy OCR cho MỌI ảnh trong 1 thư mục.

    Args:
        pipeline: đối tượng đã nạp 2 model (tránh tải model lại mỗi ảnh)
        folder: thư mục chứa ảnh
        save: có lưu ảnh vẽ khung ra Results hay không

    Returns:
        dict {tên ảnh: list[OcrResult]} — dùng sau đó để đánh giá
    """
    results_by_image = {}

    # Lấy danh sách ảnh, sắp xếp để chạy theo thứ tự cố định
    images = sorted(
        p for p in folder.iterdir()
        # Chỉ lấy file ảnh, bỏ qua file txt/csv...
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )

    if save:
        # parents=True: tạo cả thư mục CHA nếu chưa có
        # exist_ok=True: không báo lỗi nếu đã tồn tại
        RESULTS.mkdir(parents=True, exist_ok=True)

    for n, path in enumerate(images, 1):      # enumerate cho số thứ tự bắt đầu từ 1
        image = cv2.imread(str(path))         # OpenCV cần str, không nhận Path
        if image is None:
            continue                          # file hỏng -> bỏ qua, không sập

        # --- ĐO THỜI GIAN ---
        start = time.time()
        results = pipeline.run(image)
        elapsed = time.time() - start

        results_by_image[path.name] = results

        # Nối các text lại để in 1 dòng cho gọn
        texts = " | ".join(r.text for r in results)
        print(f"[{n}/{len(images)}] {path.name}: {len(results)} khung, "
              f"{elapsed:.2f}s -> {texts[:90]}")
        #                            ^:.2f = 2 chữ số thập phân
        #                             ^[:90] = cắt chuỗi, lấy 90 ký tự đầu

        if save:
            # Vẽ khung lên ảnh rồi lưu
            cv2.imwrite(str(RESULTS / f"ocr_{path.name}"),
                        draw_results(image, results))

    return results_by_image


def save_json(results_by_image: dict, name: str):
    """Lưu kết quả ra file JSON — format đọc/ghi phổ biến, giữ được dấu tiếng Việt.

    JSON dạng:
      { "im1201.jpg": [ {"box": [[x,y],...], "text": "CÀ"}, ... ], ... }

    Lưu ý ensure_ascii=False: nếu để True, mọi ký tự tiếng Việt sẽ bị
    escape thành \u1e00... khó đọc. Đặt False để giữ nguyên "CÀ PHÊ".
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name

    # OcrResult.box là numpy array -> JSON không biết đọc numpy
    # -> phải .tolist() để chuyển sang list thường trước
    data = {
        img: [{"box": r.box.tolist(), "text": r.text} for r in results]
        for img, results in results_by_image.items()
    }

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    # indent=2: thụt lề 2 khoảng trắng -> file dễ đọc bằng mắt
    print("Đã lưu JSON:", path)


def load_json(name: str) -> dict:
    """Đọc file JSON kết quả -> khôi phục lại dict {tên ảnh: list[OcrResult]}.

    Nhờ hàm này mà bạn có thể tính lại đánh giá BẤT KỲ LÚC NÀO
    mà không cần chạy lại model (tiết kiệm vài phút).
    """
    from ocr.pipeline import OcrResult      # import lỏng (local import)

    data = json.loads((RESULTS / name).read_text(encoding="utf-8"))

    # Chuyển ngược lại: list -> numpy array (vì OcrResult.box là numpy)
    return {
        img: [OcrResult(np.array(r["box"], dtype=np.int32), r["text"])
              for r in items]
        for img, items in data.items()
    }


def print_metrics(metrics: dict, title: str):
    """In bảng metric cho gọn, dùng lại được cho cả 2 chế độ đánh giá."""
    print(f"\n=== {title} ===")
    print(f"  Detection  - Precision: {metrics['det_precision']:.4f} | "
          f"Recall: {metrics['det_recall']:.4f} | "
          f"Hmean: {metrics['det_hmean']:.4f}")
    print(f"  Recognition- Accuracy : {metrics['rec_accuracy']:.4f} "
          f"({metrics['boxes_gt']} khung có chữ)")
    print(f"  End-to-End - Accuracy : {metrics['e2e_accuracy']:.4f}")
    print(f"  Số khung: GT khớp {metrics['boxes_matched']} | "
          f"dự đoán {metrics['boxes_pred']}")


def run_evaluation(results_by_image: dict):
    """In đánh giá ở CẢ 2 chế độ (word-level và line-level)."""
    from ocr.evaluate import evaluate

    print_metrics(evaluate(results_by_image),
                  "ĐÁNH GIÁ THEO TỪ (word-level) - dataset gốc")
    print_metrics(evaluate(results_by_image, by_line=True),
                  "ĐÁNH GIÁ THEO DÒNG (line-level) - công bằng với model")


def main():
    # --- 1. Khai báo các lựa chọn trên dòng lệnh ---
    parser = argparse.ArgumentParser(description="OCR: PaddleOCR + VietOCR")

    # nargs="?" nghĩa là tham số KHÔNG BẮT BUỘC (có thể không truyền)
    parser.add_argument("image", nargs="?", help="Đường dẫn 1 ảnh cần đọc")

    # action="store_true": cờ (flag) — chỉ cần gõ "--test" là True,
    #   không gõ là False. Không cần truyền giá trị.
    parser.add_argument("--test", action="store_true",
                        help="Chạy 300 ảnh test và in bảng đánh giá")
    parser.add_argument("--unseen", action="store_true",
                        help="Chạy 500 ảnh không có label")
    parser.add_argument("--eval", action="store_true",
                        help="Chỉ tính lại đánh giá từ results_test.json "
                             "(không chạy model)")
    args = parser.parse_args()

    # --- 2. Tính lại từ JSON có sẵn (nhanh nhất) ---
    if args.eval:
        results_by_image = load_json("results_test.json")
        print(f"Đã nạp {len(results_by_image)} ảnh từ results_test.json")
        run_evaluation(results_by_image)
        return                                  # thoát khỏi hàm main()

    # --- 3. Không truyền gì cả -> hiện hướng dẫn ---
    if not (args.image or args.test or args.unseen):
        parser.print_help()                     # in ra màn hình hướng dẫn
        return

    # --- 4. Nạp model (mất vài giây, làm 1 lần) ---
    pipeline = OcrPipeline()

    # --- 5. Chế độ đọc 1 ảnh ---
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            # sys.exit(msg): in thông báo rồi thoát với mã lỗi
            sys.exit(f"Không đọc được ảnh: {args.image}")

        results = pipeline.run(image)

        # In từng dòng text ra màn hình
        for r in results:
            print(r.text)

        save_json({Path(args.image).name: results}, "result.json")

        out = RESULTS / f"ocr_{Path(args.image).name}"
        cv2.imwrite(str(out), draw_results(image, results))
        print("Đã lưu ảnh kết quả:", out)
        return

    # --- 6. Chế độ chạy cả thư mục ---
    folder = TEST_IMAGES if args.test else UNSEEN_IMAGES
    #   biểu thức có điều kiện: giá trị_if_true if điều_kiện else giá trị_if_false
    #   = nếu --test thì dùng thư mục test, không thì dùng unseen

    print(f"\n=== Chạy OCR trên {folder} ===")
    results_by_image = run_folder(pipeline, folder)

    # Đặt tên file JSON theo chế độ đang chạy
    json_name = "results_test.json" if args.test else "results_unseen.json"
    save_json(results_by_image, json_name)

    # Chỉ in bảng đánh giá khi chạy trên tập CÓ label
    if args.test:
        run_evaluation(results_by_image)


# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#   Đoạn này chạy khi bạn GỌI TRỰC TIẾP file này (python run_ocr.py),
#   nhưng KHÔNG chạy khi file được import bởi file khác.
#
#   Ví dụ: "import run_ocr" -> __name__ == "run_ocr" -> KHÔNG chạy main()
#          "python run_ocr.py" -> __name__ == "__main__" -> CHẠY main()
#
#   Đây là cách Python phổ biến để file vừa dùng được như thư viện,
#   vừa chạy được như chương trình.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
