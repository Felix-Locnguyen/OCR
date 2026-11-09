"""
BƯỚC 4 — ĐÁNH GIÁ KẾT QUẢ (Evaluation)
=======================================
"Model chạy ra sao?" — trả lời bằng CON SỐ, không phải cảm tính.

Các khái niệm nền tảng:

  GROUND TRUTH (GT)  : đáp án đúng, do người annotator gán nhãn sẵn trong file label.
  PREDICTION         : thứ model dự đoán ra.

Với mỗi khung GT, ta so với khung dự đoán và phân loại:

   ┌──────────┬───────────────────┬───────────────────┐
   │          │  Dự đoán CÓ      │  Dự đoán KHÔNG   │
   ├──────────┼───────────────────┼───────────────────┤
   │ GT CÓ    │  TP (True Pos)    │  FN (False Neg)   │  <- bỏ sót
   │ GT KHÔNG │  FP (False Pos)   │  TN (True Neg)    │  <- bịa ra
   └──────────┴───────────────────┴───────────────────┘

  Precision = TP / (TP + FP)   "Dự đoán của model, bao nhiêu % là đúng?"
  Recall    = TP / (TP + FN)   "Trong số đáp án đúng, model tìm được bao nhiêu %?"
  Hmean     = trung bình điều hòa của 2 cái trên (F1-score)
"""

import numpy as np

# Shapely: thư viện hình học 2D — tính được diện tích giao/hợp của 2 hình đa giác.
# Mình dùng nó thay vì tự viết code tính IoU cho tứ giác (rất dễ sai với hình nghiêng).
from shapely.geometry import Polygon

from .config import IOU_THRESHOLD, LABELS


def read_labels(image_name: str) -> list[tuple[np.ndarray, str]]:
    """Đọc file label ground truth của 1 ảnh.

    QUY ƯỚC ĐẶT TÊN FILE (rất quan trọng, dễ sai ở đây):
        ảnh  "im1201.jpg"  <->  label  "gt_1201.txt"
        tức  im + SỐ  ->  gt_ + SỐ

    Args:
        image_name: tên file ảnh, ví dụ "im1201.jpg"

    Returns:
        Danh sách [(toạ độ 4 điểm, text), ...]

    ĐỊNH DẠNG FILE LABEL (mỗi dòng 1 khung):
        27,145,52,145,47,179,20,178,CÀ
        └────────── 8 số ──────────┘ └text┘
        = 4 điểm (x,y):  (27,145) (52,145) (47,179) (20,178)
    """
    # Bước 1: lấy số trong tên file. "im1201.jpg" -> "1201"
    #   .replace("im","")  bỏ tiền tố "im"     -> "1201.jpg"
    #   .split(".")[0]     cắt phần đuôi file  -> "1201"
    #   int()              đổi sang số nguyên   -> 1201
    num = int(image_name.replace("im", "").split(".")[0])
    path = LABELS / f"gt_{num}.txt"

    if not path.exists():
        return []              # ảnh này chưa có label

    boxes = []
    # encoding="utf-8-sig": đọc đúng tiếng Việt có dấu.
    #   "-sig" = "signature": tự xử lý ký tự BOM ẩn ở đầu file
    #   (nếu không có, ký tự ẩn này sẽ làm hỏng ký tự đầu tiên)
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            # "27,145,52,...,CÀ".split(",") -> ['27','145','52',...,'CÀ']
            parts = line.strip().split(",")

            if len(parts) < 8:      # dòng không đủ 8 tọa độ -> bỏ qua
                continue

            # parts[:8] = 8 số đầu -> reshape(4,2) để có 4 điểm (x,y)
            #   [27,145,52,145,...]  ->  [[27,145], [52,145], [47,179], [20,178]]
            pts = np.array(parts[:8], dtype=np.float32).reshape(4, 2)

            # VẤN ĐỀ: text có thể CHỨA dấu phẩy! Ví dụ: "...,LÀM,"
            # Nếu chỉ lấy parts[8] thì mất dấu phẩy -> "LÀM"
            # -> join TẤT CẢ phần từ index 8 trở đi: ",".join(parts[8:]) = "LÀM,"
            text = ",".join(parts[8:]).strip()

            boxes.append((pts, text))
    return boxes


def iou(box_a, box_b) -> float:
    """Tính IoU (Intersection over Union) giữa 2 khung chữ.

    IoU = diện tích GIAO / diện tích HỢP

        ┌────────┐                IoU = 0.5 nghĩa là
        │  ┌─────┼────┐           nửa khung chồng nhau
        │  │ GIAO│    │
        └──┼─────┘────┘

    Giá trị từ 0.0 (không chồng) đến 1.0 (trùng hoàn toàn).
    Ngưỡng thường dùng: IoU >= 0.5 coi là "tìm đúng".

    Args:
        box_a, box_b: mảng 4 điểm shape (4, 2)

    Returns:
        Số float từ 0 đến 1
    """
    # Chuyển 4 điểm thành hình đa giác của Shapely
    pa, pb = Polygon(box_a), Polygon(box_b)

    # is_valid = False khi hình tự cắt nhau (rìa tự giao nhau) -> lỗi dữ liệu
    if not pa.is_valid or not pb.is_valid:
        return 0.0

    inter = pa.intersection(pb).area   # diện tích phần GIAO
    union = pa.union(pb).area          # diện tích phần HỢP (giao + hôi)

    return inter / union if union > 0 else 0.0


def merge_to_lines(boxes: list, y_tol=20, x_gap=10) -> list:
    """
    Gộp các box CÙNG DÒNG thành 1 box lớn.

    TẠI SAO CẦN BƯỚC NÀY?
      Dataset của mình được nhãn theo TỪ:  "CÀ" | "PHÊ" | "WIFI"  (3 box)
      Model pretrained lại detect theo DÒNG: "CÀ PHÊ WIFI"        (1 box)

      Nếu so 1 box lớn với 3 box nhỏ bằng IoU thì IoU luôn THẤP
      -> kết quả đánh giá THẤP một cách oan uổng, không phản ánh model kém.

      => Gộp 3 box từ thành 1 box dòng rồi mới so -> công bằng hơn.

    Args:
        boxes: [(toạ độ, text), ...] như read_labels trả về
        y_tol: độ lệch y tối đa để coi là "cùng dòng" (pixel)
        x_gap: khoảng hở x tối đa vẫn coi là nối tiếp (pixel)

    Returns:
        [(toạ độ box dòng, text dòng gộp), ...]
    """
    lines = []       # danh sách các "dòng" đang gom được

    # Sắp theo y TĂNG DẦN để duyệt từ trên xuống (đọc như đọc sách)
    for pts, text in sorted(boxes, key=lambda b: b[0][:, 1].mean()):
        cy = pts[:, 1].mean()              # y tâm của box này
        x0, x1 = pts[:, 0].min(), pts[:, 0].max()   # x trái, x phải

        # Duyệt các dòng đã gom, xem box này thuộc dòng nào
        for ln in lines:
            # Điều kiện 1: y tâm cách nhau < y_tol  -> cùng hàng ngang
            # Điều kiện 2: KHÔNG bị tách bởi khoảng trống > x_gap
            #             (x1 < ln.x0 - gap nghĩa là box này nằm BÊN TRÁI
            #              dòng đó với khoảng cách > gap -> khác dòng)
            same_row = abs(ln["cy"] - cy) < y_tol
            overlap_x = not (x1 < ln["x0"] - x_gap or x0 > ln["x1"] + x_gap)

            if same_row and overlap_x:
                # CẬP NHẬT dòng: lấy trung bình có trọng số của y
                #   ln.cy mới = (cy cũ * n + cy mới) / (n+1)
                # Đây là TRUNG BÌNH ĐÁNG (running average), n = số box đã gom
                ln["cy"] = (ln["cy"] * ln["n"] + cy) / (ln["n"] + 1)
                ln["n"] += 1
                ln["x0"] = min(ln["x0"], x0)      # mở rộng biên trái
                ln["x1"] = max(ln["x1"], x1)      # mở rộng biên phải
                ln["text"] = f'{ln["text"]} {text}'.strip()   # nối text
                break                              # xong, khỏi xét dòng khác
        else:
            # for...else: chạy else khi vòng lặp KHÔNG bị break
            # -> nghĩa là KHÔNG dòng nào phù hợp -> TẠO DÒNG MỚI
            lines.append({"cy": cy, "x0": x0, "x1": x1, "n": 1,
                          "text": text, "pts": [pts]})

    # Gom toàn bộ điểm của từng dòng -> tính khung chữ nhật bao ngoài
    result = []
    for ln in lines:
        pts = np.vstack(ln["pts"])   # vstack: xếp chồng nhiều mảng thành 1 ma trận

        # Lấy biên cực trị -> hình chữ nhật bao
        x0, x1 = pts[:, 0].min(), pts[:, 0].max()
        y0, y1 = pts[:, 1].min(), pts[:, 1].max()

        # 4 đỉnh hình chữ nhật, theo chiều kim đồng hồ từ góc trên-trái
        box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        result.append((box, ln["text"]))
    return result


def evaluate(results_by_image: dict, by_line=False) -> dict:
    """
    Tính các metric đánh giá trên toàn bộ ảnh.

    Args:
        results_by_image: {tên ảnh: list[OcrResult]}  <- kết quả từ pipeline
        by_line: False = so theo TỪ (đúng dạng dataset gốc)
                 True  = gộp GT theo DÒNG trước khi so (công bằng hơn
                         với model pretrained vốn detect theo dòng)

    Returns:
        dict chứa các metric (xem chú thích từng cái trong phần return)
    """
    # --- Bộ đếm cho PHÁT HIỆN KHUNG (detection) ---
    tp_det = 0      # True Positive : dự đoán khớp với 1 GT  -> ĐÚNG
    fp_det = 0      # False Positive: dự đoán không khớp GT nào -> BỊA
    fn_det = 0      # False Negative: GT không có dự đoán nào khớp -> BỎ SÓT

    # --- Bộ đếm cho ĐỌC CHỮ (recognition) ---
    correct_rec = 0    # số khung đọc ĐÚNG chữ
    total_rec = 0      # tổng số khung CÓ CHỮ cần đọc (loại vùng ###)
    correct_e2e = 0    # end-to-end: vừa đúng khung vừa đúng chữ

    for image_name, results in results_by_image.items():
        gt_boxes = read_labels(image_name)
        if not gt_boxes:
            continue                     # ảnh này không có label -> bỏ qua

        # Chế độ line: gộp GT theo dòng trước khi so sánh
        if by_line:
            gt_boxes = merge_to_lines(gt_boxes)

        # Đánh dấu GT nào ĐÃ bị một dự đoán khác "chiếm mất"
        # (mỗi GT chỉ được khớp với ĐÚNG 1 dự đoán, tránh tính trùng)
        used_gt = set()

        # Với MỖI dự đoán, tìm GT trùng khung NHẤT
        for r in results:
            best_iou, best_idx = 0.0, -1

            for i, (gt_pts, _) in enumerate(gt_boxes):
                if i in used_gt:
                    continue             # GT này đã bị chiếm rồi -> bỏ qua

                score = iou(r.box, gt_pts)
                if score > best_iou:     # giữ lại GT có IoU cao nhất
                    best_iou, best_idx = score, i

            # IoU không đạt ngưỡng -> dự đoán này KHÔNG khớp GT nào
            if best_iou < IOU_THRESHOLD:
                fp_det += 1              # tính là "bịa ra"
                continue

            # --- Khớp được! ---
            used_gt.add(best_idx)        # đánh dấu GT đã dùng
            tp_det += 1
            gt_text = gt_boxes[best_idx][1]

            # "###" nghĩa là vùng này KHÔNG phải chữ (theo quy ước dataset)
            # Ở chế độ line có thể là "### ### ###" -> dùng set() để kiểm tra
            #   {"###","###"} <= {"###"}  -> True (tất cả đều là ###)
            if set(gt_text.split()) <= {"###"}:
                continue                 # không tính vào accuracy đọc chữ

            total_rec += 1
            if _same_text(r.text, gt_text):
                correct_rec += 1
                correct_e2e += 1         # đúng cả khung lẫn chữ

        # GT chưa bị dự đoán nào khớp -> đó là những cái BỎ SÓT
        fn_det += len(gt_boxes) - len(used_gt)

    # --- TÍNH METRIC ---
    # Guard (bảo vệ): chia cho 0 sẽ gây lỗi -> dùng if để tránh
    precision = tp_det / (tp_det + fp_det) if tp_det + fp_det else 0.0
    recall = tp_det / (tp_det + fn_det) if tp_det + fn_det else 0.0

    # Hmean (F1) = trung bình điều hòa của precision và recall.
    #   Trung bình thông thường (P+R)/2 sẽ "kéo" số cao bù số thấp,
    #   còn trung bình điều hòa phạt mạnh hơn khi 1 trong 2 quá thấp.
    hmean = (2 * precision * recall / (precision + recall)
             if precision + recall else 0.0)

    return {
        "mode": "line" if by_line else "word",   # chế độ đánh giá
        "images": len(results_by_image),

        # --- Detection ---
        "det_precision": round(precision, 4),    # dự đoán bao nhiêu % là đúng
        "det_recall": round(recall, 4),          # tìm được bao nhiêu % đáp án
        "det_hmean": round(hmean, 4),            # tổng thể 2 cái trên

        # --- Recognition ---
        # round(x, 4) = làm tròn 4 chữ số thập phân cho dễ đọc
        "rec_accuracy": round(correct_rec / total_rec, 4) if total_rec else 0.0,
        "e2e_accuracy": round(correct_e2e / total_rec, 4) if total_rec else 0.0,

        # --- Thông tin phụ (để debug) ---
        "boxes_gt": total_rec,                                  # khung có chữ cần đọc
        "boxes_matched": tp_det,                                # khung khớp được
        "boxes_pred": sum(len(v) for v in results_by_image.values()),  # tổng dự đoán
    }


def _same_text(pred: str, gt: str) -> bool:
    """So sánh 2 chuỗi text một cách "dễ dãi" hơn so với == thông thường.

    Vì sao cần?
      Model có thể trả "CÀ PHÊ"  mà GT là "CÀ  PHÊ"  (2 khoảng trắng)
      -> dùng == sẽ tính SAI dù nội dung giống hệt.

    Cách làm:
      " ".join(s.split())  = tách theo mọi khoảng trắng -> gom lại 1 khoảng
          "CÀ   PHÊ ".split()  -> ['CÀ', 'PHÊ']
          " ".join(...)         -> 'CÀ PHÊ'   (chuẩn hóa)
      .lower() = đổi về chữ thường để "Hello" == "hello"
    """
    return " ".join(pred.split()).lower() == " ".join(gt.split()).lower()
