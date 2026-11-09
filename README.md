---
#Thông tin cho Hugging Face Space (GitHub sẽ hiển thị 2 dòng --- này như text thường)
title: MY-OCR
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# OCR — Nhận diện văn bản tiếng Việt

Dự án học tập về Computer Vision: **phát hiện chữ** và **nhận diện chữ** trên ảnh tiếng Việt.

> 🌐 **Thử online:** [huggingface.co/spaces/LOCNGUYEN120707/MY-OCR](https://huggingface.co/spaces/LOCNGUYEN120707/MY-OCR)

## Kiến trúc

```
Ảnh → PaddleOCR (PP-OCRv5 det) → cắt từng khung chữ → VietOCR → văn bản tiếng Việt
```

- **PaddleOCR** — chỉ dùng bước *detection* (tìm khung chữ)
- **VietOCR** — bước *recognition* (đọc chữ), model Transformer pretrained tiếng Việt

## Cài đặt

Dự án dùng **2 môi trường Python** vì PaddlePaddle chưa hỗ trợ Python 3.14:

```bash
# 1. Môi trường cho PaddleOCR + VietOCR (Python 3.11)
/usr/bin/python3.11 -m venv venv_paddle
venv_paddle/bin/pip install paddleocr vietocr
venv_paddle/bin/pip install paddlepaddle-gpu==3.3.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
venv_paddle/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Model pretrained: file `weights/` (~150MB) được đẩy kèm **Hugging Face Space**;
clone từ GitHub (không có weights) thì code tự tải từ mạng về như cũ.
Cài đặt môi trường bằng `requirements.txt` (bản CPU) hoặc xem hướng dẫn dưới.

## Cách dùng

```bash
# 🖥️ Giao diện web (3 card: input / detection / text)
venv_paddle/bin/python app.py            # mở http://127.0.0.1:7860

# Đọc 1 ảnh
venv_paddle/bin/python run_ocr.py anh_can_doc.jpg

# Chạy 300 ảnh test + in bảng đánh giá
venv_paddle/bin/python run_ocr.py --test

# Tính lại đánh giá từ results_test.json (nhanh, không chạy model)
venv_paddle/bin/python run_ocr.py --eval

# Chạy 500 ảnh không có label
venv_paddle/bin/python run_ocr.py --unseen
```

Kết quả lưu tại `Data/results/` (ảnh vẽ khung + file JSON).

## Cấu trúc

```
ocr/
├── config.py       # đường dẫn, tên model, ngưỡng  (đọc ĐẦU TIÊN)
├── detector.py     # Bước 1: PaddleOCR → tìm khung, cắt crop
├── recognizer.py   # Bước 2: VietOCR → đọc chữ từ crop
├── pipeline.py     # Bước 3: gộp 2 bước + vẽ kết quả
└── evaluate.py     # Bước 4: precision/recall/accuracy so với ground truth
app.py              # 🖥️ giao diện Gradio (3 card) — chạy trên HF Space
run_ocr.py         # entry point dòng lệnh (file bạn gọi để chạy)
smoke_test.py      # kiểm tra nhanh pipeline trên 1 ảnh
requirements.txt    # dependencies (bản CPU cho Space)
weights/            # file model ~150MB (chỉ có trên HF Space, không lên GitHub)
```

Mỗi file đều có docstring giải thích **TẠI SAO** làm vậy (không chỉ "làm gì"),
kèm chú thích từng dòng ở các chỗ khó — đọc theo thứ tự đánh số ở trên.

## Gợi ý đọc code

1. **`ocr/config.py`** — làm quen đường dẫn và tham số
2. **`ocr/detector.py`** — hiểu detection và Perspective Transform
3. **`ocr/recognizer.py`** — hiểu tại sao phải đổi BGR→RGB, batch là gì
4. **`ocr/pipeline.py`** — cách ghép các bước, `@dataclass`, `@property`
5. **`ocr/evaluate.py`** — Precision/Recall/IoU (phần quan trọng nhất của ML)
6. **`run_ocr.py`** — argparse, `if __name__ == "__main__"`


## Đánh giá

Chạy `--test` sẽ so kết quả với label ground truth (300 ảnh `gt_1201.txt`…`gt_1500.txt`).
Chạy `--eval` để tính lại từ file JSON đã lưu (không cần chạy lại model).

| Metric | Ý nghĩa |
|---|---|
| Det Precision / Recall / Hmean | Tìm đúng khung chữ (IoU ≥ 0.5) |
| Rec Accuracy | Đúng chữ so với ground truth |
| E2E Accuracy | Vừa đúng khung vừa đúng chữ |

Đánh giá chạy ở **2 chế độ**:
- **word-level** — so theo từng từ, đúng dạng dataset gốc (29 khung/ảnh)
- **line-level** — gộp GT theo dòng trước khi so, công bằng hơn với model
  pretrained vốn detect theo dòng (15 khung/ảnh)

### Kết quả hiện tại (model pretrained, chưa fine-tune)

| | word-level | line-level |
|---|---|---|
| Det Precision | 0.33 | 0.29 |
| Det Recall | 0.08 | 0.14 |
| Det Hmean | 0.13 | 0.19 |
| Rec Accuracy | **0.57** | **0.51** |

**Nhận xét:** khi đã tìm được khung chữ, model đọc đúng ~51–57%.
Hai giới hạn chính:

1. **Lệch độ phân giải** — dataset nhãn theo *từ* (cao TB 45px), model detect
   theo *dòng* (box dự đoán cao TB 89px, gộp 2–3 dòng) → IoU ≥ 0.5 khó đạt,
   chỉ 29% box dự đoán vượt ngưỡng.
2. **Model pretrained** — VietOCR đọc sai dấu (`CẢ` thay vì `CÀ`), thừa ký tự.

Để cải thiện: fine-tune VietOCR trên 43k box có sẵn của dự án,
và/hoặc chỉnh ngưỡng/merge box ở bước detection.

## Dữ liệu

`Data/vietnamese_original/` — 1200 ảnh train, 300 ảnh test (có label), 500 ảnh unseen.
Định dạng label: `x1,y1,x2,y2,x3,y3,x4,y4,text` — text `###` nghĩa là vùng không phải chữ.
