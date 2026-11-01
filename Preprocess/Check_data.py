import os
import cv2
import matplotlib.pyplot as plt
import numpy as np 
def visualize_sample(image_path, label_path):
    """
    Nhận đầu vào là đường dẫn file ảnh và đường dẫn file nhãn .txt (định dạng OCR 4 điểm, phân tách bằng dấu phẩy),
    vẽ khung polygon bao quanh văn bản và trả về ảnh đã vẽ.
    """
    # Đọc ảnh gốc
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc được ảnh tại: {image_path}")
        
    # Đọc file nhãn nếu tồn tại
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                # Tách các thành phần bằng dấu phẩy
                parts = line.strip().split(',')
                if len(parts) >= 9:
                    # Lấy 8 giá trị tọa độ đầu tiên (4 điểm x,y)
                    coords = list(map(int, parts[:8]))
                    text = parts[8]  # Nội dung chữ
                    
                    # Định dạng lại thành mảng numpy cho OpenCV vẽ polygon
                    pts = np.array([
                        [coords[0], coords[1]],
                        [coords[2], coords[3]],
                        [coords[4], coords[5]],
                        [coords[6], coords[7]]
                    ], dtype=np.int32)
                    
                    # Vẽ khung đa giác (Polygon) màu xanh lá
                    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
                    
                    # Hiển thị nội dung text nhỏ lên góc trên của box
                    xmin, ymin = coords[0], coords[1]
                    cv2.putText(img, text, (xmin, max(ymin - 3, 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                                
    return img
if __name__ == "__main__":
    img_path = r"../OCR/Data/vietnamese_original/train_images/im0001.jpg"
    label_path = r"../OCR/Data/vietnamese_original/labels/gt_1.txt"

    # Gọi hàm xử lý
    img = visualize_sample(img_path, label_path)
    
    # Chuyển đổi hệ màu BGR (OpenCV) sang RGB (Matplotlib) để hiển thị chuẩn màu
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Hiển thị ảnh
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()