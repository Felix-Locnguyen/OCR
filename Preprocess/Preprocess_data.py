import os
import re
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_image_dataframe(image_dir):
    """
    DataFrame chỉ chứa duy nhất cột image_path đúng theo yêu cầu.
    """
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    data = []
    
    if not os.path.exists(image_dir):
        print(f"Thư mục không tồn tại: {image_dir}")
        return pd.DataFrame(columns=["image_path"])
        
    for f in os.listdir(image_dir):
        if f.lower().endswith(valid_extensions):
            img_path = os.path.join(image_dir, f)
            data.append({"image_path": img_path})
            
    return pd.DataFrame(data)


if __name__ == "__main__":
    # img_dir = r"../OCR/Data/vietnamese_original/train_images"
    # df = create_image_dataframe(img_dir)

    # print(f"Tổng số dòng trong DataFrame: {len(df)}")
    # print(df.head())

    img_path = r"../OCR/Data/vietnamese_original/train_images/im0005.jpg"

    # 1. Đọc ảnh và làm mờ (blur) để giảm nhiễu
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Định nghĩa ngưỡng dưới (lower) và ngưỡng trên (upper)
    # Nếu làm việc trên ảnh xám (Grayscale):
    lower = np.array([50], dtype=np.uint8)
    upper = np.array([200], dtype=np.uint8)

    # (Hoặc nếu làm việc trên không gian màu HSV để lọc màu sắc cụ thể):
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # blur = cv2.GaussianBlur(hsv, (5, 5), 0)
    # lower = np.array([30, 50, 50])
    # upper = np.array([90, 255, 255])

    # 3. Tạo mask bằng cv2.inRange
    mask = cv2.inRange(blur, lower, upper)

    # Hiển thị hoặc sử dụng mask kết quả
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()