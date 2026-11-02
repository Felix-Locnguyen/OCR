"""
Đọc ảnh -> Chuyển Grayscale.
Áp dụng Perspective Transform / Deskew (nếu ảnh bị nghiêng).
Tăng cường độ tương phản bằng CLAHE.
Resize giữ tỉ lệ với height = 32 + Pad chiều rộng về kích thước chuẩn của batch.
Scale pixel về [0, 1] và đưa vào Tensor.
"""

import cv2
import numpy as np
import pandas as pd
import os

from Preprocess.utils import ( convert_to_grayscale, load_dataset_to_dataframe,
                              find_document_corners, deskew_perspective,
                              apply_clahe, resize_and_pad,normalization,
                              )


def process_document_pipeline(
    df, 
    image_col='image_path', 
    target_height=32, 
    target_width=None, 
    clip_limit=2.0, 
    tile_grid_size=(8, 8),
    save=False,                  # Thêm tham số bật/tắt tính năng lưu ảnh
    output_dir='processed_outputs' # Thư mục lưu ảnh đầu ra
):
    """
    Pipeline xử lý ảnh toàn diện dành cho OCR từ DataFrame chứa đường dẫn ảnh.
    
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu đầu vào.
        image_col (str): Tên cột chứa đường dẫn file ảnh trong DataFrame.
        target_height (int): Chiều cao chuẩn cho model OCR (mặc định 32).
        target_width (int, optional): Chiều rộng cố định của batch. Nếu None, giữ nguyên chiều dài linh hoạt.
        clip_limit (float): Ngưỡng giới hạn độ tương phản cho CLAHE.
        tile_grid_size (tuple): Kích thước lưới chia ảnh cho CLAHE.
        save (bool): Có lưu ảnh đã xử lý xuống thư mục hay không.
        output_dir (str): Đường dẫn thư mục lưu ảnh khi save=True.
        
    Returns:
        pd.DataFrame: DataFrame bổ sung các cột:
                      - 'processed_image': mảng numpy chứa giá trị đã chuẩn hóa để feed vào model.
                      - 'saved_path': đường dẫn file ảnh đã lưu (nếu save=True).
                      - 'processing_status': trạng thái thành công hay lỗi.
    """
    # Nếu bật chế độ save, tạo thư mục lưu nếu chưa tồn tại
    if save and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    processed_images = []
    saved_paths = []
    statuses = []
    
    for idx, path in enumerate(df[image_col]):
        try:
            # Bước 1: Đọc ảnh
            if not os.path.exists(path):
                raise FileNotFoundError(f"Không tìm thấy file: {path}")
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Không thể giải mã file ảnh: {path}")
                
            # Bước 2: Tự động tìm góc và kéo thẳng tài liệu
            pts = find_document_corners(image)
            if pts is not None:
                image = deskew_perspective(image, pts)
                
            # Bước 3: Tăng cường độ tương phản bằng CLAHE
            image = apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
            
            # Bước 4: Chuyển đổi sang ảnh xám (Grayscale)
            image = convert_to_grayscale(image)
            
            # Bước 5: Chuẩn hóa kích thước (Resize & Pad)
            image = resize_and_pad(image, target_height=target_height, target_width=target_width)
            
            # --- BƯỚC LƯU ẢNH ---
            saved_path = None
            if save:
                # Đặt tên file mới dựa theo index và tên gốc để tránh trùng lặp
                file_name = f"processed_{idx}_{os.path.basename(path)}"
                saved_path = os.path.join(output_dir, file_name)
                cv2.imwrite(saved_path, image) # Lưu ảnh ở dạng uint8 (0-255 chuẩn)
            
            # Bước 6: Chuẩn hóa giá trị pixel về khoảng [0, 1] (Dùng để đưa vào Model)
            normalized_image = normalization(image)
            
            processed_images.append(normalized_image)
            saved_paths.append(saved_path)
            statuses.append("Success")
            
        except Exception as e:
            processed_images.append(None)
            saved_paths.append(None)
            statuses.append(f"Error: {str(e)}")
            
    # Đóng gói kết quả vào DataFrame mới
    result_df = df.copy()
    result_df['processed_image'] = processed_images
    if save:
        result_df['saved_path'] = saved_paths
    result_df['processing_status'] = statuses
    
    return result_df


if __name__ == "__main__":
    img_dir = r""
    label_dir = r""
    output_dir = r""

    df_input = load_dataset_to_dataframe(image_dir=img_dir, 
                                         label_dir=label_dir,
                                         columns=['image_path', 'label_path'],
                                         save = False, # có thể chỉnh nếu muốn lưu
                                         output_dir= output_dir)

    preprocess_output_dir= r""
    df_output = process_document_pipeline(
        df=df_input, 
        image_col='image_path', 
        target_height=32,
        save=True,                   # Kích hoạt lưu ảnh
        output_dir=preprocess_output_dir   # Thư mục tùy chỉnh (tuỳ chọn)
    )