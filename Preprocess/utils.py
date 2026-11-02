import os
import cv2
import numpy as np 

import os
import pandas as pd


def load_dataset_to_dataframe(
    image_dir, label_dir, column_names, save=False, output_dir=None
):
  """Đọc đường dẫn ảnh và nội dung label tương ứng, tạo pandas DataFrame,

  hiển thị 5 dòng đầu và tùy chọn lưu trữ.

  Args:
      image_dir (str): Thư mục chứa ảnh.
      label_dir (str): Thư mục chứa file label (txt).
      column_names (list): Danh sách tên cột cho DataFrame (ví dụ:
        ['image_path', 'label']).
      save (bool): Có lưu DataFrame thành file CSV hay không (Mặc định False).
      output_dir (str, optional): Thư mục để lưu file khi save=True.

  Returns:
      pandas.DataFrame: DataFrame chứa thông tin dataset.
  """
  data = []

  # Các định dạng ảnh phổ biến
  valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

  # Lấy danh sách file ảnh và sắp xếp
  if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {image_dir}")

  image_files = sorted(
      [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]
  )

  for img_file in image_files:
    base_name, _ = os.path.splitext(img_file)
    img_path = os.path.join(image_dir, img_file)

    # Giả định file label có cùng tên cơ sở với ảnh nhưng đuôi .txt
    label_file = base_name + ".txt"
    label_path = os.path.join(label_dir, label_file)

    label_content = ""
    if os.path.exists(label_path):
      with open(label_path, "r", encoding="utf-8") as f:
        label_content = f.read().strip()

    # Thêm vào danh sách dữ liệu
    data.append([img_path, label_content])

  # Tạo DataFrame với tên cột truyền vào
  if len(column_names) != 2:
    raise ValueError(
        "column_names cần cung cấp đúng 2 phần tử (ví dụ: ['image_path',"
        " 'label']) tương ứng với 2 trường dữ liệu thu thập."
    )

  df = pd.DataFrame(data, columns=column_names)

  # Hiển thị 5 dòng đầu tiên
  print("--- 5 dòng đầu của Dataset ---")
  print(df.head(5))
  print(f"Tổng số mẫu đọc được: {len(df)}")

  # Lưu file nếu save=True
  if save:
    if output_dir is None:
      raise ValueError(
          "Vui lòng cung cấp đường dẫn 'output_dir' khi bật save=True!"
      )

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "dataset_manifest.csv")

    # Lưu định dạng utf-8-sig để hỗ trợ tiếng Việt tốt trong Excel
    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu DataFrame thành công tại: {output_file_path}")

  return df

def read_image(image_path, flag=cv2.IMREAD_COLOR):
  """Đọc ảnh từ đường dẫn bằng OpenCV.

  Args:
      image_path (str): Đường dẫn tới file ảnh.
      flag (int): Chế độ đọc ảnh (Mặc định cv2.IMREAD_COLOR: ảnh màu BGR,
        cv2.IMREAD_GRAYSCALE: ảnh xám).

  Returns:
      numpy.ndarray: Mảng numpy chứa dữ liệu ảnh.
  """
  if not os.path.exists(image_path):
    raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {image_path}")

  image = cv2.imread(image_path, flag)

  if image is None:
    raise ValueError(f"Không thể đọc hoặc giải mã file ảnh: {image_path}")

  return image



def convert_to_grayscale(image):
  """Chuyển đổi ảnh màu (BGR) sang ảnh xám (Grayscale).

  Args:
      image (numpy.ndarray): Ảnh đầu vào.

  Returns:
      numpy.ndarray: Ảnh xám (1 kênh).
  """
  # Kiểm tra nếu ảnh có 3 kênh màu (BGR) thì mới chuyển
  if len(image.shape) == 3:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Nếu ảnh đã là ảnh xám rồi thì giữ nguyên
  return image




def find_document_corners(image):
  """Tự động tìm 4 góc của vật thể/tờ giấy chứa chữ dựa trên biên cạnh.

  Args:
      image (numpy.ndarray): Ảnh đầu vào (BGR).

  Returns:
      numpy.ndarray: Mảng 4 điểm [[x, y], ...] hoặc None nếu không tìm thấy.
  """
  # 1. Chuyển sang ảnh xám và làm mờ để giảm nhiễu
  if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    gray = image

  blurred = cv2.GaussianBlur(gray, (5, 5), 0)

  # 2. Dò tìm cạnh bằng Canny Edge Detection
  edged = cv2.Canny(blurred, 75, 200)

  # 3. Tìm các đường viền (contours) trên ảnh
  contours, _ = cv2.findContours(
      edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
  )
  # Sắp xếp các contour theo diện tích từ lớn đến nhỏ, lấy 5 contour lớn nhất
  contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

  screen_cnt = None
  for c in contours:
    # Tính chu vi đường viền
    peri = cv2.arcLength(c, True)
    # Xấp xỉ đa giác (lọc các đường viền có số cạnh)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # Nếu đa giác xấp xỉ có đúng 4 đỉnh -> chính là khung vật thể cần tìm
    if len(approx) == 4:
      screen_cnt = approx
      break

  if screen_cnt is not None:
    # Trả về dạng mảng (4, 2) kiểu float32
    return screen_cnt.reshape(4, 2).astype("float32")

  return None  # Không tìm thấy khung 4 góc phù hợp



def deskew_perspective(image, pts):
  """Kéo thẳng vùng ảnh chứa chữ bị nghiêng dựa trên 4 điểm góc (Perspective Transform).

  Args:
      image (numpy.ndarray): Ảnh đầu vào.
      pts (list hoặc numpy.ndarray): Mảng 4 điểm [[x, y], ...] của vùng chứa chữ.

  Returns:
      numpy.ndarray: Ảnh đã được kéo thẳng và cắt vuông vắn.
  """
  pts = np.array(pts, dtype="float32")

  # 1. Sắp xếp 4 điểm theo thứ tự chuẩn: Top-Left, Top-Right, Bottom-Right, Bottom-Left
  s = pts.sum(axis=1)
  tl = pts[np.argmin(s)]
  br = pts[np.argmax(s)]

  diff = np.diff(pts, axis=1)
  tr = pts[np.argmin(diff)]
  bl = pts[np.argmax(diff)]

  rect = np.array([tl, tr, br, bl], dtype="float32")

  # 2. Tính toán chiều rộng và chiều cao tối ưu cho ảnh đích
  widthA = np.linalg.norm(br - bl)
  widthB = np.linalg.norm(tr - tl)
  max_width = int(max(widthA, widthB))

  heightA = np.linalg.norm(tr - br)
  heightB = np.linalg.norm(tl - bl)
  max_height = int(max(heightA, heightB))

  # 3. Định nghĩa khung tọa độ đích hình chữ nhật
  dst = np.array(
      [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
      dtype="float32",
  )

  # 4. Tính ma trận biến đổi và áp dụng warp
  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(
      image, M, (max_width, max_height), flags=cv2.INTER_CUBIC
  )

  return warped


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Tăng cường độ tương phản ảnh sử dụng thuật toán CLAHE.
    
    Parameters:
        image (numpy.ndarray): Ảnh đầu vào (ảnh màu BGR hoặc ảnh xám grayscale).
        clip_limit (float): Ngưỡng giới hạn độ tương phản (mặc định: 2.0).
        tile_grid_size (tuple): Kích thước lưới chia ảnh để cân bằng cục bộ (mặc định: (8, 8)).
        
    Returns:
        numpy.ndarray: Ảnh sau khi đã áp dụng CLAHE.
    """
    # Kiểm tra xem ảnh là ảnh màu (3 kênh) hay ảnh xám (1 kênh)
    if len(image.shape) == 3:
        # Chuyển đổi từ không gian màu BGR sang LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        
        # Tách các kênh L, A, B
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Khởi tạo đối tượng CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Áp dụng CLAHE riêng lên kênh độ sáng L (Lightness)
        cl = clahe.apply(l_channel)
        
        # Gộp kênh L đã xử lý lại với kênh A và B
        merged_lab = cv2.merge((cl, a_channel, b_channel))
        
        # Chuyển đổi ngược lại không gian màu từ LAB sang BGR
        result_image = cv2.cvtColor(merged_lab, cv2.COLOR_Lab2BGR)
    else:
        # Nếu là ảnh xám, áp dụng trực tiếp
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        result_image = clahe.apply(image)
        
    return result_image


def resize_and_pad(image, target_height=32, target_width=None):
  """Resize ảnh giữ nguyên tỉ lệ với chiều cao cố định và pad chiều rộng.

  Args:
      image (numpy.ndarray): Ảnh đầu vào (ảnh xám hoặc ảnh màu).
      target_height (int): Chiều cao cố định (mặc định 32 cho các model OCR như
        CRNN/VietOCR).
      target_width (int, optional): Chiều rộng chuẩn của batch (ví dụ: 128, 256).
        Nếu để None, hàm sẽ trả về ảnh sau khi resize và pad tự nhiên theo chiều
        dài thực tế.

  Returns:
      numpy.ndarray: Ảnh đã được chuẩn hóa kích thước.
  """
  h, w = image.shape[:2]

  # 1. Tính chiều rộng mới theo tỉ lệ chuẩn của chiều cao (target_height = 32)
  new_height = target_height
  new_width = int(w * (target_height / h))

  # Resize giữ tỉ lệ khung hình
  resized = cv2.resize(
      image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
  )

  # 2. Xử lý padding chiều rộng
  if target_width is not None:
    if new_width > target_width:
      # Nếu ảnh quá dài so với chuẩn batch, tiến hành resize ép về target_width (ít khuyên dùng nhưng an toàn kích thước)
      return cv2.resize(
          resized, (target_width, target_height), interpolation=cv2.INTER_CUBIC
      )

    # Tính lượng pixel cần bù thêm vào bên phải để đủ target_width
    pad_width = target_width - new_width

    # Chọn giá trị màu padding (255 là màu trắng, phù hợp với nền văn bản thông thường)
    pad_value = [255, 255, 255] if len(image.shape) == 3 else 255

    # Thêm padding bên phải (top, bottom, left, right)
    padded_image = cv2.copyMakeBorder(
        resized, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=pad_value
    )
    return padded_image

  return resized



def normalization(image):
  """Chuẩn hóa ảnh về khoảng giá trị [0, 1].

  Args:
      image (numpy.ndarray): Ảnh đầu vào (ảnh xám hoặc ảnh màu).

  Returns:
      numpy.ndarray: Ảnh đã được chuẩn hóa.
  """
  # Chuyển đổi sang float32 để tránh tràn số khi chia
  image = image.astype(np.float32)

  # Chuẩn hóa về khoảng [0, 1]
  normalized_image = image / 255.0

  return normalized_image