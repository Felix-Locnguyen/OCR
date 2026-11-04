# config.py
TRAIN_DIR = r"../OCR/Data/vietnamese_original/train_images"
VAL_DIR = r"../OCR/Data/vietnamese_original/test_image"
LABEL_DIR = r"../OCR/Data/vietnamese_original/labels"
RESULT_DIR = r"../OCR/Data//test_results"
MODEL_PATH = r"../OCR/dbnet_detector.pth"

IMG_SIZE = (512, 512)
BATCH_SIZE = 8
EPOCHS = 20
LR = 0.001
K_BIN = 50.0
SHRINK_RATIO = 0.4