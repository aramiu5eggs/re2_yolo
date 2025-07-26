# src/ocr_processing/image_preprocess.py
import cv2
import numpy as np

def preprocess_receipt_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # コントラスト強調 (CLAHEなど) - レシートの薄い文字に有効
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_image = clahe.apply(gray)

    # 二値化 (Otsu's Binarization) - OCRによっては二値化しない方が良い場合もあるため、これはオプション
    _, binary_image = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ここでは二値化を返しますが、必要に応じて enhanced_image (グレースケール強調) を返す選択肢も考慮
    return binary_image

if __name__ == '__main__':
    # このスクリプト単体でテストする場合のコード
    # (例: data/receipt_images/receipt_001.jpg をテスト)
    # 現在のスクリプトのパスから相対的に画像を読み込む
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    sample_image_path = os.path.join(project_root, 'data', 'receipt_images', 'receipt_001.jpg')

    # ダミー画像を読み込んでテスト
    if os.path.exists(sample_image_path):
        preprocessed_img = preprocess_receipt_image(sample_image_path)
        if preprocessed_img is not None:
            cv2.imshow("Preprocessed Image", preprocessed_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(f"Processed image for: {sample_image_path}")
        else:
            print(f"Failed to preprocess: {sample_image_path}")
    else:
        print(f"Sample image not found at: {sample_image_path}")
        print("Please place a sample receipt image at data/receipt_images/receipt_001.jpg for testing.")