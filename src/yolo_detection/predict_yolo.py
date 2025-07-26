# src/yolo_detection/predict_yolo.py

import os
import sys
from ultralytics import YOLO

# プロジェクトルートをsys.pathに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD, TARGET_FOOD_YOLO_CLASSES


# YOLOv8モデルのロード（一度だけ行う）
# train.pyで学習したbest.ptモデルをロード
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"YOLOv8 prediction model loaded from {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"Error loading YOLOv8 model from {YOLO_MODEL_PATH}: {e}")
    print("Please ensure your YOLO_MODEL_PATH in src/config.py is correct and the model exists.")
    yolo_model = None


def predict_on_image(image_path, conf_threshold=YOLO_CONFIDENCE_THRESHOLD, target_classes=TARGET_FOOD_YOLO_CLASSES):
    """
    指定された画像パスの食材をYOLOv8モデルで検出し、結果を返す。

    Args:
        image_path (str): 推論対象の画像パス。
        conf_threshold (float): 検出の信頼度閾値。
        target_classes (list): 検出結果をフィルタリングするターゲット食材のYOLOクラス名リスト。

    Returns:
        list: 検出された各アイテムの辞書のリスト
              例: [{'yolo_class': 'milk', 'confidence': 0.95, 'bbox': [x1, y1, x2, y2]}, ...]
    """
    if yolo_model is None:
        print("YOLO model not loaded. Cannot perform prediction.")
        return []

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return []

    print(f"Performing YOLO prediction on: {image_path}")
    
    # YOLOv8で推論を実行
    # save=False: 結果画像を保存しない (main.pyで制御)
    # verbose=False: 詳細なログを出力しない
    results = yolo_model.predict(source=image_path, conf=conf_threshold, save=False, verbose=False, iou=0.7)
    
    detected_items = []
    # YOLOv8モデルの .names 属性からクラス名マップを取得
    class_names_map = yolo_model.names

    for r in results: # 各画像の結果
        for box in r.boxes: # 各検出されたオブジェクト
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist() # バウンディングボックス座標 [x1, y1, x2, y2]
            
            yolo_class_name = class_names_map.get(class_id, "unknown")

            # ターゲット食材クラスでフィルタリング
            if yolo_class_name in target_classes:
                detected_items.append({
                    'yolo_class': yolo_class_name,
                    'confidence': confidence,
                    'bbox': bbox,
                })
    
    print(f"YOLO prediction completed. Detected {len(detected_items)} target items.")
    return detected_items

if __name__ == '__main__':
    # ターミナルから直接推論をテストする場合の例
    # 適当な冷蔵庫の画像パスを指定
    test_image_path = os.path.join(PROJECT_ROOT, 'data', 'annotated_images', 'IMG_XXXX.jpg') # XXXXを実際のファイル名に

    # 存在確認
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}. Please update the path or provide a test image.")
    else:
        detected_results = predict_on_image(test_image_path)
        print("\n--- Detected Items ---")
        for item in detected_results:
            print(f"Class: {item['yolo_class']}, Confidence: {item['confidence']:.2f}")