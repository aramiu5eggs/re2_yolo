# src/yolo_detection/train_yolo.py

import os
import sys
from ultralytics import YOLO

# プロジェクトルートをsys.pathに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import PROJECT_ROOT, YOLO_MODEL_PATH # YOLO_MODEL_PATHはpre-trained model用
from src.config import DATA_DIR # data.yamlのパス構築に使う


def train_yolov8_model(epochs=200, imgsz=640, batch=16, model_name='yolov8n.pt', project_name='detect', run_name='train_ep200_final'): # run_nameをユニークなものに変更
    """
    YOLOv8モデルをカスタムデータセットでトレーニングします。
    """
    print(f"\n--- Starting YOLOv8 Training ({run_name}) ---")

    model = YOLO(model_name)
    data_yaml_path = os.path.join(DATA_DIR, 'datasets', 'data.yaml')

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project_name,
        name=run_name,
        exist_ok=False,
    )

    print(f"DEBUG: Results should be saved to: {results.save_dir}")

    print(f"YOLOv8 Training completed. Results saved to {results.save_dir}")
    return results.save_dir

if __name__ == '__main__':
    trained_model_dir = train_yolov8_model(epochs=200, batch=16, run_name='train_ep200_final') # ここも run_name を変更
    print(f"Initial training results in: {trained_model_dir}")