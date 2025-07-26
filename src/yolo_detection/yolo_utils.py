# src/yolo_detection/yolo_utils.py

# 現時点では、predict_yolo.pyの内部にロジックが組み込まれているため、
# このファイルは特に必須ではありませんが、
# 例えば、検出結果を画像に描画する関数などをここに追加できます。

def draw_boxes_on_image(image_path, detections, output_path):
    """
    画像に検出されたバウンディングボックスとラベルを描画し、保存する。
    (OpenCVが必要)
    """
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image for drawing: {image_path}")
        return

    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = det['yolo_class']
        conf = det['confidence']

        color = (0, 255, 0) # Green BGR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.imwrite(output_path, img)
    print(f"Detection image saved to {output_path}")