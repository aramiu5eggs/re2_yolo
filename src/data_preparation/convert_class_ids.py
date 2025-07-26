# src/data_preparation/convert_class_ids.py

import os
import shutil
import yaml 

# プロジェクトルート設定 (config.pyから読み込むのがベストだが、ここでは直接計算)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 新しい統合されたclasses.txtのパス
NEW_CLASSES_FILE = os.path.join(project_root, 'data', 'datasets', 'classes.txt')

# 元のデータセットのパス
YOUR_OLD_ANNOTATED_DIR = os.path.join(project_root, 'data', 'annotated_images') # あなたの独自アノテーション済みデータ
ROBOFLOW_TRAIN_LABELS_DIR = os.path.join(project_root, 'data', 'datasets', 'train', 'labels') # Roboflow train labels
ROBOFLOW_VAL_LABELS_DIR = os.path.join(project_root, 'data', 'datasets', 'valid', 'labels')   # Roboflow valid labels
ROBOFLOW_TEST_LABELS_DIR = os.path.join(project_root, 'data', 'datasets', 'test', 'labels')   # Roboflow test labels

# 変換後の出力先ディレクトリ（既存のものを上書き）
OUTPUT_TRAIN_LABELS_DIR = os.path.join(project_root, 'data', 'datasets', 'train', 'labels')
OUTPUT_VAL_LABELS_DIR = os.path.join(project_root, 'data', 'datasets', 'valid', 'labels')
OUTPUT_TEST_LABELS_DIR = os.path.join(project_root, 'data', 'datasets', 'test', 'labels')


def load_class_names(filepath):
    """クラス名をファイルから読み込む"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def create_class_id_map(old_classes_file, new_classes_list):
    """
    古いクラスファイルから新しいクラスリストへのIDマッピングを作成する。
    old_classes_file: 古いクラス名が書かれたファイルのパス
    new_classes_list: 新しいクラス名リスト (['apple', 'banana', ...])
    """
    old_class_names = load_class_names(old_classes_file)

    # クラス名 -> 新しいID のマップ
    new_name_to_id = {name: i for i, name in enumerate(new_classes_list)}

    # 古いID -> 新しいID のマップ
    id_map = {}
    for old_id, old_name in enumerate(old_class_names):
        if old_name in new_name_to_id:
            id_map[old_id] = new_name_to_id[old_name]
        else:
            print(f"Warning: Class '{old_name}' (old ID: {old_id}) not found in new class list. It will be skipped.")
    return id_map

def convert_annotations(input_label_dir, output_label_dir, old_id_to_new_id_map):
    """
    指定されたディレクトリ内のアノテーションファイルを新しいクラスIDに変換する。
    """
    print(f"Converting annotations in {input_label_dir}...")
    for filename in os.listdir(input_label_dir):
        if filename.endswith('.txt') and filename != 'classes.txt':
            input_filepath = os.path.join(input_label_dir, filename)
            output_filepath = os.path.join(output_label_dir, filename)

            print(f"DEBUG: Processing file: {input_filepath}")

            new_lines = []
            with open(input_filepath, 'r') as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    old_class_id = int(parts[0])
                    if old_class_id in old_id_to_new_id_map:
                        new_class_id = old_id_to_new_id_map[old_class_id]
                        new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                        new_lines.append(new_line)
                    else:
                        print(f"Skipping line for unknown old class ID {old_class_id} in {filename}")

            with open(output_filepath, 'w') as f_out:
                f_out.writelines(new_lines)

    print(f"Conversion for {input_label_dir} completed.")

if __name__ == '__main__':
    # --- 1. 新しい統合されたクラスリストを読み込む ---
    new_classes_list = load_class_names(NEW_CLASSES_FILE)
    print(f"New integrated classes: {new_classes_list}")

    new_name_to_id = {name: i for i, name in enumerate(new_classes_list)}

    # --- 2. あなたの独自データのアノテーションを変換 ---
    # ここで、あなたがアノテーションした際の classes.txt のパスを指定します。
    # 例: あなたの古い classes.txt のパス
    YOUR_OLD_CLASSES_FILE = os.path.join(project_root, 'data', 'datasets', 'old_your_classes.txt') # これを適切なパスに修正
    # ★重要: この old_your_classes.txt には、あなたがLabelImgで使っていたクラスの順番を記述しておく必要があります。

    # まず、あなたの古いclasses.txtを一時的に old_your_classes.txt として保存してください
    # 例: cp ~/re_yolo/data/datasets/classes.txt ~/re2_yolo/data/datasets/old_your_classes.txt

    your_id_map = create_class_id_map(YOUR_OLD_CLASSES_FILE, new_classes_list)
    print(f"Your old ID map: {your_id_map}")

    # 変換後のファイルを保存するディレクトリを確保
    os.makedirs(OUTPUT_TRAIN_LABELS_DIR, exist_ok=True) # ここにあなたのデータをコピーするので、出力先はtrain/labelsになります
    # あなたの独自データの画像を train/images にコピー
    YOUR_OLD_ANNOTATED_IMAGES_DIR = os.path.join(project_root, 'data', 'annotated_images')
    for img_file in os.listdir(YOUR_OLD_ANNOTATED_IMAGES_DIR):
        if img_file.endswith('.jpg'):
            shutil.copy(os.path.join(YOUR_OLD_ANNOTATED_IMAGES_DIR, img_file), 
                        os.path.join(project_root, 'data', 'datasets', 'train', 'images'))

    convert_annotations(YOUR_OLD_ANNOTATED_DIR, OUTPUT_TRAIN_LABELS_DIR, your_id_map)

    # --- 3. Roboflowデータのアノテーションを変換 ---
    # RoboflowのデータセットにもクラスIDが存在するため、それも新しいマップに合わせる必要があります。
    # Roboflowのdata.yamlに記載されているクラスリストを使用
    ROBOFLOW_DATA_YAML_PATH = os.path.join(project_root, 'data', 'datasets', 'data.yaml') # これはRoboflowのdata.yaml
    roboflow_temp_classes = load_class_names(ROBOFLOW_DATA_YAML_PATH) # data.yamlのnamesはroboflowのクラス名そのまま
    # Roboflowのnamesリストを直接クラス名として読み込む
    with open(ROBOFLOW_DATA_YAML_PATH, 'r') as f:
        roboflow_data_yaml_content = yaml.safe_load(f)
    roboflow_names_in_order = roboflow_data_yaml_content.get('names', [])

    roboflow_id_map = {}
    for old_id, old_name in enumerate(roboflow_names_in_order):
        if old_name in new_classes_list:
            roboflow_id_map[old_id] = new_name_to_id[old_name] # new_name_to_id は new_classes_list から作成したマップ
        else:
            print(f"Warning: Roboflow class '{old_name}' (old ID: {old_id}) not found in new class list. It will be skipped.")

    print(f"Roboflow old ID map: {roboflow_id_map}")

    convert_annotations(ROBOFLOW_TRAIN_LABELS_DIR, OUTPUT_TRAIN_LABELS_DIR, roboflow_id_map)
    convert_annotations(ROBOFLOW_VAL_LABELS_DIR, OUTPUT_VAL_LABELS_DIR, roboflow_id_map)
    convert_annotations(ROBOFLOW_TEST_LABELS_DIR, OUTPUT_TEST_LABELS_DIR, roboflow_id_map)

    print("\nAll annotation class IDs converted and data combined!")
    print("You can now proceed to train YOLOv8 with the combined dataset.")