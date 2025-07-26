import os
import random
import shutil

# 設定
source_dir = '../../data/annotated_images'
target_base_dir = '../../data/datasets'

train_ratio = 0.7  # トレーニングデータの割合
val_ratio = 0.15   # バリデーションデータの割合
test_ratio = 0.15  # テストデータの割合

# 各ディレクトリのパス
train_img_dir = os.path.join(target_base_dir, 'train/images')
train_label_dir = os.path.join(target_base_dir, 'train/labels')
val_img_dir = os.path.join(target_base_dir, 'val/images')
val_label_dir = os.path.join(target_base_dir, 'val/labels')
test_img_dir = os.path.join(target_base_dir, 'test/images')
test_label_dir = os.path.join(target_base_dir, 'test/labels')

# ターゲットディレクトリが存在しない場合は作成
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# 画像ファイルのリストを取得（拡張子が.jpgのファイルのみ）
image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
random.shuffle(image_files) # ファイルの順序をシャッフル

num_images = len(image_files)
num_train = int(num_images * train_ratio)
num_val = int(num_images * val_ratio)
# num_testは残りの全て
num_test = num_images - num_train - num_val

# ファイルを分割してコピー
current_idx = 0

# train
for i in range(num_train):
    img_name = image_files[current_idx]
    label_name = img_name.replace('.jpg', '.txt') # .txtファイル名を取得

    shutil.copy(os.path.join(source_dir, img_name), train_img_dir)
    shutil.copy(os.path.join(source_dir, label_name), train_label_dir)
    current_idx += 1

# val
for i in range(num_val):
    img_name = image_files[current_idx]
    label_name = img_name.replace('.jpg', '.txt')

    shutil.copy(os.path.join(source_dir, img_name), val_img_dir)
    shutil.copy(os.path.join(source_dir, label_name), val_label_dir)
    current_idx += 1

# test
for i in range(num_test):
    img_name = image_files[current_idx]
    label_name = img_name.replace('.jpg', '.txt')

    shutil.copy(os.path.join(source_dir, img_name), test_img_dir)
    shutil.copy(os.path.join(source_dir, label_name), test_label_dir)
    current_idx += 1

print(f"データ分割が完了しました。")
print(f"Train: {num_train} images")
print(f"Val: {num_val} images")
print(f"Test: {num_test} images")