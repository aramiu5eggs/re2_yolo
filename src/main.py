# src/main.py

import os
import sys
from datetime import datetime
import json
import google.generativeai as genai

# === 1. プロジェクトルートをsys.pathに追加 (全モジュールのインポートのために必須) ===
# このスクリプトがどこから実行されても、常にプロジェクトのルートディレクトリ (re2_yolo/) をsys.pathに追加する
# os.path.abspath(__file__) は現在のファイル (main.py) の絶対パス
# os.path.dirname(...) はそのディレクトリ部分 (例: /home/miu/re2_yolo/src)
# os.path.join(..., '..') で親ディレクトリ (例: /home/miu/re2_yolo) を取得
_project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# === 2. 必要なモジュールのインポート ===
# config.pyからの設定値のインポートを最優先。
# これが、コード全体で使用するPROJECT_ROOTになります。
from src.config import PROJECT_ROOT, YOLO_MODEL_PATH, OCR_CONFIDENCE_THRESHOLD, \
                       TARGET_FOOD_YOLO_CLASSES, STANDARD_TO_YOLO_CLASS_MAP, \
                       DATABASE_PATH, GEMINI_API_KEY, GEMINI_MODEL_NAME, \
                       YOLO_CLASS_CONSOLIDATION_MAP, YOLO_CLASS_ALIASES

# YOLOv8関連 - predict_on_imageがモデルロードと推論をラップ
from src.yolo_detection.predict_yolo import predict_on_image  

# OCR関連
from src.ocr_processing.run_ocr import perform_ocr  
from src.ocr_processing.receipt_parser import parse_receipt_text_simple 

# データベース関連
from src.database.db_manager import create_table, add_food_item, update_food_item_quantity, \
                                   update_food_item_details, get_all_food_items, \
                                   mark_as_consumed_or_discarded, delete_food_item, \
                                   get_db_connection, get_food_item_by_id 


# === 4. 各処理フロー関数 === 
def analyze_fridge_image(image_path): 
    """冷蔵庫画像をYOLOv8で解析し、DBを更新する""" 
    print(f"\n--- Analyzing fridge image: {image_path} ---") 
     
    # YOLO検出 
    detected_yolo_items = predict_on_image(image_path) 

    # YOLO検出結果を標準化するロジックをここに組み込む 
    standardized_yolo_items = [] 
    for item in detected_yolo_items: 
        original_yolo_class = item['yolo_class'] 
        # YOLO_CLASS_CONSOLIDATION_MAP を適用 
        consolidated_yolo_class = YOLO_CLASS_CONSOLIDATION_MAP.get(original_yolo_class, original_yolo_class) 
        standardized_yolo_items.append({ 
            'yolo_class': consolidated_yolo_class, 
            'confidence': item['confidence'], 
            'bbox': item['bbox'] 
        }) 

    print("YOLO Detected Items (standardized by YOLO_CLASS_CONSOLIDATION_MAP):") 
    if not standardized_yolo_items: 
        print(" - No target food items detected by YOLO.") 
    else: 
        for item in standardized_yolo_items: 
            print(f" - {item['yolo_class']} (Conf: {item['confidence']:.2f})") 

    # データベース更新ロジック: YOLO検出されたアイテムを既存DBと比較し、更新/追加などを判断 
    current_active_items = get_all_food_items(status='active') 
     
    yolo_counts = {} 
    for item in standardized_yolo_items: 
        yolo_counts[item['yolo_class']] = yolo_counts.get(item['yolo_class'], 0) + 1 
     
    print("YOLO Detected Counts:", yolo_counts) 

    for yolo_detected_item in standardized_yolo_items: 
        yolo_class = yolo_detected_item['yolo_class'] 
         
        found_in_db_for_yolo_update = False 
        for db_item in current_active_items: 
            if db_item['id'] in processed_db_item_ids: # ここは analyze_fridge_image には不要 (receipt_processing用)
                continue # Safety check, but processed_db_item_ids is for receipt processing

            # 1. 完全に同じ標準名（かつアクティブ）のアイテムがDBに既に存在する場合 
            #    YOLO検出で更新する対象として、最も具体的なアイテムを優先 
            if db_item['standard_name'] == yolo_class and db_item['status'] == 'active': 
                # 同じYOLOクラス名を持つアイテムがDBに存在する場合、そのlast_seen_dateを更新 
                update_food_item_details(db_item['id'], last_seen_date=datetime.now().strftime('%Y-%m-%d')) 
                print(f"Updated last seen date for existing '{db_item['standard_name']}' (ID: {db_item['id']}).") 
                found_in_db_for_yolo_update = True 
                break 
         
        if not found_in_db_for_yolo_update: 
            # YOLOで検出されたがDBにないアイテムは、新規追加として扱う 
            # 標準名はYOLOクラス名そのまま（後でレシート情報で具体化されることを期待） 
            print(f"Adding new item '{yolo_class}' from YOLO detection.") 
            add_food_item( 
                standard_name=yolo_class, 
                yolo_class=yolo_class, 
                quantity=1, # YOLO検出では一旦1個と仮定（個体識別は困難なため） 
                purchase_date=datetime.now().strftime('%Y-%m-%d'), # YOLOで検出された日を購入日とする
                detected_by='yolo' 
            ) 
    print("Fridge analysis complete.") 
    return detected_yolo_items 


def process_receipt_image(receipt_image_path): 
    """レシート画像をOCRで解析し、DBを更新する""" 
    print(f"\n--- Processing receipt image: {receipt_image_path} ---") 

    # OCRによるテキスト抽出 
    ocr_results_detail = perform_ocr(receipt_image_path, detail=1) 

    if not ocr_results_detail: 
        print("No text extracted from receipt.") 
        return [] 

    # 信頼度でフィルタリング 
    filtered_text_list = [item[1] for item in ocr_results_detail if item[2] >= OCR_CONFIDENCE_THRESHOLD] 
     
    # レシートテキストの解析 
    parsed_items_from_receipt = parse_receipt_text_simple(filtered_text_list) 

    if not parsed_items_from_receipt: 
        print("No valid food items parsed from receipt.") 
        return [] 

    print("Parsed items from receipt:", parsed_items_from_receipt) 

    current_active_items_in_db = get_all_food_items(status='active') 

    processed_db_item_ids = set() # 既にこのレシート処理で使われたDBアイテムIDを追跡
    
    for item_from_receipt in parsed_items_from_receipt: 
        standard_name_receipt = item_from_receipt['item_name'] 
        quantity_receipt = item_from_receipt['quantity'] 
         
        corresponding_yolo_class = STANDARD_TO_YOLO_CLASS_MAP.get(standard_name_receipt, 'unknown_yolo_class') 

        candidate_yolo_classes_for_match = [corresponding_yolo_class] 
        if corresponding_yolo_class in YOLO_CLASS_ALIASES: # configからインポートした YOLO_CLASS_ALIASES を使う
            candidate_yolo_classes_for_match.extend(YOLO_CLASS_ALIASES[corresponding_yolo_class]) 
        candidate_yolo_classes_for_match = list(set(candidate_yolo_classes_for_match)) # 重複を削除


        # 最適なマッチングアイテムをここで探す 
        best_match_db_item = None 
        is_exact_standard_name_match = False # 最も強いマッチタイプを追跡 

        # 優先順位1: 完全に同じ「標準名」を持つアクティブなアイテムを探す 
        for db_item in current_active_items_in_db: 
            if db_item['id'] in processed_db_item_ids: 
                continue 
            if db_item['standard_name'] == standard_name_receipt and db_item['status'] == 'active': 
                best_match_db_item = db_item 
                is_exact_standard_name_match = True 
                break # 最優先マッチが見つかったら即終了 

        # 優先順位1で見つからなかった場合のみ、優先順位2を探す 
        if best_match_db_item is None: 
            # 優先順位2: YOLOクラスがマッチング候補にあり、かつ標準名が汎用的なアイテムを探す 
            for db_item in current_active_items_in_db: 
                if db_item['id'] in processed_db_item_ids: 
                    continue 
                # YOLOクラスが候補にあり、かつDBのstandard_nameがそのyolo_class名そのまま（汎用名）の場合 
                if (db_item['yolo_class'] in candidate_yolo_classes_for_match and  
                    db_item['standard_name'] == db_item['yolo_class'] and  
                    db_item['status'] == 'active'): 
                      
                     # 念のため、レシートの品目名がYOLOクラス名と異なり、より具体的であることも確認 
                     # （標準名が既に具体的な場合、汎用マッチで上書きしないように） 
                    if standard_name_receipt != db_item['yolo_class']:  
                        best_match_db_item = db_item 
                        break # 最初に見つかった汎用マッチを優先
         # === マッチング結果に基づいて更新または新規追加 === 
        if best_match_db_item: 
            # データベースアイテムを更新 (standard_nameをレシートの具体的な名前に更新) 
            # Quantityも合算 
            update_food_item_details(best_match_db_item['id'],  
                                    standard_name=standard_name_receipt,  
                                    detected_by='both') 
            update_food_item_quantity(best_match_db_item['id'],  
                                    best_match_db_item['quantity'] + quantity_receipt,  
                                    detected_by='both') 
              
            # ログメッセージを状況に合わせて調整 
            if best_match_db_item['standard_name'] == best_match_db_item['yolo_class']: # 具体化された場合 
                print(f"Refined and updated item: '{standard_name_receipt}' (ID: {best_match_db_item['id']}) from receipt (was '{best_match_db_item['yolo_class']}').") 
            else: # 数量更新のみの場合 
                print(f"Updated quantity for existing item: '{standard_name_receipt}' (ID: {best_match_db_item['id']}) from receipt. (Already specific)") 
              
            processed_db_item_ids.add(best_match_db_item['id']) # このアイテムは処理済み 

        else: # DBにマッチするアイテムがない場合 (完全に新規の品目) 
            print(f"Adding new item '{standard_name_receipt}' from receipt.") 
            add_food_item( 
                standard_name=standard_name_receipt, 
                yolo_class=corresponding_yolo_class,  
                quantity=quantity_receipt, 
                purchase_date=datetime.now().strftime('%Y-%m-%d'), 
                detected_by='receipt' 
            ) 
    print("Receipt processing complete.")    

def display_inventory(): 
    """現在の冷蔵庫在庫を表示する""" 
    print("\n--- Current Refrigerator Inventory ---") 
    items = get_all_food_items(status='active') 
    if not items: 
        print("Your refrigerator is empty!") 
        return 

    # 見やすいように整形して表示 
    print(f"{'ID':<4} {'Name':<20} {'YOLO Class':<15} {'Qty':<5} {'Unit':<5} {'Purchase Date':<15} {'Detected By':<12}") 
    print("-" * 80) 
    for item in items: 
         # SQLite.Rowオブジェクトは辞書のようにアクセスできる 
        qty_display = item['quantity'] if item['quantity'] is not None else 0 
        unit_display = item['unit'] if item['unit'] is not None else '-' 
        purchase_date_display = item['purchase_date'] if item['purchase_date'] is not None else '-' 
        detected_by_display = item['detected_by'] if item['detected_by'] is not None else '-' 

        print(f"{item['id']:<4} {item['standard_name']:<20} {item['yolo_class']:<15} {qty_display:<5.1f} {unit_display:<5} {purchase_date_display:<15} {detected_by_display:<12}") 
    print("-" * 80) 

def recommend_recipes_with_llm(): 
    """ 
    YOLOとレシートの両方で検出された食材を使って、LLMにレシピを推薦させる。 
    """ 
    print("\n---レシピ推薦(LLM活用)---") 

     # 1. データベースから 'both' で検出された食材を取得 
    all_items = get_all_food_items()  
    both_detected_items = [] 
    for item in all_items: 
        if item['detected_by'] == 'both': 
            both_detected_items.append(item['standard_name'])  

    if not both_detected_items: 
        print("YOLOとレシートの両方で検出された食材がありません。") 
        print("まず冷蔵庫画像を解析し、その後レシートを処理して食材を紐付けてください。") 
        return 

    unique_ingredients = list(set(both_detected_items)) 
    ingredients_str = ", ".join(unique_ingredients) 

    print(f"冷蔵庫にある食材（両方で検出）: {ingredients_str}") 

     # Gemini APIの初期設定 
    genai.configure(api_key=GEMINI_API_KEY) 
    model = genai.GenerativeModel(GEMINI_MODEL_NAME) # config.pyで定義したモデル名を使用 


     # 2. LLMへのプロンプト生成 
     # LLMがJSONを生成しやすくするために、プロンプトを調整 
    prompt = f"""あなたは料理の専門家であり、レシピの提案者です。 
    冷蔵庫に以下の食材があります。これらの食材をメインに使える、おすすめのレシピを3つ提案してください。 

    # 前提条件 
    - 白米や麺類（パスタ、うどん、そば、中華麺など）は、常に家にあるものとして、自由にレシピに使用して構いません。 

    # 指示 
    - 以下の食材リストにある食材を積極的に活用してください。 
    - 各レシピについて、**食事タイプ（"朝食", "昼食", "夕食"）**、料理名、簡単な説明、主要な材料を教えてください。 
    - 出力は必ずJSON形式で、各レシピはJSON配列の要素としてください。キー名は英語（"meal_type", "name", "description", "ingredients"）で出力してください。 

    食材リスト: {ingredients_str} 

    # 出力形式の例 
    ```json 
    {{ 
      "recipes": [ 
        {{ 
        "meal_type": "朝食", 
        "name": "スクランブルエッグと野菜炒め", 
        "description": "卵と冷蔵庫の野菜で手軽に作れる栄養満点の朝食です。", 
        "ingredients": ["卵", "ピーマン", "玉ねぎ", "塩", "こしょう"] 
         }}, 
         {{ 
           "meal_type": "昼食", 
           "name": "豚肉とキャベツのうどん", 
           "description": "豚肉と野菜の旨味が詰まった、温かい和風うどんです。", 
           "ingredients": ["豚ロース肉", "キャベツ", "うどん", "めんつゆ", "ねぎ"] 
         }}, 
         {{ 
           "meal_type": "夕食", 
           "name": "鶏肉のトマト煮込み", 
           "description": "鶏むね肉をトマトソースでじっくり煮込んだ、ご飯にもパンにも合う一品です。", 
           "ingredients": ["鶏むね肉", "玉ねぎ", "トマト缶", "にんにく", "コンソメ"] 
         }} 
       ] 
     }} 
     ``` 
     出力: 
     """ 

    print("\n--- LLMにレシピをリクエスト中 ---") 

    try: 
         # LLMを呼び出し、応答を受け取る 
         # response_mime_type を application/json に指定することで、Geminiがより厳密にJSONを返そうとします 
        response = model.generate_content( 
            prompt, 
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json") 
        ) 

         # 応答からテキストを抽出（response_mime_type指定により、result.textはJSON文字列になる） 
        llm_response_text = response.text 
          
         # JSON文字列をPythonの辞書にパース 
        parsed_recipes = json.loads(llm_response_text) 
    
         # 4. レシピの表示 
        print("\n--- 🍳 おすすめの献立 🍴 ---") # タイトルを変更 
        if parsed_recipes and "recipes" in parsed_recipes: 
             # meal_typeでソート（朝→昼→夜の順番で表示するため） 
            meal_order = {"朝食": 0, "昼食": 1, "夕食": 2} 
            sorted_recipes = sorted(parsed_recipes["recipes"], key=lambda r: meal_order.get(r.get("meal_type"), 99)) 

            for recipe in sorted_recipes: 
                meal_type = recipe.get('meal_type', '食事') 
                print(f"\n--- {meal_type} ---") 
                print(f"🍽️ 料理名: {recipe.get('name', '不明なレシピ')}") 
                print(f"📝 説明: {recipe.get('description', '説明なし')}") 
                print(f"🥕 材料: {', '.join(recipe.get('ingredients', []))}") 
        else: 
            print("LLMが期待通りのレシピ情報を生成しませんでした。") 
            print("LLM Raw Response:", llm_response_text) # デバッグ用に生の応答を表示 
    except Exception as e: 
        print(f"LLMによるレシピ推薦中にエラーが発生しました: {e}") 
        print("ネットワーク接続やAPIキー、またはLLMの応答形式を確認してください。") 

 # def main(): # <- この行は削除します
     # create_table() はdb_manager.pyでDATABASE_PATHを使用するように修正済みであることを前提 
     # print(f"Database table 'food_items' ensured at {DATABASE_PATH}") # これもdb_managerで出力済み 

     # while True: # <- この行も削除します
     # print("\n--- Menu ---") 
     # print("1. Analyze Fridge Image (YOLO)") 
     # print("2. Process Receipt Image (OCR)") 
     # print("3. Display Current Inventory") 
     # print("4. Recommend Recipes (LLM)") # ★新しいメニュー項目★ 
     # print("5. Mark Item as Consumed/Discarded (Manual)")  
     # print("6. Exit") 
      
     # choice = input("Enter your choice: ") 

     # if choice == '1': 
     #     fridge_img_path = input("Enter path to fridge image (e.g., data/annotated_images/IMG_XXXX.jpg): ") 
     #     fridge_img_path_abs = os.path.join(PROJECT_ROOT, fridge_img_path) 
     #     if os.path.exists(fridge_img_path_abs): 
     #         analyze_fridge_image(fridge_img_path_abs) 
     #     else: 
     #         print(f"Error: Image not found at {fridge_img_path_abs}") 

     # elif choice == '2': 
     #     receipt_img_path = input("Enter path to receipt image (e.g., data/receipt_images/receipt_001.jpg): ") 
     #     receipt_img_path_abs = os.path.join(PROJECT_ROOT, receipt_img_path) 
     #     if os.path.exists(receipt_img_path_abs): 
     #         process_receipt_image(receipt_img_path_abs) 
     #     else: 
     #         print(f"Error: Image not found at {receipt_img_path_abs}") 

     # elif choice == '3': 
     #     display_inventory() 

     # elif choice == '4': # ★このブロックを追加 ★ 
     #     recommend_recipes_with_llm() # 新しく追加したレシピ推薦関数を呼び出す 

     # elif choice == '5': 
     #     # 手動でのアイテム消費/廃棄 
     #     display_inventory() # 選択しやすくするために現在の在庫を表示 
     #     item_id = input("Enter item ID to mark as consumed/discarded: ") 
     #     try: 
     #         item_id = int(item_id) 
     #         db_item = get_food_item_by_id(item_id) 
     #         if db_item and db_item['status'] == 'active': 
     #             status_choice = input("Mark as 'consumed' or 'discarded'? (c/d): ").lower() 
     #             if status_choice == 'c': 
     #                 mark_as_consumed_or_discarded(item_id, 'consumed') 
     #             elif status_choice == 'd': 
     #                 mark_as_consumed_or_discarded(item_id, 'discarded') 
     #             else: 
     #                 print("Invalid status choice. Please enter 'c' or 'd'.") 
     #         elif db_item and db_item['status'] != 'active': 
     #             print(f"Item ID {item_id} is already '{db_item['status']}'.") 
     #         else: 
     #             print(f"Item with ID {item_id} not found.") 
     #     except ValueError: 
     #         print("Invalid ID. Please enter a number.") 

     # elif choice == '6': 
     #     print("Exiting system. Goodbye!") 
     #     break 

     # else: 
     #     print("Invalid choice. Please try again.") 

if __name__ == '__main__': 
    print("Refrigerator Inventory Management System started.") 

    while True: 
        print("\n--- Menu ---") 
        print("1. Analyze Fridge Image (YOLO)") 
        print("2. Process Receipt Image (OCR)") 
        print("3. Display Current Inventory") 
        print("4. Recommend Recipes (LLM)") 
        print("5. Mark Item as Consumed/Discarded (Manual)")  
        print("6. Exit") 
          
        choice = input("Enter your choice: ") 

        if choice == '1': 
            fridge_img_path = input("Enter path to fridge image (e.g., data/annotated_images/IMG_XXXX.jpg): ") 
            fridge_img_path_abs = os.path.join(PROJECT_ROOT, fridge_img_path) 
            if os.path.exists(fridge_img_path_abs): 
                analyze_fridge_image(fridge_img_path_abs) 
            else: 
                print(f"Error: Image not found at {fridge_img_path_abs}") 

        elif choice == '2': 
            receipt_img_path = input("Enter path to receipt image (e.g., data/receipt_images/receipt_001.jpg): ") 
            receipt_img_path_abs = os.path.join(PROJECT_ROOT, receipt_img_path) 
            if os.path.exists(receipt_img_path_abs): 
                process_receipt_image(receipt_img_path_abs) 
            else: 
                print(f"Error: Image not found at {receipt_img_path_abs}") 

        elif choice == '3': 
            display_inventory() 

        elif choice == '4': # ★このブロックを追加 ★ 
            recommend_recipes_with_llm() # 新しく追加したレシピ推薦関数を呼び出す 

        elif choice == '5': 
             # 手動でのアイテム消費/廃棄 
            display_inventory() # 選択しやすくするために現在の在庫を表示 
            item_id = input("Enter item ID to mark as consumed/discarded: ") 
            try: 
                item_id = int(item_id) 
                db_item = get_food_item_by_id(item_id) 
                if db_item and db_item['status'] == 'active': 
                    status_choice = input("Mark as 'consumed' or 'discarded'? (c/d): ").lower() 
                    if status_choice == 'c': 
                        mark_as_consumed_or_discarded(item_id, 'consumed') 
                    elif status_choice == 'd': 
                        mark_as_consumed_or_discarded(item_id, 'discarded') 
                    else: 
                        print("Invalid status choice. Please enter 'c' or 'd'.") 
                elif db_item and db_item['status'] != 'active': 
                    print(f"Item ID {item_id} is already '{db_item['status']}'.") 
                else: 
                    print(f"Item with ID {item_id} not found.") 
            except ValueError: 
                print("Invalid ID. Please enter a number.") 

        elif choice == '6': 
            print("Exiting system. Goodbye!") 
            break 

        else: 
            print("Invalid choice. Please try again.")