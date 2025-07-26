# src/ocr_processing/run_ocr.py
import sys
import cv2
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import easyocr
import cv2
from PIL import Image 


from src.ocr_processing.image_preprocess import preprocess_receipt_image

reader = easyocr.Reader(['ja', 'en'], gpu=True) # gpu=False if no GPU or issues

def perform_ocr(image_path, detail=0):
    """
    指定された画像パスからテキストを抽出し、結果を返す。
    :param image_path: レシート画像のパス
    :param detail: 0 (テキストのみ), 1 (ボックス、テキスト、信頼度)
    :return: 抽出されたテキストのリスト
    """

    # OpenCVで直接読み込み、そのままEasyOCRに渡す
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    result = reader.readtext(img, detail=detail)
    
    # detail=0 の場合、テキストのリストを返す
    if detail == 0:
        return [item[1] for item in result]
    # detail=1 の場合、ボックス、テキスト、信頼度のリストを返す
    else:
        return result

if __name__ == '__main__':
    receipt_image_path = os.path.join(project_root, 'data', 'receipt_images', 'receipt.jpeg')

    # OCR実行
    if os.path.exists(receipt_image_path):
        print(f"Processing {receipt_image_path}...")

        # OCR実行 (detail=1 でボックス、テキスト、信頼度を取得)
        ocr_results_detail = perform_ocr(receipt_image_path, detail=1)

        if ocr_results_detail:
            print("\n--- Raw OCR Results (detail=1) ---")
            for (bbox, text, prob) in ocr_results_detail:
                print(f"Text: '{text}', Confidence: {prob:.2f}, Box: {bbox}")

            #receipt_parser.py の関数を呼び出す
            from src.ocr_processing.receipt_parser import parse_receipt_text_simple # これを追加

            # 信頼度が高いテキストのみをフィルタリングして渡す
            filtered_text_list = [
                item[1] for item in ocr_results_detail if item[2] >= 0.7 # Confidence >= 0.7
            ]

            print("\n--- Parsed Items (Simple Parser) ---")
            parsed_items = parse_receipt_text_simple(filtered_text_list) # <-- ここで呼び出しています
            for item in parsed_items:
                print(f"Item: {item['item_name']}, Quantity: {item['quantity']}, Raw Line: {item['raw_line']}")


            # もし後でLLMを使うなら、以下のように呼び出す
            # from src.ocr_processing.receipt_parser import parse_receipt_text_with_llm
            # parsed_items_llm = parse_receipt_text_with_llm("\n".join(filtered_text_list)) # LLMには結合したテキストを渡す
            # print("\n--- Parsed Items (LLM Parser) ---")
            # for item in parsed_items_llm:
            #     print(f"Item: {item['name']}, Quantity: {item['quantity']}")

        else:
            print("No text extracted for parsing.")

    else:
        print(f"Receipt image not found at: {receipt_image_path}")
        print("Please place a sample receipt image at data/receipt_images/receipt.jpeg for testing.")