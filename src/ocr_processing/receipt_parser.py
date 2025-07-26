# src/ocr_processing/receipt_parser.py
import re
import json # LLMを使用する場合に備えてインポート

# ----------------------------------------------------
# 簡易的な正規表現とキーワードマッチングによる解析関数
# ----------------------------------------------------
def parse_receipt_text_simple(extracted_text_list):
    """
    EasyOCRから抽出されたテキストリストから、品目と数量を簡易的に解析する。
    """
    parsed_items = []
    
    # 仮のキーワードリスト (あなたのプロジェクトの食材に合わせてカスタマイズしてください)
    # レシートに現れる可能性のある表記ゆれも考慮に入れると良い
    food_keywords_map = {
        '牛乳': ['牛乳', 'ぎゅうにゅう', 'ミルク', '特濃'], 
        '卵': ['たまご', '卵', '玉子', 'タマゴ', 'たまごL10コ', '鶏卵', '白M10個'], # '白M10個'のような具体的な表記もキーワードに
        '豚ロース肉': ['豚肉ローススライス', '豚肉', '豚ロース', 'ロース'], # レシートから抽出したい具体的名称
        '鶏むね肉': ['鶏むね肉', '東北産若どりむね肉', 'むね肉', '若どり'], # レシートから抽出したい具体的名称
        '肉（その他）': ['肉', '牛肉', 'もも肉', 'バラ肉'], # 汎用的な肉は「肉（その他）」のような標準名に
        '鮭': ['鮭', 'サケ', 'しゃけ'], # 具体的な魚名
        '魚（その他）': ['魚', 'マグロ', '鯛', 'ブリ'], # 汎用的な魚名
        '味噌': ['みそ', '味噌'],                             
        '豆腐': ['豆腐', 'とうふ'],                             
        'トマト': ['トマト', 'トマト袋'], # 'トマト袋'もキーワードに
        'きゅうり': ['きゅうり', '胡瓜', 'きゅうり袋'], # 'きゅうり袋'もキーワードに                       
        'なす': ['なす', 'ナス', '茄子', '長なす'],           
        'にんじん': ['にんじん', '人参'],                         
        '玉ねぎ': ['たまねぎ', '玉ねぎ', '玉葱'],                 
        'キャベツ': ['キャベツ'],                            
        'ピーマン': ['ピーマン'],                          
        'ほうれん草': ['ほうれん草', 'ホウレン草'],
        '小松菜': ['小松菜'],
        'レタス': ['レタス'], # leafy_greenとは別にレタス自体を標準名に
        'きのこ': ['きのこ', 'キノコ', 'しめじ', 'エノキ', '椎茸', 'まいたけ'],
        'もやし': ['もやし'],                           
        'ビール': ['ビール', 'BEER', 'びーる'],                    
        'チーズ': ['チーズ'],                               
        '納豆': ['納豆', 'なっとう'],                         
        'ヨーグルト': ['ヨーグルト', 'プレーンソ', 'プレーン'],      
        'ボトル飲料': ['ボトル', '水', 'お茶', 'ドリンク', 'PET'], 

        # Roboflowのクラスに対応する日本語名
        'りんご': ['りんご', 'リンゴ'],
        'バナナ': ['バナナ'],
        'ブロッコリー': ['ブロッコリー'],
        'コーン': ['コーン', 'とうもろこし'],
        'ぶどう': ['ぶどう', 'ブドウ'],
        'キウイ': ['キウイ'],
        'レモン': ['レモン'],
        'オレンジ': ['オレンジ'],
        'マンゴー': ['マンゴー'],
        'スイカ': ['スイカ'],

        # その他、YOLO学習クラスではないが、レシートから抽出したい具体的品目
        'ロイヤルブレッド': ['ロイヤルブレッド'],
        'プルーン': ['プルーン', 'TVプルーン種ぬき'], # 具体的な表記
        'おにぎり': ['おにぎり', '0尺おにぎり'], # 具体的な表記

        # 汎用的なYOLOクラス名が直接抽出された場合も考慮
        'apple': ['apple'], 'banana': ['banana'], 'broccoli': ['broccoli'], 'corn': ['corn'],
        'cucumber': ['cucumber'], 'eggplant': ['eggplant'], 'grape': ['grape'], 'kiwi': ['kiwi'],
        'lemon': ['lemon'], 'lettuce': ['lettuce'], 'mango': ['mango'], 'orange': ['orange'],
        'watermelon': ['watermelon'], 'milk': ['milk'], 'egg': ['egg'], 'meat': ['meat'],
        'fish': ['fish'], 'miso': ['miso'], 'tofu': ['tofu'], 'tomato': ['tomato'],
        'carrot': ['carrot'], 'onion': ['onion'], 'cabbage': ['cabbage'], 'bell_pepper': ['bell_pepper'],
        'leafy_green': ['leafy_green'], 'mushroom': ['mushroom'], 'bean_sprout': ['bean_sprout'],
        'beer': ['beer'], 'cheese': ['cheese'], 'natto': ['natto'], 'yogurt': ['yogurt'], 'bottle': ['bottle'],

        'meatballs': ['ミートボール'],
        'marinara sauce': ['マリナーラ'],
        'tomato soup': ['トマトスープ'],
        'chicken noodle soup': ['チキンヌードルスープ'],
        'french onion soup': ['フレンチオニオンスープ'],
        'ribs': ['リブ', 'スペアリブ'],
        'pulled pork': ['プルドポーク'],
        'hamburger': ['ハンバーガー'],
        'ロイヤルブレッド': ['ロイヤルブレッド'],
        'プルーン': ['プルーン'], 
        'おにぎり': ['おにぎり'],

    }

    reverse_keyword_map = {}
    for standard_name, keywords in food_keywords_map.items():
        for kw in keywords:
            reverse_keyword_map[kw.lower()] = standard_name

    for line_text in extracted_text_list:
        line_text_norm = line_text.lower().replace(' ', '').replace('　', '').replace('※', '')

        if re.fullmatch(r'\d+(\.\d+)?(円|※)?$|[-+]\d+%?$', line_text_norm): # 123円, 123.00, 123※, -40, 20% など
             continue

        found_standard_name = None

        sorted_keywords = sorted(reverse_keyword_map.keys(), key=len, reverse=True)
        for keyword in sorted_keywords:
            if keyword in line_text_norm: 
                found_standard_name = reverse_keyword_map[keyword]
                break
        
        if found_standard_name:
            quantity = 1
            # 1. 数字+単位 (例: 10個, 1袋)
            qty_match = re.search(r'(\d+)\s*([個袋本入組k])', line_text_norm) # 'k'はキログラムのkなどの誤認識対策
            if qty_match:
                quantity = int(qty_match.group(1))
            else:
                # 2. サイズ+数量 (例: L10コ -> 10)
                qty_match = re.search(r'([lLsSＭM])?(\d+)[コ個]', line_text_norm)
                if qty_match:
                    quantity = int(qty_match.group(2))
                else:
                    # 3. 行内の数字を数量とみなす場合 (価格ではないことを前提)
                    # ただし、「400g」のようなグラム表示は数量1とすべき
                    if re.search(r'\d+g$', line_text_norm): # '400g'のようにグラム表記で終わる場合
                        quantity = 1
                    else:
                        # 行の先頭にある数字を数量とみなす（価格ではないと判断できる場合）
                        qty_match = re.search(r'^(\d+)', line_text_norm)
                        if qty_match:
                            # ただし、その数字が単独で価格として認識される可能性がないか確認
                            # 例えば '230' だけの行は数量ではない
                            if not re.fullmatch(r'\d+', line_text_norm): # 行全体が数字だけなら数量ではない
                                quantity = int(qty_match.group(1))
                            else:
                                quantity = 1 # 数字だけの行はデフォルト1 (ただし価格の可能性が高いので注意)
                        else:
                            quantity = 1 # デフォルト

            if quantity == 0 and "おにぎり" in found_standard_name: 
                quantity = 1

                        

            price_as_item_names = [str(x) for x in range(1, 1000)]
            price_as_item_names.extend(['-40', '20%'])

            if found_standard_name in price_as_item_names:
                continue

            parsed_items.append({
                'item_name': found_standard_name,
                'quantity': quantity,
                'raw_line': line_text # デバッグ用に元の行を残す
            })

    return parsed_items

# ----------------------------------------------------
# LLM（大規模言語モデル）を用いた解析関数（推奨）
# ----------------------------------------------------
def parse_receipt_text_with_llm(ocr_raw_text):
    """
    LLM (Gemmaなど) を用いて、レシートの生テキストから品目と数量を抽出する。
    """
    # LLM API呼び出しの擬似コード
    # 実際には、Google AI StudioのGemini APIなどを利用します。
    # APIキーの管理、リクエスト形式、レスポンスのパースなどが必要です。

    # プロンプトの例（LLMへの指示文）
    prompt = f"""以下のレシートのテキストから、購入された食材の品目名と数量を抽出し、JSON配列の形式で出力してください。
    品目名は、以下のリストにある一般的な名称に正規化してください。もしリストにない場合は、テキストから最も近い一般的な名称を推測してください。
    数量が不明な場合は1としてください。

    一般的な食材の名称リスト:
    牛乳, 卵, 肉, 魚, 味噌, 豆腐, トマト, きゅうり, なす, もやし, レタス, ほうれん草, 小松菜, きのこ, しめじ, えのき, ビール, チーズ, 納豆, ヨーグルト, ボトル飲料, (その他あなたのYOLOクラスに対応する具体的な野菜名など)

    レシートテキスト:
    ```
    {ocr_raw_text}
    ```

    出力例:
    ```json
    {{
      "items": [
        {{"name": "牛乳", "quantity": 1}},
        {{"name": "卵", "quantity": 10}},
        {{"name": "ほうれん草", "quantity": 1}}
      ]
    }}
    ```
    出力:
    """

    print("\n--- LLMを活用してレシートを解析します（これはシミュレーションです）---")
    print(f"プロンプトをLLMに送信中...")
    
    # 実際にはここでLLM APIを叩く
    # 例: response = client.generate_content(prompt)
    #     json_output = response.text
    #     parsed_data = json.loads(json_output)

    # 仮のリターン値（実際のLLMの出力をシミュレート）
    simulated_llm_output = """
    {
      "items": [
        {"name": "牛乳", "quantity": 1},
        {"name": "ほうれん草", "quantity": 1},
        {"name": "卵", "quantity": 10},
        {"name": "肉", "quantity": 1}
      ]
    }
    """
    try:
        parsed_data = json.loads(simulated_llm_output)
        return parsed_data.get("items", [])
    except json.JSONDecodeError:
        print("LLMからのJSON解析に失敗しました。")
        return []

if __name__ == '__main__':
    sample_ocr_text = """
    〇〇スーパー
    2025/07/15 10:30
    牛乳          230円
    ホウレン草    198円
    たまごL10コ    250円
    豚肉ローススライス    498円
    合計          1176円
    """
    
    print("--- Simple Parser Test ---")
    parsed_items_simple = parse_receipt_text_simple([line.strip() for line in sample_ocr_text.split('\n') if line.strip()])
    for item in parsed_items_simple:
        print(f"Parsed Simple: {item['item_name']}, Quantity: {item['quantity']}")

    print("\n--- LLM Parser Test ---")
    parsed_items_llm = parse_receipt_text_with_llm(sample_ocr_text)
    for item in parsed_items_llm:
        print(f"Parsed LLM: {item['name']}, Quantity: {item['quantity']}")