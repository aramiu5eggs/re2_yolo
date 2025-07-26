# src/database/db_manager.py

import sqlite3
import os
from datetime import datetime
from src.config import DATABASE_PATH

DB_FILE = DATABASE_PATH

def get_db_connection():
    """データベースに接続し、コネクションオブジェクトを返す"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row # カラム名をキーとして値にアクセスできるようにする
    return conn

def create_table():
    """food_itemsテーブルを作成する"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS food_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            standard_name TEXT NOT NULL,
            yolo_class TEXT NOT NULL,
            quantity REAL NOT NULL,
            unit TEXT,
            purchase_date TEXT,
            expiry_date TEXT,
            detected_by TEXT NOT NULL,
            last_seen_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            notes TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database table 'food_items' ensured at {DB_FILE}")

def add_food_item(standard_name, yolo_class, quantity, detected_by, 
                  unit=None, purchase_date=None, expiry_date=None, notes=None):
    """新しい食材アイテムをデータベースに追加する"""
    conn = get_db_connection()
    cursor = conn.cursor()
    last_seen = datetime.now().strftime('%Y-%m-%d') # 今日の日付

    cursor.execute('''
        INSERT INTO food_items (standard_name, yolo_class, quantity, unit, 
                                purchase_date, expiry_date, detected_by, last_seen_date, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (standard_name, yolo_class, quantity, unit, 
          purchase_date, expiry_date, detected_by, last_seen, notes))
    conn.commit()
    item_id = cursor.lastrowid # 挿入されたアイテムのIDを取得
    conn.close()
    print(f"Added item: {standard_name} (ID: {item_id})")
    return item_id

def update_food_item_quantity(item_id, new_quantity, detected_by=None):
    """既存の食材アイテムの数量を更新する"""
    conn = get_db_connection()
    cursor = conn.cursor()
    last_seen = datetime.now().strftime('%Y-%m-%d')

    update_sql = 'UPDATE food_items SET quantity = ?'
    params = [new_quantity]

    if detected_by: # 検出方法が指定された場合のみ更新
        update_sql += ', detected_by = ?'
        params.append(detected_by)
    
    update_sql += ', last_seen_date = ? WHERE id = ?'
    params.extend([last_seen, item_id])

    cursor.execute(update_sql, tuple(params))
    conn.commit()
    conn.close()
    print(f"Updated item ID {item_id} to quantity {new_quantity}")

def update_food_item_details(item_id, **kwargs):
    """既存の食材アイテムの詳細を更新する（例: standard_name, expiry_date, notesなど）"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    set_clauses = []
    params = []
    
    allowed_fields = ['standard_name', 'yolo_class', 'quantity', 'unit', 
                      'purchase_date', 'expiry_date', 'notes', 'status', 
                      'last_seen_date', 'detected_by']

    for key, value in kwargs.items():
        if key in ['standard_name', 'yolo_class', 'unit', 'purchase_date', 'expiry_date', 'notes', 'status', 'last_seen_date']:
            set_clauses.append(f"{key} = ?")
            params.append(value)
        else:
            print(f"Warning: Invalid field '{key}' for update.")

    if not set_clauses:
        conn.close()
        return

    sql = f"UPDATE food_items SET {', '.join(set_clauses)} WHERE id = ?"
    params.append(item_id)

    cursor.execute(sql, tuple(params))
    conn.commit()
    conn.close()
    print(f"Updated details for item ID {item_id}")


def get_all_food_items(status='active'):
    """全ての食材アイテム（または指定されたステータスのアイテム）を取得する"""
    conn = get_db_connection()
    cursor = conn.cursor()
    if status == 'all':
        cursor.execute('SELECT * FROM food_items ORDER BY standard_name')
    else:
        cursor.execute('SELECT * FROM food_items WHERE status = ? ORDER BY standard_name', (status,))
    items = cursor.fetchall() # 全ての行を取得
    conn.close()
    return items

def get_food_item_by_id(item_id):
    """IDで食材アイテムを取得する"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM food_items WHERE id = ?', (item_id,))
    item = cursor.fetchone() # 1つの行を取得
    conn.close()
    return item

def delete_food_item(item_id):
    """食材アイテムをデータベースから削除する（論理削除も考慮可）"""
    conn = get_db_connection()
    cursor = conn.cursor()
    # 物理削除
    cursor.execute('DELETE FROM food_items WHERE id = ?', (item_id,))
    # 論理削除にする場合（statusを'deleted'などに変更）
    # cursor.execute("UPDATE food_items SET status = 'deleted' WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()
    print(f"Deleted item ID {item_id}")

def mark_as_consumed_or_discarded(item_id, status='consumed'):
    """食材アイテムを消費済みまたは廃棄済みにマークする"""
    if status not in ['consumed', 'discarded']:
        raise ValueError("Status must be 'consumed' or 'discarded'")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE food_items SET status = ? WHERE id = ?', (status, item_id))
    conn.commit()
    conn.close()
    print(f"Item ID {item_id} marked as {status}.")


if __name__ == '__main__':
    # データベースとテーブルを作成
    create_table()

    # テストデータ追加
    print("\n--- Adding test items ---")
    add_food_item('牛乳', 'milk', 1.0, 'manual', purchase_date='2025-07-14', expiry_date='2025-07-20')
    add_food_item('卵', 'egg', 10, 'receipt', purchase_date='2025-07-15')
    add_food_item('レタス', 'leafy_green', 1, 'yolo', purchase_date='2025-07-13')

    # 全てのアイテムを取得
    print("\n--- All active items ---")
    items = get_all_food_items()
    for item in items:
        print(f"ID: {item['id']}, Name: {item['standard_name']}, Qty: {item['quantity']}, Detected By: {item['detected_by']}")
    
    # アイテムの数量を更新
    print("\n--- Updating item quantity ---")
    # 例としてID:1のアイテムの数量を0.5にする
    # 実際のIDはadd_food_itemの戻り値やget_all_food_itemsで確認してください
    if items:
        update_food_item_quantity(items[0]['id'], 0.5, detected_by='yolo')
    
    # 更新後のアイテムを再取得
    print("\n--- Items after update ---")
    items = get_all_food_items()
    for item in items:
        print(f"ID: {item['id']}, Name: {item['standard_name']}, Qty: {item['quantity']}, Detected By: {item['detected_by']}")

    # アイテムの詳細を更新
    print("\n--- Updating item details ---")
    if items and items[1]: # 2番目のアイテム (卵) の賞味期限を追加
        update_food_item_details(items[1]['id'], expiry_date='2025-08-10', notes='Lサイズ')

    # 消費済みにマーク
    print("\n--- Marking item as consumed ---")
    if items and items[2]: # 3番目のアイテム (レタス) を消費済み
        mark_as_consumed_or_discarded(items[2]['id'], 'consumed')

    print("\n--- All items (including consumed) ---")
    all_items = get_all_food_items(status='all')
    for item in all_items:
        print(f"ID: {item['id']}, Name: {item['standard_name']}, Qty: {item['quantity']}, Status: {item['status']}")

    # アイテムを削除
    print("\n--- Deleting an item ---")
    if all_items and all_items[0]: # 最初のアイテムを削除
        delete_food_item(all_items[0]['id'])

    print("\n--- All items after deletion ---")
    items = get_all_food_items(status='all')
    for item in items:
        print(f"ID: {item['id']}, Name: {item['standard_name']}, Qty: {item['quantity']}, Status: {item['status']}")