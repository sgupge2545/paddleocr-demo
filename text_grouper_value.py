from paddleocr import PaddleOCR
import cv2
import numpy as np
import json
import unicodedata
import re


def normalize(s):
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", "", s)


def resize_image(image_path, max_size=1250):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {image_path}")

    height, width = img.shape[:2]

    if max(height, width) <= max_size:
        return img

    scale = max_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_img


def resize_image_array(img_array, max_size=1250):
    """画像配列をリサイズする関数"""
    if img_array is None:
        raise ValueError("画像配列がNoneです")

    height, width = img_array.shape[:2]

    if max(height, width) <= max_size:
        return img_array

    scale = max_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_img = cv2.resize(
        img_array, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    return resized_img


def group_texts_by_y_position(ocr_results, y_tolerance=10):
    """OCR結果をy座標でグルーピングする"""
    if not ocr_results:
        return []

    # 最初の結果を取得
    result = ocr_results[0]
    texts = result["rec_texts"]
    boxes = result["rec_polys"]
    confidences = result["rec_scores"]

    # 各テキストの中心y座標を計算
    text_items = []
    for i, (text, box, conf) in enumerate(zip(texts, boxes, confidences)):
        # バウンディングボックスの中心y座標を計算
        center_y = np.mean(box[:, 1])
        text_items.append(
            {
                "text": text,
                "y": center_y,
                "confidence": float(conf),
                "box": box.tolist(),
            }
        )

    # y座標でソート
    text_items.sort(key=lambda x: x["y"])

    # グルーピング
    groups = []
    current_group = []
    current_y = None

    for item in text_items:
        if current_y is None or abs(item["y"] - current_y) <= y_tolerance:
            # 同じグループに追加
            current_group.append(item)
            current_y = item["y"]
        else:
            # 新しいグループを開始
            if current_group:
                groups.append(current_group)
            current_group = [item]
            current_y = item["y"]

    # 最後のグループを追加
    if current_group:
        groups.append(current_group)

    return groups


def is_similar_text(text1, text2, max_diff=1):
    """2つのテキストが指定した文字数違い以内かどうかを判定する"""
    if abs(len(text1) - len(text2)) > max_diff:
        return False

    # 簡易的な類似度判定
    if text1 in text2 or text2 in text1:
        return True

    # 文字の違いを数える
    diff_count = 0
    min_len = min(len(text1), len(text2))

    for i in range(min_len):
        if text1[i] != text2[i]:
            diff_count += 1
            if diff_count > max_diff:
                return False

    # 長さの違いも考慮
    diff_count += abs(len(text1) - len(text2))

    return diff_count <= max_diff


def find_keyword_with_flexible_matching(text_items, keyword, max_attempts=3):
    """キーワードを徐々に緩い条件で検索する"""
    normalized_keyword = normalize(keyword)

    for attempt in range(max_attempts):
        max_diff = attempt + 1  # 1文字違い、2文字違い、3文字違い...

        for i, item in enumerate(text_items):
            item_text = normalize(item["text"])

            # 完全一致
            if normalized_keyword in item_text:
                return i, "完全一致"
            # 類似検索
            elif is_similar_text(item_text, normalized_keyword, max_diff):
                return i, f"類似検索({max_diff}文字違い)"

    return -1, None


def is_vehicle_number_pattern(text):
    """車両番号のパターン（日本語+数字+日本語+数字）かどうかを判定する"""
    import re

    # 日本語文字（ひらがな、カタカナ、漢字）+ 数字 + 日本語文字 + 数字 のパターン
    pattern = r"^[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+[0-9]+[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+[0-9]+$"
    return bool(re.match(pattern, text))


def find_keyword_and_value(groups, keywords):
    """特定のキーワードを含む行を見つけて、その右側の値を抽出する"""
    results = {}

    for group in groups:
        # グループ内のテキストをx座標でソート（左から右へ）
        sorted_group = sorted(
            group, key=lambda x: np.mean([point[0] for point in x["box"]])
        )

        # 各キーワードをチェック
        for keyword in keywords:
            if keyword in results:
                continue  # すでに見つかっていればスキップ

            # 柔軟なマッチングでキーワードを検索
            keyword_index, match_type = find_keyword_with_flexible_matching(
                sorted_group, keyword
            )

            if keyword_index >= 0:
                # キーワードの1つ右の要素を取得
                if keyword_index + 1 < len(sorted_group):
                    value_text = sorted_group[keyword_index + 1]["text"]

                    # 車両番号の場合、パターンマッチングをチェック
                    if keyword == "車両番号":
                        # まず現在のテキストでパターンチェック
                        if is_vehicle_number_pattern(value_text):
                            print(f"車両番号パターンマッチ: '{value_text}'")
                        else:
                            # パターンにマッチしない場合、次の要素も含めてチェック
                            if keyword_index + 2 < len(sorted_group):
                                combined_text = (
                                    value_text + sorted_group[keyword_index + 2]["text"]
                                )
                                if is_vehicle_number_pattern(combined_text):
                                    value_text = combined_text
                                    print(
                                        f"車両番号パターンマッチ（結合後）: '{value_text}'"
                                    )

                    results[keyword] = {
                        "value": value_text,
                        "group_index": groups.index(group),
                        "y_position": float(sorted_group[0]["y"]),
                        "match_type": match_type,
                    }
                    print(
                        f"キーワード '{keyword}' の値: '{value_text}' (グループ {groups.index(group)}, {match_type})"
                    )
                    if match_type == "完全一致":
                        break  # 完全一致なら以降のグループでこのキーワードは探さない

    return results


def save_groups_to_json(groups, output_file="text_groups.json"):
    """グループ化されたテキストをJSONファイルに保存"""
    output_data = {"total_groups": len(groups), "groups": []}

    for i, group in enumerate(groups):
        # グループ内のテキストをx座標でソート（左から右へ）
        sorted_group = sorted(
            group, key=lambda x: np.mean([point[0] for point in x["box"]])
        )

        group_data = {
            "group_index": i,
            "texts": [item["text"] for item in sorted_group],
            "y_positions": [float(item["y"]) for item in sorted_group],
            "confidences": [item["confidence"] for item in sorted_group],
            "combined_text": "".join([item["text"] for item in sorted_group]),
        }
        output_data["groups"].append(group_data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"グループ化結果を {output_file} に保存しました")


def visualize_groups_on_image(
    image_path, groups, output_path="grouped_text_visualization.png"
):
    """グループ化されたテキストを元画像に可視化"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"画像の読み込みに失敗しました: {image_path}")
        return

    # 画像をリサイズ（元の処理と同じサイズに）
    height, width = img.shape[:2]
    max_size = 1250
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 各グループに異なる色を割り当て
    colors = [
        (255, 0, 0),  # 青
        (0, 255, 0),  # 緑
        (0, 0, 255),  # 赤
        (255, 255, 0),  # シアン
        (255, 0, 255),  # マゼンタ
        (0, 255, 255),  # 黄色
        (128, 0, 0),  # 濃い青
        (0, 128, 0),  # 濃い緑
        (0, 0, 128),  # 濃い赤
        (128, 128, 0),  # オリーブ
    ]

    # 各グループのテキストを描画
    for i, group in enumerate(groups):
        color = colors[i % len(colors)]

        # グループ内のテキストをx座標でソート
        sorted_group = sorted(
            group, key=lambda x: np.mean([point[0] for point in x["box"]])
        )

        # グループの境界を描画
        y_positions = [item["y"] for item in sorted_group]
        min_y = min(y_positions)
        max_y = max(y_positions)

        # グループの背景を描画（半透明）
        overlay = img.copy()
        cv2.rectangle(
            overlay, (0, int(min_y - 5)), (img.shape[1], int(max_y + 5)), color, -1
        )
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

        # 各テキストのバウンディングボックスを描画
        for item in sorted_group:
            box = np.array(item["box"], dtype=np.int32)
            cv2.polylines(img, [box], True, color, 2)

        # グループ番号を描画
        cv2.putText(
            img, f"G{i}", (10, int(min_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    # 画像を保存
    cv2.imwrite(output_path, img)
    print(f"可視化結果を {output_path} に保存しました")


def main():
    # PaddleOCRの初期化
    ocr = PaddleOCR(
        lang="japan",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    # 画像をリサイズ
    resized_image = resize_image("IMG_0564.JPG", max_size=1250)

    # OCR実行
    results = ocr.predict(resized_image)

    # テキストをグルーピング
    groups = group_texts_by_y_position(results, y_tolerance=20)

    # 結果を表示
    print(f"検出されたグループ数: {len(groups)}")
    for i, group in enumerate(groups):
        texts = [item["text"] for item in group]
        y_positions = [item["y"] for item in group]
        print(f"グループ {i}: {texts} (y: {y_positions})")

    # 特定のキーワードを検索
    keywords = ["車台番号", "車両番号", "所有者の氏名又は名称", "初度登録年月"]

    # デバッグ用：各グループのテキストを表示
    print("\n=== デバッグ: 各グループのテキスト ===")
    for i, group in enumerate(groups):
        texts = [item["text"] for item in group]
        print(f"グループ {i}: {texts}")

    extracted_values = find_keyword_and_value(groups, keywords)

    print("\n=== 抽出された値 ===")
    for keyword, data in extracted_values.items():
        print(f"{keyword}: {data['value']}")

    # JSONファイルに保存
    save_groups_to_json(groups)

    # 可視化画像を生成
    visualize_groups_on_image("IMG_0564.JPG", groups)

    return groups, extracted_values


if __name__ == "__main__":
    groups, extracted_values = main()
