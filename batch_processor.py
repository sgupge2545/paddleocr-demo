import os
import json
import cv2
import numpy as np
from text_grouper_value import (
    resize_image,
    group_texts_by_y_position,
    find_keyword_and_value,
)
from paddleocr import PaddleOCR
import glob
from doc_scan import scan_document, enhance_document_quality
from PIL import Image


def process_single_image(image_path, ocr, keywords):
    """単一画像を処理して結果を返す"""
    try:
        print(f"\n=== 処理中: {image_path} ===")

        # 画像をリサイズ
        resized_image = resize_image(image_path, max_size=1250)

        # ドキュメントスキャナーで前処理
        pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        scanned_image = scan_document(pil_image, max_height=1000)

        # 画像品質向上（必要に応じて）
        enhanced_image = enhance_document_quality(
            scanned_image,
            blur=None,
            binarize=False,  # OCR前に2値化はしない
            clahe=True,  # コントラスト強調
            adaptive_block_size=25,
            adaptive_C=40,
            global_thresh=None,
        )

        # PIL画像をOpenCV形式に変換
        enhanced_cv = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

        # OCR実行
        results = ocr.predict(enhanced_cv)

        # テキストをグルーピング
        groups = group_texts_by_y_position(results, y_tolerance=20)

        # 特定のキーワードを検索
        extracted_values = find_keyword_and_value(groups, keywords)

        # 結果を整形
        result_data = {
            "image_path": image_path,
            "車両番号": extracted_values.get("車両番号", {}).get("value", ""),
            "車台番号": extracted_values.get("車台番号", {}).get("value", ""),
            "初度登録年月": extracted_values.get("初度登録年月", {}).get("value", ""),
            "所有者の氏名又は名称": extracted_values.get(
                "所有者の氏名又は名称", {}
            ).get("value", ""),
        }

        print(f"結果: {result_data}")
        return result_data

    except Exception as e:
        print(f"エラー: {image_path} の処理中にエラーが発生しました: {e}")
        return {
            "image_path": image_path,
            "車両番号": "",
            "車台番号": "",
            "初度登録年月": "",
            "所有者の氏名又は名称": "",
            "error": str(e),
        }


def process_all_images():
    """sampleディレクトリのすべての画像を処理"""
    # PaddleOCRの初期化
    ocr = PaddleOCR(
        lang="japan",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    # 検索キーワード
    keywords = ["車台番号", "車両番号", "所有者の氏名又は名称", "初度登録年月"]

    # sampleディレクトリの画像ファイルを取得
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join("sample", ext)))

    # ファイル名でソート
    image_files.sort()

    print(f"処理対象ファイル数: {len(image_files)}")
    for file in image_files:
        print(f"  - {file}")

    # 各画像を処理
    all_results = []
    for image_path in image_files:
        result = process_single_image(image_path, ocr, keywords)
        all_results.append(result)

    # 結果をJSONファイルに保存
    output_file = "batch_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n=== 処理完了 ===")
    print(f"結果を {output_file} に保存しました")
    print(f"処理した画像数: {len(all_results)}")

    # 結果のサマリーを表示
    print("\n=== 結果サマリー ===")
    for result in all_results:
        print(f"{os.path.basename(result['image_path'])}:")
        for key, value in result.items():
            if key != "image_path" and key != "error":
                if value:
                    print(f"  {key}: {value}")
        if "error" in result:
            print(f"  エラー: {result['error']}")
        print()

    return all_results


if __name__ == "__main__":
    results = process_all_images()
