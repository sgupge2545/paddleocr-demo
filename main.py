from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="japan",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

results = ocr.predict("band_02_y144-176.png")

for i, res in enumerate(results):
    res.print()  # 文字列とスコアをコンソール表示
    res.save_to_img(f"out_{i}.png")  # 枠描画済み画像を保存
    res.save_to_json(f"out_{i}.json")  # JSONで保存
