from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="japan",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True,
)

results = ocr.predict("test.jpg")

for i, res in enumerate(results):
    res.print()  # 文字列とスコアをコンソール表示
    res.save_to_img(f"out_{i}.png")  # 枠描画済み画像を保存
    res.save_to_json(f"out_{i}.json")  # JSONで保存
