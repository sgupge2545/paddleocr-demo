import cv2
import numpy as np
import os


def detect_document_rotation(img, angle_range=(-10, 10), angle_step=0.5):
    """文書の最適な回転角度を検出する"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    best_angle = 0
    max_horizontal_lines = 0

    for angle in np.arange(angle_range[0], angle_range[1] + angle_step, angle_step):
        # 回転行列を作成
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 画像を回転
        rotated = cv2.warpAffine(gray, rotation_matrix, (w, h))

        # 二値化
        bw = cv2.adaptiveThreshold(
            rotated, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 15
        )

        # エッジ検出
        edges = cv2.Canny(bw, 50, 150, apertureSize=3)

        # 水平線を検出
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=20,
            minLineLength=int(w * 0.2),
            maxLineGap=30,
        )

        horizontal_count = 0
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                line_angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if line_angle < 10:  # 水平線の閾値
                    horizontal_count += 1

        if horizontal_count > max_horizontal_lines:
            max_horizontal_lines = horizontal_count
            best_angle = angle

    return best_angle


def rotate_image(img, angle):
    """画像を指定角度で回転させる"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    return rotated


def find_horizontal_bands(
    img,
    min_len_ratio=0.2,
    angle_thresh=10,
    merge_gap=15,
    save_debug=False,
    debug_dir="debug",
    auto_rotate=True,
):
    h, w = img.shape[:2]

    # 自動回転が有効な場合、最適な角度を検出
    rotation_angle = 0
    if auto_rotate:
        print("文書の傾きを検出中...")
        rotation_angle = detect_document_rotation(img)
        print(f"検出された回転角度: {rotation_angle:.1f}度")

        if abs(rotation_angle) > 0.5:  # 0.5度以上の傾きがある場合のみ回転
            img = rotate_image(img, rotation_angle)
            if save_debug:
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(f"{debug_dir}/debug_0_rotated.png", img)
            print(f"画像を{rotation_angle:.1f}度回転しました")
        else:
            print("回転は不要です")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(f"{debug_dir}/debug_1_gray.png", gray)

    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )

    if save_debug:
        cv2.imwrite(f"{debug_dir}/debug_2_binary.png", bw)

    edges = cv2.Canny(bw, 50, 150, apertureSize=3)

    if save_debug:
        cv2.imwrite(f"{debug_dir}/debug_3_edges.png", edges)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=20,
        minLineLength=int(w * min_len_ratio),
        maxLineGap=30,
    )

    # 線検出結果を可視化
    if save_debug:
        line_img = img.copy()
        if lines is not None:
            print(f"検出された線の数: {len(lines)}")
            for x1, y1, x2, y2 in lines[:, 0]:
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if angle < angle_thresh:
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        else:
            print("線が検出されませんでした")
        cv2.imwrite(f"{debug_dir}/debug_4_lines.png", line_img)

    ys = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < angle_thresh:
                ys.append((min(y1, y2), max(y1, y2)))

    if save_debug:
        print(f"水平線の数: {len(ys)}")

    ys.sort()
    # マージ
    bands = []
    for y1, y2 in ys:
        if not bands or y1 - bands[-1][1] > merge_gap:
            bands.append([y1, y2])
        else:
            bands[-1][1] = max(bands[-1][1], y2)
    # 画像の上下端もバンドとして追加して区切る
    cuts = [0] + [int((a + b) / 2) for a, b in bands] + [h]
    strips = [
        (cuts[i], cuts[i + 1])
        for i in range(len(cuts) - 1)
        if cuts[i + 1] - cuts[i] > 5
    ]

    # 最終的なバンド区切りを可視化
    if save_debug:
        band_img = img.copy()
        for y1, y2 in strips:
            cv2.line(band_img, (0, y1), (w, y1), (255, 0, 0), 2)
            cv2.line(band_img, (0, y2), (w, y2), (255, 0, 0), 2)
        cv2.imwrite(f"{debug_dir}/debug_5_bands.png", band_img)

    return strips  # [(y_top, y_bottom), ...]


def debug_band_processing(img_path, debug_dir="debug"):
    """バンド処理のデバッグ用関数"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"画像の読み込みに失敗しました: {img_path}")
        return

    print(f"画像サイズ: {img.shape}")

    bands = find_horizontal_bands(
        img, save_debug=True, debug_dir=debug_dir, auto_rotate=False
    )
    print(f"検出されたバンド数: {len(bands)}")

    for i, (y1, y2) in enumerate(bands):
        print(f"バンド {i}: y1={y1}, y2={y2}, 高さ={y2 - y1}")

    return bands


if __name__ == "__main__":
    # デバッグ実行
    bands = debug_band_processing("IMG_0564.JPG")
