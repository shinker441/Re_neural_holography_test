"""
calibrate_homography_checkerboard.py
チェッカーボードパターンをSLMに「そのまま画像として」表示し、カメラで撮影、
cv2.findChessboardCorners で多数の対応点を一度に検出してhomographyを求める。

ゾーンプレート(ホログラム)による集光点追跡は、回折・位相のゆらぎで
再現性が低いことが分かったため、より安定なこの方式に切り替えた。

実行:
    python calibrate_homography_checkerboard.py
"""

import json
import numpy as np
import cv2
import zmq

DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 5555

SLM_W, SLM_H = 1920, 1080
ROI_RES = (880, 1600)  # (height, width)  main_citl.py の roi_res と合わせる

# チェッカーボードの内部コーナー数(交点の数。マス数-1)
BOARD_COLS, BOARD_ROWS = 3, 3   # まず最小構成で検出が通るか確認
SQUARE_PX = 250

OUTPUT_JSON = "./H_calibration.json"


def make_checkerboard(slm_w, slm_h, cols, rows, square_px):
    """SLM全面に白黒チェッカーボードを描き、ROI座標系での各内部コーナー座標も返す"""
    board_w = (cols + 1) * square_px
    board_h = (rows + 1) * square_px
    x0 = (slm_w - board_w) // 2
    y0 = (slm_h - board_h) // 2

    img = np.zeros((slm_h, slm_w), dtype="uint8")
    for j in range(rows + 1):
        for i in range(cols + 1):
            if (i + j) % 2 == 0:
                x1 = x0 + i * square_px
                y1 = y0 + j * square_px
                img[y1:y1 + square_px, x1:x1 + square_px] = 255

    # 内部コーナー(findChessboardCornersが検出する点)のSLM上座標を計算
    slm_corners = []
    for j in range(1, rows + 1):
        for i in range(1, cols + 1):
            slm_corners.append([x0 + i * square_px, y0 + j * square_px])
    slm_corners = np.array(slm_corners, dtype=np.float64)

    return img, slm_corners


def slm_to_roi_coords(slm_corners, slm_w, slm_h, roi_w, roi_h):
    """
    SLM座標 -> ROI座標 への変換(簡易: SLM全面とROIが同じ中心・比例関係にあると仮定)。
    これは「ターゲット平面でどこに見えるべきか」の理想値であり、
    実際のhomographyはこの理想値とカメラ撮影座標の対応から求める。
    """
    scale_x = roi_w / slm_w
    scale_y = roi_h / slm_h
    roi_coords = slm_corners.copy()
    roi_coords[:, 0] *= scale_x
    roi_coords[:, 1] *= scale_y
    return roi_coords


class DaemonClient:
    def __init__(self, host=DAEMON_HOST, port=DAEMON_PORT, timeout_ms=5000):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")
        print(f"[DaemonClient] connected to tcp://{host}:{port}")

    def display_and_capture(self, img_u8: np.ndarray) -> np.ndarray:
        meta = json.dumps({"height": img_u8.shape[0], "width": img_u8.shape[1]}).encode("utf-8")
        self.sock.send_multipart([meta, img_u8.tobytes()])
        meta_r, data_r = self.sock.recv_multipart()
        meta_resp = json.loads(meta_r.decode("utf-8"))
        h, w = meta_resp["height"], meta_resp["width"]
        if h == 0 or w == 0:
            raise RuntimeError("daemon側で撮影に失敗しました")
        return np.frombuffer(data_r, dtype=np.uint8).reshape(h, w)

    def close(self):
        self.sock.close()
        self.ctx.term()


def main():
    client = DaemonClient()

    board_img, slm_corners = make_checkerboard(SLM_W, SLM_H, BOARD_COLS, BOARD_ROWS, SQUARE_PX)
    roi_h, roi_w = ROI_RES
    roi_corners = slm_to_roi_coords(slm_corners, SLM_W, SLM_H, roi_w, roi_h)

    print("チェッカーボードを表示・撮影します...")
    raw_img = client.display_and_capture(board_img)
    client.close()

    cv2.imwrite("debug_checkerboard_raw.png",
                cv2.resize(raw_img, (raw_img.shape[1] // 4, raw_img.shape[0] // 4)))

    # コントラストが低いので min-max 正規化して強調する
    img_min, img_max = raw_img.min(), raw_img.max()
    if img_max > img_min:
        normalized = ((raw_img.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = raw_img
    cv2.imwrite("debug_checkerboard_normalized.png",
                cv2.resize(normalized, (normalized.shape[1] // 4, normalized.shape[0] // 4)))

    pattern_size = (BOARD_COLS, BOARD_ROWS)
    found, corners = cv2.findChessboardCorners(
        normalized, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not found:
        raise RuntimeError(
            "チェッカーボードのコーナーを検出できませんでした。"
            "debug_checkerboard_raw.png を確認し、ボードがカメラの視野に入っているか、"
            "ピントが合っているか確認してください。SQUARE_PXを変えて見え方を調整するのも有効です。"
        )

    # サブピクセル精度に補正
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(normalized, corners, (11, 11), (-1, -1), criteria)

    captured_points = corners.reshape(-1, 2)
    print(f"検出されたコーナー数: {len(captured_points)} / 期待値: {len(slm_corners)}")

    # findChessboardCornersは行/列の走査順序が反転することがあるので、
    # 両方の順序を試して、再投影誤差が小さい方を採用する
    best_H, best_err, best_mask = None, None, None
    for roi_pts in [roi_corners, roi_corners[::-1]]:
        if len(roi_pts) != len(captured_points):
            continue
        H, mask = cv2.findHomography(captured_points, roi_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        src_h = np.hstack([captured_points, np.ones((len(captured_points), 1))])
        proj = (H @ src_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        err = np.linalg.norm(proj - roi_pts, axis=1).mean()
        if best_err is None or err < best_err:
            best_H, best_err, best_mask = H, err, mask

    H = best_H
    inliers = int(best_mask.sum()) if best_mask is not None else len(captured_points)
    print(f"\nhomography計算完了。対応点 {len(captured_points)}点中 {inliers}点が有効(inlier)")
    print("H =")
    print(H)
    print(f"再投影誤差(平均): {best_err:.2f}px")

    out = {
        "H": H.tolist(),
        "dst_wh": [roi_w, roi_h],
        "margin": 0,
        "num_points": len(captured_points),
        "num_inliers": inliers,
        "reproj_error_mean_px": float(best_err),
        "method": "checkerboard",
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n保存しました: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()