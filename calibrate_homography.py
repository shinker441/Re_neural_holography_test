"""
calibrate_homography.py
ゾーンプレートで作る集光点を使って、SLM(ターゲット座標系) <-> カメラ撮影画像 の
homography行列を求める。サーバ(rtxstation)側で実行する。

考え方:
  1. ROI(roi_res)内に、既知のターゲット座標(grid点)を複数用意する
  2. 各ターゲット座標に集光するゾーンプレート位相パターンをSLMに送る
     (hw_daemon.py に直接位相を渡す。前に使ったzone plateの式そのまま)
  3. カメラ撮影画像の中で一番明るい点(輝点の重心)を検出する
  4. 「ターゲット座標 <-> 検出されたカメラ座標」の対応点をN点集める
  5. cv2.findHomography で、カメラ座標 -> ROI座標 への変換行列Hを計算する
  6. Hを JSON で保存する(後で remote_prop.RemotePhysicalProp.set_homography に渡す)

実行:
    python calibrate_homography.py

前提:
    実験室PC側で hw_daemon.py が起動していて、SSHトンネル等でこのサーバから
    tcp://127.0.0.1:5555 にアクセスできること(前のステップで確認済み)。
"""

import json
import time
import numpy as np
import cv2
import zmq

# ---- 設定 ----
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 5555

SLM_W, SLM_H = 1920, 1080
SLM_PIXEL_PITCH = 6.4e-6
WAVELENGTH = 520e-9
FOCAL_LENGTH = 0.200
PHASE_SCALE = 520 / 532  # settle_time_calib.py 等で使った値と同じ

ROI_RES = (880, 1600)  # (height, width)  main_citl.py の roi_res と合わせる

# ROI内に並べる校正点の格子（多いほど精度は上がるが時間がかかる。まずは5x5の25点）
GRID_ROWS, GRID_COLS = 5, 5
MARGIN_RATIO = 0.15  # ROIの縁からどれだけ余白を取るか

OUTPUT_JSON = "./H_calibration.json"

# ---- ゾーンプレート生成（中心からのオフセットを指定できるように一般化）----

def make_zoneplate(offset_x_m=0.0, offset_y_m=0.0):
    """SLM全面に対して、中心から(offset_x_m, offset_y_m)だけずれた点に集光するゾーンプレート位相を作る"""
    x = (np.arange(SLM_W) - SLM_W // 2) * SLM_PIXEL_PITCH - offset_x_m
    y = (np.arange(SLM_H) - SLM_H // 2) * SLM_PIXEL_PITCH - offset_y_m
    X, Y = np.meshgrid(x, y)
    phase = (np.pi * (X**2 + Y**2) / (WAVELENGTH * FOCAL_LENGTH)) % (2 * np.pi)
    return (phase / (2 * np.pi) * 255 * PHASE_SCALE).astype("uint8")


def roi_coord_to_slm_offset(roi_x, roi_y, roi_w, roi_h):
    """
    ROI内のピクセル座標(roi_x, roi_y) を、ゾーンプレートに渡す物理オフセット[m]に変換する。

    簡易的な近似: ROIの中心を光軸中心とみなし、
    ROIピクセル -> SLMピクセル相当の比率でスケールしてからピッチを掛ける。
    (厳密な光学系の倍率は後で実機の結果を見て補正してもよい)
    """
    cx, cy = roi_w / 2.0, roi_h / 2.0
    dx_px = roi_x - cx
    dy_px = roi_y - cy
    # ROIはSLM全体のうちの一部を見ている前提なので、SLM座標系にスケールする
    scale_x = SLM_W / roi_w
    scale_y = SLM_H / roi_h
    offset_x_m = dx_px * scale_x * SLM_PIXEL_PITCH
    offset_y_m = dy_px * scale_y * SLM_PIXEL_PITCH
    return offset_x_m, offset_y_m


# ---- zmq通信（hw_daemon.pyへの低レベルクライアント）----

class DaemonClient:
    def __init__(self, host=DAEMON_HOST, port=DAEMON_PORT, timeout_ms=5000):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")
        print(f"[DaemonClient] connected to tcp://{host}:{port}")

    def display_and_capture(self, phase_u8: np.ndarray) -> np.ndarray:
        meta = json.dumps({"height": phase_u8.shape[0], "width": phase_u8.shape[1]}).encode("utf-8")
        self.sock.send_multipart([meta, phase_u8.tobytes()])
        meta_r, data_r = self.sock.recv_multipart()
        meta_resp = json.loads(meta_r.decode("utf-8"))
        h, w = meta_resp["height"], meta_resp["width"]
        if h == 0 or w == 0:
            raise RuntimeError("daemon側で撮影に失敗しました")
        return np.frombuffer(data_r, dtype=np.uint8).reshape(h, w)

    def close(self):
        self.sock.close()
        self.ctx.term()


# ---- 輝点検出 ----

def find_brightest_spot(img: np.ndarray, blur_ksize=15):
    """撮影画像の中で最も明るい領域の重心座標(x, y)を返す"""
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
    # しきい値以上の領域で重心を取る(単純なmax_locよりロバスト)
    thresh_val = max(int(max_val * 0.8), 1)
    _, thresh_img = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(thresh_img > 0)
    if len(xs) == 0:
        return max_loc, max_val  # フォールバック
    cx, cy = float(xs.mean()), float(ys.mean())
    return (cx, cy), max_val


def main():
    client = DaemonClient()

    roi_h, roi_w = ROI_RES
    margin_x = roi_w * MARGIN_RATIO
    margin_y = roi_h * MARGIN_RATIO

    grid_xs = np.linspace(margin_x, roi_w - margin_x, GRID_COLS)
    grid_ys = np.linspace(margin_y, roi_h - margin_y, GRID_ROWS)

    roi_points = []      # 期待されるROI座標 (x, y)
    captured_points = [] # 撮影画像上で検出された座標 (x, y)

    total = GRID_ROWS * GRID_COLS
    count = 0

    for ry in grid_ys:
        for rx in grid_xs:
            count += 1
            offset_x_m, offset_y_m = roi_coord_to_slm_offset(rx, ry, roi_w, roi_h)
            phase = make_zoneplate(offset_x_m, offset_y_m)

            t0 = time.perf_counter()
            raw_img = client.display_and_capture(phase)
            dt = time.perf_counter() - t0

            (cx, cy), max_val = find_brightest_spot(raw_img)

            print(f"[{count}/{total}] target_roi=({rx:.1f},{ry:.1f}) "
                  f"-> captured=({cx:.1f},{cy:.1f}) max_val={max_val:.1f} ({dt*1000:.0f}ms)")

            if max_val < 30:
                print(f"    [WARN] 輝点が暗すぎます(max_val={max_val:.1f})。この点はスキップします")
                continue

            roi_points.append([rx, ry])
            captured_points.append([cx, cy])

    client.close()

    if len(roi_points) < 4:
        raise RuntimeError(f"有効な対応点が{len(roi_points)}点しかありません。最低4点必要です。"
                            f"露光/輝度/集光距離の設定を見直してください。")

    src = np.array(captured_points, dtype=np.float64)  # 撮影画像座標
    dst = np.array(roi_points, dtype=np.float64)        # ROI座標

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=5.0)

    inliers = int(mask.sum()) if mask is not None else len(src)
    print(f"\nhomography計算完了。対応点 {len(src)}点中 {inliers}点が有効(inlier)")
    print("H =")
    print(H)

    # 再投影誤差を見ておく
    src_h = np.hstack([src, np.ones((len(src), 1))])
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj - dst, axis=1)
    print(f"再投影誤差: 平均={err.mean():.2f}px, 最大={err.max():.2f}px")

    out = {
        "H": H.tolist(),
        "dst_wh": [roi_w, roi_h],
        "margin": 0,
        "num_points": len(src),
        "num_inliers": inliers,
        "reproj_error_mean_px": float(err.mean()),
        "reproj_error_max_px": float(err.max()),
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n保存しました: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()