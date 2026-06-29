"""
probe_offset_mapping.py
ゾーンプレートのオフセット[m]を変えた時、カメラ画像上で輝点がどれだけ・どちら向きに
動くかを実測する。calibrate_homography.py の座標変換(roi_coord_to_slm_offset)が
正しいスケール・符号になっているかを確認するための診断スクリプト。

実行:
    python probe_offset_mapping.py
"""

import json
import time
import numpy as np
import cv2
import zmq

DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 5555

SLM_W, SLM_H = 1920, 1080
SLM_PIXEL_PITCH = 6.4e-6
WAVELENGTH = 520e-9
FOCAL_LENGTH = 0.200
PHASE_SCALE = 520 / 532


def make_zoneplate(offset_x_m=0.0, offset_y_m=0.0):
    x = (np.arange(SLM_W) - SLM_W // 2) * SLM_PIXEL_PITCH - offset_x_m
    y = (np.arange(SLM_H) - SLM_H // 2) * SLM_PIXEL_PITCH - offset_y_m
    X, Y = np.meshgrid(x, y)
    phase = (np.pi * (X**2 + Y**2) / (WAVELENGTH * FOCAL_LENGTH)) % (2 * np.pi)
    return (phase / (2 * np.pi) * 255 * PHASE_SCALE).astype("uint8")


class DaemonClient:
    def __init__(self, host=DAEMON_HOST, port=DAEMON_PORT, timeout_ms=5000):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")

    def display_and_capture(self, phase_u8):
        meta = json.dumps({"height": phase_u8.shape[0], "width": phase_u8.shape[1]}).encode("utf-8")
        self.sock.send_multipart([meta, phase_u8.tobytes()])
        meta_r, data_r = self.sock.recv_multipart()
        meta_resp = json.loads(meta_r.decode("utf-8"))
        h, w = meta_resp["height"], meta_resp["width"]
        return np.frombuffer(data_r, dtype=np.uint8).reshape(h, w)

    def close(self):
        self.sock.close()
        self.ctx.term()


def find_brightest_spot(img: np.ndarray, blur_ksize=15, prev_pos=None):
    """
    撮影画像の中の輝点の重心座標(x, y)を返す。

    ゾーンプレートは特定のオフセットで複数の輝点(幽霊像)を同時に作ることがあるため、
    単純な「画像全体の最大輝度点」だけでは誤って幽霊像を拾うことがある。
    そこで、明るい連結領域を全部検出し、prev_pos(直前に検出した位置)が
    指定されていれば「そこに最も近い領域」を選ぶことで、誤検出を防ぐ。

    prev_pos が None の場合は、最も明るい(面積×輝度が最大の)領域を選ぶ。
    """
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    _, max_val, _, _ = cv2.minMaxLoc(blurred)
    if max_val < 10:
        return (img.shape[1] / 2, img.shape[0] / 2), float(max_val)

    thresh_val = max(int(max_val * 0.7), 1)
    _, thresh_img = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    thresh_img = thresh_img.astype("uint8")

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, connectivity=8)

    if num_labels <= 1:
        # 連結領域が見つからない場合のフォールバック
        _, _, _, max_loc = cv2.minMaxLoc(blurred)
        return max_loc, float(max_val)

    # ラベル0は背景なので除外
    candidates = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 1:
            continue
        cx, cy = centroids[lbl]
        # その領域内の最大輝度を見て「明るさ評価」にする
        mask = (labels == lbl)
        region_max = blurred[mask].max()
        candidates.append((cx, cy, area, float(region_max)))

    if not candidates:
        _, _, _, max_loc = cv2.minMaxLoc(blurred)
        return max_loc, float(max_val)

    if prev_pos is None:
        # 初回: 面積×輝度が最大の領域を選ぶ(=最も「強い」スポット)
        best = max(candidates, key=lambda c: c[2] * c[3])
    else:
        # 直前位置に最も近い領域を選ぶ(連続性によるトラッキング)
        px, py = prev_pos
        best = min(candidates, key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2)

    cx, cy, area, region_max = best
    return (float(cx), float(cy)), float(region_max)


def main():
    client = DaemonClient()

    # x方向を-600umから+600umまで7段階、その後y方向も同様に振って線形性を見る
    x_steps = np.linspace(-600e-6, 600e-6, 7)
    y_steps = np.linspace(-600e-6, 600e-6, 7)

    test_offsets_m = [("center", 0.0, 0.0)]
    for v in x_steps:
        if abs(v) > 1e-9:
            test_offsets_m.append((f"x{v*1e6:+.0f}um", float(v), 0.0))
    for v in y_steps:
        if abs(v) > 1e-9:
            test_offsets_m.append((f"y{v*1e6:+.0f}um", 0.0, float(v)))

    results = []
    center_pos = None
    prev_pos = None
    prev_axis = None
    for label, ox, oy in test_offsets_m:
        phase = make_zoneplate(ox, oy)
        t0 = time.perf_counter()
        raw_img = client.display_and_capture(phase)
        dt = time.perf_counter() - t0

        cur_axis = label[0]  # 'c'(center) / 'x' / 'y'
        if cur_axis != prev_axis:
            # 軸が切り替わったら、直前点ではなく中心点を基準にする(大きくジャンプするため)
            ref_pos = center_pos
        else:
            ref_pos = prev_pos

        (cx, cy), max_val = find_brightest_spot(raw_img, prev_pos=ref_pos)
        prev_pos = (cx, cy)
        prev_axis = cur_axis
        if label == "center":
            center_pos = (cx, cy)

        print(f"{label:10s} offset=({ox*1e6:+.0f}um,{oy*1e6:+.0f}um) "
              f"-> captured=({cx:.1f},{cy:.1f}) max_val={max_val:.1f} ({dt*1000:.0f}ms)")
        results.append((label, ox, oy, cx, cy, max_val))

        # デバッグ用に生画像を保存(縮小して見やすくする)
        small = cv2.resize(raw_img, (raw_img.shape[1] // 4, raw_img.shape[0] // 4))
        cv2.imwrite(f"debug_raw_{label}.png", small)

    client.close()

    # 中心からの移動量を計算
    cx0, cy0 = results[0][3], results[0][4]
    print("\n--- 中心からの移動量(px) ---")
    for label, ox, oy, cx, cy, max_val in results[1:]:
        dpx_x = cx - cx0
        dpx_y = cy - cy0
        print(f"{label:10s}: dpx_x={dpx_x:+.1f}px, dpx_y={dpx_y:+.1f}px  (max_val={max_val:.1f})")

    print("\nこの結果から、x方向・y方向それぞれの「物理オフセット[m] -> 画像上の移動量[px]」の")
    print("比率と符号を読み取って、calibrate_homography.py の roi_coord_to_slm_offset を調整してください。")


if __name__ == "__main__":
    main()