"""
calibrate_homography_manual.py
チェッカーボード(または何らかの既知パターン)をSLMに表示・撮影し、
自動検出ではなく「人間が画像を見て4点をクリックする」方式でhomographyを求める。

使い方:
    python calibrate_homography_manual.py

操作:
    表示されたウィンドウで、ボードの4つの角を「左上 -> 右上 -> 右下 -> 左下」の順に
    左クリック。4点クリックすると自動でウィンドウが閉じ、homographyを計算する。
    クリックミスしたら 'r' キーでリセットしてやり直せる。 'q' で中断。

前提:
    実験室PC側で hw_daemon.py が起動していて、SSHトンネル等で
    tcp://127.0.0.1:5555 にアクセスできること。

注意:
    このスクリプトはGUIウィンドウ(cv2.imshow)を表示する必要があるため、
    rtxstation上で直接実行するとディスプレイが無くて失敗する可能性がある。
    その場合は、撮影だけ先に行って画像を保存し、その画像をVSCode等で見ながら
    座標を手入力する run_with_saved_image() の使い方に切り替えること(下部参照)。
"""

import json
import numpy as np
import cv2
import zmq

DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 5555

SLM_W, SLM_H = 1920, 1080
ROI_RES = (880, 1600)  # (height, width)  main_citl.py の roi_res と合わせる

# ボードの位置・サイズ(calibrate_homography_checkerboard.py と同じ設定でよい)
BOARD_COLS, BOARD_ROWS = 5, 4
SQUARE_PX = 250

OUTPUT_JSON = "./H_calibration.json"

# クリック結果を入れるグローバル
clicked_points = []
display_img = None


def make_checkerboard(slm_w, slm_h, cols, rows, square_px):
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

    # ボード全体の4隅のSLM座標(理想の四角形)
    board_corners_slm = np.array([
        [x0, y0],                                   # 左上
        [x0 + (cols + 1) * square_px, y0],           # 右上
        [x0 + (cols + 1) * square_px, y0 + (rows + 1) * square_px],  # 右下
        [x0, y0 + (rows + 1) * square_px],           # 左下
    ], dtype=np.float64)

    return img, board_corners_slm


def slm_to_roi_coords(points, slm_w, slm_h, roi_w, roi_h):
    scale_x = roi_w / slm_w
    scale_y = roi_h / slm_h
    out = points.copy()
    out[:, 0] *= scale_x
    out[:, 1] *= scale_y
    return out


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


def mouse_callback(event, x, y, flags, param):
    global clicked_points, display_img
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        cv2.circle(display_img, (x, y), 8, (0, 0, 255), 2)
        cv2.putText(display_img, str(len(clicked_points)), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Click 4 corners (TL->TR->BR->BL)", display_img)


def compute_and_save_homography(captured_points_full_res, board_corners_roi, roi_w, roi_h):
    H, _ = cv2.findHomography(np.array(captured_points_full_res, dtype=np.float64), board_corners_roi)

    src_h = np.hstack([captured_points_full_res, np.ones((4, 1))])
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj - board_corners_roi, axis=1)

    print("\nH =")
    print(H)
    print(f"再投影誤差: 各点={err}, 平均={err.mean():.2f}px")

    out = {
        "H": H.tolist(),
        "dst_wh": [roi_w, roi_h],
        "margin": 0,
        "num_points": 4,
        "num_inliers": 4,
        "reproj_error_mean_px": float(err.mean()),
        "method": "manual_4point",
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n保存しました: {OUTPUT_JSON}")


def main():
    global clicked_points, display_img

    client = DaemonClient()
    board_img, board_corners_slm = make_checkerboard(SLM_W, SLM_H, BOARD_COLS, BOARD_ROWS, SQUARE_PX)
    roi_h, roi_w = ROI_RES
    board_corners_roi = slm_to_roi_coords(board_corners_slm, SLM_W, SLM_H, roi_w, roi_h)

    print("チェッカーボードを表示・撮影します...")
    raw_img = client.display_and_capture(board_img)
    client.close()

    # コントラスト正規化(見やすくするため。クリック座標は元画像スケールに戻して使う)
    img_min, img_max = raw_img.min(), raw_img.max()
    if img_max > img_min:
        normalized = ((raw_img.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = raw_img
    cv2.imwrite("debug_for_manual_click.png", normalized)

    # 画面に収まるよう縮小して表示する(クリック座標は後で元スケールに戻す)
    scale = 0.25
    small = cv2.resize(normalized, (int(normalized.shape[1] * scale), int(normalized.shape[0] * scale)))
    display_img = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    cv2.namedWindow("Click 4 corners (TL->TR->BR->BL)")
    cv2.setMouseCallback("Click 4 corners (TL->TR->BR->BL)", mouse_callback)

    print("\nウィンドウ上で、ボードの4つの角を 左上->右上->右下->左下 の順にクリックしてください。")
    print("'r' でリセット、'q' で中断、4点クリックすると自動で進みます。")
    print("(GUIが開かない/エラーになる場合は、このスクリプトの下部にある")
    print(" run_with_saved_image_and_manual_coords() を使う方法に切り替えてください)\n")

    while True:
        cv2.imshow("Click 4 corners (TL->TR->BR->BL)", display_img)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            clicked_points = []
            display_img = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("中断しました")
            return
        if len(clicked_points) == 4:
            cv2.waitKey(500)
            break

    cv2.destroyAllWindows()

    captured_points_full_res = np.array(clicked_points, dtype=np.float64) / scale

    print("クリックされた座標(フル解像度換算):")
    for i, p in enumerate(captured_points_full_res):
        print(f"  点{i+1}: ({p[0]:.1f}, {p[1]:.1f})")

    compute_and_save_homography(captured_points_full_res, board_corners_roi, roi_w, roi_h)


def run_with_saved_image_and_manual_coords():
    """
    rtxstation上にGUI(ディスプレイ)が無く cv2.imshow が使えない場合の代替手段。

    1. まず main() の中の撮影部分だけ実行して画像を保存する(下のコードで自動で保存される)
    2. 保存された debug_for_manual_click.png を VSCode 等の画像ビューアで開く
    3. 画像上でマウスを4隅にあわせ、VSCodeの座標表示やペイントソフト等でpiexl座標を読む
       (または、十字の目盛りを画像に重ねた版を別途出力することもできる)
    4. 読み取った4点の座標を、下の CAPTURED_POINTS に直接書き込んで、このブロックを実行する
    """
    board_img, board_corners_slm = make_checkerboard(SLM_W, SLM_H, BOARD_COLS, BOARD_ROWS, SQUARE_PX)
    roi_h, roi_w = ROI_RES
    board_corners_roi = slm_to_roi_coords(board_corners_slm, SLM_W, SLM_H, roi_w, roi_h)

    # ↓ debug_for_manual_click.png を見て、フル解像度での(x,y)座標をここに書く
    # 順番: 左上 -> 右上 -> 右下 -> 左下
    CAPTURED_POINTS = np.array([
        [0, 0],   # 左上 (要修正)
        [0, 0],   # 右上 (要修正)
        [0, 0],   # 右下 (要修正)
        [0, 0],   # 左下 (要修正)
    ], dtype=np.float64)

    compute_and_save_homography(CAPTURED_POINTS, board_corners_roi, roi_w, roi_h)


if __name__ == "__main__":
    main()