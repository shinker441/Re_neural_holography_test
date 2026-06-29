"""
remote_prop.py
サーバ(rtxstation)側で使う、Pengの PhysicalProp 互換クラス。

PCのhw_daemon.pyにzmq経由で「位相を表示して撮影して」を依頼し、
撮影画像を振幅テンソルとして返す。

main.py / train_model.py / eval.py からは、
本家の PhysicalProp と同じ呼び方(forward(slm_phase) -> captured_amp)で使える。

使い方の例（main.pyのCITLブロックを想定）:
    from remote_prop import RemotePhysicalProp
    camera_prop = RemotePhysicalProp(
        daemon_host="127.0.0.1",   # SSHトンネル経由なのでlocalhost
        daemon_port=5555,
        roi_res=(1080, 1920),
    )
    captured_amp = camera_prop(slm_phase)   # slm_phase: torch.Tensor [1,1,H,W], 0-1
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
import zmq


class RemotePhysicalProp(nn.Module):
    """
    Pengの PhysicalProp と同じインターフェースを持つ、
    実機(PC側のhw_daemon)をネットワーク越しに叩くモジュール。

    forward(slm_phase) -> captured_amp
        slm_phase:    torch.Tensor, shape [1,1,H,W], 値範囲 0-1 (phaseの正規化済みテンソル)
        captured_amp: torch.Tensor, shape [1,1,roi_h,roi_w], 値範囲 0-1 (撮影振幅)
    """

    def __init__(
        self,
        daemon_host="127.0.0.1",
        daemon_port=5555,
        roi_res=(1080, 1920),
        homography_matrix=None,   # 3x3 numpy array。Noneなら素通し(後でcalibration後に設定)
        timeout_ms=5000,
        device=None,
    ):
        super().__init__()
        self.roi_res = roi_res  # (height, width)
        self.homography_matrix = homography_matrix
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{daemon_host}:{daemon_port}")

        print(f"[RemotePhysicalProp] connected to tcp://{daemon_host}:{daemon_port}")

    # ---- 内部ユーティリティ ----

    def _tensor_to_phase_u8(self, slm_phase: torch.Tensor) -> np.ndarray:
        """torch.Tensor [1,1,H,W] (0-1) -> numpy uint8 [H,W]"""
        arr = slm_phase.detach().squeeze().cpu().numpy()
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).astype(np.uint8)

    def _send_and_receive(self, phase_u8: np.ndarray) -> np.ndarray:
        """デーモンに位相を送り、撮影画像(numpy, uint8)を受け取る"""
        meta = json.dumps({"height": phase_u8.shape[0], "width": phase_u8.shape[1]}).encode("utf-8")
        self.sock.send_multipart([meta, phase_u8.tobytes()])

        meta_r, data_r = self.sock.recv_multipart()
        meta_resp = json.loads(meta_r.decode("utf-8"))
        h, w = meta_resp["height"], meta_resp["width"]
        if h == 0 or w == 0:
            raise RuntimeError("[RemotePhysicalProp] daemon側で撮影に失敗しました(空画像が返りました)")
        img = np.frombuffer(data_r, dtype=np.uint8).reshape(h, w)
        return img

    def _postprocess(self, raw_img: np.ndarray) -> torch.Tensor:
        """
        撮影直後の生画像(numpy, uint8, フル解像度) -> 振幅テンソル(torch, [1,1,roi_h,roi_w], 0-1)

        現状は homography_matrix が未設定なら、中心クロップ+リサイズで簡易対応。
        homography校正が済んだら self.homography_matrix を設定し、ここでcv2.warpPerspectiveを通す。
        """
        import cv2

        if self.homography_matrix is not None:
            roi_h, roi_w = self.roi_res
            img = cv2.warpPerspective(
                raw_img, self.homography_matrix, (roi_w, roi_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        else:
            # 簡易フォールバック: 中心クロップ後にROI解像度へリサイズ
            h, w = raw_img.shape
            roi_h, roi_w = self.roi_res
            # アスペクトを保った中心クロップ
            target_aspect = roi_w / roi_h
            src_aspect = w / h
            if src_aspect > target_aspect:
                new_w = int(h * target_aspect)
                x0 = (w - new_w) // 2
                cropped = raw_img[:, x0:x0 + new_w]
            else:
                new_h = int(w / target_aspect)
                y0 = (h - new_h) // 2
                cropped = raw_img[y0:y0 + new_h, :]
            img = cv2.resize(cropped, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

        intensity = img.astype(np.float32) / 255.0
        amp = np.sqrt(np.clip(intensity, 0.0, 1.0))
        amp_tensor = torch.from_numpy(amp).float().unsqueeze(0).unsqueeze(0).to(self.device)
        return amp_tensor

    # ---- PhysicalProp互換インターフェース ----

    def forward(self, slm_phase: torch.Tensor) -> torch.Tensor:
        phase_u8 = self._tensor_to_phase_u8(slm_phase)
        raw_img = self._send_and_receive(phase_u8)
        amp = self._postprocess(raw_img)
        return amp

    def set_homography(self, H: np.ndarray):
        """homography校正完了後にこれで行列を設定する"""
        self.homography_matrix = H

    def close(self):
        self.sock.close()
        self.ctx.term()


if __name__ == "__main__":
    # 単体動作確認用（カメラ実機が必要、サーバ側で実行する想定）
    prop = RemotePhysicalProp(daemon_host="127.0.0.1", daemon_port=5555, roi_res=(1080, 1920))

    test_phase = torch.zeros(1, 1, 1080, 1920)
    t0 = time.perf_counter()
    amp = prop(test_phase)
    print(f"roundtrip: {(time.perf_counter()-t0)*1000:.1f} ms")
    print(f"captured amp shape: {amp.shape}, range: [{amp.min().item():.3f}, {amp.max().item():.3f}]")

    prop.close()