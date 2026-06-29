# Neural Holography (CITL) — 個人研究用フォーク

このリポジトリは [computational-imaging/neural-holography](https://github.com/computational-imaging/neural-holography)（Peng et al., *"Neural Holography with Camera-in-the-loop Training"*, SIGGRAPH Asia 2020）をベースに、筑波大学の研究室環境で動かすために改変したものです。

元のコード・研究の著作権は原著者に帰属します。本リポジトリは研究・学習目的の個人的な改変版であり、原著作の権利を主張するものではありません。

## 改変の概要

元のコードは、SLM・カメラ・レーザー制御を1台のPC上で行う構成を前提としています。本フォークでは、計算資源（GPU）とハードウェア（SLM・カメラ）を別々のマシンに分離した環境に対応させています。

```
┌─────────────────────────┐         ┌─────────────────────────┐
│  GPUサーバー              │  SSHトンネル │  実験室PC                │
│  (計算: CITL最適化等)     │◀───────────▶│  (SLM表示 + カメラ撮影)   │
│  Docker + PyTorch         │   (zmq RPC)  │  Windows                 │
└─────────────────────────┘             └─────────────────────────┘
```

### 主な変更点

- **`remote_prop.py`**（新規）
  元の `utils/modules.py` の `PhysicalProp` と同じインターフェースを持つ、リモート版の物理伝播クラス。SLM表示・カメラ撮影を実験室PC側のデーモンプロセスに委譲し、結果(撮影画像)をネットワーク越しに受け取る。

- **実験室PC側のハードウェアデーモン**（別管理、本リポジトリ外）
  SLM表示・整定時間待機・カメラ撮影をzmqの`REP`ソケットで待ち受け、GPUサーバー側からのリクエストに応答する常駐スクリプト。GPUサーバーとはSSHのリバースポートフォワーディング経由で通信する。

- **`calibrate_homography.py` / `calibrate_homography_checkerboard.py` / `calibrate_homography_manual.py`**（新規）
  実機のSLM-カメラ間のホモグラフィ変換を求めるためのキャリブレーションスクリプト群。

- **`main_citl.py`**（新規）
  リモートのハードウェアデーモンを使ったCamera-in-the-loop最適化のエントリポイント。

- **`eval.py` / `utils/utils.py`**（修正）
  リモート版`PhysicalProp`との連携、および評価指標まわりの調整。

- **`probe_offset_mapping.py`**（新規）
  SLM-カメラ間の対応関係を調べるための補助スクリプト。

## 実行環境

- GPUサーバー: Docker (PyTorch, CUDA), GPUは研究室の共用GPUサーバーを使用
- 実験室PC: Windows, SLM (Jasper Display EDK), カメラ (Basler, `pypylon`)
- 両者はSSHのリバーストンネル経由でzmq通信

## 注意

- 本リポジトリは特定の研究室のハードウェア構成（SLM・カメラの型番、ネットワーク環境）に強く依存したコードを含みます。そのままでは他の環境では動作しません。
- 元のPengらのコードの大部分（伝播モデル・最適化アルゴリズム・損失関数等）は無改造のまま使用しています。

## 元の研究について

```
@article{Peng:2020:NeuralHolography,
author = {Yifan Peng, Suyeon Choi, Nitish Padmanaban, Gordon Wetzstein},
title = {{Neural Holography with Camera-in-the-loop Training}},
journal = {ACM Trans. Graph. (SIGGRAPH Asia)},
year = {2020},
}
```

詳細は元のリポジトリ・論文を参照してください。
