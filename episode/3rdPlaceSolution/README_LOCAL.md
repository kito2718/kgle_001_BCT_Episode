# Kaggle Biohub 3位解法 ローカル環境ガイド

Kaggle コンペティション「**Biohub - Cell Tracking During Development**」の3位入賞ソリューション（`ren4yu` 氏、submission 060: Private 0.96918 / Public 0.97832）のローカル環境および再現リポジトリです。

参考URL:
- [Kaggle Notebook: biohub-inference-run](https://www.kaggle.com/code/ren4yu/biohub-inference-run)
- [Kaggle Dataset: ren4yu/biohub-inference](https://www.kaggle.com/datasets/ren4yu/biohub-inference)
- [Kaggle Dataset: ren4yu/biohub-weights](https://www.kaggle.com/datasets/ren4yu/biohub-weights)

---

## 1. ディレクトリ構成

```text
3rdPlaceSolution/
├── .venv/                      # Python 3.12 仮想環境 (uv管理、PyTorch CUDA 12.4等)
├── .vscode/
│   ├── settings.json          # VSCode インタープリター設定・検索パス
│   └── launch.json            # デバッグ実行構成
├── biohub-inference/          # パッケージ本体 (biohub_infer/) および offline wheels
├── biohub-weights/            # 全70個の重みファイル (md5sums.txt完全検証済)
├── input/                     # ディレクトリジャンクション (c:\work\aaa\s9\input へのリンク)
│   └── test/                  # 公開テスト用 4動画 (.zarr)
├── biohub-inference-run.ipynb # 推論ノートブック
├── biohub_inference_run.py    # 推論実行スクリプト (ノートブックのPython移植版)
├── requirements.txt           # 仮想環境再構築用ファイル
├── verify_local_env.py        # ローカル環境の健全性検証スクリプト
└── README_LOCAL.md            # 本ドキュメント
```

---

## 2. 環境スペックとインストール済みライブラリ

- **OS**: Windows 11
- **GPU**: NVIDIA GeForce RTX 4060 (8GB VRAM / Compute Capability 8.9)
- **Python**: 3.12.13 (uv 管理)
- **PyTorch**: 2.6.0 + CUDA 12.4
- **主要ライブラリ**:
  - `tensorrt-cu12`: 11.4.0.106
  - `zarr`: 3.3.0
  - `numcodecs`: 0.16.5
  - `monai`: 1.5.2
  - `lightgbm`: 4.7.0
  - `xgboost`: 3.2.0
### 仮想環境の再構築コマンド

```powershell
# uv を使用する場合 (推奨・高速)
uv venv .venv --python 3.12
uv pip install -r requirements.txt

# 通常の pip を使用する場合
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. 環境検証スクリプトの実行

ローカル環境の健全性（GPU認識、テストデータ、重みファイルのMD5チェック、PyTorch Matcher/Flowモデルのロード等）は、以下のコマンドで一発で確認できます。

```powershell
.venv\Scripts\python.exe verify_local_env.py
```

実行結果例:
```text
============================================================
Biohub 3rd Place Solution - Local Environment Verification
============================================================
[1] Device: cuda
    GPU: NVIDIA GeForce RTX 4060
    Capability: (8, 9)
    Torch Version: 2.6.0+cu124

[2] Test Movies: 4 found in .../input/test
    - 44b6_0113de3b.zarr: shape (100, 64, 256, 256), chunks (1, 64, 256, 256)
    - 44b6_0b24845f.zarr: shape (100, 64, 256, 256), chunks (1, 64, 256, 256)
    - 6bba_05b6850b.zarr: shape (100, 64, 256, 256), chunks (1, 64, 256, 256)
    - 6bba_05db0fb1.zarr: shape (100, 64, 256, 256), chunks (1, 64, 256, 256)

[3] Weights Directory: .../input/datasets/ren4yu/biohub-weights
    Successfully resolved all 70 required weight files!

[4] Testing Matcher checkpoints loading...
loaded weights from ...\dl_401_fold0_last.ckpt
    - Fold 0 matcher loaded: EmbedUnet
...
[5] Testing SSL Flow model loading...
    - Flow model loaded: FlowNet3D
============================================================
```

---

## 4. 実行に関する重要事項（TensorRT エンジンとハードウェア仕様）

3位解法は Kaggle の実行環境（**Tesla T4 x2, TensorRT 10.9.0.34**）に合わせて最適化されています。

1. **プリコンパイル済み TensorRT エンジン (`.engine`)**:
   - 検出器（15個）および分裂モデル（40個）の `.engine` ファイルは、NVIDIA Tesla T4 (Turing アーキテクチャ / sm_75) 向けにコンパイルされています。
   - NVIDIA TensorRT の仕様上、シリアライズされた `.engine` バイナリはコンパイル時と異なる GPU アーキテクチャ（RTX 4060 は Ada Lovelace / sm_89）やプラットフォームでは直接 deserialize できません。
2. **PyTorch ネイティブコンポーネント**:
   - 追跡マッチャー（5-fold `EmbedUnet` チェックポイント）、3D Dense Flow（`FlowNet3D`）、GBDT較正器（LightGBM / XGBoost）、および Lineage LP（整数計画法による系譜グラフ最適化・後処理）はすべてローカル環境で直接動作します。
3. **Kaggle 上での完全実行**:
   - 本ノートブック `biohub-inference-run.ipynb` を Kaggle 上で「Tesla T4 x2」を指定して実行することで、公式の公開テスト4動画に対する完全な提出ファイル（`submission.csv`）が約20分で生成されます。
