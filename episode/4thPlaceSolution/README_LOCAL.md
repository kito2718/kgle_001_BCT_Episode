# Kaggle Biohub 4位解法 ローカル環境ガイド

Kaggle コンペティション「**Biohub - Cell Tracking During Development**」の4位入賞ソリューション（`songqizhou` 氏、submission: Private 0.962 / Public 0.969）のローカル環境および再現リポジトリです。

参考URL:
- [Kaggle Notebook: biohub-4th-place-solution](https://www.kaggle.com/code/songqizhou/biohub-4th-place-solution)
- [Kaggle Dataset: songqizhou/biohub-4th-place-artifacts](https://www.kaggle.com/datasets/songqizhou/biohub-4th-place-artifacts)

---

## 1. ディレクトリ構成

```text
4thPlaceSolution/
├── .venv/                      # Python 3.12 仮想環境 (uv管理、PyTorch CUDA 12.4等)
├── .vscode/
│   └── settings.json          # VSCode インタープリター設定・検索パス
├── biohub-4th-place-artifacts/ # 4位解法アーティファクト (スクリプト群 & 学習済み重み models/)
│   ├── models/                # 3D Net, Edge Transformer, Division CNN, CatBoost等の重み
│   ├── *.py                   # detection, downstream, ILP, tracking 実行スクリプト
│   └── README.md
├── input/                     # ディレクトリジャンクション (..\input へのリンク)
│   └── test/                  # 公開テスト用 4動画 (*.zarr)
├── working/                   # 推論実行時の中間生成物および最終成果物 (submission.csv)
├── biohub-4th-place-solution.ipynb # 元のKaggle推論ノートブック
├── biohub-4th-place-solution.py    # 推論実行スクリプト (ローカル・Kaggle両対応版)
├── requirements.txt           # 仮想環境再構築用ファイル
└── README_LOCAL.md            # 本ドキュメント
```

---

## 2. 環境スペックとインストール済みライブラリ

- **OS**: Windows 11
- **GPU**: NVIDIA GeForce RTX 4060 (8GB VRAM / CUDA Compute Capability 8.9)
- **Python**: 3.12.13 (uv 管理)
- **PyTorch**: 2.6.0 + CUDA 12.4
- **主要ライブラリ**:
  - `tracksdata`: 0.1.0rc12
  - `pyscipopt`: 6.3.0
  - `ilpy`: 0.6.0
  - `polars`: 2.1.0 (`polars-runtime-32` 含む)
  - `lightgbm`: 4.7.0
  - `catboost`: 1.2.10
  - `blosc2`: 4.14.1
  - `zarr`: 3.4.0
  - `numcodecs`: 0.15.2
  - `scikit-image`: 0.26.0

### 仮想環境の再構築コマンド

```powershell
# uv を使用する場合 (推奨・高速)
uv venv .venv --python 3.12
uv pip install -r requirements.txt --python .venv\Scripts\python.exe

# 通常の pip を使用する場合
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. 推論の実行方法

仮想環境の Python インタープリターで `biohub-4th-place-solution.py` を実行します。

```powershell
.venv\Scripts\python.exe biohub-4th-place-solution.py
```

### パイプラインの流れ

1. **密度ルーター (Density router)**:
   - 古典的 DoG (Difference of Gaussians) ブロブトラッカーで各動画の平均細胞間隔を推定。
   - 間隔 >= 19.7 μm の動画を「疎 (sparse)」として分類。
2. **3D Net 検出 (Detection with 3D Net)**:
   - 3D Net による細胞確率と中心単位ベクトル場の予測。
   - ベクトル場に沿った移流 (advection) とクラスタリング、適応的 NMS。
   - 擬似ラベル学習モデル (`net3d_128_pl.pt`) によるリファインメント。
   - 疎な動画には高解像度モデルに加え低解像度 3D Net (`net3d_64.pt`) も併用。
3. **リンク候補・特徴量抽出 (Linking)**:
   - 空間近傍リンクの探索、幾何学的・運動特徴量の計算。
   - Edge Transformer によるリンク確率、対照学習 (Contrastive Cell Embedding) 類似度。
   - LightGBM モデルによるリンク確率の統合。
4. **分裂事前確率 (Division prior)**:
   - 5フレーム 3D Division CNN アンサンブル。
   - CatBoost ノード分裂モデル、LightGBM + CatBoost 母娘ペアモデル。
5. **整数線形計画法 (ILP)**:
   - `tracksdata` + `SCIP` ソルバーによる大域的最適化追跡。
   - 運動補償スムージング、CatBoost 分岐 (fork) 検証、ギャップ補間。
6. **トラックサポートゲート (Track-support gate)**:
   - 疎な動画に対し、DoG / 3D Net-64 のサポートのない偽陽性トラックをフィルタリング。
7. **最終出力**:
   - `working/submission.csv` に出力。
