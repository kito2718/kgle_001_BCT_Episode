# Kaggle Biohub 5位解法 ローカル環境ガイド

Kaggle コンペティション「**Biohub - Cell Tracking During Development**」の5位入賞ソリューション（`hirotetsu` / Tang 氏: **3D U-Net + Transformer Linker + Multi-stage ILP Tracking**）および、そのベースとなった公開ベースライン（`thibautgoldsborough` 氏: **UNet + Transformer baseline**）のローカル実行・再現環境です。

参考URL:
- [5位解法 Writeup: 5th Place: 3D U-Net + Transformer Linker + Multi-stage ILP Tracking](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/5th-place-3d-u-net-transformer-linker-multi-s)
- [5位解法 Kaggle Kernel: ctdd-11model-final_v2-ilp1-rematch-adj-off](https://www.kaggle.com/code/hirotetsu/ctdd-11model-final-v2-ilp1-rematch-adj-off)
- [5位解法 コードデータセット: hirotetsu/ctdd-final-v2-ilp1-rematch-adj-off-code](https://www.kaggle.com/datasets/hirotetsu/ctdd-final-v2-ilp1-rematch-adj-off-code)
- [5位解法 重みデータセット: hirotetsu/ctdd-run53-run74-run79-ensemble-code](https://www.kaggle.com/datasets/hirotetsu/ctdd-run53-run74-run79-ensemble-code)
- [公開ベースライン Notebook: unet-baseline-inference-submission](https://www.kaggle.com/code/thibautgoldsborough/unet-baseline-inference-submission)
- [公開ベースライン Artifacts: thibautgoldsborough/cellmot-baseline-artifacts](https://www.kaggle.com/datasets/thibautgoldsborough/cellmot-baseline-artifacts)

---

## 1. ディレクトリ構成

```text
5thPlaceSolution/
├── .venv/                                     # Python 3.12 仮想環境 (uv管理、PyTorch CUDA 12.4等)
├── .vscode/
│   ├── settings.json                          # VSCode インタープリター設定・補完検索パス
│   └── launch.json                            # デバッグ実行構成 (5位解法・ベースライン・環境検証)
├── ctdd-final-v2-ilp1-rematch-adj-off-code/   # 5位解法リポジトリ本体 (config, models, inference, utils 等)
│   ├── vendor/repo/                           # ベンダー同梱の tracking_cellmot パッケージ・学習スクリプト
│   ├── config.py                              # BaseConfig / InferConfig / SubmitConfig / TrainConfig
│   ├── dataset.py                             # ZebraHub事前学習・Competitionデータローダ・4x縮小処理
│   ├── models.py                              # TemporalUNet3D + NodeTransformer (v1 / v2)
│   ├── inference.py                           # マルチモデル・8x TTA推論 & 多段ILPグラフ構築
│   ├── inference_fast.py                      # OOM自動チャンクフォールバック・GPU/CPU非同期スケジューラ
│   └── utils.py                               # HiGHS/SCIP ILPソルバー、ILP3再リンク、ILP4ギャップ結合、スムージング
├── ctdd-run53-run74-run79-ensemble-code/      # 5位解法の学習済みチェックポイント群 (全11モデル、SHA-256検証済)
│   └── models/
│       ├── run53/fold_0..4/                   # NodeTransformer v1 (5-fold)
│       ├── run74/fold_0..4/                   # NodeTransformer v2 (5-fold)
│       └── run79/fold_0/                      # Residual TemporalUNet3D + NodeTransformer v2 (fold 0)
├── cellmot-baseline-artifacts/                # 公開ベースラインのコード & 重み (edge_predictor_best.pth)
├── input/                                     # ディレクトリジャンクション (..\input へのリンク)
│   └── test/                                  # 公開テスト用 4動画 (*.zarr)
├── working/                                   # 推論実行時の中間生成物 (.geff) および最終提出ファイル (submission.csv)
├── ctdd-11model-final-v2-ilp1-rematch-adj-off.py # 5位解法 推論実行スクリプト (ローカル・Kaggle両対応)
├── unet-baseline-inference-submission.ipynb   # 元の公開ベースライン推論ノートブック
├── unet_baseline_inference.py                 # 公開ベースライン 推論実行スクリプト (ローカル・Kaggle両対応)
├── verify_local_env.py                        # ローカル環境の健全性検証スクリプト (SHA-256・GPU・モデルロード確認)
├── requirements.txt                           # 仮想環境再構築用ファイル
└── README_LOCAL.md                            # 本ドキュメント
```

---

## 2. 環境スペックとインストール済みライブラリ

- **OS**: Windows 11
- **GPU**: NVIDIA GeForce RTX 4060 (8GB VRAM / CUDA Compute Capability 8.9)
- **Python**: 3.12.13 (uv 管理)
- **PyTorch**: 2.6.0 + CUDA 12.4
- **主要ライブラリ**:
  - `tracksdata`: 0.1.0rc12
  - `geff`: 1.4.0.1.3 (`geff-spec`: 1.3.1)
  - `pyscipopt`: 6.3.0
  - `ilpy`: 0.6.0
  - `rustworkx`: 0.18.1
  - `polars`: 2.1.0 (`polars-runtime-32` 含む)
  - `scipy`: 1.18.1 (HiGHS MILP ソルバー含む)
  - `zarr`: 3.4.0 / `numcodecs`: 0.15.2 / `blosc2`: 4.14.1

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

## 3. 環境検証スクリプトの実行

GPU認識、必須パッケージ、テスト動画（4本）、5位解法の全コード・11個のモデル重みのSHA-256ハッシュ検証、ベースライン重みの存在確認、およびモデルのロード確認は以下のコマンドで実行できます。

```powershell
.venv\Scripts\python.exe verify_local_env.py
```

---

## 4. 推論の実行方法

### A. 5位解法（11モデルアンサンブル + 8x TTA + 多段ILPトラッキング）の実行

```powershell
.venv\Scripts\python.exe ctdd-11model-final-v2-ilp1-rematch-adj-off.py
```
- 出力先: `working/submission.csv`（各動画の予測グラフは `working/predictions/*.geff`）
- `InferenceUNetRunner` により、8GB VRAM 環境で OOM が発生した場合でも自動的に TTA チャンクサイズ（`8 -> 4 -> 2 -> 1`）を縮小して全8ビューの推論を完走します。

### B. 公開ベースライン（単一モデル + ILP）の実行

```powershell
.venv\Scripts\python.exe unet_baseline_inference.py
```
- 出力先: `working/submission_baseline.csv`
- 最初の1動画のみ高速テストしたい場合は環境変数 `BASELINE_SLICE=":1"` を指定できます。

---

## 5. 公開ベースラインから5位解法への主な進化ポイント

1. **入力スケーリングと3D U-Net検出器**:
   - フル解像度 `(64, 256, 256)` を 3D Average Pooling で XY 方向に 1/4 縮小し、等方的な `(64, 64, 64)` ボリュームとして `TemporalUNet3D` に入力。
   - 検出ピークに対して各軸方向の対数放物線フィッティング (`refine_peaks_parabolic`) を適用し、サブボクセル精度の座標を算出。
2. **Transformer リンカーの4出力化 (`NodeTransformer` / `NodeTransformerV2`)**:
   - ベースラインのリンク確率 (`edge`) に加え、U-Net特徴量から抽出した16次元の外見埋め込み類似度 (`identity_embedding`)、新規細胞スコア (`appearance_head`)、および高重み付け損失で学習した細胞分裂スコア (`division_head`) の計4要素を予測。
   - `v2` (`run74`, `run79`) ではマルチスケール特徴量の Trilinear サンプリングと、フレーム内 Self-Attention + フレーム間 Cross-Attention、およびマイクロメートル単位の変位ベクトル入力を導入。
3. **ZebraHub 事前学習 + アンサンブル + 8x TTA**:
   - 4つの公開 ZebraHub データセット (`ZSNS001`, `ZSNS003`, `ZSNS004`, `ZSNS005`) で事前学習後、コンペデータ（クリップ種別 `44b6`/`6bba` と分裂数で層化した 5-fold CV）でファインチューニング。
   - XY 平面の反転・回転による8通りの TTA (`8fixed`) とモデルアンサンブル（`run53` 5-fold + `run74` 5-fold + `run79` fold0 の計11モデル）を統合。
4. **多段ILPトラッキングと軌跡修復・スムージング (`solve_graph`)**:
   - **ノード確信度コスト**: 検出確信度 `q` に応じて `-ilp_node_weight * (q - ilp_node_bias)` のノードコストを付与し、確信度の高い細胞や外見類似度の高いリンク、新規出現・分裂ヘッドの予測を ILP 目的関数に直接反映。
   - **多段ILP (`ILP1` / `ILP2` / `ILP3`)**: 確信度 `p >= 0.6` の候補を含めてギャップを埋めつつ、ノードと分岐を固定して弱いリンク（足場リンク `scaffold`）を補助的に用いて再リンク (`ILP3`) し、解決後に足場リンクを除去。
   - **ギャップ結合 (`ILP4` / `close_gaps`)**: 2〜10フレーム離れたトラック終端と始端を速度整合性と距離コストに基づき1対1の MILP で結合し、中間フレームを補間（空き検出があればスナップ）。
   - **運動保存スムージング (`smooth_track_coords`) + 1フレーム欠損補間 (`fill_track_gaps`)**: フレーム全体の共通運動を保護したうえで前後4フレームの直線フィッティング（重み `0.825`、最大シフト上限 `4.5 μm`）で座標のブレを平滑化し、最後に1フレーム欠損を補間。さらに最終CSV出力時は丸め誤差によるスコア低下を防ぐため小数第3位までの浮動小数点で出力。
