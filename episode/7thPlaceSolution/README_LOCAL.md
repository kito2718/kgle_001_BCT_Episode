# Biohub 7th Place Solution (`tatsutaka`) - Local Execution Environment

Kaggle コンペティション **Biohub - Cell Tracking During Development** の7位解法 ([Writeup](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/7th-place-solution) / [Inference Notebook](https://www.kaggle.com/code/tatsutaka/biohub-7th-place-inference)) をローカル Windows + NVIDIA GPU 環境で再現・実行するための環境です。

---

## 1. ディレクトリ構成

```text
7thPlaceSolution/
├── .venv/                              # Python 3.12 仮想環境 (PyTorch 2.6.0+cu124, SCIP/ilpy, OR-Tools, LightGBM 等)
├── .vscode/                            # VS Code 実行・デバッグ構成 (settings.json / launch.json)
├── datasets/                           # Kaggle 添付データセット一式 (7データセット)
│   ├── biohub-motion-nnunet-ilp-artifacts-1500-v001/  # Scratch 1500ep 3D ResEncM + nnU-Net カスタム Trainer + vendor/tools
│   ├── biohub-synthetic-native1500-0921/              # Synthetic Pretrained (1000ep) + Real (1500ep) 3D ResEncM
│   ├── biohub-division-runtime-20260908-v001/         # 分裂パイプライン (2x BiGRU, ExtraTrees, OR-Tools/SCIP ランタイム)
│   ├── biohub-division-swa-image5-assets-0926/        # SWA MLP (nn_h.pt) + 5-seed LightGBM 娘ペアモデル (91特徴量)
│   ├── biohub-motion-postprocess-v001/                # グラフ後処理・安全性回帰テスト
│   ├── biohub-choice957-offline-deps-20260911/        # Kaggle オフラインホイール
│   └── biohub-m4000-division-trt-t4-engines-0923/     # Kaggle Tesla T4 用ビルド済み TensorRT エンジン
├── input/                              # ..\input へのジャンクション (test/*.zarr, sample_submission.csv)
├── working/                            # 推論時の展開コード・中間グラフ (.geff)・最終 submission.csv 出力先
├── biohub-7th-place-inference.ipynb    # Kaggle オリジナル推論ノートブック
├── biohub-7th-place-inference.py       # ローカル Windows / Kaggle 両対応のスタンドアロン推論スクリプト
├── verify_local_env.py                 # 環境・モデル SHA-256・各コンポーネントの動作検証スクリプト
└── requirements.txt                    # 依存パッケージ定義
```

---

## 2. パイプライン概要 (4ステージ構成)

1. **Stage 1: 3D Detector & Backward Motion (3D ResEncM nnU-Net)**
   - 入力: 4チャネル 3D ボリューム `(I_{t-d}, I_t, I_{t+d}, d/2)` (推論時は `d=1`, 物理ボクセル間隔 `[1.625, 0.40625, 0.40625] µm`)。
   - 最高解像度デコーダ段を省略した Half-XY 出力 (`nnUNetTrainerBiohubColabM2000`) を3次元線形補間で元解像度へ戻し、平面 D4 (8視点) TTA で検出ロジットを統合、恒等視点から3D逆方向変位ベクトル $(\Delta z, \Delta y, \Delta x)$ (`×8 µm`) を推論。
   - 3D局所最大値 (`threshold=0.30` または `0.28`) と物理距離 Greedy NMS (`2.0 µm`) により候補ノードを抽出。
   - ガウス重み付き近傍サンプリング (`sigma=1 µm`, `radius=2 µm`) による逆方向投影後、`7.0 µm` 以内の最大 Top-5 候補へ遷移エッジを張り、$\exp(-d_{ij}^2/(2\times 3^2))\sqrt{q_i q_j}$ でエッジ確率を算出。

2. **Stage 2: 専用細胞分裂 (Division) パイプライン**
   - **親ノード画像スコアラー (`ParentScorer` / `SlidingT3Residual`)**: `SparseLocalFlow` で位置合わせした前後5フレーム (`t-2..t+2`) の `12×48×48` クロップに対し、生画像・差分7ch + 再構成ガウスヒートマップ7ch の計14chを入力とする 3D CNN + BiGRU (2シードアンサンブル) で親分裂確率を推論。
   - **娘ペアランキング (`PortableForest` & `daughter_image_lgb5`)**: 73個の幾何・局所軌跡特徴量による ExtraTrees と、娘ペア周辺の画像強度・コントラスト特徴18個を加えた91特徴量による 5-seed LightGBM アンサンブルで娘ペア候補を評価。
   - **分裂イベント統合 MLP (`nn_h.pt`)**: 73幾何・軌跡特徴 + 7競合親・継続コスト特徴 + 親検出スコア + BiGRU ロジット + ExtraTrees 確率の計83次元を入力とする 2層 SiLU MLP (SWA 重み) により、分裂ペア報酬ロジットを算出。

3. **Stage 3: 大域的 ILP 最適化 (`CertifiedPairSolver`)**
   - Google OR-Tools の最小費用循環流 (`SimpleMinCostFlow`) と 1対1 貪欲森で初期実行可能解 (Warmstart) を構築し、SCIP (`ilpy` / `pyscipopt`, `gap=0`) で大域的最適解を厳密保証。
   - 出次数2の親ノードに対する娘ペア報酬折りたたみ (`BIOHUB_PAIR_FOLD_MODE='fold'`) およびノード数 $\le 30,000$ での `SCIP_PARAMSETTING.OFF` ガード付き高速化を適用。

4. **Stage 4: グラフ後処理 (`final_post.py`)**
   - 1フレーム欠損のギャップ結合 (`<= 9.0 µm`、孤立ノード再利用 `<= 3.2 µm` または局所輝度補正付き中点ノード挿入)。
   - 短トラック除去 (`min_len=9`、分裂を含む連結成分の保護、高信頼度8ノード成分の保持、および `boundary_mean1_post.py` による動画境界の3〜8ノード高相対スコア成分の復元)。
   - 隣接フレーム同一時の座標投影 (`first_projection.py`) と、移動上限 `2.0 µm` 付き局所線形平滑化 (`window=4`, `weight=0.6`)。

---

## 3. 実行方法

### 3.1 環境・チェックポイント検証 (`verify_local_env.py`)

GPU・必須ライブラリ・全モデル重みの SHA-256 ハッシュ・各推論モジュールのスモークテストを実行します。

```powershell
.\.venv\Scripts\python.exe verify_local_env.py
```

### 3.2 高速スモークテスト (1動画・5フレーム制限)

1本のテスト動画 (`44b6_0113de3b`) の先頭5フレームのみを対象に、Stage 1〜4 から `working/submission.csv` 生成までを一気通貫で検証します (RTX 4060 で約40秒)。

```powershell
.\.venv\Scripts\python.exe biohub-7th-place-inference.py --movies 44b6_0113de3b --max-frames 5
```

### 3.3 フル推論の実行 (全4テスト動画)

7位最終選択サブミッション設定 (`pretrained` 検出器 + `det-threshold=0.30`) で全4動画の推論を実行します。

```powershell
.\.venv\Scripts\python.exe biohub-7th-place-inference.py
```

Writeup で言及されている各バリエーションも CLI 引数で切り替え可能です:

```powershell
# 検出閾値 0.28 (Pretrained: Public LB 0.903 / Private LB 0.873)
.\.venv\Scripts\python.exe biohub-7th-place-inference.py --detector pretrained --det-threshold 0.28

# Scratch (合成事前学習なし 1500ep) 検出器 + 閾値 0.28 (Public LB 0.906 / Private LB 0.878)
.\.venv\Scripts\python.exe biohub-7th-place-inference.py --detector scratch --det-threshold 0.28

# 2モデルアンサンブル (Pretrained + Scratch)
.\.venv\Scripts\python.exe biohub-7th-place-inference.py --detector ensemble --det-threshold 0.30
```
