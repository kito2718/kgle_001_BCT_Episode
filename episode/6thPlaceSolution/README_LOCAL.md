# Kaggle Biohub 6位解法 ローカル環境ガイド

Kaggle コンペティション「**Biohub - Cell Tracking During Development**」の6位入賞ソリューション（`jamalsaeedi` / Jamal Saeedi 氏: **Detect, Link, Solve, Repeat — 検出・リンク一体型ネットワーク + 大域的ILP + 3%アノテーションに基づく教師-生徒反復学習**）のローカル実行・再現環境です。

参考URL:
- [6位解法 Writeup: 6th Place: Detect, Link, Solve, Repeat](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/6th-place-solution)
- [6位解法 GitHub リポジトリ: jamal-saeedi/biohub_cell_tracking_kaggle](https://github.com/jamal-saeedi/biohub_cell_tracking_kaggle)
- [6位解法 Kaggle Models (重み & Re-scorers): jamalsaeedi/biohub-cell-tracking](https://www.kaggle.com/models/jamalsaeedi/biohub-cell-tracking)
- [6位解法 Kaggle 推論ノートブック: biohub-cell-tracking-inference](https://www.kaggle.com/code/jamalsaeedi/biohub-cell-tracking-inference)
- [6位解法 コードデータセット: jamalsaeedi/biohub-cell-tracking-kaggle](https://www.kaggle.com/datasets/jamalsaeedi/biohub-cell-tracking-kaggle)

---

## 1. ディレクトリ構成

```text
6thPlaceSolution/
├── .venv/                                     # Python 3.12 仮想環境 (uv管理、PyTorch CUDA 12.4等)
├── .vscode/
│   ├── settings.json                          # VSCode インタープリター設定・補完検索パス
│   └── launch.json                            # デバッグ実行構成 (fin推論 / best推論 / Smoke Test / 環境検証)
├── biohub_cell_tracking_kaggle/               # 6位解法公式GitHubリポジトリ (editable install済)
│   ├── docs/                                  # 解説ドキュメント (SOLUTION.md, MODEL_SUMMARY.md) & 図表
│   ├── notebooks/                             # 公式推論・学習ウォークスルーノートブック
│   ├── recipes/                               # 全学習系譜のレシピ (train/*.json, ensembles/*.json)
│   ├── src/biohub_tracking/                   # パッケージ本体
│   │   ├── models/                            # IsotropicLineageNet & MultiScaleLineageNet
│   │   ├── isotropic/                         # デコード、TTA、EncoderCache、TreeEnsemble再スコアラー、LP-first ILP
│   │   ├── postprocess/                       # ドリフト補正付き直線フィット平滑化
│   │   └── training/                          # 3状態ターゲット、カウント事前分布、擬似ラベル学習、LightGBM学習
│   └── tools/                                 # download_models.py, smoke_test.py, reproduce_training.sh 等
├── models/                                    # Kaggle Models から取得・SHA-256検証済みのモデル群
│   ├── mres-fin-444322/1/                     # 最終選択提出 (Private 0.953 / Public 0.961): 6モデル + 6 Re-scorers
│   └── mres-best-444322/1/                    # 最良検証チェックポイント (Private 0.955 / Public 0.958): 6モデル + 6 Re-scorers
├── input/                                     # ディレクトリジャンクション (..\input へのリンク)
│   └── test/                                  # 公開テスト用 4動画 (*.zarr)
├── working/                                   # 推論実行時の中間生成物 (predictions/*.geff) および最終提出 (submission.csv)
├── SETTINGS.json                              # ローカル / Kaggle 入出力パス設定ファイル
├── biohub-cell-tracking-inference.ipynb       # 元のKaggle推論ノートブック
├── biohub-cell-tracking-inference.py          # 6位解法 推論実行スクリプト (ローカル・Kaggle両対応、CLI引数対応)
├── verify_local_env.py                        # ローカル環境の健全性検証スクリプト (SHA-256・GPU・モデル・Numba・HiGHS確認)
├── requirements.txt                           # 仮想環境再構築用ファイル
└── README_LOCAL.md                            # 本ドキュメント
```

---

## 2. 環境スペックとインストール済みライブラリ

- **OS**: Windows 11
- **GPU**: NVIDIA GeForce RTX 4060 (8GB VRAM / CUDA Compute Capability 8.9 / Native bf16対応)
- **Python**: 3.12.13 (uv 管理)
- **PyTorch**: 2.6.0 + CUDA 12.4
- **主要ライブラリ**:
  - `biohub_tracking`: 1.0.0 (`./biohub_cell_tracking_kaggle` editable install)
  - `tracksdata`: 0.1.0rc12
  - `geff`: 1.4.0.1.3 (`geff-spec`: 1.3.1)
  - `pyscipopt`: 6.3.0 / `ilpy`: 0.6.0 / `scipy`: 1.18.1 (HiGHS LP/MILP ソルバー)
  - `numba`: 0.68.1 (フラット配列化されたLightGBM決定木の高速並列推論用)
  - `lightgbm`: 4.7.0 (エッジ再スコアラー学習用)
  - `polars`: 2.1.0 (`polars-runtime-32` 含む)
  - `zarr`: 3.4.0 / `numcodecs`: 0.15.2 / `blosc2`: 4.15.0
  - `kagglehub`: 1.0.2

### 仮想環境の再構築とモデルダウンロード手順

```powershell
# 1. uv で仮想環境を作成しパッケージをインストール
uv venv .venv --python 3.12
uv pip install -r requirements.txt --python .venv\Scripts\python.exe
uv pip install -e ./biohub_cell_tracking_kaggle --no-deps --python .venv\Scripts\python.exe

# 2. Kaggle Models から fin / best 両バリアントの重み・再スコアラーをダウンロード & SHA-256検証
.venv\Scripts\python.exe biohub_cell_tracking_kaggle\tools\download_models.py --settings SETTINGS.json
```

---

## 3. 環境検証スクリプトとスモークテストの実行

### A. ローカル環境の健全性検証 (`verify_local_env.py`)
GPU認識、必須パッケージ、テスト動画(4本)、`fin`/`best` 全24ファイルのSHA-256ハッシュ検証、`IsotropicLineageNet` と `MultiScaleLineageNet` のチェックポイントロード、Numba JITツリー探索、およびHiGHS LP-firstソルバーの動作確認を行います。

```powershell
.venv\Scripts\python.exe verify_local_env.py
```

### B. 合成データによる全工程スモークテスト (`smoke_test.py`)
合成データ生成 → 教師モデル学習 → 擬似ラベル生成 → 生徒モデル学習 → LightGBM再スコアラー学習 → 推論・ILP求解・CSV検証までの全パイプラインをCPU上で数分でテストします。

```powershell
.venv\Scripts\python.exe biohub_cell_tracking_kaggle\tools\smoke_test.py
```

---

## 4. 推論の実行方法

### A. 最終選択モデル (`fin`: Private 0.953 / Public 0.961) で推論

```powershell
.venv\Scripts\python.exe biohub-cell-tracking-inference.py --variant fin
```

### B. 最良検証チェックポイント (`best`: Private 0.955 / Public 0.958) で推論

```powershell
.venv\Scripts\python.exe biohub-cell-tracking-inference.py --variant best
```

### C. 特定の動画のみを指定して推論 (高速動作確認用)

```powershell
.venv\Scripts\python.exe biohub-cell-tracking-inference.py --variant fin --movies 44b6_0113de3b
```

- 出力先: `working/submission.csv`（各動画の解グラフは `working/predictions/*.geff`）
- CLI エントリポイント `.venv\Scripts\biohub-predict.exe` も同じオプションで利用可能です。

---

## 5. 6位解法のアーキテクチャと技術的ハイライト

1. **検出とリンクを単一ネットワークで同時学習 (`IsotropicLineageNet` & `MultiScaleLineageNet`)**:
   - ネイティブ解像度の3フレーム `(t-1, t, t+1)` (`64 × 256 × 256`) を入力とし、フレーム単位の3Dエンコーダ（異方性4:1を解消する水平2段階縮小＋3D縮小）で特徴抽出。推論時は `EncoderCache` により各フレーム・各TTAビューのエンコード結果をキャッシュして3つのスライディングウィンドウ間で再利用し、1.4〜1.9倍高速化。
   - **時間的融合 (Temporal Fusion)**: 隣接フレームへの有界な変位を予測して特徴量をサンプリング・位置合わせした上でアテンション統合。
   - **関連付けヘッド (Association Head)**: 連続2フレームの双方向k-NN候補グラフ (`k = 4`, `<= 20 μm`) 上でスパースグラフアテンションを実行し、各細胞の候補親に対するソフトマックス確率（明示的な **「新規細胞 (new cell)」クラス** を含む）、親の分裂ロジット、娘ペアロジット、および速度・不確実性を予測。
2. **3%スパースアノテーションへの対策（3状態ターゲット + カウント事前分布 + パイプライン全体のNoisy Student自己学習）**:
   - アノテーション中心から3 μm以内を正例、フレーム中央値より暗くアノテーションから6 μm以上離れた領域を検証済み背景、それ以外の未アノテーション細胞領域を損失重み0（未知）とする3状態Focal Lossを採用し、さらに主催者発表の動画別推定細胞数にヒートマップ総和を引き寄せる **Count Prior** で過剰検出を抑制。
   - 単一モデルではなく **「アンサンブル + 再スコアラー + 大域的ILP」のパイプライン全体** を教師とし、ILPが解いた高精度なトポロジー（接続関係）と平滑化前の生検出座標（撮像コホート別の体系的Zオフセット補正付き）から約500万細胞・490万リンクの擬似ラベルを生成して反復学習。
3. **2アーキテクチャ × 6モデルアンサンブル + メンバー専用LightGBM再スコアラー (`TreeEnsemble`)**:
   - 異なる動画分割・学習系譜を持つ `IsotropicLineageNet` 3モデル (A, B, C: 4 TTA views) と `MultiScaleLineageNet` 3モデル (D: 3 views, E, F: 2 views) の中心ヒートマップとオフセットを平均化して**単一の共有細胞セット**を抽出。
   - 共有細胞間の候補リンクに対し、各モデル固有のヘッド出力と23次元の幾何・競合・局所密度・ドリフト補正特徴量を入力とする**メンバー専用のLightGBM再スコアラー**（新規細胞確率は維持したまま親候補のロジットを再ランク付け、推論時はNumba JITで高速評価）を適用し、6モデルの確率分布を平均化。
4. **ドリフト補正距離付き大域的ILP (`LP-first` + `SCIP`) とドリフト補償平滑化**:
   - 高信頼リンク (`P >= 0.7`) の変位中央値からフレーム間の顕微鏡ステージドリフトを推定・除去した距離をリンクコストに加算 (+0.009改善)。またネットワークの較正された「新規細胞」確率をそのまま出現コスト (`-log P_T(new cell) + 0.5`) として利用することで、手作業の修復ヒューリスティックを一切不要化。
   - 制約行列が分裂行を除き全ユニモジュラ（ネットワーク行列）である性質を活かし、**HiGHSでLP緩和を解いて整数変数を固定し、非整数変数とその3-hop近傍のみを小さなMILPで解く `LP-first` ソルバー (`solve_lpfix`)** を実装（SCIP単体比で38〜166倍高速、ギャップ0.5%超の場合のみSCIPへフォールバック）。
   - 最終出力前に累積ステージドリフトを一旦除去し、分裂点で止まる前後2フレームの直線フィッティング (`0.2 × 元座標 + 0.8 × フィット座標`) で主にZ軸の量子化誤差を平滑化した後、ドリフトを再加算してボリューム内にクランプ。
