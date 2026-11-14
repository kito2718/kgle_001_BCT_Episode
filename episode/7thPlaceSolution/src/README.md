# 7th Place Solution (`tatsutaka`) - 展開済み全ソースコード (`src/`)

このディレクトリには、`datasets/` 内の基底モジュールと `biohub-7th-place-inference.py` 内の動的パッチ (Cell 1〜13) をすべて統合した **最終実行状態の全 Python ソースコード (66ファイル)** が格納されています。

---

## 1. ディレクトリ構成と主要モジュール一覧

### Stage 1: 3D Detector & Backward Motion (3D ResEncM nnU-Net)
- `custom_trainers/nnUNetTrainerBiohubColabM2000.py`: 最高解像度デコーダ段を省略した Half-XY 出力 + 3D 逆方向 Motion 回帰ヘッドを持つカスタム nnU-Net Trainer 定義 (`nnUNetTrainerBiohubTemporalDilatedBackwardMotionCurriculumXYHalfIter250Long1500`)。
- `runtime/capture.py` / `runtime/capture_ensemble.py`: シングルモデル (`pretrained` / `scratch`) または 2モデルアンサンブルによる候補ノード・候補エッジの抽出ランナー。
- `runtime/legacy/biohub_motion_submission.py`: 4チャネル入力 `(I_{t-d}, I_t, I_{t+d}, d/2)` のフルボリューム推論、平面 D4 (8視点) TTA、3D 局所最大値検出、物理距離 Greedy NMS (`2.0 µm`)。
- `runtime/legacy/biohub_planar_d4_tta.py`: XY 平面の 8視点二面体群 ($D_4$) 変換と逆変換。
- `runtime/legacy/two_model_shared_forward.py`: 2つの 3D ResEncM ネットワークの共有入力 D4 TTA フォワード。
- `runtime/legacy/biohub_motion_ilp_fold.py`: ガウス重み付き近傍サンプリングによる逆方向 Motion 投影と候補エッジ生成 (`add_candidate_edges`)、および `DivisionPairILPSolver` の基底定義。

### Stage 2: 専用細胞分裂 (Division) パイプライン
- **親ノード分裂画像分類 (`ParentScorer` / 3D CNN + BiGRU)**:
  - `runtime/parent.py`: 出次数2以上の親候補に対する5フレーム (`t-2..t+2`) 14チャネルクロップ生成と 2シード BiGRU 推論。
  - `runtime/legacy/biohub_division_sliding_t3_mil.py`: `SlidingT3Residual` (4ブロック 3D CNN + BiGRU + 補助カウント・オフセットヘッド) のネットワーク定義・学習コード。
  - `runtime/legacy/biohub_division_flow_aligned_diff_classifier.py`: `SparseLocalFlow` によるフレーム間の局所剛体シフト推定。
  - `runtime/legacy/biohub_division_bigru_shift9_ensemble.py`: 生画像・フレーム間差分 7ch + 再構成ガウスヒートマップ 7ch の計14chテンソル構築 (`make_image`)。
- **娘ペア特徴量 & ランキング (`ExtraTrees` & `5-seed LightGBM`)**:
  - `runtime/pairs.py`: 親候補ごとの娘ペア候補列挙と73特徴量算出。
  - `runtime/legacy/biohub_division_physical_pair_cpu.py`: 幾何学的対称性・距離・角度・前後フレーム (`t-1..t+4`) の局所軌跡整合性を含む 73特徴量ビルダー (`PhysicalFeatureBuilder`)。
  - `runtime/portable_forest.py`: NumPy 配列化された ExtraTrees 推論エンジン (`PortableForest`)。
  - `runtime/daughter_image_lgb5_runtime.py`: 娘ペア周辺の局所画像強度・コントラスト特徴18個を追加した計91特徴量による 5-seed LightGBM 再評価 (`rescore`)。
- **分裂イベント統合 MLP (`nn_h.pt`)**:
  - `runtime/preilp_pairs.py`: 事前線形サロゲート (`models/parent_surrogate_all40.json`) と LightGBM Top-3 ペアの和集合による Pre-ILP 候補ペア提案。
  - `runtime/choice_features.py`: 競合する別親候補や継続トラックとの距離マージン特徴量 (7個) の算出。
  - `runtime/choice_net.py` & `runtime/division_new_transfer.py`: 83次元入力 (`73 geom/traj + 7 alt + parent_score + th_logit + forest_prob`) を受ける 2層 SiLU MLP (`nn_h.pt`, SWA重み) による分裂報酬ロジット推論および TH/Fusion ブリッジ。

### Stage 3: 大域的 ILP 最適化 (`CertifiedPairSolver`)
- `runtime/biohub_solver_runtime/flow_solver.py`: Google OR-Tools `SimpleMinCostFlow` による最小費用循環流ソルバー (SCIP の初期解生成)。
- `runtime/solve.py`: 1対1 貪欲森 + 最小費用流の2段階 Warmstart、出次数2親ノードの娘ペア報酬折りたたみ (`BIOHUB_PAIR_FOLD_MODE='fold'`)、および SCIP (`gap=0`) による大域的厳密最適化 (`CertifiedPairSolver`)。
- `runtime/solve_fusion.py` & `runtime/solve_choice.py` & `runtime/terms.py`: 親分裂コスト・娘ペア報酬・MLP 学習済みペア報酬ロジットを ILP 目的関数へ統合するソルバーエントリポイント。
- `runtime/choice_worker.py`: 動画ごとの Stage 2〜Stage 4 実行・動的スケジューリング・SCIP `SEPARATION_OFF` (ノード数 $\le 30,000$) と標準 SCIP フォールバックを制御するメインワーカー。

### Stage 4: グラフ後処理 & 提出 CSV 変換
- `runtime/final_post.py`: 最終グラフ後処理のエントリポイント。
- `runtime/postbest.py`: 1フレーム欠損のギャップ結合 (`<= 9.0 µm`)、短トラック除去 (`min_len=9`、分裂保護・高信頼8ノード保持)、移動上限 `2.0 µm` 付き局所線形軌道平滑化 (`window=4`, `weight=0.6`)。
- `runtime/boundary_mean1_post.py`: 動画開始・終了フレームに接する 3〜8 ノード短トラックのうち、周囲 `7.0 µm` 以内の確定トラックに対する相対検出スコア比が $\ge 1.0$ の成分を復元。
- `runtime/first_projection.py`: 隣接する3Dフレームが完全同一画像 (フリーズフレーム) の場合に座標を投影する補正処理。
- `tools/biohub_geff_to_submission.py`: 各動画の `.geff` グラフからトポロジー制約 (入次数 $\le 1$, 出次数 $\le 2$, 境界内座標) を検証しつつ最終 `submission.csv` へ変換。
