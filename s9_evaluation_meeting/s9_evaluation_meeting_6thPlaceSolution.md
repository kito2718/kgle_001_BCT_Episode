**6th Place: Detect, Link, Solve, Repeat**  
第6位: 検出、リンク、求解、反復

**One network that detects and links cells, a global ILP, and teacher → student rounds on 3 % annotation**  
細胞の検出とリンクを同時に行う1つのネットワーク、大域的整数線形計画法(ILP)、そして3%のアノテーションに基づく教師-生徒の反復学習

**[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) Solution Writeup · 6th place · Sep 30, 2026**  
[Biohub - Cell Tracking During Development] 解法ライトアップ · 第6位 · 2026年9月30日

**Biohub Cell Tracking During Development: 6th place solution**  
Biohub 発生期細胞追跡コンペティション: 第6位解法

**Private 0.953 (6th place), public 0.961.**  
プライベート 0.953(第6位)、パブリック 0.961。

**A second submission with the same pipeline and best-validation checkpoints scored private 0.955.**  
同一パイプラインで検証スコア最良のチェックポイントを用いた2つ目の提出は、プライベート 0.955を記録しました。

**Code (GitHub)**  
コード(GitHub)  
https://github.com/jamal-saeedi/biohub_cell_tracking_kaggle

**Models (Kaggle Models, MIT)**  
モデル(Kaggle Models, MIT)  
https://www.kaggle.com/models/jamalsaeedi/biohub-cell-tracking

**Code dataset (package + pinned wheels)**  
コードデータセット(パッケージ + バージョン固定済みwheel)  
https://www.kaggle.com/datasets/jamalsaeedi/biohub-cell-tracking-kaggle

**Inference notebook (Kaggle)**  
推論ノートブック(Kaggle)  
https://www.kaggle.com/code/jamalsaeedi/biohub-cell-tracking-inference

**Competition**  
コンペティション  
https://www.kaggle.com/competitions/biohub-cell-tracking-during-development

**Predicted tracks on public test movie 44b6_0113de3b (z projection, every second frame).**  
パブリックテスト動画 44b6_0113de3b に対する予測軌跡(z射影、2フレームごと)。

**Each colour is one track; a red ring marks a division.**  
各色は1つの軌跡を表し、赤色のリングは分裂を示しています。

**Two observations shaped the design.**  
2つの観察結果がこの設計を決定づけました。

**First, only about 3 % of the cells are annotated, so most of the image is neither a known cell nor known background, and a model trained on the annotation alone is never told when it links a cell to the wrong, unannotated neighbour.**  
第1に、細胞の約3%しかアノテーションされていないため、画像の大部分は既知の細胞でも既知の背景でもなく、アノテーションのみで学習したモデルは未アノテーションの誤った近傍細胞とリンクさせてもペナルティを受けません。

**Second, the hard part of linking is telling apart neighbours that look alike, and a linker that sees only coordinates cannot do that.**  
第2に、リンク処理における難所は見た目が酷似した近傍同士を見分ける点にあり、座標しか見ないリンカーにはそれが不可能です。

**So:**  
したがって、以下の構成としました:

* **One network detects and links.**  
  1つのネットワークが検出とリンクを担います。
* **It reads three frames at native resolution, and its linker scores candidate links from features sampled out of the same image features that found the cells;**  
  ネイティブ解像度で3フレームを読み込み、リンカーは細胞を検出したのと同じ画像特徴量からサンプリングされた特徴を用いて候補リンクをスコアリングします。
* **the network outputs probabilities a solver can use directly, including an explicit "new cell" class, so a global ILP needs no hand-tuned repair rules;**  
  ネットワークは明示的な「新規細胞」クラスを含むソルバーが直接利用可能な確率を出力するため、大域的ILP(整数線形計画法)に人手で調整した修復ルールは不要です。
* **the whole pipeline teaches the next model.**  
  パイプライン全体が次のモデルを指導します。
* **Its tracks on the training movies become pseudo-labels for the 97 % of cells nobody annotated.**  
  訓練動画上で生成された軌跡が、誰にもアノテーションされなかった97%の細胞に対する擬似ラベルとなります。

---

### 1. The task / 1. タスク

**Input. 3D time-lapse movies of developing embryos: 100 frames of 64 × 256 × 256 voxels, anisotropic spacing 1.625 µm in z and 0.406 µm in y and x, with up to about 400 cells per frame.**  
入力: 発生期胚の3次元タイムラプス動画。64 × 256 × 256ボクセルの100フレームで構成され、z方向に1.625 µm、yおよびx方向に0.406 µmの異方性ボクセル間隔を持ち、1フレームあたり最大約400個の細胞が存在します。

**Output. Every cell centre in every frame, and the links from each cell at t to its successor(s) at t + 1.**  
出力: すべてのフレームにおける全細胞の中心、および各フレームtの細胞からt + 1における後続細胞へのリンク。

**A cell with two successors divided.**  
後続が2つある細胞は分裂したことを意味します。

**This is the Cell Tracking Challenge setting [2], run as a Kaggle competition [1].**  
これはCell Tracking Challenge [2]の設定であり、Kaggleコンペティション [1]として実施されました。

**Metric. Per movie, an edge Jaccard index with a penalty on the node count, then a weighted mean over movies, plus a division term:**  
評価指標: 動画ごとにノード数ペナルティを伴うエッジJaccard係数を計算し、各動画の加重平均を取った上で、さらに分裂に関する項を加算します。

**$N_{est}$ is the organisers' estimate of the number of cells in the movie.**  
$N_{est}$ は主催者が推定した動画内の細胞数です。

**The annotation is extremely sparse.**  
アノテーションは極めて疎(スパース)です。

**The 199 training movies carry 133k annotated cells, about 3 % of the cells present, as whole lineages, and only 151 divisions.**  
199本の訓練動画には、存在する細胞の約3%にあたる13万3千個の細胞が完全な系統としてアノテーションされており、分裂はわずか151回しか含まれていません。

**One frame of a public test movie: the cells the final pipeline found (cyan) and the annotated cells (red).**  
パブリックテスト動画の1フレーム: 最終パイプラインが検出した細胞(シアン)とアノテーションされた細胞(赤)。

**Two consequences shaped everything:**  
このことから生じる2つの帰結が、全体の設計を決定づけました:

* **Most of the image is neither a known positive nor a known negative, so ordinary dense detection targets would teach the detector to suppress real cells.**  
  画像の大部分は既知の正例でも既知の負例でもないため、通常の密な検出ターゲットを与えると検出器は実在する細胞を抑制するように学習してしまいます。
* **Most linking errors are identity swaps with unannotated neighbours, which ground truth alone never penalises.**  
  リンクエラーの大部分は未アノテーションの近傍細胞との入れ替わりであり、正解ラベルのみではこれにペナルティを与えることができません。

---

### 2. The pipeline at a glance / 2. パイプラインの概要

* **Six networks of two architectures.**  
  2種類のアーキテクチャによる計6つのネットワーク。
* **Each detects cells and scores their links to the previous frame, in one model.**  
  各ネットワークが、1つのモデル内で細胞の検出と前フレームへのリンクのスコアリングを同時に行います。
* **One shared cell set.**  
  1つの共有細胞セット。
* **The six centre maps are averaged; each network then scores the same candidate links with its own head and its own gradient-boosted re-scorer; the six link distributions are averaged.**  
  6つの中心ヒートマップを平均化し、各ネットワークが自身のエッジヘッドと勾配ブースティングによる再スコアラーを用いて同じ候補リンクをスコアリングし、6つのリンク確率分布を平均化します。
* **A global integer linear program turns cells and link probabilities into tracks and divisions; a drift-compensated smoother refines positions.**  
  大域的整数線形計画法(ILP)が細胞とリンク確率を軌跡および分裂へと変換し、ドリフト補正付きスムーザーが位置を微調整します。
* **Training went through several teacher → student rounds in which the teacher is the whole pipeline above.**  
  学習は、上記のパイプライン全体を教師とする数ラウンドの教師-生徒(Teacher → Student)学習を経て行われました。

---

### 3. Models / 3. モデル

#### 3.1 Common design / 3.1 共通設計

**One frame through the first stages: input, centre heatmap, detected cells and the candidate links to the next frame coloured by the predicted parent probability.**  
初期ステージを通過する1フレーム: 入力、中心ヒートマップ、検出された細胞、および予測された親確率で色分けされた次フレームへの候補リンク。

**This frame pair spans a stage jump, so most links are long and parallel.**  
このフレームペアはステージ移動(顕微鏡ステージの急なズレ)を跨いでいるため、ほとんどのリンクが長く平行になっています。

**Both architectures read a 3-frame window (t − 1, t, t + 1) at native resolution: no resampling, no cropping at inference.**  
どちらのアーキテクチャもネイティブ解像度で3フレームのウィンドウ(t − 1, t, t + 1)を読み込みます。推論時のリサンプリングやクロップは行いません。

**Each frame is min–max scaled and z-scored on its own.**  
各フレームは個別にMin-MaxスケーリングおよびZスコア正規化されます。

**A network has two parts: a detector (a 3D U-Net-like [3, 4] encoder–decoder with temporal fusion) and an association head that links the detected cells of two consecutive frames.**  
ネットワークは2つの部分で構成されます。検出器(時間的融合機構を持つ3D U-Net類似[3, 4]のエンコーダ-デコーダ)と、連続する2フレームの検出細胞同士をリンクする関連付けヘッドです。

* **Encoder (per frame).**  
  エンコーダ(フレーム単位)。
* **A 3D convolutional stack (Conv3d → GroupNorm [5] → GELU [6]) that handles the 4:1 anisotropy in two steps: two lateral-only reductions reach a near-isotropic 1.625 µm grid, then a 3D reduction reaches a coarse 3.25 µm grid.**  
  4:1の異方性を2段階で処理する3D畳み込みスタック(Conv3d → GroupNorm [5] → GELU [6])であり、まず水平方向のみの2回の解像度削減によってほぼ等方的な1.625 µmグリッドに到達し、その後の3D削減によって粗い3.25 µmグリッドに到達します。
* **Frames are encoded independently, so at inference each frame is encoded once and reused by the three windows that contain it.**  
  各フレームは独立してエンコードされるため、推論時は各フレームを1度だけエンコードし、それを含む3つのウィンドウで再利用されます。
* **Temporal fusion.**  
  時間的融合。
* **On the coarse grid each voxel of frame t predicts a bounded displacement into each neighbouring frame, samples the neighbour's features there and attention-weights them against its own.**  
  粗いグリッド上において、フレームtの各ボクセルは各隣接フレームへの有界な変位を予測し、その位置における隣接フレームの特徴をサンプリングして自身の特徴とのアテンション重み付けを行います。
* **This aligns moving cells before they are compared.**  
  これにより、移動する細胞が比較される前に位置合わせされます。
* **At initialisation the displacements are zero and the attention is uniform.**  
  初期化時、変位はゼロでありアテンションは一様です。
* **Detection.**  
  検出。
* **The decoder returns to a 64 × 128 × 128 grid with skip connections and predicts a centre heatmap (cells = non-maximum-suppressed peaks above 0.5) and a sub-voxel offset per cell.**  
  デコーダはスキップ接続によって64 × 128 × 128グリッドへと復元し、中心ヒートマップ(細胞 = 0.5以上の非最大値抑制ピーク)と細胞ごとのサブボクセルオフセットを予測します。
* **The two heads run in float32.**  
  これら2つのヘッドはfloat32で動作します。
* **Association head (tracking).**  
  関連付けヘッド(追跡)。
* **For each pair of consecutive frames: each cell descriptor (features sampled at the cell's continuous centre) predicts a velocity and a per-axis uncertainty;**  
  連続する各フレームペアに対して: 各細胞記述子(細胞の連続中心座標でサンプリングされた特徴)が、速度と軸ごとの不確実性を予測します。
* **each candidate link gets the displacement and the uncertainty-normalised residual displacement − velocity as geometric features;**  
  各候補リンクは、幾何学的特徴として変位、および不確実性で正規化された残差(変位 − 速度)を受け取ります。
* **several layers of bidirectional sparse graph attention [7, 8] run over the candidate links, so cells at t and t + 1 update each other;**  
  候補リンク上で双方向スパースグラフアテンション[7, 8]が複数層実行され、フレームtとt + 1の細胞が相互に特徴を更新し合います。
* **the head outputs, per cell, a softmax over its candidate parents plus a "new cell" class, a division score per parent, a score per daughter pair, and the velocity.**  
  このヘッドは細胞ごとに、候補親細胞群に「新規細胞」クラスを加えたソフトマックス確率、親ごとの分裂スコア、娘細胞ペアごとのスコア、および速度を出力します。

**The explicit "new cell" probability is well calibrated, and becomes the solver's appearance cost (§6.5).**  
明示的な「新規細胞」確率は適切に較正されており、ソルバーにおける出現コストとなります(§6.5)。

**Scoring links with attention over cell descriptors is related to Trackastra [9]; ours is sparse, runs on a k-NN candidate graph, and is trained jointly with the detector.**  
細胞記述子に対するアテンションでリンクをスコアリングする手法はTrackastra [9]と関連していますが、本手法はスパースであり、k-NN候補グラフ上で動作し、検出器とエンドツーエンドで共同学習されます。

---

#### 3.2 The two architectures / 3.2 2つのアーキテクチャ

| 項目 | IsotropicLineageNet | MultiScaleLineageNet |
| :--- | :--- | :--- |
| **Temporal fusion** / 時間的融合 | **one learned sample per neighbour frame, coarse grid** / 隣接フレームごとに1つの学習されたサンプリング点、粗グリッド | **fusion at two scales (coarse + isotropic), several learned samples per neighbour and a per-voxel gate; starts as the identity (zero-initialised projection)** / 2つのスケール(粗 + 等方)での融合、隣接フレームごとに複数の学習されたサンプリング点およびボクセルごとのゲート機構。恒等写像(ゼロ初期化射影)から開始 |
| **Detection vs linking features** / 検出用とリンク用の特徴量 | **shared** / 共有 | **separate residual task adapters** / 個別の残差タスクアダプタ |
| **Cell descriptor** / 細胞記述子 | **centre sample + local mean** / 中心サンプル + 局所平均 | **+ attention-pooled samples at learned offsets (within ± 3 µm per axis), as in deformable sampling [10]** / + 変形可能サンプリング[10]のように学習されたオフセット(各軸±3 µm以内)でのアテンションプーリングサンプル |
| **Association head** / 関連付けヘッド | **attention blocks** / アテンションブロック | **+ in-graph motion refinement: a soft assignment to candidate successors updates velocity and uncertainty midway** / + グラフ内運動微調整: 候補後続細胞へのソフト割り当てが途中で速度と不確実性を更新 |
| **Linker training input** / リンカー学習入力 | **annotated positions with 0.5 µm jitter** / 0.5 µmのジッターを加えたアノテーション座標 | **75 % of matched annotated cells moved onto the detector's own detections** / マッチしたアノテーション細胞の75%を検出器自身の検出位置へシフト |
| **Parameters** / パラメータ数 | **2.5–4.8 M** / 2.5〜4.8M | **5.7 M** / 5.7M |

---

#### 3.3 The six models of the ensemble / 3.3 アンサンブルを構成する6モデル

| モデル | アーキテクチャ | チャンネル / リンカー | パラメータ数 | 学習 | TTAビュー数 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** (R3-ft24-noisy-lc-s1) | **Isotropic** | 48 / 192 × 3 | 2.5 M | **third teacher → student round** / 第3教師-生徒ラウンド | 4 |
| **B** (B3-v11-ft32-noisy-lc-s31) | **Isotropic, wide** | 64 / 256 × 3 | 4.4 M | **fine-tuned twice on ensemble labels** / アンサンブルラベル上で2回ファインチューニング | 4 |
| **C** (D2-v11-ft32-noisy-lc-s33) | **Isotropic, wide + deep** | 64 / 256 × 4 | 4.8 M | **trained on v10 labels, fine-tuned on v11** / v10ラベルで学習、v11でファインチューニング | 4 |
| **D** (EX-ms-lc-s5) | **MultiScale** | 64 / 256 × 4 | 5.7 M | **from scratch on ensemble labels** / アンサンブルラベル上でスクラッチ学習 | 3 |
| **E** (FX-ms-lc-s6) | **MultiScale** | 64 / 256 × 4 | 5.7 M | **from scratch on ensemble labels** / アンサンブルラベル上でスクラッチ学習 | 2 |
| **F** (GX-ms-lc-s7) | **MultiScale** | 64 / 256 × 4 | 5.7 M | **from scratch on ensemble labels** / アンサンブルラベル上でスクラッチ学習 | 2 |

**Every model has its own random, movie-disjoint train/validation split, so the members make different mistakes.**  
すべてのモデルが動画単位で重複しない個別のランダム訓練/検証分割を持っているため、各メンバーはそれぞれ異なる誤りを犯します。

**The full per-model table is in §10.**  
モデルごとの詳細な一覧表は§10に記載しています。

---

#### 3.4 Parameters per component / 3.4 コンポーネントごとのパラメータ数

| コンポーネント | A (Iso, c 48, h 192, L 3) | B (Iso, c 64, h 256, L 3) | C (Iso, c 64, h 256, L 4) | D–F (MultiScale) |
| :--- | :--- | :--- | :--- | :--- |
| **encoder (stem → coarse)** / エンコーダ(語幹 → 粗) | 0.78 M | 1.39 M | 1.39 M | 1.61 M |
| **temporal fusion** / 時間的融合 | 0.51 M | 0.91 M | 0.91 M | 1.13 M (粗 0.90 + 等方 0.23) |
| **decoder, adapters, centre / offset heads** / デコーダ、アダプタ、中心/オフセットヘッド | 0.28 M | 0.50 M | 0.50 M | 0.72 M |
| **association head** / 関連付けヘッド | 0.90 M | 1.59 M | 1.99 M | 2.22 M |
| **of which the daughter-pair head (training only)** / うち娘細胞ペアヘッド(学習時のみ) | 0.11 M | 0.20 M | 0.20 M | 0.20 M |
| **total** / 合計 | 2.47 M | 4.38 M | 4.78 M | 5.68 M |

**c = feature channels, h = association width, L = graph-attention blocks.**  
c = 特徴量チャンネル数、h = 関連付け層の幅、L = グラフアテンションブロック数。

---

### 4. Training on 3 % annotation / 4. 3%アノテーションでの学習

#### 4.1 Three-state detection targets / 4.1 3状態の検出ターゲット

**A training crop: the annotated cell gets a Gaussian target (middle).**  
学習用クロップ: アノテーションされた細胞にはガウスターゲットが付与されます(中央)。

**Only the red (positive) and green (verified background) voxels carry loss; the grey region, where the unannotated cells are, is ignored.**  
赤(正例)と緑(検証済み背景)のボクセルのみに損失が発生し、未アノテーション細胞が存在する灰色の領域は無視されます。

| 状態 | 定義 | 損失重み |
| :--- | :--- | :--- |
| **positive** / 正例 | **within 3 µm of an annotated centre; Gaussian target (σ 1.5 µm), 1.0 at the nearest grid cell** / アノテーション中心から3 µm以内。ガウスターゲット(σ 1.5 µm)、最近傍グリッドセルで1.0 | 1.0 (full) |
| **verified background** / 検証済み背景 | **darker than the frame median and more than 6 µm from every annotation** / フレーム中央値より暗く、すべてのアノテーションから6 µm以上離れている | 1.0 (full) |
| **unknown** / 未知 | **everything else** / それ以外のすべて | 0 |

**A count prior supplies what the unknown region lacks: the summed heatmap mass of each crop-frame is pulled towards the organisers' per-movie cell-count estimate, scaled to the crop.**  
未知領域に欠落している情報を細胞数事前分布が補填します。各クロップフレームのヒートマップ値の総和は、クロップサイズに合わせてスケールされた主催者発表の動画別推定細胞数へと引き寄せられます。

**Without it the detector over-fires in the unknown region.**  
これがないと、検出器は未知領域で過剰検出を起こしてしまいます。

---

#### 4.2 Linking samples / 4.2 リンクサンプル

**The linker trains on k-NN candidate graphs (k = 4 both ways, ≤ 20 µm), the same graphs as at inference, plus any annotated parent link the k-NN graph missed.**  
リンカーは推論時と同一のk-NN候補グラフ(双方向k = 4、20 µm以下)に、k-NNグラフが取りこぼしたアノテーション済み親リンクを加えたグラフ上で学習します。

**Distractors: bright intensity peaks without an annotation (up to 256 per frame; 64 in the last fine-tunes, B and C) are added as nodes.**  
ディストラクター(妨害要素): アノテーションのない高輝度ピーク(1フレームあたり最大256個、最終ファインチューンのBおよびCでは64個)がノードとして追加されます。

**As a source a distractor is a verified wrong parent; as a target its parent is unknown.**  
始点ノードとしてのディストラクターは検証済みの「誤った親」であり、終点ノードとしてはその親が「未知」として扱われます。

**15 % of annotated parents are removed from the source set and labelled "new cell", which teaches the explicit null class.**  
アノテーションされた親の15%を始点集合から除外して「新規細胞」とラベル付けし、明示的なnullクラスを学習させます。

**Losses are computed per frame pair, unbatched.**  
損失はバッチ化せず、フレームペアごとに計算されます。

---

#### 4.3 Losses / 4.3 損失関数

**The centre and parent losses are computed separately on ground truth and on pseudo-labels (§5) and combined as $L = L_{GT} + 0.5 \cdot L_{pseudo}$, so the sparse ground truth is never drowned out.**  
中心損失と親リンク損失は正解データと擬似ラベル(§5)で個別に計算され、$L = L_{GT} + 0.5 \cdot L_{pseudo}$ として結合されるため、疎な正解データがかき消されることはありません。

**The sub-voxel offset, division, daughter-pair and velocity terms use ground truth only; the count prior uses the organisers' cell-count estimate.**  
サブボクセルオフセット、分裂、娘細胞ペア、速度の各項は正解データのみを使用し、細胞数事前分布には主催者の推定細胞数が使用されます。

| 損失項 | 重み |
| :--- | :--- |
| **penalty-reduced focal loss on centres [11, 12]** / 中心に対するペナルティ低減型Focal Loss [11, 12] | 1.0 |
| **sub-voxel offset (smooth L1, annotated cells only)** / サブボクセルオフセット(Smooth L1、アノテーション細胞のみ) | 1.0 |
| **parent cross-entropy over candidates + "new cell"** / 候補 + 「新規細胞」に対する親クロスエントロピー | 1.0 |
| **division BCE (positive weight 20, ground truth only)** / 分裂BCE(正例重み20、正解データのみ) | 0.5 |
| **daughter-pair BCE** / 娘細胞ペアBCE | 0.25 |
| **velocity Gaussian NLL (ground truth only)** / 速度ガウス負の対数尤度(正解データのみ) | 0.1 |
| **count prior** / 細胞数事前分布 | 1.0 |

**The four association losses are off for the first 300 optimizer steps and ramp in over the next 300 (from-scratch runs), so the linker never trains on features that cannot localise anything yet.**  
4つの関連付け損失は最初の300オプティマイザステップではオフにされ、その後の300ステップをかけて徐々に導入されるため(スクラッチ学習時)、位置特定すらできていない特徴量でリンカーが学習することはありません。

---

#### 4.4 Samples and augmentation / 4.4 サンプルとデータ拡張

**Samples: full-frame 3-frame crops (64 × 256 × 256).**  
サンプル: フルフレームの3フレームクロップ(64 × 256 × 256)。

**15 % are placed uniformly; of the others, 25 % are centred on a division and the rest on an annotated cell.**  
15%は一様ランダムに配置され、残りのうち25%は分裂点を中心に、その他はアノテーション細胞を中心として切り出されます。

**Augmentation: lateral D4 only (z is never flipped: light attenuation makes it directional);**  
データ拡張: 水平方向のD4対称変換(回転・反転)のみ(光の減衰により方向性があるため、z軸の反転は一切行わない)。

* **gamma and Gaussian noise;**  
  ガンマ変換およびガウスノイズ。
* **synthetic stage drift: frames shifted laterally as a random walk, up to 6 µm per step (most models);**  
  人工的なステージドリフト: ランダムウォークとしてフレームを水平方向にステップあたり最大6 µmシフト(大半のモデルで適用)。
* **noisy-student noise [13]: shot noise, a mean-preserving depth gain, and for the fine-tuned isotropic models channel dropout 0.2 and weight decay 0.01;**  
  Noisy Studentノイズ[13]: ショットノイズ、平均値を維持する深度ゲイン、およびファインチューニングされたIsotropicモデルにはチャンネルドロップアウト0.2とウェイトディケイ0.01。
* **low-contrast haze: each frame blended with a 17 µm box blur of itself at a random contrast, mimicking hazy, deep movies (+0.005 on validation).**  
  低コントラスト霞(ヘイズ): 各フレームを自身の17 µmボックスブラー画像とランダムなコントラストでブレンドし、霞んだ深部の動画を模倣(検証スコア+0.005)。

---

#### 4.5 Optimisation / 4.5 最適化

**AdamW [14], linear warm-up and cosine decay [15], gradient clipping at 1, and an EMA of the weights [16] (decay 0.999) that is the model used at inference.**  
AdamW [14]、線形ウォームアップおよびコサイン減衰[15]、勾配クリッピング(閾値1)、および重みのEMA(指数移動平均)[16](減衰率0.999)を採用し、このEMAモデルを推論時に使用します。

**bf16 mixed precision [17] on RTX 3090/4090.**  
RTX 3090/4090上でbf16混合精度学習[17]を実施。

**From scratch: learning rate 1e-4, 32 epochs of 2,048 crops (48 for DX; F stopped after 24).**  
スクラッチ学習時: 学習率 1e-4、2,048クロップからなる32エポック(DXは48エポック、Fは24エポック後に停止)。

**Fine-tuning: 3e-5, 24–48 epochs.**  
ファインチューニング時: 学習率 3e-5、24〜48エポック。

**Checkpoints are selected on validation loss (B2 and DX: validation parent loss, with early stopping).**  
チェックポイントは検証損失に基づいて選択されます(B2とDXはEarly Stoppingを伴う検証親損失で選択)。

---

### 5. Pseudo-labels and teacher → student rounds / 5. 擬似ラベルと教師-生徒ラウンド

**With 3 % of cells annotated, a model trained on ground truth alone never learns to separate touching cells and is never penalised for linking to an unannotated neighbour.**  
細胞の3%しかアノテーションされていない状況では、正解データのみで学習したモデルは接触する細胞の分離を学習できず、未アノテーションの近傍細胞へリンクさせてもペナルティを受けません。

**Offline noisy-student self-training [13, 18] fixed both.**  
オフラインのNoisy Student自己学習[13, 18]がこれら双方の問題を解決しました。

**The teacher is the whole pipeline, not one network.**  
ここで教師となるのは単一のネットワークではなく、パイプライン全体です。

---

#### 5.1 How a label set is made (biohub-pseudo-labels) / 5.1 ラベルセットの作成方法

* **Run the teacher pipeline on every training movie: decode, link, re-score and solve the ILP exactly as at inference.**  
  推論時と全く同じ手順で、すべての訓練動画に対して教師パイプライン(デコード、リンク、再スコアリング、ILP求解)を実行します。
* **Topology from the ILP solution.**  
  トポロジー(接続関係)はILPの解から取得します。
* **Its links are far more precise than the network's per-cell argmax (0.949 vs 0.885 on annotated lineages).**  
  そのリンク精度は、ネットワークの細胞ごとargmax出力よりも遥かに高精度です(アノテーション済み系統で0.949 vs 0.885)。
* **Positions from the raw detections, not the smoothed tracks: the smoother helps the metric but moves points away from the true centres.**  
  座標は平滑化された軌跡ではなく、生の検出結果から取得します。平滑化は評価指標を改善するものの、点を真の中心から遠ざけてしまうためです。
* **Align positions per imaging cohort.**  
  撮像コホートごとに座標の位置合わせを行います。
* **The teacher's cells sit at a small systematic offset from where the annotators put the same cells (up to 0.5 µm in z, cohort-dependent).**  
  教師モデルの細胞座標には、アノテータが同一の細胞を配置した位置との間にわずかな体系的オフセットが存在します(コホート依存でz方向に最大0.5 µm)。
* **It is measured on validation movies and subtracted.**  
  このオフセットを検証動画上で計測し、差し引きます。
* **Keep the teacher's probabilities (centre probability per cell, parent probability per link) as loss weights.**  
  教師の確率(細胞ごとの中心確率、リンクごとの親確率)を損失の重みとして保持します。

---

#### 5.2 How labels are merged in training / 5.2 学習時におけるラベルの統合方法

* **A pseudo cell within 4 µm of an annotated cell in the same frame is that cell.**  
  同一フレーム内でアノテーション細胞から4 µm以内にある擬似細胞は、そのアノテーション細胞と同一視されます。
* **Pseudo links that contradict an annotated parent or child are dropped.**  
  アノテーションされた親や子と矛盾する擬似リンクは除外されます。
* **Pseudo cells and links below probability 0.5 are dropped.**  
  確率0.5未満の擬似細胞および擬似リンクは除外されます。
* **Divisions, velocities and sub-voxel offsets are supervised by ground truth only.**  
  分裂、速度、およびサブボクセルオフセットは正解データのみによって教師あり学習されます。
* **The 4 public test movies are never labelled.**  
  4本のパブリックテスト動画には一切ラベル付けを行いません。
* **The final label set holds about 5.0 M cells and 4.9 M links over 195 movies, about 38 × the annotated cells.**  
  最終的なラベルセットは195本の動画にわたり約500万個の細胞と490万本のリンクを含み、アノテーション細胞数の約38倍に達します。

---

#### 5.3 The lineage of the final models / 5.3 最終モデルの系譜

**The first round gave the largest single gain: +0.008 on validation over the ground-truth-only model.**  
最初のラウンドで最大の単一改善が得られ、正解ラベルのみのモデルに対して検証スコアが+0.008向上しました。

**Plain self-distillation then flattened; later gains came from new architectures, new splits and ensemble teachers.**  
単純な自己蒸留の効果はその後横ばいになりましたが、その後の利得は新規アーキテクチャ、新しい分割、およびアンサンブル教師によってもたらされました。

**The two ensemble teachers (v10, v11) are 3-model pipelines with member-own re-scorers; they scored 0.958 and 0.959 on the public leaderboard when submitted.**  
2つのアンサンブル教師(v10, v11)は各メンバー専用の再スコアラーを備えた3モデルのパイプラインであり、提出時のパブリックリーダーボードで0.958および0.959を記録しました。

**Every step has a recipe in recipes/ and tools/reproduce_training.sh runs the lineage in order.**  
すべての工程は recipes/ 内にレシピとして定義されており、tools/reproduce_training.sh によってこの系譜が順次実行されます。

---

### 6. Inference, step by step / 6. 推論の各ステップ

#### 6.1 Decoding and test-time augmentation / 6.1 デコードとテスト時データ拡張(TTA)

**Each frame is decoded from the 3-frame window centred on it.**  
各フレームは、それを中心とする3フレームのウィンドウからデコードされます。

**Each model runs 2–4 lateral views (90° rotations).**  
各モデルは2〜4通りの水平ビュー(90度刻みの回転)を実行します。

**Outputs, including the offset vectors, are mapped back and averaged in logit space.**  
オフセットベクトルを含む出力は逆変換され、ロジット空間で平均化されます。

**Descriptors are sampled from the view-averaged features, so linking benefits from TTA too.**  
記述子はビュー平均化された特徴量からサンプリングされるため、リンク処理もTTAの恩恵を受けます。

**Going from 1 to 4 views on an early single model was worth +0.05 score.**  
初期の単一モデルにおいて、ビュー数を1から4に増やすことでスコアが+0.05向上しました。

**The per-frame encoder output is cached (in fp16 under fp16 autocast, as on the T4) and reused across windows and views (1.4–1.9 × faster decoding).**  
フレームごとのエンコーダ出力はキャッシュされ(T4と同様にfp16 autocast下でfp16として保持)、ウィンドウ間およびビュー間で再利用されます(デコードが1.4〜1.9倍高速化)。

---

#### 6.2 One set of cells, six link opinions / 6.2 1つの細胞セットと6つのリンク予測

**Cells: the six models' centre logits and offsets are averaged and peaks are extracted once.**  
細胞: 6つのモデルの中心ロジットとオフセットを平均化し、ピーク抽出を一度だけ行います。

**Links: each model samples its own descriptors at the shared cells, scores the same candidate links with its own head, and its own re-scorer re-ranks them.**  
リンク: 各モデルが共有された細胞座標において自身の記述子をサンプリングし、自身のヘッドで同じ候補リンクをスコアリングした上で、自身の再スコアラーがそれらを再ランク付けします。

**The six parent distributions (including "new cell") are averaged in probability space.**  
これら6つの親分布(「新規細胞」を含む)が確率空間で平均化されます。

---

#### 6.3 Candidate links / 6.3 候補リンク

**The union of the 4 nearest cells at t for each cell at t + 1 and the 4 nearest at t + 1 for each cell at t, within 20 µm.**  
フレームt + 1の各細胞から見たtの最近傍4細胞と、フレームtの各細胞から見たt + 1の最近傍4細胞の和集合(距離20 µm以内)。

**The true parent is a candidate for about 99.4 % of annotated links.**  
アノテーションされたリンクの約99.4%において、真の親が候補に含まれます。

**Candidates are not pruned by probability: the softmax already has a "new cell" option.**  
確率による候補の枝刈りは行いません。ソフトマックスにすでに「新規細胞」の選択肢が含まれているためです。

---

#### 6.4 Member-own edge re-scorers / 6.4 各メンバー専用のエッジ再スコアラー

**Identity swaps between neighbours are the largest error class (51–57 % of edge errors on validation).**  
近傍同士の入れ替わり(同一性スワップ)は最大のエラー要因です(検証時におけるエッジエラーの51〜57%)。

**Each model has a LightGBM [19] re-ranker of each cell's candidate parents (300 trees, 31 leaves, binary objective) with 23 features per candidate link:**  
各モデルは各細胞の親候補に対するLightGBM [19]による再スコアラー(300木、31葉、二値分類目的)を持ち、候補リンクごとに23次元の特徴量を用います:

* **5 dims: the model's log-probability, its rank among the cell's candidates, the margin to the best rival, the number of candidates, the "new cell" log-probability**  
  5次元: モデルの対数確率、細胞の候補内での順位、最良のライバルとの差分、候補数、「新規細胞」の対数確率
* **2 dims: distance, raw and drift-corrected**  
  2次元: 距離(生の値およびドリフト補正後)
* **4 dims: z and lateral displacement, raw and after removing the frame's stage drift**  
  4次元: z方向および水平方向の変位(生の値およびフレームのステージドリフト除去後)
* **2 dims: velocity residual, total and in z**  
  2次元: 速度残差(全体およびz方向)
* **3 dims: centre logit of both cells, division logit of the parent**  
  3次元: 両細胞の中心ロジット、親細胞の分裂ロジット
* **3 dims: competition for the parent: its best score to another cell, how many cells rank it first, its number of candidate children**  
  3次元: 親細胞に対する競合度(他セルに対する最高スコア、その親を第1位とした細胞数、候補子細胞の数)
* **2 dims: local density (cells within 10 µm) around both cells**  
  2次元: 両細胞周辺の局所密度(10 µm以内の細胞数)
* **2 dims: depth (z) of the cell and relative time in the movie**  
  2次元: 細胞の深度(z)および動画内での相対時刻

**The re-ranked distribution keeps the "new cell" probability untouched:**  
再ランク付けされた分布においても、「新規細胞」の確率は変更せずそのまま維持されます。

**Training (biohub-train-rescorer).**  
学習 (biohub-train-rescorer)。

**Each member has its own trees, fitted on that member's own scores on the ensemble's shared cells, over its own held-out validation movies.**  
各メンバーは独自の決定木群を持ち、アンサンブルの共有細胞に対する自身のスコアに基づき、自身がホールドアウトした検証動画上でフィッティングされます。

**A row is a candidate parent of a cell whose annotated cell and annotated parent both match decoded cells within 7 µm; the true parent is the positive.**  
アノテーション細胞とアノテーションされた親の双方が7 µm以内でデコード細胞と一致した細胞の親候補を行データとし、真の親を正例とします。

**During validation the trees are applied out-of-fold (5 folds by movie), so no movie is scored by trees that saw it.**  
検証中、決定木はアウトオブフォールド(動画単位の5分割交差検証)で適用されるため、学習時に参照した決定木でスコアリングされる動画はありません。

**Worth +0.004 to +0.007 per model on validation.**  
モデルあたり検証スコアで+0.004〜+0.007の改善をもたらしました。

**At inference the trees are evaluated from flat arrays with a compiled tree walk [20], so no gradient-boosting library is needed.**  
推論時にはコンパイル済みツリー探索[20]を用いてフラットな配列から決定木が評価されるため、勾配ブースティングライブラリは不要です。

---

#### 6.5 Global ILP / 6.5 大域的整数線形計画法(ILP)

**Cyan dots: cells at t; orange crosses: cells at t + 1.**  
シアン色の点: 時刻tの細胞。オレンジ色の十字: 時刻t + 1の細胞。

**Left: candidate links and their probabilities.**  
左図: 候補リンクとその確率。

**Right: the submitted links after the ILP and smoothing (no division in this window).**  
右図: ILP求解および平滑化後の最終リンク(このウィンドウ内には分裂なし)。

**Each cell has binary variables exists, appears, disappears, divides; each candidate link one variable.**  
各細胞は二値変数(存在、出現、消失、分裂)を持ち、各候補リンクは1つの変数を持ちます。

**This is the conservation-tracking family of models [21, 22].**  
これは保存則追跡(conservation-tracking)モデル群[21, 22]に属する設計です。

**Flow conservation:**  
フロー保存則:
$$\text{appear}_j + \sum_i \text{edge}_{ij} = \text{node}_j \qquad \text{disappear}_i + \sum_j \text{edge}_{ij} = \text{node}_i + \text{div}_i \qquad \text{div}_i \le \text{node}_i$$

| 項 | コスト |
| :--- | :--- |
| **cell** / 細胞 | **− centre logit (break-even at p = 0.5)** / − 中心ロジット(p = 0.5で損益分岐) |
| **appearance** / 出現 | **− log P_T(new cell) + 0.5 (first-frame cells appear for free)** / − log P_T(新規細胞) + 0.5(第1フレームの細胞はコスト0で出現) |
| **disappearance** / 消失 | 12 |
| **division** / 分裂 | **max(0, − division logit)** / max(0, − 分裂ロジット) |
| **link** / リンク | **− log P_T(parent) + 0.2 · drift-corrected distance (µm)** / − log P_T(親) + 0.2 · ドリフト補正後距離(µm) |

**$P_T$ is the re-scored, ensemble-averaged distribution with temperature T = 0.9.**  
$P_T$ は温度 T = 0.9 を適用した、再スコアリング・アンサンブル平均後の分布です。

**Drift-corrected distance.**  
ドリフト補正後の距離。

**The microscope stage drifts between frames (median 1.3 µm, 99th percentile 7.3 µm), and identity swaps concentrate on those frames.**  
フレーム間で顕微鏡ステージがドリフトし(中央値1.3 µm、99パーセンタイル7.3 µm)、細胞の入れ替わりエラーはそのようなフレームに集中します。

**For each pair of frames the global shift is estimated as the median displacement of the confident links (P ≥ 0.7); each link pays for its distance after that shift is removed (+0.009 on validation).**  
フレームペアごとに、高信頼リンク(P ≥ 0.7)の変位の中央値として大域的シフト量を推定し、各リンクはそのシフトを除去した後の距離に対してコストを支払います(検証スコア+0.009)。

**Whole-field displacement between consecutive frames on the four public test movies.**  
4本のパブリックテスト動画における連続フレーム間の視野全体の変位量。

**LP-first solving.**  
線形計画(LP)先行求解。

**Apart from the division rows the constraint matrix is a network matrix, so the LP relaxation is almost integral.**  
分裂の制約行を除けば制約行列はネットワーク行列であるため、LP緩和解はほぼ整数解となります。

**The LP is solved with HiGHS [23], every integral variable is fixed, and a small MILP over the fractional variables and their neighbourhood finishes the job: 38–166 × faster than branch-and-bound with SCIP [24], and within 0.5 % of the LP bound (otherwise SCIP runs).**  
HiGHS [23]でLPを解き、すべての整数変数を固定した上で、非整数変数とその近傍に対して小さな混合整数線形計画法(MILP)を解くことで求解を完了します。SCIP [24]による分枝限定法と比べて38〜166倍高速であり、LP緩和界の0.5%以内の解が得られます(満たさない場合はSCIPを実行)。

**A greedy solver is the last resort.**  
貪欲法ソルバーは最後のフェイルセーフとして用意されています。

**Because every cell and every appearance has a price, the ILP leaves no dangling fragments and no repair heuristics are needed.**  
すべての細胞および出現にコストが設定されているため、ILPの解に宙に浮いた断片が残ることはなく、ヒューリスティックな修復ルールは一切不要です。

---

#### 6.6 Drift-compensated smoothing / 6.6 ドリフト補正付き平滑化

**z is quantised at 1.625 µm and carries most of the localisation error, so positions are smoothed along tracks: for each cell, up to 2 neighbours each way along its track (stopping at divisions), a straight-line fit per axis, and the cell moves to 0.2 · original + 0.8 · fit.**  
z軸は1.625 µm単位で量子化されており位置特定誤差の大半を占めるため、軌跡に沿って座標を平滑化します。各細胞について、軌跡に沿って前後最大2つの近傍(分裂点で停止)を用いて軸ごとに直線フィッティングを行い、細胞座標を「0.2 × 元座標 + 0.8 × フィット座標」へと移動させます。

**Smoothing is done after removing the cumulative stage drift and the drift is added back afterwards (+0.005 over plain smoothing).**  
平滑化は累積ステージドリフトを除去した後に実行され、その後ドリフトが加算されて元に戻されます(通常の平滑化に対して+0.005の改善)。

**Positions are clamped to the volume and rounded.**  
座標は画像ボリューム内にクリップされ、丸め処理が行われます。

---

### 7. Running in 12 hours on two T4s / 7. 2枚のT4 GPUによる12時間以内の実行

* **One worker per GPU, one shared queue.**  
  GPUあたり1つのワーカー、1つの共有キュー。
* **Each worker loads the six models once and claims movies by atomically renaming a per-movie file.**  
  各ワーカーは6つのモデルを一度だけロードし、動画ごとのファイルをアトミックにリネームすることで処理対象の動画を確保します。
* **fp16, not bf16, on the T4 (no native bf16) [17]: 3.5 × faster prediction than emulated bf16.**  
  T4上ではネイティブbf16非対応のため、bf16ではなくfp16を使用[17]: エミュレートされたbf16と比較して3.5倍高速に予測できます。
* **Deadline guard.**  
  デッドラインガード(時間超過防止機構)。
* **Before each movie a worker estimates its cost from its recent history and picks the richest setting that fits: all views, 75 %, 50 %, then one view.**  
  各動画の処理前に、ワーカーは直近の処理履歴から所要時間を推定し、制限時間に収まる最もリッチな設定(全ビュー、75%、50%、単一ビュー)を選択します。
* **Every movie gets a prediction.**  
  すべての動画が確実に予測結果を得られます。
* **A failed movie is retried at the next cheaper level, then re-run with one view, then with the primary model alone, then with a model-free tracker (difference-of-Gaussians detection [25] and drift-corrected nearest-neighbour links), and finally a placeholder.**  
  失敗した動画は一段階軽量な設定でリトライされ、それでも失敗した場合は1ビュー、主要モデル単体、モデル不要のトラッカー(ガウシアンの差分[DoG]検出[25]およびドリフト補正付き最近傍リンク)、そして最終手段としてプレースホルダーの順でフォールバックされます。
* **The 4 public test movies take about 15 minutes; the hidden set ran in about 10 hours.**  
  4本のパブリックテスト動画の処理には約15分かかり、秘匿されたテストセットは約10時間で完走しました。

---

### 8. Validation and results / 8. 検証と結果

* **Local metric: our re-implementation of the official metric, including the per-movie weighting (not part of this repository).**  
  ローカル指標: 動画ごとの重み付けを含めて公式評価指標を独自に再実装したもの(本リポジトリには含まれていません)。
* **The CSV our Kaggle notebook writes for the 4 public test movies scores exactly the same locally.**  
  Kaggleノートブックが4本のパブリックテスト動画に対して出力するCSVは、ローカルでも完全に同一のスコアを記録します。
* **Splits: each model trains on 165 movies and validates on 30 of its own; the 4 public test movies are held out from every model.**  
  分割: 各モデルは165本の動画で学習し、独自の30本で検証を行います。4本のパブリックテスト動画は全モデルから除外されています。
* **Model A's 30 validation movies rank every change, including ensembles.**  
  アンサンブルを含むあらゆる変更の優劣比較には、モデルAの30本の検証動画が用いられます。
* **Tuning: each model or inference change gets a small solver grid (disappearance, appearance bias, distance weight, temperature, smoothing), compared with 2-fold cross-validation over the validation movies and a paired bootstrap over movies.**  
  チューニング: 各モデルや推論処理の変更ごとに、検証動画上での2分割交差検証および動画単位の対応のあるブートストラップ法を用いて、ソルバーの小規模グリッドサーチ(消失コスト、出現バイアス、距離の重み、温度、平滑化)を実施しました。

| 提出 | 検証 (adj. J) | 4つのパブリックテスト動画 | パブリックLB | プライベートLB |
| :--- | :--- | :--- | :--- | :--- |
| **fin (selected)** / 最終選択 | 0.940 | 0.928 | 0.961 | 0.953 |
| **best** / 最良検証モデル | 0.939 | 0.933 | 0.958 | 0.955 |

**The division term is left out of the local numbers: with 151 divisions in the whole corpus it is too noisy to rank changes by.**  
ローカルの数値から分裂の評価項は除外されています。全データ中に151回の分裂しかなく、変更の優劣を評価するにはノイズが大きすぎるためです。

---

### 9. What mattered / 9. 効果のあった取り組み

| 変更点 | 測定された利得 |
| :--- | :--- |
| **test-time augmentation, 1 → 4 views (single model)** / テスト時データ拡張、1 → 4ビュー(単一モデル) | スコア +0.05 |
| **first teacher → student round** / 第1回の教師-生徒ラウンド | 検証 +0.008 |
| **drift-corrected link distance** / ドリフト補正リンク距離 | 検証 +0.009 |
| **member-own edge re-scorer** / 各メンバー専用エッジ再スコアラー | モデルあたり +0.004〜+0.007 |
| **drift-compensated smoothing (over plain smoothing)** / ドリフト補正付き平滑化(通常の平滑化に対して) | +0.005 |
| **low-contrast haze augmentation** / 低コントラスト霞データ拡張 | +0.005 |
| **ensembles of models on different splits and architectures** / 異なる分割とアーキテクチャによるモデルのアンサンブル | パブリック 0.954 → 0.959(1 → 3モデル) |
| **fp16 instead of emulated bf16 on the T4** / T4上でのエミュレートbf16に代わるfp16の採用 | 予測が3.5倍高速化 |

**Key takeaways:**  
主な要点:

* **Let the network output probabilities the solver can use.**  
  ソルバーが直接利用可能な確率をネットワークに出力させること。
* **An explicit "new cell" class gives calibrated appearance and link costs and makes repair heuristics unnecessary.**  
  明示的な「新規細胞」クラスが適切に較正された出現コストとリンクコストを提供し、ヒューリスティックな修復処理を不要にします。
* **Take pseudo-labels from the whole pipeline (ILP topology, raw positions, ground truth authoritative) and regenerate them from the current best pipeline.**  
  パイプライン全体(ILPによるトポロジー、生の検出座標、正解ラベル最優先)から擬似ラベルを抽出し、その時点の最良パイプラインから再生成すること。
* **Diversify splits as well as architectures.**  
  アーキテクチャだけでなく、データ分割も多様化させること。
* **Model the camera, not only the cells: stage drift in the solver, the smoother and the augmentation.**  
  細胞だけでなくカメラ系(装置)もモデル化すること: ソルバー、スムーザー、データ拡張におけるステージドリフトの考慮。
* **Engineer the time budget: fp16, the encoder cache, two-GPU sharding and a graded deadline guard made a six-model ensemble with TTA fit in 12 hours.**  
  実行時間バジェットを徹底的に管理すること: fp16、エンコーダキャッシュ、2-GPUシャーディング、段階的デッドラインガードにより、TTAを伴う6モデルアンサンブルを12時間以内に収めました。

---

### 10. Model summary / 10. モデルのまとめ

**All runs: 3-frame full-frame crops (64 × 256 × 256), 2,048 crops per epoch, effective batch 8, AdamW, EMA 0.999, count prior weight 1.0, pseudo-label weight 0.5, low-contrast haze 0.3 (from R3 on), shot noise 0.2 and depth gain 0.15 (from R2 on).**  
全ラン共通設定: 3フレームのフルフレームクロップ(64 × 256 × 256)、エポックあたり2,048クロップ、実効バッチサイズ8、AdamW、EMA 0.999、細胞数事前分布重み1.0、擬似ラベル重み0.5、低コントラスト霞0.3(R3以降)、ショットノイズ0.2および深度ゲイン0.15(R2以降)。

*(表の注記部分)*  
**\* early stopping on validation parent loss (patience 10).**  
\* 検証親損失に基づくEarly Stopping(忍耐数10)。

**Checkpoints of the two submissions: fin uses the last checkpoint of B–E, best their best-validation-loss checkpoints.**  
2つの提出におけるチェックポイント: finはB〜Eの最終チェックポイントを使用し、bestは検証損失最良のチェックポイントを使用しています。

**A and F are the same in both: F's run stopped after 24 of its 32 scheduled epochs, and its last checkpoint was also its best.**  
AとFは双方で共通です。Fの実行は予定されていた32エポック中24エポックで停止し、最終チェックポイントが最良でもありました。

**Each submission has its own six re-scorers.**  
各提出はそれぞれ固有の6つの再スコアラーを持っています。

**best-e23 / best-e20 are the best checkpoints up to the 24th / 21st epoch of BX / CX, kept as snapshots for the v10 teacher.**  
best-e23 / best-e20 は、BX / CX のそれぞれ第24エポック / 第21エポックまでの最良チェックポイントであり、v10教師モデル用のスナップショットとして保持されたものです。

**Compute: one RTX 3090/4090 per run: 4–10 h per isotropic run, about 17 h per MultiScale run (15 GiB at batch 1).**  
計算資源: 1ランあたりRTX 3090/4090を1基使用。Isotropicモデルは1ランあたり4〜10時間、MultiScaleモデルは約17時間(バッチサイズ1で15 GiB)。

**Inference: about 7 minutes per movie per T4 for the six-model ensemble.**  
推論時間: 6モデルアンサンブルにおいて、T4 1基あたり動画1本につき約7分。

---

### 11. Code / 11. コード

| ステップ | コマンド | コードリンク |
| :--- | :--- | :--- |
| **predict (fin / best, or your own models)** / 予測(fin / best、または独自モデル) | `biohub-predict` | [cli.py](https://github.com/jamal-saeedi/biohub_cell_tracking_kaggle/blob/main/src/biohub_tracking/cli.py), [isotropic/](https://github.com/jamal-saeedi/biohub_cell_tracking_kaggle/tree/main/src/biohub_tracking/isotropic) |
| **train one model from a recipe** / レシピから1モデルを学習 | `biohub-train --recipe recipes/train/<run>.json` | [training/](https://github.com/jamal-saeedi/biohub_cell_tracking_kaggle/tree/main/src/biohub_tracking/training) |
| **pseudo-labels from a teacher ensemble** / 教師アンサンブルからの擬似ラベル生成 | `biohub-pseudo-labels --ensemble recipes/ensembles/<set>-teacher.json` | [labels.py](https://github.com/jamal-saeedi/biohub_cell_tracking_kaggle/blob/main/src/biohub_tracking/labels.py) |
| **member-own re-scorers** / 各メンバー専用の再スコアラー | `biohub-train-rescorer --ensemble recipes/ensembles/<spec>.json --names ...` | [training/rescorer.py](https://github.com/jamal-saeedi/biohub_cell_tracking_kaggle/blob/main/src/biohub_tracking/training/rescorer.py) |
| **the whole training lineage** / 学習系譜全体の実行 | `bash tools/reproduce_training.sh` | [recipes/](https://github.com/jamal-saeedi/biohub_cell_tracking_kaggle/tree/main/recipes) |
| **every stage on synthetic data, CPU** / 合成データを用いた全工程テスト(CPU) | `python tools/smoke_test.py` | [notebooks/training.ipynb](https://github.com/jamal-saeedi/biohub_cell_tracking_kaggle/blob/main/notebooks/training.ipynb) |

**The models are on [Kaggle Models](https://www.kaggle.com/models/jamalsaeedi/biohub-cell-tracking) and the inference notebook on [Kaggle](https://www.kaggle.com/code/jamalsaeedi/biohub-cell-tracking-inference).**  
モデルは[Kaggle Models](https://www.kaggle.com/models/jamalsaeedi/biohub-cell-tracking)で、推論ノートブックは[Kaggle](https://www.kaggle.com/code/jamalsaeedi/biohub-cell-tracking-inference)で公開されています。

---

### References / 参考文献

* **Biohub. Biohub - Cell Tracking During Development. Kaggle competition, 2026.**  
  Biohub. Biohub - Cell Tracking During Development. Kaggle コンペティション, 2026.
* **M. Maška et al. The Cell Tracking Challenge: 10 years of objective benchmarking. Nature Methods 20, 1010–1020, 2023.**  
  M. Maška 他. The Cell Tracking Challenge: 10 years of objective benchmarking. Nature Methods 20, 1010–1020, 2023.
* **O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.**  
  O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.
* **Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, O. Ronneberger. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. MICCAI 2016.**  
  Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, O. Ronneberger. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. MICCAI 2016.
* **Y. Wu, K. He. Group Normalization. ECCV 2018.**  
  Y. Wu, K. He. Group Normalization. ECCV 2018.
* **D. Hendrycks, K. Gimpel. Gaussian Error Linear Units (GELUs). arXiv:1606.08415, 2016.**  
  D. Hendrycks, K. Gimpel. Gaussian Error Linear Units (GELUs). arXiv:1606.08415, 2016.
* **P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio. Graph Attention Networks. ICLR 2018.**  
  P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio. Graph Attention Networks. ICLR 2018.
* **A. Vaswani et al. Attention Is All You Need. NeurIPS 2017.**  
  A. Vaswani 他. Attention Is All You Need. NeurIPS 2017.
* **B. Gallusser, M. Weigert. Trackastra: Transformer-based cell tracking for live-cell microscopy. ECCV 2024. arXiv:2405.15700.**  
  B. Gallusser, M. Weigert. Trackastra: Transformer-based cell tracking for live-cell microscopy. ECCV 2024. arXiv:2405.15700.
* **J. Dai, H. Qi, Y. Xiong, Y. Li, G. Zhang, H. Hu, Y. Wei. Deformable Convolutional Networks. ICCV 2017.**  
  J. Dai, H. Qi, Y. Xiong, Y. Li, G. Zhang, H. Hu, Y. Wei. Deformable Convolutional Networks. ICCV 2017.
* **H. Law, J. Deng. CornerNet: Detecting Objects as Paired Keypoints. ECCV 2018.**  
  H. Law, J. Deng. CornerNet: Detecting Objects as Paired Keypoints. ECCV 2018.
* **X. Zhou, D. Wang, P. Krähenbühl. Objects as Points. arXiv:1904.07850, 2019.**  
  X. Zhou, D. Wang, P. Krähenbühl. Objects as Points. arXiv:1904.07850, 2019.
* **Q. Xie, M.-T. Luong, E. Hovy, Q. V. Le. Self-training with Noisy Student improves ImageNet classification. CVPR 2020.**  
  Q. Xie, M.-T. Luong, E. Hovy, Q. V. Le. Self-training with Noisy Student improves ImageNet classification. CVPR 2020.
* **I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization. ICLR 2019.**  
  I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization. ICLR 2019.
* **I. Loshchilov, F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR 2017.**  
  I. Loshchilov, F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR 2017.
* **B. T. Polyak, A. B. Juditsky. Acceleration of Stochastic Approximation by Averaging. SIAM Journal on Control and Optimization 30(4), 838–855, 1992.**  
  B. T. Polyak, A. B. Juditsky. Acceleration of Stochastic Approximation by Averaging. SIAM Journal on Control and Optimization 30(4), 838–855, 1992.
* **P. Micikevicius et al. Mixed Precision Training. ICLR 2018.**  
  P. Micikevicius 他. Mixed Precision Training. ICLR 2018.
* **D.-H. Lee. Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks. ICML 2013 Workshop on Challenges in Representation Learning.**  
  D.-H. Lee. Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks. ICML 2013 Workshop on Challenges in Representation Learning.
* **G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 2017.**  
  G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 2017.
* **S. K. Lam, A. Pitrou, S. Seibert. Numba: A LLVM-based Python JIT Compiler. LLVM-HPC 2015.**  
  S. K. Lam, A. Pitrou, S. Seibert. Numba: A LLVM-based Python JIT Compiler. LLVM-HPC 2015.
* **B. X. Kausler, M. Schiegg, B. Andres, M. Lindner, U. Köthe, H. Leitte, J. Wittbrodt, L. Hufnagel, F. A. Hamprecht. A Discrete Chain Graph Model for 3d+t Cell Tracking with High Misdetection Robustness. ECCV 2012.**  
  B. X. Kausler, M. Schiegg, B. Andres, M. Lindner, U. Köthe, H. Leitte, J. Wittbrodt, L. Hufnagel, F. A. Hamprecht. A Discrete Chain Graph Model for 3d+t Cell Tracking with High Misdetection Robustness. ECCV 2012.
* **M. Schiegg, P. Hanslovsky, B. X. Kausler, L. Hufnagel, F. A. Hamprecht. Conservation Tracking. ICCV 2013, 2928–2935.**  
  M. Schiegg, P. Hanslovsky, B. X. Kausler, L. Hufnagel, F. A. Hamprecht. Conservation Tracking. ICCV 2013, 2928–2935.
* **Q. Huangfu, J. A. J. Hall. Parallelizing the dual revised simplex method. Mathematical Programming Computation 10, 119–142, 2018.**  
  Q. Huangfu, J. A. J. Hall. Parallelizing the dual revised simplex method. Mathematical Programming Computation 10, 119–142, 2018.
* **S. Bolusani et al. The SCIP Optimization Suite 9.0. arXiv:2402.17702, 2024.**  
  S. Bolusani 他. The SCIP Optimization Suite 9.0. arXiv:2402.17702, 2024.
* **D. G. Lowe. Distinctive Image Features from Scale-Invariant Keypoints. IJCV 60(2), 91–110, 2004.**  
  D. G. Lowe. Distinctive Image Features from Scale-Invariant Keypoints. IJCV 60(2), 91–110, 2004.
