**7th Place Solution**  
7位解法

**3D Cell Tracking with Joint Detection and Motion Prediction, Dedicated Division Models, and Lineage Optimization**  
検出・動き予測の同時学習、分裂専用モデル、および系譜最適化による3D細胞トラッキング

**[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) Solution Writeup · 7th place · Oct 1, 2026**  
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) 解法まとめ · 7位 · 2026年10月1日

**I placed 7th in Biohub Cell Tracking During Development.**  
Biohub Cell Tracking During Developmentコンペティションで7位になりました。

**Thank you to the organizers, the data contributors, and everyone who shared discussions and code during the competition.**  
主催者、データ提供者、そしてコンペ期間中にディスカッションやコードを共有してくださったすべての方々に感謝いたします。

**My approach used a single 3D nnU-Net (ResEncM) to predict cell locations and motion, dedicated models to evaluate cell division, and integer linear programming to construct a globally consistent lineage graph.**  
私のアプローチでは、単一の3D nnU-Net(ResEncM)を用いて細胞の位置と動きを予測し、細胞分裂を評価するための専用モデルを使用し、整数線形計画法(ILP)によって大域的に整合性のとれた系譜グラフを構築しました。

**I then applied postprocessing to address missing detections, short spurious tracks, and position jitter.**  
その後、検出漏れ、短い擬陽性のトラック、位置のジッター(ブレ)に対処するための後処理を適用しました。

**Rather than building a larger detection ensemble, I focused on improving a single detector and the steps that turn its predictions into correct connections.**  
大規模な検出アンサンブルを構築するのではなく、単一の検出器の改善と、その予測結果を正しい接続へと変換するステップの改善に注力しました。

**Figure 1: Pipeline overview.**  
図1: パイプラインの概要。

**Volumetric images from multiple time points are used to predict cell centers and motion, then construct candidate continuation and division links.**  
複数時点のボリュメトリック画像を用いて細胞中心と動きを予測し、継続および分裂の候補リンクを構築します。

**After optimizing lineage consistency, postprocessing addresses gaps, short spurious tracks, and coordinates.**  
系譜の整合性を最適化した後、後処理によってギャップ、短い擬陽性トラック、座標の補正を行います。

**The cells and lineages are schematic.**  
細胞と系譜は模式図です。

---

### 1. Cell Detection and Motion Prediction
1. 細胞検出と動き予測

#### 1.1 Joint Learning with 3D nnU-Net (ResEncM)
1.1 3D nnU-Net(ResEncM)による同時学習

**The backbone was [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), using the 3D ResEncM configuration from the [residual encoder presets](https://arxiv.org/abs/2404.09556).**  
バックボーンには[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)を採用し、[残差エンコーダのプリセット](https://arxiv.org/abs/2404.09556)から3D ResEncM構成を使用しました。

**It takes three 3D images, including the preceding and following time points, plus a channel encoding the frame interval.**  
前後の時点を含む3枚の3D画像に加えて、フレーム間隔をエンコードした1チャンネルを入力とします。

**It predicts a cell-center heatmap and a 3D backward displacement toward the preceding time point.**  
モデルは細胞中心のヒートマップと、直前の時点へ向かう3Dの後方変位(動き)を予測します。

**Cell-center candidates are extracted from the heatmap, and motion predictions guide the links between frames.**  
ヒートマップから細胞中心の候補が抽出され、動き予測によってフレーム間のリンクが誘導されます。

**Sharing a feature extractor between detection and motion allows the network to learn cell appearance together with temporal changes.**  
検出と動きの間で特徴抽出器を共有することにより、ネットワークは細胞の見た目と時間的変化を一緒に学習できるようになります。

**The training patch size was 48 × 128 × 128 (Z, Y, X).**  
学習パッチサイズは48×128×128(Z, Y, X)でした。

**I omitted the highest-resolution decoder stage, produced detection and motion outputs at half the XY resolution, and interpolated them back to the original resolution.**  
最高解像度のデコーダステージを省略し、XY解像度を半分にして検出と動きの出力を生成した上で、元の解像度に補間して戻しました。

**This reduced computation and was also intended to avoid excessive sensitivity to small annotation offsets.**  
これにより計算量が削減され、小さなアノテーションのズレに対する過度な敏感さを避けることも意図しました。

**Figure 2: An actual inference example, with inputs and outputs shown separately.**  
図2: 実際の推論例(入力と出力を個別に表示)。

**Inputs are the previous, current, and next 3D images and the frame interval.**  
入力は前・現在・次の3D画像とフレーム間隔です。

**Outputs are the cell-center heatmap, X/Y/Z motion components, and the resulting vectors.**  
出力は細胞中心ヒートマップ、X/Y/Zの動き成分、および合成されたベクトルです。

**Images and heatmaps are maximum-intensity projections over the same Z range.**  
画像とヒートマップは同一のZ範囲における最大値投影(MIP)です。

**At each pixel, the motion components are taken from the Z position with the highest detection probability.**  
各ピクセルにおいて、動き成分は最も検出確率が高いZ位置から取得されています。

**Arrows show current-to-past XY displacement at true scale, with color indicating the Z component.**  
矢印は実スケールでの現在から過去へのXY変位を示し、色はZ成分を表しています。

**Detection uses eight-view TTA; motion uses the untransformed input.**  
検出には8視点のTTAを使用し、動きには変換なしの入力を使用しています。

---

#### 1.2 Target Heatmaps and Sparse Annotations
1.2 ターゲットヒートマップと疎なアノテーション

**The detection target is a Gaussian heatmap around each annotated cell center.**  
検出ターゲットは、アノテーションされた各細胞中心を中心とするガウシアンヒートマップです。

**However, treating every unannotated region as negative could train the detector to suppress real cells that simply lack annotations.**  
しかし、アノテーションのない領域をすべて陰性(ネガティブ)として扱うと、単にアノテーションが欠落しているだけの実際の細胞を抑制するように検出器が学習してしまう可能性があります。

**I therefore restricted the regions contributing to the loss based on the annotations, rather than strongly supervising unannotated cells as background.**  
そのため、アノテーションのない細胞を背景として強く教師付けするのではなく、アノテーションに基づいて損失に寄与する領域を制限しました。

**I also tuned the heatmap width.**  
また、ヒートマップの幅も調整しました。

**An overly broad Gaussian makes neighboring peaks overlap and cells harder to separate.**  
ガウシアンが広すぎると隣接するピーク同士が重なり合い、細胞の分離が難しくなります。

**An overly narrow one is less tolerant of positional offsets and more sensitive to annotation variability, which I expected could make training less stable.**  
狭すぎると位置のズレに対する許容度が低下し、アノテーションのばらつきに過敏になって学習が不安定になる恐れがあると考えました。

**Based on validation, I used a Gaussian standard deviation of σ = 2 µm, truncated at 3σ = 6 µm.**  
検証に基づき、ガウシアンの標準偏差としてσ = 2 µmを採用し、3σ = 6 µmで打ち切る設定としました。

**Figure 3: Schematic heatmaps for the same two cell centers with different Gaussian widths.**  
図3: ガウシアンの幅を変えた同一の2つの細胞中心に対する模式的ヒートマップ。

**The middle panel uses the selected σ = 2 µm.**  
中央のパネルは採用したσ = 2 µmを使用しています。

**Narrow targets increase sensitivity to annotation offsets, while broad targets make neighboring cells harder to separate.**  
狭いターゲットはアノテーションのズレに対する敏感さを高め、広いターゲットは隣接する細胞の分離を困難にします。

---

#### 1.3 Motion Targets and Loss
1.3 動きのターゲットと損失関数

**Motion supervision uses the locations of a current cell and its parent or ancestor at an earlier time point, where their correspondence is known from the ground-truth lineage.**  
動きの教師付けには、正解系譜から対応関係が既知である、現在の細胞とより早い時点におけるその親または祖先の位置を使用します。

**With frame interval d and physical coordinates p, the backward-motion target is v = (p[t−d] − p[t]) / d.**  
フレーム間隔をd、物理座標をpとすると、後方への動きのターゲットは v = (p[t−d] − p[t]) / d となります。

**Dividing by d keeps the target in displacement per frame even when training with more widely separated images.**  
dで割ることで、より時間間隔が離れた画像で学習する場合でも、ターゲットを1フレームあたりの変位として維持できます。

**I then divided by 8 µm for numerical scaling and trained the network to regress this normalized value.**  
その後、数値のスケーリングのために8 µmで割り、この正規化された値を回帰するようにネットワークを学習させました。

**At inference, I multiply by 8 µm and use a frame interval of one to predict the location in the preceding frame.**  
推論時には8 µmを掛け、フレーム間隔を1として直前フレームにおける位置を予測します。

**The motion loss is computed within 2 µm of a cell center with a known correspondence.**  
動きの損失は、対応関係が分かっている細胞中心から2 µm以内の範囲で計算されます。

**A Gaussian weight with σ = 1 µm emphasizes positions near the center.**  
中心付近の位置を重視するために、σ = 1 µmのガウシアン重みを適用します。

**The weighted Smooth L1 loss on the XYZ components, with β = 0.25 in normalized units, is added to the detection loss.**  
正規化単位でβ = 0.25としたXYZ成分に対する重み付きSmooth L1損失が、検出損失に加えられます。

**Where supervision regions overlap, the target from the nearer center is used.**  
教師領域が重複する場所では、より近い中心からのターゲットが使用されます。

**Correspondence points receive the same rotations and reflections as the images, and motion targets are constructed from the transformed coordinates.**  
対応点は画像と同じ回転や反転を受け、変換された座標から動きのターゲットが構築されます。

**Cells with unknown correspondences, or correspondences that are no longer available after cropping or transformation, are excluded from the motion loss rather than assigned zero motion.**  
対応関係が不明な細胞や、クロッピングや幾何変換によって対応関係が失われた細胞は、動きゼロとして割り当てるのではなく、動きの損失計算から除外されます。

**For division, the representation preserves the correspondence of two daughters to their shared parent.**  
分裂の場合、この表現形式によって2つの娘細胞とそれらが共有する親細胞との対応関係が保持されます。

**Motion prediction helps construct candidate links; it does not determine whether a division occurred on its own.**  
動き予測は候補リンクの構築を補助するものであり、それ単体で分裂が起きたかどうかを決定するものではありません。

---

#### 1.4 Augmentation
1.4 データ拡張

**Spatial augmentation included rotation, scaling, and reflection in the XY plane.**  
空間的なデータ拡張には、XY平面内での回転、スケーリング、反転が含まれます。

**Because Z resolution is coarser than XY resolution, rotations requiring interpolation were confined to the XY plane, while reflections were also used along Z.**  
Z解像度がXY解像度よりも粗いため、補間を伴う回転はXY平面内に限定し、反転はZ軸方向にも適用しました。

**Spatial transformations were applied consistently across time points.**  
空間変換は複数の時点間で一貫して適用されました。

**As noted in the competition discussions, some sequences contained freeze intervals, during which images changed very little, followed by large cell movements.**  
コンペのディスカッションでも指摘されていたように、画像がほとんど変化しない静止区間(フリーズ区間)の後に細胞の大きな動きが続くシーケンスが存在しました。

**To address this, I used temporal augmentation with frames farther apart than consecutive frames.**  
これに対処するため、連続するフレームよりも時間間隔の広いフレームを用いた時間的データ拡張を行いました。

**The aim was to expose the model to larger displacements and improve its ability to associate cells after a freeze interval.**  
その目的は、モデルをより大きな変位に晒すことで、フリーズ区間後の細胞を関連付ける能力を向上させることでした。

**For interval d, the inputs at time t are t−d, t, and t+d.**  
間隔dに対して、時刻tでの入力は t−d、t、t+d となります。

**Training starts with d = 1, then introduces d = 2 and later d = 3.**  
学習は d = 1 から開始し、その後に d = 2、さらに d = 3 を導入しました。

**The interval is also provided as an input channel so that the model can distinguish identical apparent displacements observed over different time intervals.**  
異なる時間間隔で観察される同一の見かけの変位をモデルが識別できるように、間隔情報も入力チャンネルとして与えられます。

---

#### 1.5 Pretraining on Synthetic Data
1.5 合成データによる事前学習

**I generated synthetic data using code shared by José Freitas ([josefreitasalvesneto](https://www.kaggle.com/josefreitasalvesneto)), from his [discussion on synthetic 3D microscopy data](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103) and [Biohub Synthetic Dataset notebook](https://www.kaggle.com/code/josefreitasalvesneto/biohub-synthetic-dataset).**  
私はJosé Freitas氏([josefreitasalvesneto](https://www.kaggle.com/josefreitasalvesneto))が共有してくれたコードを使用し、同氏の[3D顕微鏡合成データに関するディスカッション](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103)および[Biohub Synthetic Datasetノートブック](https://www.kaggle.com/code/josefreitasalvesneto/biohub-synthetic-dataset)を参考にして合成データを生成しました。

**I am grateful for this contribution of labeled synthetic data, including divisions.**  
分裂を含むラベル付き合成データを提供してくださったことに感謝しています。

**In the pretrained configuration, the detector was pretrained for 1,000 epochs on synthetic data and then trained for 1,500 epochs on real data.**  
事前学習を行う構成では、検出器を合成データで1,000エポック事前学習させた後、実データで1,500エポック学習させました。

**Figure 4: Synthetic pretraining followed by training on real images.**  
図4: 合成データによる事前学習とそれに続く実画像での学習。

**The left image is a schematic illustration of the training process, the middle image is real data, and the right image is a heatmap produced by the trained model.**  
左の画像は学習プロセスの模式図、中央の画像は実データ、右の画像は学習済みモデルによって生成されたヒートマップです。

**Synthetic-data generation was based on José Freitas's shared code.**  
合成データの生成はJosé Freitas氏の共有コードに基づいています。

**The public and private leaderboards did not lead to the same conclusion about pretraining.**  
パブリックとプライベートのリーダーボードでは、事前学習について同じ結論には至りませんでした。

**In a comparison with division modeling, graph optimization, and postprocessing held fixed, the public score increased from 0.960 to 0.966, while the private score decreased from 0.956 to 0.947.**  
分裂モデル、グラフ最適化、後処理を固定した比較において、パブリックスコアは0.960から0.966へと向上した一方で、プライベートスコアは0.956から0.947へと低下しました。

**This was not a fully controlled comparison of all factors, including total training effort, but it did not establish improved generalization to unseen data from pretraining.**  
これは総学習量などすべての要素を完全に統制した比較ではありませんが、事前学習によって未知データへの汎化性能が向上したとは確認できませんでした。

**Figure 5: Comparison of models trained with and without synthetic pretraining.**  
図5: 合成データ事前学習の有無によるモデルの比較。

**Conditions outside the detection and motion model were held fixed, but total training effort, including pretraining, differed.**  
検出および動きモデル以外の条件は固定されましたが、事前学習を含む総学習量は異なっていました。

---

#### 1.6 Inference
1.6 推論

**For detection, I used eight views formed by rotations and reflections in the XY plane.**  
検出には、XY平面での回転と反転によって形成される8つの視点を使用しました。

**After transforming predictions back to the original coordinate system, I averaged the heatmap logits—the values before conversion to probabilities.**  
予測結果を元の座標系に戻した後、確率に変換する前の値であるヒートマップのロジットを平均化しました。

**This test-time augmentation (TTA) of a single detector was more useful than adding detection models.**  
単一の検出器に対するこのテスト時データ拡張(TTA)は、検出モデルを追加するよりも効果的でした。

**Applying the same averaging strategy to motion vectors did not improve the results.**  
動きベクトルに対して同様の平均化戦略を適用しても、結果は改善しませんでした。

**For motion, I used the prediction from the untransformed input.**  
動きに関しては、変換なしの入力からの予測を使用しました。

**Each frame is Z-score normalized, and the temporal interval is fixed to one at inference.**  
各フレームはZスコア正規化され、推論時の時間間隔は1に固定されます。

**Cell centers are extracted from local heatmap maxima, followed by non-maximum suppression with a physical distance of 2 µm.**  
細胞中心はヒートマップの局所的最大値から抽出され、続いて2 µmの物理距離によるNon-Maximum Suppression(NMS)が適用されます。

**Motion outputs are converted back to displacements in µm and used to associate detections with candidates in the previous frame.**  
動きの出力はµm単位の変位に戻され、前フレームの候補と検出結果を対応付けるために使用されます。

**Based on validation and public leaderboard results, I used 0.30 as the baseline detection threshold.**  
検証とパブリックリーダーボードの結果に基づき、ベースラインの検出閾値として0.30を採用しました。

**I later tested lower values, including 0.28, and used 0.28 with the additional postprocessing.**  
後に0.28を含むより低い値をテストし、追加の後処理を施した設定では0.28を採用しました。

**Looking back after the competition, lower thresholds tended to perform better on the private leaderboard.**  
コンペ終了後に振り返ると、プライベートリーダーボードではより低い閾値の方が良好なパフォーマンスを示す傾向がありました。

**One possibility is that harder cells shifted the heatmap outputs toward lower confidence, but this remains a hypothesis: I could not inspect the private images or prediction distributions.**  
一つの可能性として、より難しい細胞によってヒートマップの出力が低信頼度側にシフトしたことが考えられますが、これは推測にとどまります。プライベートの画像や予測分布を確認することはできなかったためです。

**Figure 6: Submission scores at different detection thresholds.**  
図6: 異なる検出閾値における提出スコア。

**The left panel uses a late-training model without pretraining; the right panel uses a pretrained model with additional postprocessing.**  
左のパネルは事前学習なしの後期学習モデルを使用しており、右のパネルは追加の後処理を施した事前学習済みモデルを使用しています。

**The 0.30 point in the right panel also differs in execution configuration, so it is not a threshold-only comparison.**  
右パネルの0.30のプロットは実行構成も異なっているため、閾値のみの比較ではありません。

**The settings preferred by the public and private leaderboards did not always agree.**  
パブリックとプライベートのリーダーボードで好まれた設定は、必ずしも一致しませんでした。

---

#### 1.7 Building Temporal Link Candidates from Motion Predictions
1.7 動き予測からの時間的リンク候補の構築

**After detecting cells in each frame, the next question is which cell in the previous frame corresponds to each current cell.**  
各フレームで細胞を検出した後の次の問題は、現在の各細胞が前フレームのどの細胞に対応しているかです。

**Choosing the nearest cell to the current location can fail under large motion or in crowded regions.**  
現在の位置に最も近い細胞を選択する方法は、大きな動きがある場合や過密な領域では破綻する可能性があります。

**Instead, I used the distance between the motion-projected location and the cells actually detected in the previous frame.**  
その代わりに、動きによって投影された位置と、前フレームで実際に検出された細胞との間の距離を使用しました。

**The model predicts a backward 3D displacement from the current frame toward the previous one.**  
モデルは、現在のフレームから前フレームへ向かう後方の3D変位を予測します。

**I sample the motion field around each detection center with Gaussian weighting and add the resulting vector to the current location.**  
各検出中心の周囲からガウシアン重み付けを用いて動き場をサンプリングし、得られたベクトルを現在の位置に加算します。

**This gives a predicted correspondence location in the previous frame.**  
これにより、前フレームにおける対応予測位置が得られます。

**Distances are computed in µm, accounting for the different Z, Y, and X voxel spacings.**  
距離は、Z, Y, Xの異なるボクセル間隔を考慮してµm単位で計算されます。

**I retain up to the five nearest detections within 7 µm of the projected location.**  
投影された位置から7 µm以内にある最も近い検出結果を最大5個まで保持します。

**This 7 µm gate applies to the residual distance from the prediction, not to the total distance a cell is allowed to move.**  
この7 µmのゲート値は、細胞が移動を許容される総移動距離ではなく、予測からの残差距離に適用されます。

**Figure 7: Project a current cell into the previous frame and evaluate residual distances to nearby detections.**  
図7: 現在の細胞を前フレームに投影し、近傍の検出との残差距離を評価する。

**Multiple candidate edges and scores are retained, leaving the final selection to ILP.**  
複数の候補エッジとスコアが保持され、最終的な選択はILPに委ねられます。

**Motion arrows point from current to past, while lineage edges point from past to current.**  
動きの矢印は現在から過去を指すのに対し、系譜エッジは過去から現在を指します。

**Distances, confidence values, and scores in the figure are illustrative examples, not measurements.**  
図中の距離、信頼度、スコアは説明用の例示であり、実測値ではありません。

**The implementation uses 3D distances.**  
実装では3次元距離を使用しています。

**The link score is a Gaussian function of residual distance, multiplied by the geometric mean of the detection confidences at the two endpoints.**  
リンクスコアは残差距離のガウス関数であり、両端点における検出信頼度の幾何平均を掛け合わせたものです。

**For previous-frame candidate i and current cell j, with residual distance d_ij and detection confidences q_i and q_j, the score is:**  
前フレームの候補iと現在の細胞jについて、残差距離をd_ij、検出信頼度をq_iおよびq_jとすると、スコアは次のようになります:

$$s_{ij} = \exp\left[-\frac{d_{ij}^2}{2 \times (3\,\mu\text{m})^2}\right] \times \sqrt{q_i \times q_j}$$

**The nearest candidate is not selected immediately.**  
最も近い候補が直ちに選択されるわけではありません。

**I keep the candidate edge i→j and its score in the graph, and pass −s_ij as its link cost to the ILP objective.**  
グラフ内に候補エッジ i→j とそのスコアを保持し、そのリンクコストとして −s_ij をILPの目的関数に渡します。

**Higher-scoring links are favored, while appearance, disappearance, and division costs and constraints against multiple parents are considered jointly across the lineage.**  
より高スコアのリンクが優先される一方で、出現・消失・分裂のコストや、単一細胞に対する複数の親を禁止する制約が系譜全体で同時に考慮されます。

**The following sections describe division evaluation and graph optimization in more detail.**  
以下のセクションでは、分裂の評価とグラフ最適化についてさらに詳しく説明します。

---

### 2. A Dedicated Pipeline for Cell Division
2. 細胞分裂専用のパイプライン

**Solving division entirely within the detection network would simplify the system.**  
検出ネットワーク内だけで分裂を完全に解決できればシステムは簡素化されます。

**However, division examples were limited, and I felt they were difficult to emphasize sufficiently while learning ordinary cells and motion.**  
しかし、分裂の事例は限られており、通常の細胞や動きを学習する中で分裂を十分に強調することは難しいと感じました。

**I therefore built a separate pipeline focused on learning division.**  
そこで、分裂の学習に特化した別のパイプラインを構築しました。

**It has three main roles:**  
これには主に3つの役割があります:

**Assess whether the local images around a candidate parent suggest division.**  
親候補の周囲の局所画像が分裂を示唆しているかを評価すること。

**Rank pairs of candidate daughters.**  
娘細胞候補のペアを順位付けすること。

**Use trajectory and geometric information to assess whether division or ordinary continuation is the more plausible explanation.**  
軌跡と幾何情報を用いて、分裂と通常の継続のどちらがより妥当な説明であるかを評価すること。

**Figure 8: A schematic example with parent P and daughter candidates A, B, and C.**  
図8: 親Pと娘細胞候補A, B, Cによる模式例。

**Local images provide division evidence, daughter combinations are evaluated separately, and an MLP combines image, pair, and trajectory features.**  
局所画像が分裂の証拠を提供し、娘細胞の組み合わせが個別に評価され、MLPが画像・ペア・軌跡の特徴を統合します。

**These outputs evaluate candidates; the final connections are chosen under ILP constraints.**  
これらの出力が候補を評価し、最終的な接続はILP制約のもとで選択されます。

---

#### 2.1 Align Motion Before Taking Image Differences
2.1 画像差分を取る前の動き合わせ(アライメント)

**Inspection of images around divisions revealed characteristic changes in brightness and shape.**  
分裂前後の画像を観察すると、輝度と形状に特徴的な変化が見られました。

**However, raw temporal differences also contained strong signals from ordinary cell motion.**  
しかし、単純な時間差分には通常の細胞の動きに起因する強いシグナルも含まれていました。

**For image-based division classification, I first estimate local motion from surrounding cells and align the preceding and following 3D images.**  
画像ベースの分裂分類のために、まず周囲の細胞から局所的な動きを推定し、前後の3D画像を位置合わせ(アライメント)しました。

**Taking differences after alignment suppresses signals caused by ordinary translation and makes division-related changes in shape and brightness easier to see.**  
位置合わせ後に差分を取ることで、通常の並進移動によるシグナルが抑制され、分裂に伴う形状や輝度の変化を捉えやすくなります。

**Inputs include aligned images at multiple time points, signed temporal differences, their absolute values, and heatmaps reconstructed from detection candidates.**  
入力には、複数時点で位置合わせされた画像、符号付き時間差分、それらの絶対値、および検出候補から再構成されたヒートマップが含まれます。

**Raw images describe shape, signed differences show where intensity increases or decreases, and absolute differences describe the magnitude of change.**  
元画像は形状を表し、符号付き差分は強度の増減箇所を示し、絶対値差分は変化の大きさを示します。

**The heatmaps are Gaussians reconstructed from detected centers and their confidence scores; they provide information about the candidate location and the surrounding cell arrangement.**  
ヒートマップは検出された中心座標と信頼度スコアから再構成されたガウシアンであり、候補の位置や周囲の細胞配置に関する情報を提供します。

**Figure 9: Top left: actual images before and after division.**  
図9: 左上: 分裂前後の実際の画像。

**Yellow arrows identify the parent and cyan arrows identify the daughters.**  
黄色の矢印は親細胞を、シアンの矢印は娘細胞を示しています。

**Bottom left: raw differences and differences after the same 3D local-motion estimation used in the submitted pipeline, using saved detection candidates.**  
左下: 保存された検出候補を使用し、単純な差分と、提出パイプラインで使用されたものと同じ3D局所運動推定後の差分。

**For display, differences are computed from Z maximum-intensity projections, with a common color scale.**  
表示用として、差分は共通のカラースケールを持つZ最大値投影(MIP)から計算されています。

**The Gaussian schematic on the right illustrates a negative signal at the old parent location and positive signals at the two daughter locations.**  
右側のガウシアン模式図は、元の親細胞位置における負のシグナルと、2つの娘細胞位置における正のシグナルを示しています。

**Actual patterns depend on division orientation and intensity changes.**  
実際のパターンは、分裂の方向や強度の変化に依存します。

---

#### 2.2 Division Image Classifier and Auxiliary Supervision
2.2 分裂画像分類器と補助的な教師付け(Auxiliary Supervision)

**The image classifier is a small 3D CNN followed by a bidirectional GRU (BiGRU).**  
画像分類器は、小さな3D CNNとそれに続く双方向GRU(BiGRU)で構成されています。

**It takes 12 × 48 × 48 (Z, Y, X) crops around a candidate parent.**  
親候補の周囲から切り出した12×48×48(Z, Y, X)のクロップを入力とします。

**Five time points, t−2 through t+2, form three overlapping three-frame windows.**  
t−2からt+2までの5つの時点が、重複する3つの3フレームウィンドウを形成します。

**The aim is to capture the transition from one cell to two rather than a single static shape.**  
単一の静的な形状ではなく、1つの細胞から2つへの遷移を捉えることが目的です。

**Each window contains seven image channels—three images, two signed differences, and two absolute differences—plus seven channels constructed in the same way from detection heatmaps, for 14 channels in total.**  
各ウィンドウは、7つの画像チャンネル(3枚の画像、2つの符号付き差分、2つの絶対値差分)に加えて、検出ヒートマップから同様に構築された7チャンネルを含み、合計で14チャンネルとなります。

**The heatmap channels use t−1, t, and t+1 and are shared across the three image windows.**  
ヒートマップチャンネルには t−1、t、t+1 が使用され、3つの画像ウィンドウ間で共有されます。

**A shared 3D CNN maps each window to a 128-dimensional feature vector.**  
共有された3D CNNが各ウィンドウを128次元の特徴ベクトルにマッピングします。

**The BiGRU combines temporal information, which is added residually to the central window's features.**  
BiGRUが時間情報を統合し、中央ウィンドウの特徴量に残差(Residual)として加算されます。

**The CNN consists of four blocks with 24→48→96→128 channels, spatial pooling, and global average pooling.**  
CNNは24→48→96→128チャンネルの4つのブロック、空間プーリング、およびグローバル平均プーリングで構成されています。

**Figure 10: The upper panel shows 14 actual input channels reconstructed from saved detections and real images using the submitted preprocessing code.**  
図10: 上段パネルは、提出した前処理コードを用いて、保存された検出と実画像から再構成された実際の14個の入力チャンネルを示しています。

**It shows a Z slice from the central three-frame window, before channel-wise standardization.**  
チャンネルごとの標準化を行う前の、中央の3フレームウィンドウからのZスライスを示しています。

**The lower panel shows one model's architecture.**  
下段パネルはあるモデルのアーキテクチャを示しています。

**Its main output is a division logit; auxiliary outputs predict the number and relative positions of successor cells.**  
その主出力は分裂ロジットであり、補助出力は後続細胞の数と相対位置を予測します。

**Outputs marked “(Aux)” receive auxiliary supervision.**  
「(Aux)」と表記された出力は補助教師付けを受けます。

**To encourage features that capture morphological change, I supervised not only division but also the number of successor cells in the next frame (0, 1, or 2) and the relative XYZ positions of up to two successors.**  
形態変化を捉える特徴量を促すため、分裂だけでなく、次フレームにおける後続細胞の数(0, 1, または2)と、最大2個の後続細胞の相対XYZ位置も教師付けしました。

**The loss adds count cross-entropy and position Smooth L1, each weighted by 0.25, to the division BCE loss.**  
損失関数は、分裂のBCE損失に、それぞれ0.25の重みを付けた個数のクロスエントロピー損失と位置のSmooth L1損失を加えたものです。

**With sparse annotations, auxiliary losses are computed only where valid targets are available.**  
アノテーションが疎であるため、補助損失は有効なターゲットが存在する場所でのみ計算されます。

**I also fed count and position predictions back into the main classifier to adjust its division logit.**  
また、個数と位置の予測をメイン分類器にフィードバックし、その分裂ロジットを調整しました。

**Gradients into the auxiliary predictions are stopped along this feedback path.**  
このフィードバック経路では、補助予測への勾配は停止(Stop Gradient)されます。

**The auxiliary outputs support division classification; their predicted positions are not directly adopted as the final daughters or edges.**  
補助出力は分裂分類を補助するためのものであり、予測された位置がそのまま最終的な娘細胞やエッジとして直接採用されるわけではありません。

**Actual daughters are selected from detected candidates.**  
実際の娘細胞は検出された候補から選択されます。

---

#### 2.3 Identifying a Dividing Parent and Choosing Two Daughters Are Different Problems
2.3 分裂する親の特定と2つの娘細胞の選択は異なる問題である

**A correct division decision is still wrong as a lineage event if one daughter is misidentified.**  
分裂の判断自体が正しくても、片方の娘細胞の特定を誤れば、系譜イベントとしては誤りになってしまいます。

**I therefore used the image classifier to assess the parent and an ExtraTrees classifier to rank daughter pairs.**  
そのため、画像分類器を用いて親を評価し、ExtraTrees分類器を用いて娘細胞のペアを順位付けしました。

**Pairs are formed from detections in the next frame around the same candidate parent.**  
ペアは、同一の親候補の周囲にある次フレームの検出結果から形成されます。

**Important features include parent-to-daughter distances, daughter separation, the angle between the two parent-to-daughter directions, the offset between the parent and the daughters' midpoint, nearby cell counts, and detection confidence.**  
重要な特徴量には、親と娘の距離、娘同士の離間距離、親から2つの娘へ向かう方向のなす角、親と娘たちの中点とのオフセット、近傍の細胞数、および検出信頼度が含まれます。

**Features derived from neighboring time points describe whether the daughters plausibly share a parent or are better explained by separate tracks.**  
隣接する時点から得られる特徴量は、娘細胞たちが妥当に親を共有しているのか、それとも別々のトラックとして説明する方が適切なのかを表します。

**There are 73 geometric and temporal features in total.**  
幾何学的および時間的な特徴量は合計で73個あります。

**Figure 11: A schematic with three daughter candidates.**  
図11: 3つの娘細胞候補が存在する模式図。

**Even with high confidence that the parent divides, the correct daughter pair must be evaluated separately.**  
親が分裂するという確信度が高くても、正しい娘細胞ペアは個別に評価されなければなりません。

**Restricting division assessment to branches already selected in an initial tracking graph cannot recover events that were missed there.**  
初期トラッキンググラフで既に選択された分岐のみに分裂評価を限定してしまうと、そこで見落とされたイベントを回復することができません。

**I therefore expanded assessment to existing link candidates as well.**  
そのため、評価対象を既存のリンク候補にまで拡大しました。

**This change increased the public score from 0.969 to 0.971, while both private scores in that comparison were 0.947.**  
この変更によりパブリックスコアは0.969から0.971へと向上しましたが、その比較におけるプライベートスコアはいずれも0.947でした。

---

#### 2.4 Combining Local Images with Trajectory and Geometric Information
2.4 局所画像と軌跡・幾何情報の統合

**Local images alone cannot fully assess consistency with surrounding cells or the plausibility of a trajectory.**  
局所画像だけでは、周囲の細胞との整合性や軌跡の妥当性を十分に評価することはできません。

**I therefore used a small MLP with 83 inputs, 32 hidden units, SiLU activation, and one output to further evaluate division hypotheses involving daughter pairs.**  
そこで、入力83、隠れ層32ユニット、SiLU活性化関数、出力1の小型MLPを使用して、娘ペアを含む分裂仮説をさらに評価しました。

**Its 83 inputs consist of 73 geometric and temporal features; seven additional features such as distance to another candidate parent and the cost of an ordinary-continuation explanation; the parent's detection confidence; the image classifier's division logit; and the ExtraTrees pair score.**  
その83個の入力は、73個の幾何・時間特徴量、別の親候補への距離や通常継続の説明コストなどの7個の追加特徴量、親の検出信頼度、画像分類器の分裂ロジット、およびExtraTreesのペアスコアで構成されています。

**For example, if the two proposed daughters are each close to a different parent, an apparent division may actually be two ordinary tracks.**  
例えば、提案された2つの娘細胞がそれぞれ別の親細胞に近い場合、一見分裂に見えるものは実際には2本の通常のトラックである可能性があります。

**These alternative explanations are assessed together with image-based division evidence.**  
このような代替的な説明は、画像ベースの分裂の証拠とともに総合的に評価されます。

**Features are standardized and extreme values are clipped.**  
特徴量は標準化され、極端な値はクリッピングされます。

**The MLP output is converted into costs for division hypotheses and passed to ILP.**  
MLPの出力は分裂仮説のコストに変換され、ILPに渡されます。

**A high score for an individual hypothesis does not determine the links by itself: the selected graph must also satisfy global constraints, such as not assigning multiple parents to one daughter.**  
個別の仮説に対するハイスコア単独でリンクが決まるわけではありません。選択されるグラフは、1つの娘細胞に複数の親を割り当てないといった大域的な制約も満たす必要があります。

**In another configuration, I replaced the daughter-pair ranker with LightGBM using the 73 geometric and temporal features plus 18 image features describing brightness and contrast near the daughters, for 91 features in total.**  
別の構成では、娘ペアのランキングモデルをLightGBMに置き換え、73個の幾何・時間特徴量に加えて、娘細胞付近の輝度やコントラストを表す18個の画像特徴量を使用し、合計91個の特徴量としました。

**The image features include each daughter's center intensity, the mean, maximum, and standard deviation in a small neighborhood, the mean in a surrounding region, and local-to-surrounding contrast.**  
画像特徴量には、各娘細胞の中心輝度、狭い近傍における平均・最大値・標準偏差、周囲領域における平均、および局所対周囲のコントラストが含まれます。

**Taking the minimum, maximum, and absolute difference across the two daughters makes these features independent of daughter ordering.**  
2つの娘細胞間で最小値、最大値、絶対差を取ることにより、これらの特徴量は娘細胞の並び順に依存しないようになります。

**The MLP evaluates the division event, while the image-feature LightGBM ranks daughter pairs.**  
MLPが分裂イベント自体を評価し、画像特徴量を用いたLightGBMが娘ペアを順位付けします。

**This configuration achieved a public score of 0.970 and a private score of 0.952.**  
この構成はパブリックスコア0.970、プライベートスコア0.952を達成しました。

**However, it also differed in other model and optimization choices, so this submission alone does not isolate the contribution of any individual component.**  
ただし、この構成は他のモデルや最適化の選択肢も異なっていたため、この提出結果のみから各コンポーネント個別の寄与度を切り分けることはできません。

---

### 3. Selecting a Consistent Lineage with Integer Linear Programming
3. 整数線形計画法(ILP)による一貫した系譜の選択

**Selecting candidates independently can produce inconsistent graphs, such as assigning multiple parents to one daughter.**  
候補を独立して選択すると、1つの娘細胞に複数の親が割り当てられるなど、不整合なグラフが生成される可能性があります。

**I converted candidate evaluations into costs and used integer linear programming (ILP) to select connections jointly.**  
そこで、候補の評価値をコストに変換し、整数線形計画法(ILP)を用いて接続を同時に一括選択しました。

#### 3.1 The First ILP Produces a Provisional Lineage
3.1 最初のILPによる暫定系譜の生成

**The first ILP combines candidate edges from detection and motion with image-based division scores and daughter-pair scores.**  
第1段階のILPは、検出と動きから得られた候補エッジを、画像ベースの分裂スコアおよび娘ペアスコアと統合します。

**Division is already included at this stage, alongside ordinary continuation.**  
この段階で、通常の継続と並んで分裂も既に組み込まれています。

**Appearance, disappearance, and division costs are combined with constraints such as no multiple parents per cell to produce a provisional lineage that is consistent across the video.**  
出現、消失、分裂のコストが、細胞あたり親は1つまでといった制約と組み合わされ、動画全体で一貫した暫定的な系譜が生成されます。

#### 3.2 Re-evaluating Division Hypotheses
3.2 分裂仮説の再評価

**Starting from branches in the provisional lineage, I evaluate alternative daughter pairs.**  
暫定系譜の分岐点を起点として、代替となる娘ペアを評価します。

**For example, P may appear to divide into A and B, but if another parent Q is close to B, ordinary links P→A and Q→B may be a more natural explanation.**  
例えば、PがAとBに分裂しているように見えても、別の親QがBの近くにあれば、通常のリンク P→A および Q→B とする方がより自然な説明である場合があります。

**The MLP re-evaluates the division hypothesis using image evidence, daughter-pair geometry, distances to alternative parents, and the cost of an ordinary-continuation explanation.**  
MLPは、画像の証拠、娘ペアの幾何構造、代替親への距離、および通常の継続として説明した場合のコストを用いて、分裂仮説を再評価します。

**Evaluating only selected branches would prevent recovery of divisions missed by the first ILP.**  
選択された分岐のみを評価していては、最初のILPで見逃された分裂を回復できません。

**I therefore also include existing division candidates that have not yet been re-evaluated.**  
そのため、まだ再評価されていない既存の分裂候補も含めるようにしました。

**These additional hypotheses must still use parent–daughter combinations consistent with the candidate edges constructed from detection and motion.**  
これらの追加仮説も、検出と動きから構築された候補エッジと矛盾しない親娘の組み合わせを使用する必要があります。

#### 3.3 Updating the Costs and Optimizing Again
3.3 コストの更新と再最適化

**The MLP output is converted into rewards or penalties for daughter pairs and added to the cost of selecting both parent–daughter edges together.**  
MLPの出力は娘ペアに対する報酬またはペナルティに変換され、両方の親娘エッジを同時に選択する際のコストに加算されます。

**Plausible divisions are encouraged, while hypotheses better explained by ordinary continuation are penalized.**  
妥当な分裂は促進され、通常の継続として説明する方が適している仮説にはペナルティが課されます。

**I then solve ILP again on the original candidate graph, reconsidering competing links.**  
その後、元の候補グラフ上で競合するリンクを再検討しながら、再びILPを解きます。

**The key is global re-optimization with updated scores, rather than freezing the first lineage and locally editing its branches.**  
重要なのは、最初の系譜を固定してその枝を局所的に編集するのではなく、更新されたスコアを用いて大域的な再最適化を行うことです。

**Figure 12: Two-stage ILP.**  
図12: 2段階ILP。

**The first solve already includes division and produces a provisional lineage.**  
初回の計算で既に分裂が含まれており、暫定的な系譜を生成します。

**Division hypotheses and alternatives are then re-evaluated, and links are selected again on the same candidate graph.**  
その後、分裂仮説と代替案が再評価され、同一の候補グラフ上でリンクが再度選択されます。

**In the example on the right, an apparent division by P is reinterpreted as a continuation from another parent Q.**  
右側の例では、Pによる見かけ上の分裂が、別の親Qからの継続として再解釈されています。

**This is a hypothetical illustration, not a measured result.**  
これは仮説的な図解であり、実測結果ではありません。

**Another configuration used a single optimization stage with division candidates included.**  
別の構成では、分裂候補を含めた単一の最適化ステージを使用しました。

**Switching to one stage can change the selected graph; it is not merely a change in execution order.**  
1ステージへの変更は選択されるグラフを変化させる可能性があり、単なる実行順序の変更にとどまりません。

**Score differences between configurations therefore include differences in optimization as well as division models.**  
したがって、構成間のスコアの差には、分裂モデルだけでなく最適化手法の違いも含まれています。

---

### 4. Errors Addressed by Postprocessing
4. 後処理によって対処したエラー

**I added postprocessing in response to observed errors, also drawing on public notebooks.**  
公開ノートブックも参考にしながら、観察されたエラーに対応して後処理を追加しました。

**The main targets were brief detection gaps, weak short tracks, tracks near divisions or video boundaries, and coordinate jitter, including during freeze intervals.**  
主な対象は、短時間の検出ギャップ(欠落)、微弱で短いトラック、分裂付近や動画境界付近のトラック、そしてフリーズ区間中を含む座標のジッター(ブレ)でした。

#### 4.1 Gap Filling, Short-Component Selection, and Coordinate Correction
4.1 ギャップ補間、短い成分の選択、および座標補正

**The configuration with additional postprocessing used the following operations.**  
追加の後処理を施した構成では、以下の操作を行いました。

**Short-track decisions consider the number of nodes in a connected component, not just the length of an individual branch.**  
短いトラックかどうかの判定では、個々のブランチの長さだけでなく、連結成分内のノード数を考慮します。

---

**Target: One missing frame**  
対象: 1フレームの欠落

**Decision and correction: Match track ends at t to track starts at t+2 one-to-one, within 9 µm between endpoints. Reuse an isolated node within 3.2 µm of the interpolated midpoint if one exists; otherwise insert a node whose position is refined using the image near that midpoint.**  
判定と補正: 時刻tでのトラック終端と時刻t+2でのトラック開始を、端点間距離9 µm以内で1対1マッチングします。補間された中点から3.2 µm以内に孤立ノードが存在する場合はそれを再利用し、存在しない場合はその中点付近の画像を用いて位置を微調整したノードを挿入します。

**Safeguards: Do not reuse a start or intermediate node more than once. Limit new nodes to approximately 2% of the original node count and at most 2,000.**  
安全策: 開始ノードや中間ノードを2回以上再利用しないこと。新規ノードの追加は元のノード数の約2%かつ最大2,000個に制限すること。

---

**Target: Short isolated components**  
対象: 短い孤立成分

**Decision and correction: By default, remove connected components with fewer than nine nodes.**  
判定と補正: デフォルトでは、ノード数が9個未満の連結成分を削除します。

**Safeguards: Protect components containing division. An eight-node component can be retained if its mean link score is at least 0.5.**  
安全策: 分裂を含む成分は保護します。8ノードの成分は、平均リンクスコアが0.5以上であれば保持できます。

---

**Target: Restoring short boundary and interior components**  
対象: 境界付近および内部の短い成分の復元

**Decision and correction: Restore components of three to eight nodes touching the first or last frame when their confidence is high relative to nearby detections. Interior components of five to seven consecutive nodes are also considered when link scores and relative detection confidence are high.**  
判定と補正: 最初または最後のフレームに接している3〜8ノードの成分は、近傍の検出に対して相対的に信頼度が高い場合に復元します。連続する5〜7ノードの内部成分も、リンクスコアと相対的な検出信頼度が高い場合には検討対象とします。

**Safeguards: Require a mean confidence ratio of at least one, with a ratio of at least one for a majority of nodes. Interior restoration is limited to components sharing no nodes with the existing graph.**  
安全策: 平均信頼度比が1以上であり、過半数のノードで比率が1以上であることを要求します。内部成分の復元は、既存グラフとノードを共有していない成分に限定されます。

---

**Target: Excessive restoration**  
対象: 過剰な復元の抑制

**Decision and correction: For boundary and eight-node restoration candidates, remove a component if a majority of its nodes lie within 7 µm of baseline-graph nodes at the same time points.**  
判定と補正: 境界成分および8ノードの復元候補について、同一時点においてノードの過半数がベースライングラフのノードから7 µm以内にある場合はその成分を削除します。

**Safeguards: If there are at least eight interior eight-node components, also remove those whose mean detection confidence falls below the 25th-percentile threshold.**  
安全策: 内部の8ノード成分が8個以上ある場合、平均検出信頼度が第25パーセンタイル閾値を下回るものも削除します。

---

**Target: Coordinate jitter along a track**  
対象: トラックに沿った座標のジッター(ブレ)

**Decision and correction: Fit a line to the uniquely traceable neighborhood, up to four steps in each direction, and blend 40% of the original coordinate with 60% of the fitted coordinate.**  
判定と補正: 各方向に最大4ステップまでの、一意に追跡可能な近傍に対して直線をフィッティングし、元の座標40%とフィッティングされた座標60%をブレンドします。

**Safeguards: Cap each node's correction at 2 µm to avoid excessive displacement.**  
安全策: 過度な変位を防ぐため、各ノードの補正量を2 µmに制限(キャップ)します。

---

**Removing every short track would also discard real cells just after division or near video boundaries.**  
短いトラックをすべて削除してしまうと、分裂直後や動画の境界付近にある実際の細胞まで破棄してしまいます。

**I therefore combined removal of spurious components with protection and restoration of short components supported by evidence.**  
そのため、擬陽性成分の削除と、証拠によって裏付けられた短い成分の保護・復元を組み合わせました。

**The table describes this additional-postprocessing configuration; these settings were not shared by every experiment.**  
上記の表はこの追加後処理構成を示したものであり、すべての実験で共有されていたわけではありません。

---

#### 4.2 Aligning Coordinates During Freezes Without Changing Connections
4.2 接続を変えずにフリーズ中の座標を整列させる

**A freeze is detected when adjacent 3D images are exactly equal at the pixel level.**  
フリーズ(静止)は、隣接する3D画像がピクセルレベルで完全に一致している場合に検出されます。

**Within such intervals, one-to-one chains are aligned to the coordinate of their first actual detection, excluding branch points and their immediate neighbors.**  
そのような区間内では、分岐点とその直傍を除き、1対1の鎖状トラックが最初に実際に検出された座標に整列されます。

**Node counts, timestamps, and edges remain unchanged, and nodes inserted by gap filling are not used as coordinate anchors.**  
ノード数、タイムスタンプ、エッジは変更されず、ギャップ補間によって挿入されたノードが座標のアンカーとして使われることはありません。

**Even correct connections can exhibit coordinate jitter during a freeze.**  
正しい接続であっても、フリーズ中に座標のブレが生じることがあります。

**Correcting coordinates without adding graph links was useful in this case.**  
この場合、グラフのリンクを追加することなく座標のみを補正することが有効でした。

**On the same 40 development videos, the overall score improved from 0.960897 to 0.962707 without changing node or division counts.**  
同一の開発用40動画において、ノード数や分裂数を変えることなく、総合スコアが0.960897から0.962707へと向上しました。

**Coordinate correction also changed matching to ground truth: Edge TP increased by 17 and FP decreased by 34.**  
座標補正によってグラウンドトゥルース(正解)とのマッチングも変化し、エッジのTP(真陽性)が17増加し、FP(偽陽性)が34減少しました。

**This was a comparison on a set used during development, not an independent evaluation on unseen data.**  
これは開発中に使用されたデータセットでの比較であり、未知データに対する独立した評価ではありません。

**Figure 13: A schematic of postprocessing objectives.**  
図13: 後処理の目的を示す模式図。

**It does not show actual before-and-after predictions or the exact selected parameters.**  
実際の処理前後の予測や、厳密に選択されたパラメータを示しているわけではありません。

**Two submissions at threshold 0.28 both scored 0.971 on the public leaderboard.**  
閾値0.28での2つの提出は、どちらもパブリックリーダーボードで0.971を記録しました。

**The private score was 0.949 with additional postprocessing and 0.947 without it.**  
プライベートスコアは、追加の後処理ありで0.949、なしで0.947でした。

**However, the same postprocessing did not necessarily help after changing the detector.**  
しかしながら、検出器を変更した後は、同じ後処理が必ずしも役立つとは限りませんでした。

**Missing-detection and false-positive patterns can change, making end-to-end evaluation important.**  
検出漏れや偽陽性のパターンが変化する可能性があるため、エンドツーエンドの評価が重要になります。

---

### 5. Why Averaging Motion Predictions Did Not Help
5. なぜ動き予測の平均化が効果をもたらさなかったのか

**I tested motion TTA as well as detection TTA.**  
検出のTTAだけでなく、動きのTTAもテストしました。

**In addition to averaging vectors after transforming them back to the original coordinate system, I tried voting for the parent candidate indicated by each transformed prediction and weighting those votes by distance.**  
元の座標系に逆変換した後にベクトルを平均化することに加え、各変換予測が指し示す親候補に投票し、その投票を距離で重み付けする方法も試しました。

**When different views point toward different parents, their average vector can point between the candidates.**  
異なる視点が異なる親を指している場合、それらの平均ベクトルは候補の中間を指してしまうことがあります。

**If another cell lies near that average endpoint, averaging can favor a connection that the original predictions did not support.**  
もしその平均終点の近くに別の細胞が存在していた場合、平均化によって元の予測が支持していなかった接続が優位になってしまう可能性があります。

**Reducing vector variance is not the same as identifying the correct parent.**  
ベクトルの分散を減らすことは、正しい親を特定することと同義ではありません。

**The final connection also depends on other costs and ILP constraints, so the cell near the average endpoint is not necessarily selected.**  
最終的な接続は他のコストやILP制約にも依存するため、平均終点付近の細胞が必ずしも選択されるわけではありません。

**Figure 14: Predictions from individual transforms point toward A or B, while another cell C lies near their average endpoint.**  
図14: 個々の変換からの予測がAまたはBを指している一方で、別の細胞Cがそれらの平均終点の近くに位置している様子。

**The purple arrow is an averaged motion prediction, not a confirmed link.**  
紫色の矢印は平均化された動き予測であり、確定したリンクではありません。

**If C belongs to a different trajectory, averaging could increase the risk of a wrong connection.**  
もしCが別の軌跡に属している場合、平均化によって誤った接続のリスクが高まる可能性があります。

**The layout is illustrative and does not establish the main cause of the measured regression.**  
この配置は説明用のものであり、測定された性能低下の主な原因を確定的に証明するものではありません。

**I compared these methods on 12 development videos, using 100 frames per video and fixed division-candidate scores.**  
開発用12動画(各動画100フレーム)を用い、分裂候補スコアを固定してこれらの手法を比較しました。

**Mean Edge Jaccard across videos, including the node-count correction, was 0.91931 for the untransformed prediction, 0.91619 for vector averaging, 0.90218 for parent voting, and 0.91433 for distance-weighted voting.**  
ノード数補正を含めた動画全体の平均Edge Jaccardは、無変換の予測で0.91931、ベクトル平均で0.91619、親投票で0.90218、距離重み付き投票で0.91433でした。

**The untransformed motion prediction performed best.**  
無変換の動き予測が最も良好な性能を示しました。

**This was a diagnostic comparison on development data, not an unseen-data evaluation.**  
これは開発データでの診断的比較であり、未知データでの評価ではありません。

**Since these methods did not improve the result, I used untransformed motion predictions.**  
これらの手法は結果を改善しなかったため、変換なしの動き予測を使用しました。

**Detection and temporal association responded differently to prediction averaging.**  
予測の平均化に対して、検出と時間的な対応付けとでは異なる反応を示しました。

---

### 6. Differences Between Public and Private Results
6. パブリック結果とプライベート結果の相違

**Reviewing the submission history after the competition showed that improvements on the public leaderboard did not always translate into private improvements.**  
コンペ終了後に提出履歴を振り返ると、パブリックリーダーボードでの改善が必ずしもプライベートでの改善に結びついていないことが分かりました。

**Figure 15: The 111 scored submissions in chronological order.**  
図15: スコアが付与された111件の提出の時系列推移。

**Submissions without scores, such as failed runs, are excluded.**  
エラー終了などスコアのない提出は除外されています。

**These are repeated submissions to the same evaluation sets, not independent experiments.**  
これらは同一の評価セットに対する繰り返しの提出であり、独立した実験ではありません。

**The development process can be roughly divided into the periods below.**  
開発プロセスはおおまかに以下の期間に分けることができます。

**Dates are UTC.**  
日付はUTCです。

**Scores are public/private pairs from representative configurations, not combinations of separate best scores within each period.**  
スコアは各期間内の個別のベストスコアの組み合わせではなく、代表的な構成におけるパブリック/プライベートのペアです。

---

**Period: Late July–August**  
期間: 7月下旬〜8月  
**Main focus: Detection and motion, eight-view detection TTA, gap filling, coordinate correction**  
主な注力点: 検出と動き、8視点検出TTA、ギャップ補間、座標補正  
**Representative public/private scores: .877/.880 → .909/.910**  
代表的なパブリック/プライベートスコア: .877/.880 → .909/.910  
**Observed pattern: Both improved, with a small gap.**  
観察された傾向: 小さなギャップを保ちながら双方が向上。

**Period: September 7–12**  
期間: 9月7日〜12日  
**Main focus: Division costs, image-based division classification, daughter-pair selection, two-stage optimization**  
主な注力点: 分裂コスト、画像ベースの分裂分類、娘ペア選択、2段階最適化  
**Representative public/private scores: .939/.937 → .957/.956 → .959/.959**  
代表的なパブリック/プライベートスコア: .939/.937 → .957/.956 → .959/.959  
**Observed pattern: Better handling of division coincided with large improvements on both.**  
観察された傾向: 分裂の処理向上に伴い、双方で大幅な改善が見られた。

**Period: September 13–16**  
期間: 9月13日〜16日  
**Main focus: Comparing division classifiers, rejecting incorrect pairs, boundary handling and postprocessing**  
主な注力点: 分裂分類器の比較、誤ったペアの棄却、境界処理と後処理  
**Representative public/private scores: .960/.956**  
代表的なパブリック/プライベートスコア: .960/.956  
**Observed pattern: Some public gains did not carry over to private.**  
観察された傾向: パブリックでの向上の一部がプライベートに反映されなかった。

**Period: September 17–21**  
期間: 9月17日〜21日  
**Main focus: Late-training adjustments without pretraining, detection-threshold search**  
主な注力点: 事前学習なしでの後期学習調整、検出閾値の探索  
**Representative public/private scores: .964/.957 → .966/.956**  
代表的なパブリック/プライベートスコア: .964/.957 → .966/.956  
**Observed pattern: Public improved while private remained roughly flat.**  
観察された傾向: パブリックが改善する一方で、プライベートはほぼ横ばいだった。

**Period: September 21–22**  
期間: 9月21日〜22日  
**Main focus: Switching to a synthetically pretrained detection and motion model**  
主な注力点: 合成データで事前学習した検出・動きモデルへの切り替え  
**Representative public/private scores: Paired comparison: .960/.956 → .966/.947**  
代表的なパブリック/プライベートスコア: 一対比較: .960/.956 → .966/.947  
**Observed pattern: The gap widened substantially.**  
観察された傾向: ギャップが大幅に拡大した。

**Period: September 23–24**  
期間: 9月23日〜24日  
**Main focus: Retraining division classifiers, expanding the set of assessed division candidates**  
主な注力点: 分裂分類器の再学習、評価対象とする分裂候補セットの拡大  
**Representative public/private scores: .969/.947 → .971/.947**  
代表的なパブリック/プライベートスコア: .969/.947 → .971/.947  
**Observed pattern: Public improved; private did not change.**  
観察された傾向: パブリックは向上したが、プライベートは変化しなかった。

**Period: September 26–29**  
期間: 9月26日〜29日  
**Main focus: Division and daughter-pair refinements, optimization and runtime changes, additional postprocessing**  
主な注力点: 分裂および娘ペアの改良、最適化と実行時の変更、追加の後処理  
**Representative public/private scores: Examples: .970/.952 and .971/.949**  
代表的なパブリック/プライベートスコア: 例: .970/.952 および .971/.949  
**Observed pattern: Some configurations recovered part of the private score.**  
観察された傾向: いくつかの構成でプライベートスコアの一部が回復した。

---

**For a late-training model without pretraining, threshold 0.28 gave .963/.959, compared with .966/.956 at threshold 0.32.**  
事前学習なしの後期学習モデルにおいて、閾値0.28は.963/.959を与え、これは閾値0.32での.966/.956と対照的でした。

**There were therefore signs that public and private preferred different settings even before the switch to pretraining.**  
したがって、事前学習へ切り替える前から、パブリックとプライベートで好まれる設定が異なっている兆候がありました。

**This timeline is descriptive; it does not attribute each period's changes to a single factor.**  
このタイムラインは経緯を記述したものであり、各期間の変化を単一の要因に帰するものではありません。

**Early changes associated with large private improvements included adjusting division costs and explicitly modeling daughter pairs in graph selection.**  
プライベートでの大幅な改善に関連した初期の変更には、分裂コストの調整や、グラフ選択における娘ペアの明示的なモデル化が含まれていました。

**In contrast, later changes that increased public scores, such as pretraining and expanded division assessment, did not produce corresponding private gains in the paired comparisons.**  
対照的に、事前学習や分裂評価の対象拡大など、パブリックスコアを向上させた後期の変更は、一対比較においてそれに見合うプライベートの改善をもたらしませんでした。

**Figure 16: Comparisons between submissions before and after changes.**  
図16: 変更前後の提出スコアの比較。

**Adjusting division costs increased private from 0.910 to 0.937; a change including daughter-pair modeling and two-stage optimization increased it from 0.942 to 0.956.**  
分裂コストの調整によりプライベートスコアは0.910から0.937へと向上し、娘ペアのモデル化と2段階最適化を含む変更によって0.942から0.956へと向上しました。

**Comparisons changing multiple components, including the latter, cannot isolate the contribution of a single component.**  
後者を含め、複数のコンポーネントを変更した比較では、単一コンポーネントの寄与を切り分けることはできません。

---

### 7. Closing Remarks
7. おわりに

**The main focus of this solution was how individual cell and division candidates are selected within a complete lineage.**  
この解法の主眼は、完全な系譜の中で個々の細胞および分裂の候補がどのように選択されるかにありました。

**I combined detection training that accounts for sparse annotations, temporal-interval augmentation, dedicated division models, globally consistent graph selection, and postprocessing tailored to observed errors.**  
疎なアノテーションを考慮した検出学習、時間間隔データ拡張、専用の分裂モデル、大域的に一貫したグラフ選択、そして観察されたエラーに合わせた後処理を組み合わせました。

**I also found that adding models, averaging predictions, and improving the public score did not necessarily improve performance on unseen data.**  
また、モデルの追加、予測の平均化、パブリックスコアの向上が、必ずしも未知データに対する性能向上に直結しないことも分かりました。

**It was important to inspect detection, continuation, and division separately, then evaluate the actual output graph end to end.**  
検出、継続、分裂を個別に精査し、その上で実際に出力されるグラフをエンドツーエンドで評価することが極めて重要でした。

---

### References
参考文献

**Isensee et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18, 203–211. [Paper](https://doi.org/10.1038/s41592-020-01008-z) · [Code](https://github.com/MIC-DKFZ/nnUNet)**  
Isenseeら (2021). nnU-Net: 深層学習ベースの生体医用画像セグメンテーションのための自己設定型手法. Nature Methods, 18, 203–211. [論文](https://doi.org/10.1038/s41592-020-01008-z) · [コード](https://github.com/MIC-DKFZ/nnUNet)

**Isensee et al. (2024). nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation. [Paper](https://arxiv.org/abs/2404.09556). This work introduces the residual encoder presets, including ResEncM used in this solution.**  
Isenseeら (2024). nnU-Net再考: 3D医用画像セグメンテーションにおける厳密な検証の提唱. [論文](https://arxiv.org/abs/2404.09556)。本研究では、本解法で使用されたResEncMを含む残差エンコーダプリセットが導入されています。

**José Freitas's [synthetic-data discussion](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103) and [generation notebook](https://www.kaggle.com/code/josefreitasalvesneto/biohub-synthetic-dataset).**  
José Freitas氏の[合成データに関するディスカッション](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103)および[生成ノートブック](https://www.kaggle.com/code/josefreitasalvesneto/biohub-synthetic-dataset)。

**Figures labeled “schematic” illustrate the underlying ideas.**  
「schematic(模式図)」と記された図は、根底にあるアイデアを図解したものです。

**Figures showing actual images or evaluation results specify the corresponding data and comparison conditions in their captions.**  
実際の画像や評価結果を示す図については、キャプションに対応するデータと実験比較条件を明記しています。

---

**Author [tatsutaka](https://www.kaggle.com/tatsutaka) tatsutaka**  
著者: [tatsutaka](https://www.kaggle.com/tatsutaka) tatsutaka

**Share**  
共有

**Citation**  
引用

**tatsutaka. 7th Place Solution. https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/7th-place-solution. 2026. Kaggle**  
tatsutaka. 7th Place Solution. https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/7th-place-solution. 2026. Kaggle
