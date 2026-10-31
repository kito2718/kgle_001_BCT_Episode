### タイトル・メタ情報

**3rd Place Solution**  
3位解法  

**Cell Tracking with 2.5D/3D Ensembles and Lineage Graph Optimization**  
2.5D/3Dアンサンブルと系譜グラフ最適化による細胞追跡  

**[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) Solution Writeup · 3rd place · Sep 30, 2026**  
[Biohub - Cell Tracking During Development] 解法まとめ・3位・2026年9月30日  

**Thank you to the organizers and to everyone who shared discussions and notebooks.**  
主催者の皆様、そしてディスカッションやノートブックを共有してくださったすべての方々に感謝いたします。  

**Our solution focused on detecting cells from sparse annotations and combining motion and division evidence into a consistent lineage graph.**  
私たちの解法は、疎(スパース)なアノテーションから細胞を検出し、運動と分裂のエビデンスを統合して一貫した系譜グラフを構築することに注力しました。  

---

### 1. Overview (概要)

**1. Overview**  
1. 概要  

**Our pipeline has six stages:**  
私たちのパイプラインは6つのステージで構成されています:  

- **Cell detection: ensemble heatmaps from 2.5D U-Nets and a 3D SegResNet to detect cell centers.**  
  細胞検出: 2.5D U-Net群と3D SegResNetからのヒートマップをアンサンブルし、細胞中心を検出します。  
- **Dense flow: estimate a 3D displacement field between consecutive frames.**  
  デンスフロー: 連続するフレーム間の3次元変位場を推定します。  
- **Cell matching: score candidate correspondences using image features and positions corrected for motion.**  
  細胞マッチング: 画像特徴量と運動補正後の位置を用いて、対応候補のスコアを算出します。  
- **Division identification: identify dividing parents from original and motion-aligned images, with auxiliary models for the stages before and after division.**  
  分裂同定: 元画像および運動アライメント済み画像から分裂親細胞を同定し、分裂前後のステージに対する補助モデルも併用します。  
- **Lineage graph optimization: jointly select ordinary links and division events.**  
  系譜グラフ最適化: 通常の接続リンクと分裂イベントを同時に選択します。  
- **Post-processing: fill short gaps, remove small components, and refine node coordinates.**  
  後処理: 短いギャップを埋め、小さな連結成分を除去し、ノード座標を微小調整します。  

**We put particular effort into detection accuracy because every downstream stage depends on the detected cells.**  
後続のすべてのステージが検出された細胞に依存するため、私たちは特に検出精度に力を注ぎました。  

**For tracking, we trained the flow, matching, and division models separately, then combined their outputs through graph optimization to determine the connections between cells.**  
トラッキングに関しては、フローモデル、マッチングモデル、分裂モデルを個別に学習させ、それらの出力をグラフ最適化によって統合して細胞間の接続を決定しました。  

**Our final submission used an ensemble of relatively large detection backbones, and detection accounted for 55% of the total runtime.**  
最終提出では比較的大規模な検出バックボーンのアンサンブルを採用し、検出処理が総実行時間の55%を占めました。  

**Our final solution achieved a CV score of 0.977801 (0.540107), a public leaderboard score of 0.977 (0.52), and a private leaderboard score of 0.967 (0.47).**  
私たちの最終解法は、CVスコア0.977801 (0.540107)、パブリックLBスコア0.977 (0.52)、プライベートLBスコア0.967 (0.47)を達成しました。  

**Values in parentheses indicate Division Jaccard.**  
カッコ内の数値は分裂Jaccard(Division Jaccard)を表しています。  

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F745525%2F11838c91b2611653d8c246657fd05b02%2Fpipeline.png?generation=1790727945719645&alt=media)

---

### 2. Data and Validation (データと検証)

**2. Data and Validation**  
2. データと検証  

**The training set contains 199 videos cropped from two embryos.**  
トレーニングセットには、2つの胚から切り出された199本の動画が含まれています。  

**Each video has 100 frames of 64 × 256 × 256 voxels.**  
各動画は64×256×256ボクセルのフレームが100フレームで構成されています。  

**Voxel spacing is 1.625 µm along z and 0.40625 µm along x and y, so we compute distances in physical coordinates.**  
ボクセル間隔はz方向に1.625 µm、xおよびy方向に0.40625 µmであるため、距離は物理座標系で計算します。  

**The main challenge is that the annotations cover only a small fraction of the cells.**  
主な課題は、アノテーションが細胞全体のごく一部しかカバーしていない点です。  

**There are approximately 133,000 annotated nodes, about 2.8% of the total estimated by the organizers.**  
アノテーションされたノードは約133,000個であり、これは主催者が推定した全体の約2.8%にすぎません。  

**Many clearly visible cells have no annotation, and the training set contains only 151 annotated divisions.**  
明らかに視認できる多くの細胞にアノテーションが付いておらず、トレーニングセットに含まれるアノテーション済み分裂はわずか151件でした。  

**This sparsity also matters for evaluation.**  
この疎(スパース)さは評価指標においても重要になります。  

**The official score is the sum of the node-count-adjusted edge Jaccard and 0.1 times the division Jaccard.**  
公式スコアは、ノード数調整済みエッジJaccardと、分裂Jaccardの0.1倍の和です。  

**Links entirely within unannotated regions are not automatically false positives; incorrect links that compete with annotated connections are penalized.**  
未アノテーション領域内に完全に収まるリンクは自動的に偽陽性(False Positive)とされるわけではなく、アノテーション済み接続と競合する誤ったリンクがペナルティを受けます。  

**A separate adjustment accounts for the number of predicted nodes.**  
予測ノード数に応じた個別の調整も行われます。  

**Detection precision or local link classification accuracy alone therefore does not describe the final score.**  
したがって、検出精度や局所的なリンク分類精度だけでは最終スコアを説明できません。  

**We used five-fold cross-validation at the video level, distributing videos from both embryos across the folds and keeping all frames of a video together.**  
私たちは動画単位での5分割交差検証(5-fold CV)を採用し、同一動画の全フレームを同じフォールドに保持した上で、両方の胚からの動画を各フォールドに分散させました。  

**We generated out-of-fold (OOF) predictions for the detectors, matcher, and division models.**  
検出器、マッチャー、および分裂モデルに対してOut-of-Fold (OOF)予測を生成しました。  

**The downstream calibration models were also trained with the evaluation fold excluded.**  
下流のキャリブレーションモデルも、評価対象のフォールドを除外して学習させました。  

**Both test embryos are different from the training embryos.**  
テスト対象の2つの胚は、どちらもトレーニング用の胚とは異なります。  

**We considered training on one embryo and validating on the other, but used video-level five-fold CV for model selection based on the improvement trends observed in CV and on the public leaderboard.**  
一方の胚で学習し他方の胚で検証することも検討しましたが、CVおよびパブリックLBで観察された改善傾向に基づき、モデル選定には動画レベルの5分割CVを使用しました。  

**Crops from the same embryo can overlap in space and time across folds, so this validation is not fully independent.**  
同一胚からの切り出し領域はフォールド間で空間的・時間的に重複し得るため、この検証は完全に独立しているわけではありません。  

**Agreement with the public leaderboard also does not establish generalization to the private test embryo.**  
また、パブリックLBとの一致がプライベートテストの胚への汎化を保証するものでもありません。  

**The four embryos are summarized below.**  
4つの胚の概要は以下の通りです。  

**Training prefixes and video counts come from the provided data; test prefixes, counts, density ranges, and public/private split assignments were identified through probing.**  
トレーニング用のプレフィックスと動画数は提供データに基づきます。テスト用のプレフィックス、本数、密度範囲、およびパブリック/プライベートの割り当てはプロービング(探針)によって特定しました。  

| Embryo prefix | Videos | Split | Density p10 | Density p50 (median) | Density p90 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **44b6** | 71 | Train | 134 | 355 | 627 |
| **6bba** | 128 | Train | 54 | 114 | 445 |
| **fdad** | 60 | Public test | [40, 60) | [160, 260) | [260, 420) |
| **ea36** | 106 | Private test | [40, 60) | [60, 100) | [160, 260) |

**Here, density means detections per frame, used as a proxy for the number of cells.**  
ここで密度とはフレームあたりの検出数を意味し、細胞数の代替指標として使用しています。  

**Using the same early version of our detector for all videos, we counted detections in four evenly spaced frames per video, averaged the counts, and computed percentiles across videos.**  
すべての動画に対して同じ初期バージョンの検出器を用い、動画ごとに等間隔の4フレームで検出数をカウントして平均し、動画全体でのパーセンタイルを計算しました。  

**Training values are rounded measurements, while test values are intervals inferred from probing.**  
トレーニングの値は丸められた実測値であり、テストの値はプロービングから推測された区間です。  

**The public and private test embryos differ substantially in cell density.**  
パブリックテストとプライベートテストの胚では、細胞密度が大きく異なっています。  

---

### 3. Cell Detection (細胞検出)

**3. Cell Detection**  
3. 細胞検出  

#### Backbones (バックボーン)

**Backbones**  
バックボーン  

**We combined 2.5D and fully 3D detectors.**  
私たちは2.5D検出器と完全3D検出器を組み合わせました。  

**For the 2.5D models, we reused the backbone architecture from our [CZII - CryoET Object Identification solution](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/writeups/yu4u-tattaka-4th-place-solution-source-codes-submi).**  
2.5Dモデルには、私たちの[CZII - CryoET Object Identification解法]のバックボーン構造を再利用しました。  

**A pretrained 2D encoder extracts features from z slices, features are pooled along z at each stage, and a 3D decoder combines them.**  
事前学習済みの2Dエンコーダが各zスライスから特徴量を抽出し、各ステージでz方向に特徴量をプーリングし、3Dデコーダがそれらを結合します。  

**We used EfficientNetV2-L and EfficientNet-B7 encoders.**  
エンコーダにはEfficientNetV2-LとEfficientNet-B7を使用しました。  

**Each encoder input stacks three neighboring z slices as channels, allowing the first layer to use local depth information.**  
エンコーダの各入力は隣接する3つのzスライスをチャンネルとしてスタックし、第1層から局所的な深度情報を利用できるようにしています。  

**These slices provide spatial context only; temporal information is introduced later in the tracking pipeline.**  
これらのスライスは空間的なコンテキストのみを提供し、時間情報は後のトラッキングパイプラインで導入されます。  

**The fully 3D detector uses MONAI's SegResNet implementation, a residual encoder-decoder built from 3D convolutions.**  
完全3D検出器には、3D畳み込みで構築された残差エンコーダ・デコーダであるMONAIのSegResNet実装を使用しました。  

**We used 32 initial channels, four encoder stages with one residual block per stage, and Group Normalization.**  
初期チャンネル数は32、エンコーダステージは4つ(各ステージに残差ブロック1つ)、そしてGroup Normalizationを採用しました。  

**The submission used the following ensemble:**  
提出モデルでは以下のアンサンブルを使用しました:  

| Detector | Architecture | Heatmap weight |
| :--- | :--- | :--- |
| **EfficientNetV2-L** | 2.5D U-Net | 0.3 |
| **EfficientNet-B7** | 2.5D U-Net | 0.4 |
| **SegResNet (MONAI)** | 3D residual encoder-decoder | 0.3 |

#### Training with sparse annotations (疎なアノテーションでの学習)

**Training with sparse annotations**  
疎なアノテーションでの学習  

**The target heatmap places a Gaussian with a standard deviation of 2 µm at each annotated center.**  
ターゲットヒートマップは、アノテーションされた各中心に標準偏差2 µmのガウス分布を配置します。  

**Applying MSE directly to this target would also train the detector to predict zero at unannotated cells.**  
このターゲットに直接MSEを適用すると、アノテーションされていない細胞に対してもゼロを出力するように検出器が学習されてしまいます。  

**To avoid treating these cells as negatives, we extracted candidate centers using Difference of Gaussians (DoG) with a low threshold.**  
これらの細胞を負例として扱ってしまうのを避けるため、低めの閾値でDifference of Gaussians (DoG)を用いて中心候補を抽出しました。  

**We excluded regions within 6 µm of unmatched candidates from the loss.**  
アノテーションと一致しなかった候補から6 µm以内の領域を損失計算から除外しました。  

**DoG candidates within 2 µm of an annotated center were treated as duplicates and did not create exclusion regions.**  
アノテーション中心から2 µm以内にあるDoG候補は重複とみなし、除外領域を作成しませんでした。  

**Positive regions around annotated centers remained supervised even when they overlapped an exclusion region.**  
アノテーション中心の周囲の正例領域は、除外領域と重複した場合でも教師あり学習の対象のままとしました。  

**DoG thus identified uncertain background regions to mask, without adding positive pseudo-labels.**  
このようにしてDoGは、正例の疑似ラベル(pseudo-labels)を追加することなく、マスクすべき不確実な背景領域を特定しました。  

**We averaged MSE separately over positive and background regions, with a background weight of 0.5.**  
MSEは正例領域と背景領域で個別に平均を取り、背景の重みを0.5としました。  

**This prevents the much larger background volume from overwhelming the supervision around cell centers.**  
これにより、はるかに体積の大きい背景が細胞中心周囲の教師信号を圧倒してしまうのを防ぎます。  

**Image intensities were normalized using the 0.1st and 99.9th percentiles of each video.**  
画像の輝度値は、各動画の0.1パーセンタイルおよび99.9パーセンタイルを用いて正規化しました。  

**Training crops were 48 × 192 × 192 voxels.**  
学習用のクロップサイズは48×192×192ボクセルとしました。  

**Augmentations included flips and rotations in the xy plane, intensity changes, and blur.**  
データ拡張(Augmentation)には、xy平面での反転や回転、輝度変化、ブラー(ぼかし)を含めました。  

**Scaling and rotation augmentations degraded performance in our experiments.**  
私たちの実験では、スケーリングや(3Dでの)回転拡張は性能を低下させました。  

#### Inference (推論)

**Inference**  
推論  

**For each detector, we averaged heatmaps from the five fold models and used two test-time views: the original image and an image flipped along both x and y.**  
各検出器について、5つのフォールドモデルからのヒートマップを平均し、オリジナル画像とx・y両軸で反転させた画像の2つのTest-Time View(TTA)を使用しました。  

**We combined the detector heatmaps with the weights above, then extracted centers using non-maximum suppression with a (3, 7, 7) kernel and a threshold of 0.2.**  
上記の重みで検出器のヒートマップを合成した後、(3, 7, 7)カーネルおよび閾値0.2のNon-Maximum Suppression (NMS)を用いて中心を抽出しました。  

#### A0: Detection with geometric linking (幾何的リンクによるベースライン)

**A0: Detection with geometric linking**  
A0: 幾何的リンクを伴う検出  

**The baseline in Table 1 links detections in consecutive frames using physical distance and a one-to-one Hungarian assignment with a 7 µm distance gate.**  
表1のベースラインは、物理距離と7 µmの距離ゲートを用いた1対1のハンガリアン法により、連続するフレーム間の検出同士をリンクします。  

**We estimate collective translation, or drift, from the median displacement of matched points and repeat drift correction and assignment twice.**  
マッチした点の中央値変位から集団的な並進移動(ドリフト)を推定し、ドリフト補正と割り当てを2回繰り返します。  

**This baseline uses no dense flow, learned matcher, division predictions, or post-processing.**  
このベースラインでは、デンスフロー、学習ベースのマッチャー、分裂予測、後処理は一切使用していません。  

**Simple linking provides a starting point for evaluation: detections alone cannot produce true positive edges, while ordinary links can earn an edge score without predicting divisions.**  
単純なリンク処理は評価の出発点となります。検出だけでは真陽性のエッジを生成できませんが、通常のリンクであれば分裂を予測しなくてもエッジスコアを獲得できるためです。  

---

### 4. Dense Flow and Cell Matching (デンスフローと細胞マッチング)

**4. Dense Flow and Cell Matching**  
4. デンスフローと細胞マッチング  

#### A1: Dense flow (デンスフロー)

**A1: Dense flow**  
A1: デンスフロー  

**Cells move both individually and collectively.**  
細胞は個別にも集団的にも運動します。  

**We estimate flow from consecutive 3D images to predict where each cell will move in the next frame.**  
連続する3D画像からフローを推定し、各細胞が次のフレームでどこに移動するかを予測します。  

**Downsampling by (z, y, x) = (1, 4, 4) gives an isotropic grid with 1.625 µm spacing.**  
(z, y, x) = (1, 4, 4)でダウンサンプリングすることで、1.625 µm間隔の等方性グリッドが得られます。  

**A small 3D encoder-decoder operates on this grid.**  
小型の3Dエンコーダ・デコーダがこのグリッド上で動作します。  

**Local correlations and a soft-argmax produce an initial displacement, which the network refines with a residual prediction.**  
局所相関とsoft-argmaxにより初期変位を生成し、ネットワークが残差予測によってそれを洗練させます。  

**The encoder uses 32, 64, and 128 channels without normalization layers.**  
エンコーダは正規化層を持たず、32、64、128チャンネルを使用します。  

**On real image pairs, training combines image and local feature consistency after warping, forward-backward consistency, and smoothness losses.**  
実画像ペアでの学習では、ワーピング後の画像および局所特徴量の一致性、前方・後方(forward-backward)の一致性、および平滑性損失を組み合わせます。  

**We also directly supervise displacement on synthetic pairs generated with known translations and smooth deformations.**  
また、既知の並進移動と滑らかな変形によって生成された合成ペアを用いて、変位を直接教師あり学習させました。  

**Annotated links are used for endpoint-error evaluation and checkpoint selection, but not in the training loss.**  
アノテーションされたリンクはエンドポイント誤差の評価やチェックポイントの選定に使用されますが、学習損失には含まれません。  

**The estimated flow supports both correspondence search and alignment of the images supplied to the division model.**  
推定されたフローは、対応関係の探索と、分裂モデルに入力される画像のアライメントの両方をサポートします。  

**In A1, we add dense flow to the source coordinates, estimate residual drift twice, and retain the same 7 µm gate and one-to-one assignment as A0.**  
A1では、元の座標にデンスフローを加算し、残差ドリフトを2回推定した上で、A0と同じ7 µmゲートと1対1割り当てを維持します。  

**This measures the effect of motion compensation before introducing division modeling.**  
これにより、分裂モデリングを導入する前の運動補正の効果を測定できます。  

#### A2: Cell matching (細胞マッチング)

**A2: Cell matching**  
A2: 細胞マッチング  

**The matcher takes two consecutive frames and their detections, and scores which cells correspond across time.**  
マッチャーは連続する2フレームとその検出結果を入力とし、時間軸を跨いでどの細胞同士が対応するかをスコアリングします。  

**It uses appearance and positions corrected by flow to predict candidate correspondence scores and a null score representing the absence of a match.**  
外見特徴とフローで補正された位置情報を用いて、対応候補スコアと「一致なし」を表すnullスコアを予測します。  

**These scores narrow the candidate set; graph optimization later selects the final connections.**  
これらのスコアにより候補セットを絞り込み、後段のグラフ最適化が最終的な接続を選択します。  

**The matcher has its own 2.5D U-Net with an EfficientNetV2-S encoder, separate from the detectors.**  
マッチャーは検出器とは独立した、EfficientNetV2-Sエンコーダを持つ独自の2.5D U-Netを備えています。  

**It uses the same (1, 4, 4) downsampling as the flow model.**  
フローモデルと同じ(1, 4, 4)のダウンサンプリングを使用します。  

**We sample 64-channel features at each detection from an intermediate decoder stage, avoiding computation of the full-resolution decoder output.**  
フル解像度のデコーダ出力の計算を回避するため、中間デコーダステージから各検出位置で64チャンネルの特徴量をサンプリングします。  

**Self-attention and cross-attention process cell features and positional information.**  
Self-attentionとCross-attentionが細胞特徴量と位置情報を処理します。  

**The matching head refines scores based on appearance cosine similarity and uses role embeddings to distinguish the two frames.**  
マッチングヘッドは外見のコサイン類似度に基づいてスコアを洗練させ、2つのフレームを区別するためにロール埋め込み(role embeddings)を使用します。  

**It also predicts the null option.**  
また、nullの選択肢(マッチなし)も予測します。  

**For training, detections matched to annotated cells act as anchors, and annotated edges provide targets for bidirectional cross-entropy.**  
学習時には、アノテーションされた細胞に一致する検出がアンカーとして機能し、アノテーションされたエッジが双方向クロスエントロピーのターゲットを提供します。  

**Unannotated cells remain competing candidates, but absence of annotation does not make them null targets.**  
アノテーションされていない細胞は競合する候補のままであり、アノテーションがないからといってnullターゲットとして扱われることはありません。  

**For a dividing parent, we construct a target for each daughter and exclude the other daughter from the negative set.**  
分裂親細胞に対しては各娘細胞向けにターゲットを構築し、もう一方の娘細胞を負例セットから除外します。  

**For ordinary links, we retain candidates among the matcher's top five probabilities whose distance after flow compensation is at most 12 µm.**  
通常のリンクについては、マッチャーの確率上位5位以内の候補のうち、フロー補正後の距離が12 µm以下のものを保持します。  

**We pass their ranks, forward and reverse probabilities, and margins over competing candidates to the next stage.**  
それらのランク、順方向・逆方向の確率、および競合候補に対するマージンを次のステージへ渡します。  

**In A2, we use these candidates and scores in a Hungarian assignment.**  
A2では、これらの候補とスコアをハンガリアン法による割り当てに使用します。  

**A0 and A1 consider all pairs within 7 µm after drift correction; A2 applies the same gate to the matcher candidates.**  
A0とA1はドリフト補正後に7 µm以内にあるすべてのペアを考慮しますが、A2はマッチャーの候補に対して同じゲートを適用します。  

**The A1-to-A2 change therefore includes candidate filtering.**  
したがって、A1からA2への変更には候補のフィルタリングが含まれています。  

**In the complete pipeline, joint graph optimization replaces this local assignment and uses candidates within the 12 µm flow distance gate.**  
完全なパイプラインでは、この局所的な割り当てが結合グラフ最適化に置き換わり、12 µmフロー距離ゲート内の候補が使用されます。  

---

### 5. Division Identification (A3) (分裂同定)

**5. Division Identification (A3)**  
5. 分裂同定 (A3)  

**The division model predicts whether a detected cell will split into two daughters in the next frame, using 3D images from neighboring time points.**  
分裂モデルは、近隣時点の3D画像を用いて、検出された細胞が次のフレームで2つの娘細胞に分裂するかどうかを予測します。  

**We call this the parent model.**  
これを「親モデル(parent model)」と呼びます。  

**Its output is combined with matching scores and daughter geometry, and graph optimization chooses the parent-to-daughter connections.**  
その出力はマッチングスコアや娘細胞の幾何情報と組み合わされ、グラフ最適化が親から娘への接続を選択します。  

**Both division and ordinary motion change the appearance of a local image region.**  
分裂と通常の運動の双方が、局所的な画像領域の外見を変化させます。  

**To help distinguish them, we align the neighboring frames to the central frame before passing them to the model.**  
これらを区別しやすくするため、近隣フレームを中心に位置するフレームへとアライメント(位置合わせ)してからモデルに入力します。  

**We also retain the original images because a single displacement field cannot fully represent a one-to-two split.**  
また、単一の変位場では1から2への分裂を完全に表現できないため、元の画像もそのまま保持します。  

**The main parent model receives five volumes from three time points:**  
メインの親モデルは、3つの時点から得られる5つのボリュームを受け取ります:  

**[original t−1, t−1 aligned to t, t, t+1 aligned to t, original t+1]**  
[元のt−1, tへアライメントされたt−1, t, tへアライメントされたt+1, 元のt+1]  

**The division model uses (1, 2, 2) downsampling.**  
分裂モデルは(1, 2, 2)のダウンサンプリングを使用します。  

**When converting flow to this grid, we account for both displacement units and voxel-center offsets introduced by pooling.**  
フローをこのグリッドに変換する際、変位単位とプーリングによって生じるボクセル中心のオフセットの両方を考慮します。  

**We sample the next frame using forward flow.**  
次のフレームは前方フロー(forward flow)を用いてサンプリングします。  

**For the previous frame, we approximate the inverse displacement by negating the preceding forward flow field.**  
前のフレームについては、直前の前方フロー場を反転(符号反転)させることで逆変位を近似します。  

**A shared EfficientNetV2-S encoder processes each input.**  
共有のEfficientNetV2-Sエンコーダが各入力を処理します。  

**Features from corresponding stages are concatenated in input order and passed to a 3D decoder with Group Normalization.**  
対応するステージからの特徴量を入力順に連結し、Group Normalizationを備えた3Dデコーダに渡します。  

**The decoder predicts a parent score at each detection.**  
デコーダは各検出位置における親スコアを予測します。  

**The flow model remains frozen.**  
フローモデルはフリーズ(重み固定)されたままです。  

**Positive examples are annotated parents with two children in the next frame.**  
正例は、次のフレームに2つの子細胞を持つアノテーション済みの親細胞です。  

**Negatives include ordinary continuing cells and daughters immediately after division.**  
負例には、通常の継続細胞や分裂直後の娘細胞が含まれます。  

**Detections without a GT match do not enter the negative loss.**  
GT(正解)と一致しない検出は負例損失には入りません。  

**We average positive and negative BCE separately.**  
正例と負例のBCE(二値交差エントロピー)は別々に平均を取ります。  

**During training, 25% of samples come from frames containing positive parents, 25% from hard-negative frames around divisions, and the remainder from ordinary sampling.**  
学習中、サンプルの25%は正例の親を含むフレームから、25%は分裂周辺のハードネガティブフレームから、残りは通常のサンプリングから抽出されます。  

**We use AdamW with a learning rate of 1e-4, 20 epochs, and EMA decay of 0.995.**  
オプティマイザにはAdamWを使用し、学習率1e-4、20エポック、EMA減衰率0.995としました。  

**We also use independent pre and post models.**  
また、独立したpreモデルとpostモデルも使用します。  

**For a division with the parent at time t and daughters at t+1, pre identifies the precursor at t−1, while post identifies the daughters at t+1.**  
時刻tの親とt+1の娘細胞からなる分裂に対し、preモデルはt−1の前駆体を同定し、postモデルはt+1の娘細胞を同定します。  

**Each model receives three original frames centered on the cell's evaluation time.**  
各モデルは、細胞の評価時刻を中心とした3つの元フレームを受け取ります。  

**Their scores provide additional evidence for division candidates.**  
これらのスコアは分裂候補に対する追加のエビデンスを提供します。  

**The submission uses the parent model with both original and aligned images to build the primary division candidate pool.**  
提出解法では、元画像とアライメント済み画像の両方を用いた親モデルを使用して、一次分裂候補プールを構築します。  

**A second parent model takes only three original frames.**  
2つ目の親モデルは3つの元フレームのみを受け取ります。  

**Its scores, together with pre and post scores, support ordinary-link calibration and an additional division candidate pool with a 16 µm radius.**  
そのスコアはpreおよびpostスコアとともに、通常のリンクのキャリブレーションおよび半径16 µmの追加分裂候補プールを支援します。  

**The two division pools are calibrated separately.**  
これら2つの分裂プールは個別にキャリブレーションされます。  

**Division inference averages five fold models and four rotations in the xy plane; OOF evaluation uses the corresponding held-out model for each video.**  
分裂推論では、5つのフォールドモデルとxy平面での4回転を平均化します。OOF評価には各動画に対応するホールドアウトモデルを使用します。  

---

### 6. Lineage Graph Optimization (A3) (系譜グラフ最適化)

**6. Lineage Graph Optimization (A3)**  
6. 系譜グラフ最適化 (A3)  

#### Calibrating candidate scores (候補スコアのキャリブレーション)

**Calibrating candidate scores**  
候補スコアのキャリブレーション  

**We combine distances and model outputs to estimate probabilities for ordinary links and division candidates.**  
距離とモデル出力を統合し、通常のリンクおよび分裂候補の確率を推定します。  

**These probabilities let the graph optimizer compare competing choices.**  
これらの確率により、グラフ最適化アルゴリズムが競合する選択肢を比較できるようになります。  

**Candidates are either ordinary links or division events consisting of one parent and two daughters.**  
候補は、通常のリンク、または1つの親と2つの娘からなる分裂イベントのいずれかです。  

**A parent score threshold of 0.1 controls division candidate generation; final selection is handled by the optimizer.**  
親スコアの閾値0.1によって分裂候補の生成を制御し、最終的な選択は最適化器が担当します。  

**Calibration features include distance, bidirectional matching scores, candidate ranks, margins, detection confidence, and parent/pre/post scores.**  
キャリブレーション用の特徴量には、距離、双方向マッチングスコア、候補ランク、マージン、検出信頼度、および親/pre/postスコアが含まれます。  

**We aggregate pre evidence over several possible predecessors.**  
いくつかの考えられる前駆細胞にわたってpreモデルのエビデンスを集約します。  

**Examples include the maximum of the pre score multiplied by the matching probability, and an average weighted by matching probabilities.**  
例として、preスコアにマッチング確率を乗じたものの最大値や、マッチング確率で重み付けした平均値などが挙げられます。  

**To handle sparse annotations, we estimate two probabilities:**  
疎なアノテーションに対処するため、私たちは2つの確率を推定します:  

- **Probability of being evaluated: whether a candidate is subject to scoring given the available annotations.**  
  評価対象となる確率: 利用可能なアノテーションを前提とした場合、その候補が採点対象となるかどうか。  
- **Conditional probability of being correct: whether it is a true positive, given that it is evaluated.**  
  正解である条件付き確率: 評価対象であるという条件下で、それが真陽性(True Positive)であるかどうか。  

**Writing these as a and q, we use a × q as the expected true-positive contribution and a × (1 − q) as the expected false-positive contribution.**  
これらをaおよびqと表記すると、a × qを期待真陽性寄与度として、a × (1 − q)を期待偽陽性寄与度として使用します。  

**This avoids treating all unevaluated candidates as errors during training.**  
これにより、学習中に未評価の候補すべてを誤りとして扱ってしまう事態を防ぎます。  

**Ordinary links use logistic regression.**  
通常のリンクにはロジスティック回帰を使用します。  

**For the conditional division probability, we train LightGBM and XGBoost from a linear model's initial logit and average their predicted probabilities with equal weights.**  
条件付き分裂確率については、線形モデルの初期ロジットからLightGBMとXGBoostを学習させ、それらの予測確率を等しい重みで平均化します。  

**The primary and additional division pools have separate models.**  
一次分裂プールと追加分裂プールには別々のモデルを用意します。  

**Features include aggregated predecessor evidence, motion, and daughter separation.**  
特徴量には、集約された前駆エビデンス、運動量、娘細胞間の距離が含まれます。  

**We retain the LightGBM probabilities for constructing the initial graph and use the averaged probabilities when evaluating choices during optimization.**  
初期グラフの構築にはLightGBMの確率を保持し、最適化中に選択肢を評価する際には平均化された確率を使用します。  

**Division calibration targets require exact matches to the parent and both daughters, which differs from the official metric's local lineage criterion.**  
分裂キャリブレーションのターゲットには、親および両方の娘細胞との完全一致が求められますが、これは公式評価指標の局所系譜基準とは異なります。  

#### Selecting links and divisions jointly (リンクと分裂の同時選択)

**Selecting links and divisions jointly**  
リンクと分裂の同時選択  

**If ordinary links are fixed first, a correct daughter may be assigned to another parent, preventing a valid division from being added later.**  
通常のリンクを先に固定してしまうと、正しい娘細胞が別の親に割り当てられてしまい、後から妥当な分裂を追加できなくなる可能性があります。  

**We therefore jointly choose which nodes to retain, which ordinary links to use, and which daughter pairs to select as divisions.**  
そのため、保持するノード、使用する通常のリンク、そして分裂として選択する娘ペアを同時に選択します。  

**The main constraints are:**  
主な制約条件は以下の通りです:  

- **Each cell has at most one parent.**  
  各細胞が持つ親は高々1つであること。  
- **Each cell has at most one outgoing event: an ordinary link or a division.**  
  各細胞からの出力イベントは高々1つ(通常のリンクまたは分裂)であること。  
- **A division selects both daughters together.**  
  分裂は両方の娘細胞を同時に選択すること。  
- **A dividing parent must have an ordinary incoming link, and each daughter must have an ordinary outgoing link.**  
  分裂親細胞は通常の入力リンクを持たなければならず、各娘細胞は通常の出力リンクを持たなければならないこと。  
- **A daughter cannot divide again at the immediately following time boundary.**  
  娘細胞はその直後の時間境界で再び分裂することはできないこと。  

**The last two constraints reflect the temporal resolution of this dataset.**  
最後の2つの制約は、このデータセットの時間解像度を反映したものです。  

**They also prevent implausible divisions from supporting one another and strengthen the LP relaxation.**  
これらはまた、あり得ない分裂同士が相互に支持し合うのを防ぎ、線形計画(LP)緩和を強化します。  

**The objective is a surrogate of the competition metric computed from expected true positives, false positives, and node counts.**  
目的関数は、期待される真陽性数、偽陽性数、およびノード数から計算されるコンペ評価指標の代理(サロゲート)関数です。  

**We linearize it around the current solution to obtain rewards for true positives and costs for false positives and nodes.**  
現在の解の周囲でこれを線形化し、真陽性に対する報酬と、偽陽性およびノードに対するコストを取得します。  

**We optimize each video using these coefficients, aggregate expected counts across videos, and update the coefficients for up to three rounds.**  
これらの係数を用いて各動画を最適化し、動画全体で期待カウント数を集約した上で、最大3ラウンドまで係数を更新します。  

**We accept an updated solution only if it improves the surrogate objective.**  
代理目的関数を改善する場合にのみ、更新された解を採用します。  

**We use HiGHS, first solving the linear programming (LP) relaxation.**  
ソルバーにはHiGHSを使用し、まず線形計画(LP)緩和を解きます。  

**If fractional decisions remain, we solve a mixed-integer linear program (MILP) with binary division variables.**  
非整数の決定変数が残る場合は、分裂変数をバイナリとした混合整数線形計画法(MILP)を解きます。  

**If the time limit is reached, we retain a feasible solution.**  
制限時間に達した場合は、実行可能解を保持します。  

**Optimization uses predicted probabilities; the resulting graphs are evaluated separately with the official scoring implementation.**  
最適化には予測確率が使用され、得られたグラフは公式の採点実装によって個別に評価されます。  

---

### 7. Post-processing (後処理)

**7. Post-processing**  
7. 後処理  

#### A4: Gap closing (ギャップ充填)

**A4: Gap closing**  
A4: ギャップ充填  

**To recover temporarily missed cells, we connect track endpoints to the starts of later tracks.**  
一時的に見失われた細胞を復元するため、トラックの終点と後続のトラックの始点を接続します。  

**We estimate collective motion from selected links, compensate for it, and use Hungarian assignment to select connections.**  
選択されたリンクから集団運動を推定して補正を行い、ハンガリアン法による割り当てを用いて接続を選択します。  

**We process gaps of one, two, and three missing frames in that order.**  
1フレーム、2フレーム、3フレームの欠損ギャップをこの順序で処理します。  

**For g missing frames, the distance gate is 3√(g+1) µm.**  
gフレームの欠損に対する距離ゲートは 3√(g+1) µm です。  

**We insert nodes by linearly interpolating between the original endpoint coordinates.**  
元の端点座標間を線形補間することによりノードを挿入します。  

#### A5: Short component removal (短い連結成分の除去)

**A5: Short component removal**  
A5: 短い連結成分の除去  

**After gap closing, we remove weakly connected components with fewer than six nodes.**  
ギャップ充填後、ノード数が6個未満の弱連結成分を除去します。  

**Gap closing comes first because a short fragment may become part of a longer track once missing detections are filled in.**  
欠損した検出が埋められることで短い断片がより長いトラックの一部になる可能性があるため、ギャップ充填を先に行います。  

#### A6: Affine refinement of detected coordinates (検出座標のアフィン微小調整)

**A6: Affine refinement of detected coordinates**  
A6: 検出座標のアフィン微小調整  

**Even with correct connections, localization jitter can affect matching to GT points.**  
接続が正しくても、位置特定のジッター(ブレ)がGT点とのマッチングに影響を与えることがあります。  

**We refine coordinates while holding the graph connections fixed.**  
グラフの接続関係は固定したまま、座標を微小調整します。  

**For each pair of consecutive frames, we fit an affine model that predicts displacement from position using ordinary links.**  
連続するフレームの各ペアについて、通常のリンクを用いて位置から変位を予測するアフィンモデルを適合(フィット)させます。  

**A quadratic optimization then balances consistency with the predicted motion against staying close to the original detection coordinates.**  
その後、二次計画最適化により、予測された運動との整合性と、元の検出座標から離れすぎないこととのバランスを取ります。  

**Link weights are the calibrated conditional probabilities of being correct.**  
リンクの重みには、キャリブレーションされた「正解である条件付き確率」を使用します。  

**We keep division parents and daughters, gap endpoints, and interpolated nodes fixed.**  
分裂の親・娘細胞、ギャップの端点、および補間ノードは固定したままにします。  

**Refined positions are rounded to integer voxel coordinates.**  
調整された位置は整数のボクセル座標に丸められます。  

**Changes that leave the volume or create coordinate collisions within a frame are rejected.**  
ボリュームの外に出てしまう変更や、フレーム内で座標の衝突を引き起こす変更は破棄(リジェクト)されます。  

**Node counts and edges remain unchanged.**  
ノード数とエッジは変更されません。  

#### A7: Affine refinement of interpolated coordinates (補間座標のアフィン微小調整)

**A7: Affine refinement of interpolated coordinates**  
A7: 補間座標のアフィン微小調整  

**We then refine the nodes inserted into gaps using the same affine motion fields, fitted from ordinary links before coordinate refinement.**  
次に、座標調整前に通常のリンクから適合させた同一のアフィン運動場を用いて、ギャップに挿入されたノードを微小調整します。  

**Starting at one endpoint, we propagate positions through the motion fields and distribute the final endpoint discrepancy linearly across the gap so that both endpoints are preserved.**  
一方の端点から開始して運動場を通じて位置を伝播させ、両端点が保持されるように最終的な端点のズレをギャップ全体に線形に分配します。  

**Only the interpolated nodes move in this step; detected nodes, gap endpoints, and graph connections remain fixed.**  
このステップでは補間されたノードのみが移動し、検出ノード、ギャップ端点、およびグラフ接続は固定されたままです。  

**We again round coordinates and reject out-of-bounds positions and collisions.**  
ここでも座標を丸め、範囲外の位置や衝突を破棄します。  

**The submission uses both A6 and A7.**  
提出解法ではA6とA7の両方を使用しています。  

---

### 8. Submission and Results (提出と結果)

**8. Submission and Results**  
8. 提出と結果  

**We accelerated inference with TensorRT for the detectors and division models, mixed precision, and parallel execution on two GPUs.**  
検出器および分裂モデルへのTensorRT適用、混合精度演算(Mixed Precision)、そして2基のGPUでの並列実行により推論を高速化しました。  

**The matcher computes only the intermediate features it needs, and the flow fields are reused for matching and division-image alignment.**  
マッチャーは必要な中間特徴量のみを計算し、フロー場はマッチングと分裂画像アライメントの両方に再利用されます。  

**Division models also reuse encoder features of original frames across overlapping temporal windows.**  
また、分裂モデルは重複する時間ウィンドウ間で元フレームのエンコーダ特徴量を再利用します。  

**We reuse these features in the models that take three original frames and in the original-frame branches of the parent model.**  
これらの特徴量を、3つの元フレームを受け取るモデル群と、親モデルの元フレームブランチで再利用しています。  

**The latter additionally processes aligned images.**  
後者はさらにアライメント済み画像も処理します。  

**Our final solution achieved a CV score of 0.977801 (0.540107), a public leaderboard score of 0.977 (0.52), and a private leaderboard score of 0.967 (0.47).**  
私たちの最終解法は、CVスコア0.977801 (0.540107)、パブリックLBスコア0.977 (0.52)、プライベートLBスコア0.967 (0.47)を達成しました。  

**Values in parentheses indicate Division Jaccard.**  
カッコ内の数値は分裂Jaccard(Division Jaccard)を表しています。  

**Table 1. Building the complete pipeline.** Starting from A0, we progressively add or replace stages to reach the final configuration. Every row starts from the same OOF detections for all 199 training videos, with the videos, folds, and scoring implementation fixed. Optimization and post-processing can change which nodes are retained, add interpolated nodes, and refine coordinates.
表1. 完全なパイプラインの構築。A0から開始し、段階的にステージを追加または置き換えることで最終構成に到達します。  
すべての行は、動画、フォールド、および採点実装を固定した状態で、199本の全トレーニング動画に対する同一のOOF検出結果からスタートしています。最適化と後処理により、どのノードを保持するかが変更され、補間ノードが追加され、座標が洗練されます。  

| ID | Configuration / change | Adjusted edge Jaccard | Division Jaccard | Total score |
| :--- | :--- | :--- | :--- | :--- |
| **A0** | Detection + Hungarian assignment with drift correction | 0.890211 | 0.000000 | 0.890211 |
| **A1** | + Dense flow for motion compensation | 0.901919 | 0.000000 | 0.901919 |
| **A2** | + Matcher candidate filtering and correspondence scores | 0.901953 | 0.000000 | 0.901953 |
| **A3** | + Division models and calibration; joint link/division optimization replaces Hungarian assignment | 0.911017 | 0.534759 | 0.964493 |
| **A4** | + Gap closing | 0.916111 | 0.534759 | 0.969587 |
| **A5** | + Removal of components with fewer than six nodes | 0.918105 | 0.531915 | 0.971297 |
| **A6** | + Affine refinement of detected coordinates | 0.923141 | 0.540107 | 0.977151 |
| **A7** | + Affine refinement of interpolated coordinates: complete pipeline | 0.923790 | 0.540107 | 0.977801 |

**The total score is Adjusted edge Jaccard + 0.1 × Division Jaccard.**  
トータルスコアは「調整済みエッジJaccard + 0.1 × 分裂Jaccard」です。  

**A0–A2 produce no branches, so their division scores are zero.**  
A0〜A2は分岐(枝分かれ)を生成しないため、それらの分裂スコアはゼロになります。  

**Because ordinary-link calibration also uses parent, pre, and post features, A3 introduces division modeling and graph optimization together.**  
通常のリンクのキャリブレーションでも親、pre、postの特徴量を使用するため、A3では分裂モデリングとグラフ最適化を同時に導入しています。  

**The Hungarian cost is the corrected distance d in A0 and A1, and d + 12(1 − p) in A2, where p is the matching probability.**  
ハンガリアン法のコストは、A0およびA1では補正後の距離dであり、A2ではd + 12(1 − p)です(ここでpはマッチング確率)。  

**A0–A2 retain all detections, including isolated ones.**  
A0〜A2では、孤立した検出を含め、すべての検出を保持します。  

**Moving from A2 to A3 also changes the candidate gate from 7 µm to the final 12 µm flow distance gate, alongside calibration, division modeling, node selection, and joint optimization.**  
A2からA3への移行では、キャリブレーション、分裂モデリング、ノード選択、同時最適化とともに、候補ゲートも7 µmから最終的な12 µmのフロー距離ゲートへと変更されます。  

**Adding flow and matching scores to Hungarian assignment produced modest gains.**  
ハンガリアン法にフローとマッチングスコアを追加したことによるゲインは控えめなものでした。  

**The larger improvement came at A3, when division models, calibration, and joint optimization were introduced.**  
より大きな改善は、分裂モデル、キャリブレーション、および同時最適化が導入されたA3でもたらされました。  

**Distances after flow compensation and matching scores also provide evidence to the downstream calibration models, which in turn guide the LP's selection of connections.**  
フロー補正後の距離やマッチングスコアは、下流のキャリブレーションモデルにもエビデンスを提供し、それがひいてはLPによる接続の選択を導きます。  

**The gains at A1 and A2 alone therefore do not capture their full role in the pipeline.**  
したがって、A1およびA2単体でのゲインだけでは、パイプライン全体におけるそれらの役割の全貌を捉えきれていません。  

**We interpret the A3 improvement as the combined benefit of these features, division evidence, and graph constraints.**  
私たちは、A3での改善をこれらの特徴量、分裂エビデンス、およびグラフ制約の複合的なメリットであると解釈しています。  

**Gap closing, component removal, and coordinate refinement further improved the final score.**  
ギャップ充填、成分除去、および座標の微小調整により、最終スコアがさらに向上しました。  

---

### Author / Citation (著者・引用)

**Author [yu4u](https://www.kaggle.com/ren4yu) ren4yu**  
著者 [yu4u](https://www.kaggle.com/ren4yu) ren4yu  

**Share**  
共有  

**Citation**  
引用  

**yu4u. 3rd Place Solution. https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/3rd-place-solution. 2026. Kaggle**  
yu4u. 3rd Place Solution. https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/3rd-place-solution. 2026. Kaggle
