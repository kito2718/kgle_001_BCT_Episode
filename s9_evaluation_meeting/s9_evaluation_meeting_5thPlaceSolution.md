**5th Place: 3D U-Net + Transformer Linker + Multi-stage ILP Tracking**  
5位: 3D U-Net + Transformer リンカー + 多段ILPトラッキング

**[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) Solution Writeup · 5th place · Sep 30, 2026**  
[Biohub - Cell Tracking During Development] 解法ライトアップ · 5位 · 2026年9月30日

---

### Overview (概要)

**Overview**  
概要

**The pipeline has three parts:**  
パイプラインは3つのパートから構成されています:

**Detector: a 3D U-Net that finds the cells.**  
検出器: 細胞を検出する3D U-Net。

**Linker: a transformer that links cells between two frames.**  
リンカー: 2つのフレーム間で細胞同士をリンクするTransformer。

**Tracking: ILPs that build the tracks.**  
トラッキング: トラック(追跡軌跡)を構築するILP(整数線形計画法)。

**The detector and the linker are used as a 5-fold ensemble with TTA.**  
検出器とリンカーは、TTA(テスト時データ拡張)を伴う5-foldアンサンブルとして使用されます。

**The outputs of the 5 models and the 8 TTA views are averaged.**  
5つのモデルと8つのTTAビューの出力が平均化されます。

---

### Validation design (バリデーション設計)

**Validation design**  
バリデーション設計

**5-fold CV over clips, stratified by clip type (44b6 or 6bba) and by the number of divisions per clip.**  
クリップ単位での5-fold CVで、クリップのタイプ(44b6または6bba)およびクリップあたりの細胞分裂数によって層化しました。

**The training data has two kinds of problem frames:**  
トレーニングデータには2種類の間題のあるフレームが存在します:

**Duplicated frames: a frame is exactly the same as the next one.**  
重複フレーム: あるフレームが次のフレームと完全に同一であるもの。

**Drift frames: the whole image jumps between two frames, as in the figure.**  
ドリフトフレーム: 図に示すように、2つのフレーム間で画像全体が大きくズレて跳んでいるもの。

**Drift between two frames.**  
2つのフレーム間のドリフト。

**After shifting back by 14 µm, the cells match.**  
14 µm元に戻すようにシフトさせると、細胞が一致します。

**In validation, I removed the links of these frames from both the prediction and the ground truth.**  
バリデーションでは、予測と正解ラベル(Ground Truth)の双方からこれらのフレームのリンクを除外しました。

**My drift fix gave a big gain in CV, but no gain on the LB.**  
このドリフト修正はCVを大きく改善させましたが、LB(リーダーボード)では改善が見られませんでした。

**So the hidden test data has no large drift.**  
したがって、隠されたテストデータには大きなドリフトは存在しません。

**I think this is also one reason why many people saw CV and LB that did not match.**  
これが、多くの参加者がCVとLBの不一致を経験した理由の1つでもあると考えています。

**In my experiments, training without the duplicated frames and with the drift corrected raised the private score of the same fold-0 model by 0.011, but the public score dropped.**  
実験では、重複フレームを除外しドリフトを補正した状態で学習を行うと、同一のfold-0モデルのプライベートスコアが0.011向上しましたが、パブリックスコアは低下しました。

**So I did not use it in the final model.**  
そのため、最終モデルにはこれを採用しませんでした。

---

### Model (モデル)

**Model**  
モデル

**U-Net: the temporal 3D U-Net from the [public baseline](https://www.kaggle.com/code/thibautgoldsborough/unet-baseline-inference-submission).**  
U-Net: [公開ベースライン]の時系列3D U-Netを使用しています。

**I use average pooling to make each frame 4× smaller in X and Y before the U-Net.**  
U-Netに入力する前に、Average Poolingを用いて各フレームのX方向およびY方向のサイズを4分の1に縮小しています。

**Linker: a transformer on the cells of frames t and t+1 (self-attention inside each frame, cross-attention between the two frames).**  
リンカー: フレームtとフレームt+1の細胞に対するTransformer(各フレーム内でのSelf-Attentionと、2つのフレーム間でのCross-Attention)。

**It learns four things:**  
これは以下の4つの要素を学習します:

**Link score: for every pair of cells in t and t+1, is it the same cell?**  
リンクスコア: tとt+1におけるすべての細胞ペアについて、同一の細胞であるかどうか？

**Identity similarity: do two cells look alike?**  
外見の類似度(Identity similarity): 2つの細胞の見た目が似ているかどうか？

**Each cell gets a 16-dim appearance vector from the U-Net features, and the same cell in t and t+1 should have similar vectors.**  
各細胞はU-Netの特徴量から16次元の外見ベクトルを取得し、tとt+1の同一細胞は類似したベクトルを持つはずです。

**This helps to separate cells that are close together.**  
これが互いに近接している細胞同士を識別・分離するのに役立ちます。

**New cell: is this cell in t+1 new, with no matching cell in t (for example, it just came into view)?**  
新規細胞: t+1におけるこの細胞は、tに対応する細胞が存在しない新規のもの(例えば、視野内に入ってきたばかりのもの)であるかどうか？

**Division: does this cell in t divide into two?**  
分裂: tにおけるこの細胞が2つに分裂するかどうか？

**Divisions are rare and easy to miss, so a separate head, with a higher weight on division examples in the loss, gives a direct signal for them.**  
細胞分裂は稀で見落としやすいため、損失関数において分裂サンプルの重みを高く設定した独立したヘッドを設けることで、それらに直接的なシグナルを与えています。

**The linker.**  
リンカー。

**Top: structure.**  
上段: 構造。

**Bottom: its three outputs on a small example, where cell B divides into 2 and 3, and cell 4 is new.**  
下段: 細胞Bが2と3に分裂し、細胞4が新規に出現している小さな具体例における3つの出力結果。

**Pretraining: I first trained on 4 public ZebraHub datasets.**  
事前学習: まず4つの公開ZebraHubデータセットで学習を行いました。

**Then I fine-tuned on the competition data.**  
その後、コンペティションのデータでファインチューニングしました。

**Training: AdamW, lr 1e-4, cosine schedule, batch size 8.**  
学習設定: AdamW、学習率 1e-4、コサインスケジューラ、バッチサイズ 8。

**Augmentation: flips and rotations in XY, gamma and brightness shift.**  
データ拡張: XY平面での反転および回転、ガンマ補正および明暗シフト。

---

### Tracking (トラッキング)

**Tracking**  
トラッキング

**An ILP picks which cells and links to keep, and where tracks start, end or split.**  
ILP(整数線形計画法)により、どの細胞とリンクを残すか、そしてトラックがどこで開始・終了・分岐(分裂)するかを決定します。

**Confident cells, high-score links and links between cells with similar identity vectors are cheaper to keep.**  
確信度の高い細胞、ハイスコアなリンク、および類似した外見ベクトルを持つ細胞間のリンクほど、維持するためのコストが低く設定されます。

**A track start is cheaper when the new-cell score is high, and a split is cheaper when the division score is high.**  
トラックの開始は新規細胞スコアが高いほどコストが低くなり、トラックの分岐は分裂スコアが高いほどコストが低くなります。

**ILP1: solve with confident cells only (detection confidence p ≥ 0.9).**  
ILP1: 確信度の高い細胞(検出確信度 p ≥ 0.9)のみを用いて最適化を解きます。

**ILP2: solve again with more cells (p ≥ 0.6).**  
ILP2: より多くの細胞(p ≥ 0.6)を含めて再度最適化を解きます。

**It fills gaps, but also adds some false divisions.**  
これによりギャップ(途切れた箇所)が埋まりますが、誤った分裂もいくつか追加されてしまいます。

**No new divisions: remove every division that ILP1 did not have.**  
新規分裂の除外: ILP1に含まれていなかったすべての分裂を除去します。

**The extra cell keeps its own track.**  
除外された余分な細胞は、自身の独立したトラックを保持します。

**ILP3 (re-link): keep the cells and divisions, and solve only the links again.**  
ILP3(再リンク): 細胞と分裂を固定して保持したまま、リンクのみを再度最適化して解きます。

**Weak links (score 0.2 to 0.3) can help during the solve, and are removed after.**  
弱いリンク(スコア0.2〜0.3)は解く過程で手助けとなり、最適化後に除去されます。

**Repairs:**  
修復処理:

**ILP4 joins a track end and a track start that are 2 to 5 frames apart, and fills in the missing positions.**  
ILP4は、2〜5フレーム離れたトラックの終了地点と開始地点を結合し、欠損している位置を補間します。

**Unclear links are matched again, using how similar the cells look.**  
不確実なリンクについては、細胞の外見がどれだけ似ているかに基づいて再度マッチングを行います。

**A track that ends at frame t and one that starts at t+1 are joined by a weak link (score 0.1 to 0.3) if the cells look similar.**  
細胞の外見が似ている場合、フレームtで終了するトラックとフレームt+1で開始するトラックは、弱いリンク(スコア0.1〜0.3)によって結合されます。

**Smoothing: the detected positions jump around a little from frame to frame.**  
スムージング: 検出された位置はフレーム間でわずかにブレて跳ぶことがあります。

**For each cell, I fit a straight line through the positions of its track, from 4 frames before to 4 frames after.**  
各細胞について、前後4フレーム(4フレーム前から4フレーム後まで)にわたるトラック上の位置に対して直線をフィッティングします。

**The new position is 80% the line and 20% the original position.**  
新しい位置は、直線上の位置を80%、元の位置を20%として合成します。

**After that, if a track misses one frame, I fill in the missing position.**  
その後、トラックに1フレームの欠損がある場合は、欠損した位置を補間して埋めます。

---

### Results (結果)

**Results**  
結果

**The main improvements, as the change in LB score:**  
主な改善点とLBスコアの変化:

| Method (手法) | Public LB | Private LB |
| :--- | :--- | :--- |
| **Linker: New cell and Division heads**<br>(リンカー: 新規細胞および分裂ヘッド) | +0.020 | +0.024 |
| **Pretraining on ZebraHub**<br>(ZebraHubによる事前学習) | +0.014 | +0.016 |
| **5-fold ensemble**<br>(5-foldアンサンブル) | +0.008 | +0.007 |
| **TTA with 8 flips and rotations**<br>(8つの反転および回転によるTTA) | +0.007 | +0.005 |
| **Tracking: detection confidence as the cell cost in the ILP**<br>(トラッキング: ILPにおける細胞コストとしての検出確信度の利用) | +0.008 | +0.020 |
| **Tracking steps 2–3: ILP2 and No new divisions**<br>(トラッキング ステップ2–3: ILP2および新規分裂の除外) | +0.003 | +0.007 |
| **Tracking step 5: Repairs**<br>(トラッキング ステップ5: 各種修復処理) | +0.001 | +0.002 |
| **Tracking step 6: Smoothing**<br>(トラッキング ステップ6: スムージング) | +0.004 | +0.005 |

---

### Author / Citation (著者 / 引用)

**Author [Tang](https://www.kaggle.com/hirotetsu) hirotetsu**  
著者: [Tang](https://www.kaggle.com/hirotetsu) hirotetsu

**Share**  
共有

**Citation**  
引用

**Tang. 5th Place: 3D U-Net + Transformer Linker + Multi-stage ILP Tracking. https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/5th-place-3d-u-net-transformer-linker-multi-s. 2026. Kaggle**  
Tang. 5th Place: 3D U-Net + Transformer Linker + Multi-stage ILP Tracking. https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/5th-place-3d-u-net-transformer-linker-multi-s. 2026. Kaggle
