**4th Place Solution**  
4位の解法

**3D Net detection, learned linking, and a division prior inside the ILP**  
3D Netによる検出、学習ベースのリンク付け、そしてILP内部での細胞分裂事前分布

**[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) Solution Writeup · 4th place · Oct 1, 2026**  
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) 解法まとめ · 4位 · 2026年10月1日

**Public 0.969 / Private 0.962.**  
パブリック 0.969 / プライベート 0.962。

**Notebook: [kaggle.com/code/songqizhou/biohub-4th-place-solution](https://www.kaggle.com/code/songqizhou/biohub-4th-place-solution)**  
ノートブック: [kaggle.com/code/songqizhou/biohub-4th-place-solution](https://www.kaggle.com/code/songqizhou/biohub-4th-place-solution)

**Many thanks to the organisers for a challenging and well-designed competition, and to everyone who shared notebooks, support packs and ideas on the forum.**  
難易度が高く素晴らしい設計のコンペを主催してくださった運営の皆様、そしてフォーラムでノートブックやサポートパック、アイデアを共有してくださったすべての方々に深く感謝いたします。

---

### Overview (概要)

**Our tracker has five stages: learned models for detection, links and divisions, a global ILP that puts them together, and rule-based post-processing.**  
私たちのトラッカーは5つのステージで構成されています: 検出・リンク・分裂のための学習モデル、それらを統合する大域的ILP(整数線形計画法)、そしてルールベースの後処理です。

| stage (ステージ) | what it does (処理内容) | section (セクション) |
| :--- | :--- | :--- |
| **1. Detection** <br> 1. 検出 | **3D Net finds nuclei; a pseudo-label model refines them; sparse movies get a stricter threshold** <br> 3D Netが細胞核を見つけ、疑似ラベルモデルがそれらを洗練し、疎な動画にはより厳しい閾値を適用します | 2 |
| **2. Linking** <br> 2. リンク付け | **every candidate link gets a learned probability** <br> すべての候補リンクに学習された確率を付与します | 3 |
| **3. Divisions** <br> 3. 分裂 | **every node gets a learned division score s** <br> すべてのノードに学習された分裂スコア s を付与します | 4 |
| **4. ILP** <br> 4. ILP | **solves the tracks; each node pays its own division cost 6 − 16·s (base 5 in sparse movies)** <br> トラックを解きます。各ノードは個別の分裂コスト 6 − 16·s(疎な動画ではベースが5)を支払います | 4, 5 |
| **5. Post-processing** <br> 5. 後処理 | **fork verifier, smoothing, gap bridging, track-support gate for sparse movies** <br> 分岐検証器、平滑化、ギャップ補間、疎な動画向けのトラック支持ゲート | 4, 5 |

**The per-node division cost in the ILP was our largest single gain (+0.035).**  
ILPにおけるノードごとの分裂コストが、単一の改善としては最大のものでした(+0.035)。

---

### Our approach: model the biology (私たちのアプローチ: 生物学をモデル化する)

**Our compute was modest.**  
私たちの計算リソースは控えめなものでした。

**Every network in the pipeline is small (the largest, the 3D Net, has 18 M parameters), and all the models that combine evidence (the edge model, the division graph and pair models, the fork verifier) are gradient-boosted trees on CPU.**  
パイプライン内のすべてのネットワークは小型であり(最大のものである3D Netでもパラメータ数は18M)、各種の手がかりを統合するモデル(エッジモデル、分裂グラフ/ペアモデル、分岐検証器)はすべてCPU上の勾配ブースティング木です。

**Most of our gains therefore came from looking closely at the biology and at the metric, and turning each observation into a small, targeted model:**  
そのため、私たちのスコア向上の大部分は、生物学的特徴と評価指標を綿密に観察し、個々の観察結果をターゲットを絞った小さなモデルへと落とし込んだことから得られました:

| observation (観察事実) | what we built (構築したもの) | section (セクション) |
| :--- | :--- | :--- |
| **Only about 1% of the cells are annotated; most real cells and most real divisions carry no label** <br> アノテーションされている細胞は約1%に過ぎず、実際の細胞や分裂のほとんどにはラベルが付いていません | **a loss with separate tiers for annotated, background and unannotated voxels; division labels only from annotated tracks** <br> アノテーション済み、背景、未アノテーションのボクセルを個別の階層(ティア)に分けた損失関数。アノテーション済みトラックからのみ分裂ラベルを作成 | 2.1, 4.2 |
| **Two touching nuclei form one bright blob, but their centres are distinct** <br> 接触している2つの細胞核は1つの明るい塊を形成しますが、中心は明確に分かれています | **a vector field that points each voxel to its own centre** <br> 各ボクセルをそれぞれの中心に向かわせるベクトル場 | 2.1 |
| **Nucleus size and spacing differ strongly between embryos** <br> 胚によって細胞核のサイズや間隔が大きく異なります | **a DoG density router that sets the detection threshold per movie** <br> 動画ごとに検出閾値を設定するDoG(Difference of Gaussians)密度ルーター | 2.3 |
| **A mother rounds up and brightens, its chromatin splits, and two compact daughters appear** <br> 母細胞は丸くなって明るさを増し、クロマチンが分裂して、2つの引き締まった娘細胞が現れます | **a division score built from these cues, used as a per-node division cost in the ILP** <br> これらの手がかりから構築した分裂スコア。ILP内でノードごとの分裂コストとして使用 | 4 |
| **Real daughters keep moving apart for several frames** <br> 本物の娘細胞は数フレームにわたって離れ続けます | **a fork verifier that follows both daughters in the raw images** <br> 元画像内で両方の娘細胞を追跡する分岐検証器 | 4.3 |
| **The whole tissue drifts between frames** <br> 組織全体がフレーム間でドリフト(平行移動)します | **motion-compensated smoothing of the tracks** <br> トラックの動き補償平滑化 | 5.2 |
| **A model trained on more (pseudo-)labels finds more cells, but also invents some on a new embryo** <br> より多くの(疑似)ラベルで訓練されたモデルはより多くの細胞を見つけますが、新しい胚では存在しない細胞を捏造することもあります | **keep its tracks only where an independent model agrees** <br> 独立したモデルが一致した箇所でのみトラックを保持 | 2.2, 5.3 |

---

### 1. Data, metric and validation (データ、評価指標、検証)

**Data.**  
データ。

**199 training movies from two embryos (71 and 128 movies).**  
2つの胚から得られた199本の訓練用動画(71本と128本)。

**Each movie is 100 frames of 64 × 256 × 256 voxels at 1.625 × 0.406 × 0.406 µm.**  
各動画は100フレームで構成され、解像度1.625 × 0.406 × 0.406 µmの64 × 256 × 256ボクセルです。

**Only a handful of lineages per movie are annotated, about 1% of the cells.**  
動画ごとにアノテーションされている細胞系譜はごくわずかで、全細胞の約1%にすぎません。

**The whole training set contains only 151 annotated divisions.**  
訓練セット全体でも、アノテーションされた分裂はわずか151回しか含まれていません。

**One embryo is dense with small nuclei (DoG nucleus spacing about 13–19 µm), the other mostly sparse with large nuclei (about 17–31 µm).**  
一方の胚は小さな細胞核が密集しており(DoGによる細胞核の間隔は約13〜19 µm)、もう一方は主に疎で大きな細胞核を持っています(約17〜31 µm)。

**Metric.**  
評価指標。

**adjusted edge Jaccard + 0.1 · division Jaccard.**  
調整済みエッジJaccard + 0.1 · 分裂Jaccard。

**A predicted edge is a false positive only if it touches an annotated node, so most unannotated cells are "free".**  
予測されたエッジは、アノテーション済みノードに接触している場合にのみ偽陽性(False Positive)となるため、アノテーションされていないほとんどの細胞はペナルティなし("free")となります。

**A node-count adjustment penalises predicting more nodes than the estimated number of cells, so the node count still matters.**  
ノード数調整によって推定細胞数よりも多くのノードを予測するとペナルティが課されるため、ノード数は依然として重要です。

**Validation rules.**  
検証ルール。

**Score everything with the official evaluator.**  
すべて公式の評価スクリプトでスコアリングすること。

**An early custom node-level metric reversed two of our conclusions.**  
開発初期に使用していた自作のノードレベルの指標では、結論が2回も逆転してしまいました。

**Out-of-fold everything.**  
すべてをOut-of-fold(交差検証の予測値)で統一すること。

**Detections for all 199 movies come from 5-fold copies of each detector, and every downstream model is trained on these out-of-fold detections.**  
全199動画の検出結果は各検出器の5分割フォールドモデルから得られたものであり、下流のすべてのモデルはこれらのOut-of-fold検出値を使って訓練されています。

**The deployed detectors are trained on all 199 movies.**  
本番デプロイ用の検出器は全199動画で訓練されています。

**Check detector changes across embryos (train on one embryo, test on the other), not only on random folds.**  
ランダムなフォールドだけでなく、胚間(一方の胚で訓練し、もう一方の胚でテスト)でも検出器の変更点を確認すること。

**Same-embryo validation over-rated changes that fit the seen embryos better.**  
同一胚内での検証は、学習済みの胚により過剰適合した変更点を過大評価してしまいました。

**Change one component at a time and compare against the deployed chain's own output.**  
変更は一度に1つのコンポーネントのみとし、本番パイプライン自体の出力と比較すること。

---

### 2. Detection (検出)

#### 2.1 3D Net

**Idea.**  
アイデア。

**For every voxel, predict a cell probability and a unit vector pointing to the centre of its nucleus (the Cellpose flow idea), then follow the vectors to find the nuclei.**  
すべてのボクセルについて、細胞確率と核の中心を指す単位ベクトル(Cellposeのflowのアイデア)を予測し、そのベクトルを辿って細胞核を見つけ出します。

**Network.**  
ネットワーク。

**Adapted from a dual-encoder 3D U-Net with temporal attention from the literature.**  
先行研究の時間的アテンションを持つデュアルエンコーダー型3D U-Netをベースに改良しました。

**One encoder sees the current frame, the other the difference between the next and the previous frame.**  
一方のエンコーダーは現在のフレームを見通し、もう一方は次フレームと前フレームの差分を見ます。

**Adding the neighbouring frames clearly helps: removing this input dropped the detection score from 0.93 to 0.73 in cross-embryo validation, while a wider five-frame window gave no further gain.**  
隣接フレームの追加は明らかに効果的でした: 胚間検証においてこの入力を取り除くと検出スコアが0.93から0.73に低下しましたが、5フレームに広げたウィンドウではさらなる向上は見られませんでした。

**Decoding: our choices.**  
デコーディング: 私たちの選択。

**Strict seeds.**  
厳格なシード。

**Only voxels with probability > 0.97 are followed (0.99 in sparse movies, section 2.3), so dim background rarely turns into a nucleus.**  
確率 > 0.97 のボクセルのみを追跡対象とすることで(疎な動画では0.99、セクション2.3参照)、薄暗い背景が細胞核に誤認識されることはほとんどなくなります。

**Short, binned convergence.**  
短距離かつビン分割による収束。

**Seeds move 40 steps of 0.5 µm.**  
シードは0.5 µm刻みで40ステップ移動します。

**Seeds whose end points fall into the same 1 µm cell form one nucleus; duplicates are left to the NMS.**  
終点が同じ1 µmのセル内に収まるシードは1つの核を形成し、重複したものはNMS(非極大抑制)に任せられます。

**Probability-ranked, adaptive NMS.**  
確率ランク順のアダプティブNMS。

**Nuclei are ranked by their highest probability, and the NMS radius is set per movie: 4 µm, or 7 µm when the median nearest-neighbour distance is ≥ 10 µm (large nuclei).**  
核は最高確率順にランク付けされ、NMSの半径は動画ごとに設定されます: 通常は4 µm、最近傍距離の中央値が10 µm以上の場合は7 µm(大きな核向け)とします。

**Loss: our choices.**  
損失関数: 私たちの選択。

**The vector field gets an L1 loss and the probability a weighted BCE with three tiers: annotated nuclei, confident background, and everything else (only a very weak push towards background).**  
ベクトル場にはL1損失を、確率には3つの階層(アノテーション済み核、確実な背景、その他のすべて[背景方向へのごく弱い誘導のみ])を持つ重み付きBCEを適用します。

**What matters is how the tiers are built:**  
重要なのは、それらの階層をどのように構築するかです:

**Tier normalisation.**  
階層の正規化。

**Each tier's weight is divided by its voxel count, so the weights set how much each tier counts as a whole.**  
各階層の重みはそのボクセル数で除算されるため、重みは各階層全体としての寄与度を設定することになります。

**With per-voxel weights the few annotated voxels carried less than 0.5% of the loss, and the probability head collapsed towards zero.**  
ボクセルごとの重み付けでは、わずかなアノテーション済みボクセルが損失全体の0.5%未満しか占めず、確率予測ヘッドがゼロへと潰れてしまいました。

**Where the vectors are supervised.**  
ベクトルを教師あり学習させる領域。

**An annotation owns the voxels that are nearest to it, within 5.5 µm and among the brightest 5% of the frame, plus a 2 µm core that is always kept.**  
アノテーションは、自身に最も近く、5.5 µm以内で、かつフレーム内で最も明るい上位5%に含まれるボクセルと、常に保持される2 µmのコア領域を自身の領域とします。

**The cap keeps an unannotated neighbour from being pulled into the region.**  
この上限設定によって、隣接するアノテーションされていない細胞がその領域に巻き込まれるのを防ぎます。

**Local background.**  
局所的背景。

**A voxel is confident background only if it is darker than 0.4 × the local maximum within 3.5 µm, so the rim of a real cell in a dense frame is not labelled background.**  
ボクセルは、3.5 µm以内の局所的最大値の0.4倍より暗い場合にのみ「確実な背景」とみなされるため、密集したフレーム内の本物の細胞の縁が背景とラベル付けされるのを防ぎます。

**We trained three 3D Nets:**  
私たちは3つの3D Netを訓練しました:

| model (モデル) | training data (訓練データ) | role (役割) |
| :--- | :--- | :--- |
| **3D Net-128 (XY pooled by 2)** <br> 3D Net-128 (XYを2分の1にプーリング) | **ground truth** <br> 正解ラベル | **main detector** <br> メイン検出器 |
| **3D Net-128-PL** | **ground truth + pseudo-labels** <br> 正解ラベル + 疑似ラベル | **refines the nodes (2.2)** <br> ノードの洗練 (2.2) |
| **3D Net-64 (XY pooled by 4)** <br> 3D Net-64 (XYを4分の1にプーリング) | **ground truth** <br> 正解ラベル | **independent check in sparse movies (5.3)** <br> 疎な動画での独立チェック (5.3) |

**Training.**  
訓練。

**The deployed models are trained on all 199 movies with learning rate 3e-4.**  
本番モデルは学習率3e-4で全199動画を用いて訓練されています。

**Each detector is trained several times (five folds for the out-of-fold detections plus the final model), so with our limited GPU compute we kept every run short: 10 epochs for the 128 models, using the average of the epoch 7–10 weights, and 50 for the 64 model, which is about four times cheaper per epoch.**  
各検出器は複数回訓練されるため(Out-of-fold検出用の5フォールド分に加えて最終モデル)、限られたGPUリソースのもとで各実行を短く抑えました: 128モデルは10エポック(7〜10エポックの重みの平均を使用)、エポックあたりの計算コストが約4分の1である64モデルは50エポックとしました。

**We do not claim that 10 epochs is optimal: many teams train for hundreds.**  
10エポックが最適であると主張するつもりはありません。多くのチームは何百エポックも訓練しています。

**In our runs, however, recall on unseen embryos peaked early in training, so our short-trained detector leans towards recall and predicts somewhat more nodes than there are cells (about 9% more).**  
しかし私たちの実行では、未知の胚に対する再現率(Recall)は訓練の初期にピークに達したため、短時間で訓練した検出器は再現率寄りの傾向を持ち、実際の細胞数よりもやや多くのノードを予測しました(約9%過剰)。

**Much of the rest of the pipeline is about removing those extra nodes downstream: the strict seed threshold, the pseudo-label check (2.2), the ILP with its short-track filter and the track-support gate (5.3).**  
パイプラインの後続部分の多くは、下流でこれらの余分なノードを排除することを目的としています: 厳格なシード閾値、疑似ラベルチェック(2.2)、ショートトラックフィルターを備えたILP、そしてトラック支持ゲート(5.3)です。

**In that sense, several of these steps compensate for short training; with more GPU compute we would rather put that effort into the detector itself, which would be the more elegant route.**  
その意味では、これらのステップのいくつかは短時間訓練を埋め合わせるためのものです。より多くのGPU計算リソースがあれば、検出器そのものにその労力を注ぎ込む方を選んだでしょうし、そちらの方がよりエレガントな方法と言えます。

**Resolution.**  
解像度。

**In a quick check on one random fold with the same simple linker, the 128 × 128 XY grid scored 0.856 against 0.853 for the 64 × 64 grid: at 64 crowded nuclei merge in dense tissue.**  
同じ単純なリンカーを用いた1つのランダムフォールドでの簡単な確認では、128 × 128のXYグリッドが0.856を記録したのに対し、64 × 64グリッドは0.853でした: 64では密集した組織内で混み合った細胞核が結合してしまいます。

**In sparse movies, however, the 64 grid did best locally.**  
しかしながら、疎な動画においては64グリッドがローカルで最良の結果を出しました。

**Rather than switching detectors per movie, which changes the node count, we use 3D Net-64 as an independent check there (5.3).**  
動画ごとに検出器を切り替えるとノード数が変動してしまうため、その代わりに3D Net-64をそこでの独立したチェック機構として利用しています(5.3)。

**Full resolution (256 × 256) costs 4.4× the compute of 128 and needs a smaller batch.**  
フル解像度(256 × 256)は128の4.4倍の計算コストがかかり、バッチサイズも小さくする必要があります。

**We could train it only once, on one fold and for 10 epochs; that single short run showed no clear gain, and whether it pays off when trained to convergence remains open.**  
私たちは1つのフォールド、10エポックで1度しか訓練できませんでした。その1回の短い実行では明確な向上は見られず、収束するまで訓練した場合に見返りがあるかどうかは未検証のままです。

**No heatmap detector.**  
ヒートマップ検出器は不採用。

**The public heatmap U-Net needs several hundred epochs to converge, which was beyond our compute, so we could not train our own folds of it.**  
公開されているヒートマップU-Netは収束までに数百エポックを必要とし、私たちの計算能力を超えていたため、独自のフォールドを訓練することができませんでした。

**Its public weights were trained on all 199 training movies, so we had no way to cross-validate an ensemble with them.**  
その公開済みの重みは全199本の訓練用動画ですでに学習されていたため、それらを用いたアンサンブルを交差検証する手段がありませんでした。

**Without a reliable local score, the risk outweighed the expected gain for us, and we left it out.**  
信頼できるローカルスコアが得られない以上、私たちにとってリスクが見込み利益を上回っていたため、採用を見送りました。

**No test-time augmentation for the 3D Net.**  
3D Netに対するテスト時拡張(TTA)は不採用。

**4- or 8-view TTA multiplies the cost of every validation run.**  
4視点または8視点のTTAは、すべての検証にかかるコストを何倍にも跳ね上げます。

**A quick test on our chain gave no reliable gain (within noise locally, lower on the leaderboard), so we left it out.**  
私たちのパイプラインで簡単にテストしたところ、信頼できる向上は得られず(ローカルではノイズの範囲内、リーダーボードでは低下)、除外しました。

**Other teams report gains from TTA, so this may depend on the pipeline.**  
他のチームはTTAによる改善を報告しているため、これはパイプラインの構成に依存する可能性があります。

---

#### 2.2 Pseudo-label refinement (疑似ラベルによる洗練)

**Problem.**  
課題。

**The ground-truth-only model is conservative: few false tracks, but it misses dim nuclei in some frames.**  
正解ラベルのみで訓練したモデルは保守的です: 誤ったトラックは少ないものの、一部のフレームで薄暗い核を見逃してしまいます。

**A model trained with pseudo-labels finds more nuclei, but used alone it also invents tracks on a new embryo (−0.003 to −0.004 on the leaderboard).**  
疑似ラベルで訓練したモデルはより多くの核を見つけますが、単独で使用すると新しい胚で存在しないトラックを捏造してしまいます(リーダーボードで−0.003〜−0.004)。

**Pseudo-labels.**  
疑似ラベル。

**The out-of-fold detections of all 199 movies, plus the detections on one unlabelled external Zebrahub embryo.**  
全199動画のOut-of-fold検出結果に加えて、ラベル付けされていない1つの外部Zebrahub胚に対する検出結果。

**3D Net-128-PL is trained on ground truth plus these pseudo-labels.**  
3D Net-128-PLは正解ラベルとこれらの疑似ラベルを合わせて訓練されています。

**Method.**  
手法。

**Use the pseudo-label model's nodes, but only on tracks the ground-truth model agrees with:**  
正解モデルが合意したトラック上でのみ、疑似ラベルモデルのノードを使用します:

**Run both models on the movie.**  
動画に対して両方のモデルを実行します。

**A pseudo-label node is confirmed if it is matched one-to-one to a ground-truth-model node within 6 µm in the same frame.**  
疑似ラベルノードは、同一フレーム内で6 µm以内の正解モデルノードと1対1で一致した場合に「確認済み」と判定されます。

**Link the pseudo-label nodes frame to frame into tracks (greedy matching, 7 µm gate).**  
疑似ラベルノードをフレーム間でリンクしてトラックを作成します(貪欲マッチング、7 µmゲート)。

**Keep a track only if at least 50% of its nodes are confirmed.**  
構成ノードの50%以上が「確認済み」であるトラックのみを保持します。

**The output is the pseudo-label nodes of the kept tracks.**  
出力は、保持されたトラックの疑似ラベルノードとなります。

**Effect.**  
効果。

**Frames where the ground-truth model missed a nucleus are filled in, and tracks it never saw disappear: +0.004 on the leaderboard, for the cost of one extra training run rather than an ensemble.**  
正解モデルが細胞核を見逃していたフレームが埋められ、正解モデルが検出しなかった余計なトラックは消滅します: アンサンブルを行う代わりに1回余分に訓練を実行するだけのコストで、リーダーボード上で+0.004の向上となりました。

---

#### 2.3 Density-aware threshold (DoG router) (密度に応じた閾値処理: DoGルーター)

**Problem.**  
課題。

**In sparse movies, large nuclei produce extra seeds at the normal threshold.**  
疎な動画では、大きな細胞核が通常の閾値だと余分なシードを生み出してしまいます。

**Method.**  
手法。

**A simple rule-based DoG blob tracker counts nuclei and gives each movie a nucleus spacing.**  
単純なルールベースのDoGブロブトラッカーが核をカウントし、各動画に核の間隔を割り当てます。

**Movies with spacing ≥ 19.7 µm are re-detected with a stricter seed threshold (0.99 instead of 0.97).**  
間隔が19.7 µm以上の動画は、より厳しいシード閾値(0.97ではなく0.99)で再検出されます。

**Why DoG.**  
なぜDoGなのか。

**Our first version used the detector's own node count to decide.**  
私たちの初期バージョンでは、検出器自身のノード数を使って判定していました。

**But a detector's count drifts on an unseen embryo, and a router should not depend on the model it routes.**  
しかし、検出器によるカウントは未知の胚では変動してしまいますし、ルーターは自身が振り分けを行う対象のモデルに依存すべきではありません。

**The detector-independent DoG router gained +0.002 on public with no change in local validation.**  
検出器から独立したDoGルーターにより、ローカル検証では変化がなかったものの、パブリックで+0.002のゲインを得ました。

---

### 3. Linking (リンク付け)

#### 3.1 Candidate links (候補リンク)

**For every node at frame t: all nodes at t+1 within 10 µm and its 5 nearest successors.**  
フレーム t のすべてのノードに対して: 10 µm以内の t+1 のすべてのノード、および最も近い5つの後続ノード。

**In addition, every node at t+1 is linked to its 3 nearest predecessors.**  
さらに、t+1 のすべてのノードは最も近い3つの先行ノードとリンクされます。

#### 3.2 Evidence for each link (各リンクの手がかり・特徴量)

**Geometry (26 features).**  
幾何学的特徴(26特徴量)。

**Displacement, distance ranks and gaps among competing candidates in both directions, nearest-neighbour distances, detection confidence and cluster size, the residual against the node's velocity from a first greedy linking pass, the residual against the local tissue flow (median displacement of confident neighbouring links), and whether the nodes continue into the past / future.**  
変位、両方向における競合候補間の距離ランクとギャップ、最近傍距離、検出信頼度とクラスターサイズ、最初の貪欲リンクパスから得られたノード速度との残差、局所的な組織の流れ(信頼度の高い隣接リンクの変位の中央値)との残差、そしてノードが過去/未来へと継続しているかどうか。

**Transformer probability.**  
Transformerによる確率。

**We retrained the competition baseline's linker architecture (a temporal U-Net encoder plus a node transformer that scores all node pairs of two consecutive frames) on all 199 movies.**  
コンペのベースラインとなったリンカー構造(時間的U-Netエンコーダーと、連続する2フレームのすべてのノードペアをスコアリングするノードTransformer)を全199動画で再学習させました。

**It is run in both directions; the two softmax-normalised probabilities and their weighted harmonic mean are features.**  
これを双方向で実行し、ソフトマックスで正規化された2つの確率とそれらの重み付き調和平均を特徴量とします。

**Cell embedding.**  
細胞の埋め込み表現。

**A small 3D CNN encodes a 16 × 32 × 32 raw patch around each node.**  
小さな3D CNNが、各ノード周辺の16 × 32 × 32の生パッチをエンコードします。

**It is trained with InfoNCE so that the same annotated cell at t and t+1 is closer than its competing candidates.**  
InfoNCE損失で訓練されており、t と t+1 における同一のアノテーション済み細胞が競合候補よりも近くなるように学習されます。

**The cosine similarity of the two ends of a link is a feature.**  
リンクの両端のコサイン類似度を特徴量とします。

#### 3.3 Edge model (エッジモデル)

**LightGBM on all of the above (47 features), trained on out-of-fold detections, gives the link probability p used everywhere downstream.**  
Out-of-fold検出値で訓練された、上記のすべて(47特徴量)を入力とするLightGBMが、下流のすべての処理で使われるリンク確率 p を出力します。

**Lesson.**  
得られた教訓。

**Out-of-fold stacking under-values evidence from models trained on all data.**  
Out-of-foldによるスタッキングは、全データで訓練されたモデルからの手がかりを過小評価してしまいます。

**Locally we could only use fold versions of the transformer, trained on less data, and its feature looked worth almost nothing.**  
ローカル環境では、より少ないデータで訓練されたフォールド版のTransformerしか使用できず、その特徴量はほとんど価値がないように見えました。

**On the leaderboard, the version trained on all 199 movies was worth +0.005.**  
しかしリーダーボード上では、全199動画で訓練されたバージョンは+0.005の価値がありました。

---

### 4. Divisions (細胞分裂)

#### 4.1 Why a per-node division cost (なぜノードごとの分裂コストなのか)

**In the ILP a division means that one node keeps two outgoing edges.**  
ILPにおいて分裂とは、1つのノードが2つの出エッジを保持することを意味します。

**The second daughter's edge is usually weak: that daughter is further away and looks different from its mother.**  
2番目の娘細胞のエッジは通常弱くなります: その娘細胞は母細胞からより遠くにあり、母細胞とは外見も異なるためです。

**With one global division cost, the solver either never opens forks, or opens false ones wherever two cells sit close together.**  
一律の単一の分裂コストを用いると、ソルバーは分岐を全く作らなくなるか、あるいは2つの細胞が近接しているあらゆる場所で誤った分岐を作ってしまいます。

**We therefore give each node its own division cost, cost = 6 − 16·s (5 − 16·s in sparse movies), where s is the learned probability that the node is a dividing mother.**  
そのため、私たちは各ノードに個別の分裂コスト、cost = 6 − 16·s(疎な動画では 5 − 16·s)を与えました。ここで s はノードが分裂中の母細胞であるという学習済みの確率です。

**Forks open only where the evidence supports them.**  
証拠がそれを支持する場所にのみ、分岐が開かれます。

#### 4.2 The division score s (分裂スコア s)

**Divisions are rare and subtle.**  
分裂は稀であり、かつ捉えにくい現象です。

**The mother rounds up and brightens, its chromatin splits, and two compact, bright daughters appear that keep moving apart for a few frames:**  
母細胞は丸くなって明るさを増し、クロマチンが分裂し、2つの引き締まった明るい娘細胞が現れて数フレームにわたって互いに離れ続けます:

**Labels.**  
ラベル。

**A node with two annotated children is positive, a node with one annotated child is negative.**  
2つのアノテーションされた子ノードを持つノードを陽性(Positive)とし、1つのアノテーションされた子ノードを持つノードを陰性(Negative)とします。

**Unannotated nodes are never used as negatives, because most real divisions are unannotated.**  
アノテーションされていないノードは陰性としては決して使用しません。本物の分裂のほとんどがアノテーションされていないためです。

**For the same reason, hard-negative mining hurt badly: the "hardest negatives" are mostly real, unannotated divisions.**  
同じ理由から、ハードネガティブマイニングは大きな悪影響を及ぼしました: 最も「紛らわしい陰性(Hard Negative)」の正体は、大半が実際のアノテーションされていない本物の分裂だったからです。

**Three scorers.**  
3つのスコア算出器。

| scorer (算出器) | input (入力) | model (モデル) |
| :--- | :--- | :--- |
| **Division CNN** <br> 分裂CNN | **5-frame crop (t−2 … t+2) of 9 × 33 × 33 voxels around the node** <br> ノード周辺の9 × 33 × 33ボクセルの5フレームクロップ (t−2 … t+2) | **6 CNNs (3 seeds, with and without copy-paste augmentation), 4 flips** <br> 6つのCNN (3つの乱数シード、コピペ拡張の有無)、4方向フリップ |
| **Graph model** <br> グラフモデル | **53 features of the node in the candidate graph: its best links and how contested they are, where its candidate daughters are (distance, opposite sides, symmetry), whether they continue, motion, local density, the CNN score** <br> 候補グラフ内のノードの53個の特徴量: 最良リンクと競合度合い、候補娘細胞の位置(距離、反対側にあるか、対称性)、それらの継続性、動き、局所密度、CNNスコア | **CatBoost** |
| **Pair model** <br> ペアモデル | **candidate (mother, daughter, daughter) triplets with 174 image features, plus graph, embedding and link features of both daughter edges** <br> 候補となる(母細胞、娘細胞、娘細胞)のトリプレット。174個の画像特徴量に加え、両方の娘エッジのグラフ、埋め込み、リンク特徴量 | **LightGBM + CatBoost** |

**How the pair model works:**  
ペアモデルの仕組み:

**For every node, candidate daughter pairs are taken from the next frame: both within 13.5 µm, sisters 4.5–17 µm apart, the mother near their midpoint, the daughters on opposite sides.**  
すべてのノードについて、次フレームから候補となる娘細胞ペアを抽出します: 両方とも13.5 µm以内、姉妹間の距離は4.5〜17 µm、母細胞がそれらの中点付近にあり、娘細胞が反対側に位置していること。

**The 174 image features follow the biology: the mother's brightness, texture, compactness and size over her last 7 frames, the intensity drop at her position at the split, both daughters over their first 3 frames, sister separation and its growth, movement along the division axis.**  
174個の画像特徴量は生物学的現象に忠実です: 過去7フレームにおける母細胞の輝度、テクスチャ、引き締まり具合、サイズ、分裂時における母細胞の位置での輝度低下、最初の3フレームにおける両娘細胞の挙動、姉妹細胞間の離間距離とその増加、分裂軸に沿った運動。

**A node's score is its best pair.**  
あるノードのスコアは、そのノードの最良ペアのスコアとなります。

**The final score is s = max(pair model, graph model), which added +0.003 over the graph model alone.**  
最終スコアは s = max(ペアモデル, グラフモデル) とし、グラフモデル単体に対して+0.003の改善をもたらしました。

#### 4.3 Divisions inside and after the ILP (ILP内部および後処理での分裂処理)

**Relaxed admission.**  
参入条件の緩和。

**A link enters the ILP graph if p > 0.3 − 0.5·s, so likely mothers keep their weaker second link.**  
p > 0.3 − 0.5·s である場合にリンクがILPグラフに組み込まれるため、母細胞である可能性が高いノードはより弱い2番目のリンクも保持できるようになります。

**Second-daughter rescue.**  
第2娘細胞の救済。

**For nodes with s ≥ 0.2, links that are not the node's best link take max(p, transformer probability), because the transformer is often more confident about the second daughter.**  
s ≥ 0.2 のノードについては、そのノードの最良リンク以外のリンクは max(p, transformer確率) を採用します。Transformerの方が第2娘細胞に対してより高い確信度を持つことが多いためです。

**Fork verifier.**  
分岐検証器。

**After the ILP, both daughters of every fork are traced in the raw images from 3 frames before to 5 frames after the fork (motion-compensated DoG peak search).**  
ILPの処理後、すべての分岐における両方の娘細胞を、分岐の3フレーム前から5フレーム後まで元画像内で追跡します(動き補償を施したDoGピーク探索)。

**157 features describe their separation over time, peak quality and displacement.**  
157個の特徴量が、時間経過に伴う分離挙動、ピーク品質、変位を表現します。

**A CatBoost model removes the weaker daughter edge when its score is below 0.25, unless the two daughters clearly move apart along the fork axis.**  
CatBoostモデルは、2つの娘細胞が分岐軸に沿って明らかに離れていっている場合を除き、スコアが0.25未満の弱い方の娘エッジを削除します。

**+0.002, and +0.003 more from tuning the threshold.**  
これにより+0.002、さらに閾値の調整により追加で+0.003の向上となりました。

**Division base cost.**  
分裂の基本コスト。

**The optimum is broad, around 5–6; dense movies use 6, sparse movies 5.**  
最適値の範囲は広く、およそ5〜6の間です。密集した動画では6を、疎な動画では5を使用します。

**Effect.**  
効果。

**Turning on the division score took the leaderboard from 0.913 to 0.948 (+0.035 on both public and private).**  
分裂スコアを有効にしたことで、リーダーボードのスコアは0.913から0.948へと跳ね上がりました(パブリック・プライベートともに+0.035)。

---

### 5. Tracking and post-processing (トラッキングと後処理)

#### 5.1 ILP

**tracksdata with the SCIP solver, one movie per process:**  
SCIPソルバーを用いてtracksdataを解きます(1プロセスにつき1動画):

**edge reward p;**  
エッジ報酬 p ;

**appearance cost 0.5, disappearance cost 3.2;**  
出現コスト 0.5、消失コスト 3.2 ;

**per-node division cost 6 − 16·s, or 5 − 16·s in sparse movies (section 4).**  
ノードごとの分裂コスト 6 − 16·s、疎な動画では 5 − 16·s(セクション4)。

#### 5.2 Cleaning the tracks (トラックのクリーニング)

**Short-track filter.**  
ショートトラックフィルター。

**Isolated single nodes are removed.**  
孤立した単一ノードを削除します。

**Motion-compensated smoothing.**  
動き補償平滑化。

**The per-frame tissue translation is estimated from confident non-division links.**  
フレームごとの組織の平行移動量を、信頼性の高い非分裂リンクから推定します。

**Coordinates are then smoothed along each track with a local line fit (±3 frames) after removing that translation.**  
その後、その平行移動量を除去した上で、局所直線フィッティング(±3フレーム)により各トラックに沿って座標を平滑化します。

**Gap bridging.**  
ギャップ補間。

**A track ending at t and a nearby track starting at t+2 or t+3 are joined through linearly interpolated nodes (+0.002).**  
t で終了するトラックと、近傍の t+2 または t+3 で始まるトラックを線形補間ノードによって結合します(+0.002)。

#### 5.3 Track-support gate (sparse movies) (トラック支持ゲート: 疎な動画向け)

**Problem.**  
課題。

**In sparse movies the main remaining error is over-counting: long, dim tracks that are not real nuclei.**  
疎な動画において残存する主なエラーは過剰カウントです: 本物の細胞核ではない、長く薄暗いトラックが存在することです。

**Method.**  
手法。

**The same principle as in 2.2, now applied to the final tracks with two independent detectors, 3D Net-64 and a sensitive DoG detector:**  
2.2と同じ原理を、3D Net-64と高感度DoG検出器という2つの独立した検出器を用いて最終トラックに適用します:

**Cut the tracks into division-free segments.**  
トラックを分裂を含まないセグメントに分割します。

**A node is confirmed if an independent detection is close by in the same frame.**  
同一フレーム内で独立した検出結果が近くにあれば、そのノードは「確認済み」と判定されます。

**Remove a segment if too few of its nodes are confirmed:**  
確認済みノードがあまりに少ないセグメントを削除します:

| movie spacing (動画の核間隔) | a node is confirmed if … (ノードが確認済みとなる条件) | remove the segment if confirmed < (セグメントを削除する確認率の閾値) |
| :--- | :--- | :--- |
| **19.7–25 µm** | **a DoG detection within 4 µm or a 3D Net-64 node within 5 µm** <br> 4 µm以内にDoG検出があるか、5 µm以内に3D Net-64ノードがある | **40%** |
| **> 25 µm** | **a DoG detection within 5 µm and a 3D Net-64 node within 5 µm** <br> 5 µm以内にDoG検出があり、かつ5 µm以内に3D Net-64ノードがある | **20%** |

**Dense movies are not touched.**  
密集した動画には手を加えません。

**Effect.**  
効果。

**Positive in local validation and neutral on public (0.969), so it went into the final submission.**  
ローカル検証ではプラス、パブリックでは横ばい(0.969)だったため、最終サブミッションに採用しました。

**On private it cost 0.001, while a more lenient setting gained 0.001: like other changes to the node count, it transferred to new embryos less predictably than the rest of the pipeline.**  
プライベートでは0.001のマイナスとなりましたが、より寛容な設定では0.001プラスとなっていました: ノード数を変更する他の修正と同様、パイプラインの他の部分に比べて新しい胚への汎化の予測が困難でした。

---

#### 5.4 Model sizes and cost (モデルサイズと計算コスト)

| model (モデル) | size (サイズ) | input (入力) |
| :--- | :--- | :--- |
| **3D Net-128 / 3D Net-128-PL / 3D Net-64** | **18 M parameters each (10 / 10 / 50 epochs)** <br> 各18Mパラメータ (10 / 10 / 50 エポック) | **three-frame window of the whole volume** <br> ボリューム全体の3フレームウィンドウ |
| **Edge transformer** <br> エッジTransformer | **2.1 M parameters** <br> 2.1Mパラメータ | **two consecutive frames, downsampled** <br> ダウンサンプリングされた連続2フレーム |
| **Division CNN** <br> 分裂CNN | **1.4 M parameters × 6** <br> 1.4Mパラメータ × 6 | **5-frame patch of 9 × 33 × 33 voxels** <br> 9 × 33 × 33ボクセルの5フレームパッチ |
| **Cell embedding** <br> 細胞埋め込み | **0.24 M parameters** <br> 0.24Mパラメータ | **16 × 32 × 32 patch** <br> 16 × 32 × 32 パッチ |
| **Edge model, division graph and pair models, fork verifier** <br> エッジモデル、分裂グラフ/ペアモデル、分岐検証器 | **gradient-boosted trees (CPU)** <br> 勾配ブースティング木 (CPU) | **tabular features** <br> テーブル形式の特徴量 |

**The whole chain runs in one Kaggle notebook on 2 × T4, about 31 minutes on the 4 visible movies.**  
パイプライン全体は2 × T4を搭載した1つのKaggleノートブック上で動作し、可視化されている4本の動画に対して約31分で完了します。

**The heaviest parts are the division CNN (split over both GPUs by node count), the transformer link scoring and the ILP.**  
最も負荷の高い処理は、分裂CNN(ノード数に応じて両方のGPUに分散処理)、Transformerによるリンクのスコアリング、そしてILPです。

---

### 6. Results (結果)

**Each row adds one component, and every row is a single submission.**  
各行はコンポーネントを1つずつ追加した結果であり、各行が単一のサブミッションに対応しています。

**The detector work (0.875 → 0.906) and the division score (0.913 → 0.948) are the two big blocks.**  
検出器の改良(0.875 → 0.906)と分裂スコア(0.913 → 0.948)が、2つの大きな柱でした。

**Linking and post-processing add the rest.**  
リンク付けと後処理が残りの部分を押し上げました。

**The final submission scored 0.969 / 0.962.**  
最終サブミッションのスコアは 0.969 / 0.962 でした。

**Our best private score was 0.969 / 0.964.**  
私たちのプライベートでのベストスコアは 0.969 / 0.964 でした。

**It came from a more lenient setting of the same track-support gate, which tied on public, so we did not pick it.**  
それは同じトラック支持ゲートをより寛容に設定したものでしたが、パブリックでスコアが同着だったため選択しませんでした。

**Across all submissions of our chain, private stayed close to public:**  
私たちのパイプラインの全提出を通じて、プライベートスコアはパブリックスコアと近い値を保ち続けました:

---

### 7. What did not help (効果がなかったこと)

| idea (アイデア) | effect on the leaderboard (リーダーボードへの影響) |
| :--- | :--- |
| **The pseudo-label 3D Net as the detector itself** <br> 疑似ラベル3D Netを検出器そのものとして単独使用 | **−0.003 to −0.004** |
| **Registering frames before computing link geometry** <br> リンクの幾何学的特徴を計算する前にフレームの位置合わせ(レジストレーション)を行う | **−0.003** |
| **A DoG-only support gate (without 3D Net-64)** <br> DoGのみによる支持ゲート (3D Net-64なし) | **−0.001 public, −0.011 private** <br> パブリックで−0.001、プライベートで−0.011 |
| **TensorRT FP16 for all networks** <br> 全ネットワークにTensorRT FP16を適用 | **42% faster, −0.001; we kept fp32** <br> 42%高速化するも−0.001。fp32を維持 |
| **Hard-negative mining for division models, re-admitting nodes the ILP dropped, constant coordinate offsets** <br> 分裂モデルに対するハードネガティブマイニング、ILPが除外したノードの再採用、固定値での座標オフセット | **negative in local validation** <br> ローカル検証でスコア低下 |

---

### 8. Takeaways (得られた知見・教訓)

**Look at the biology first.**  
まずは生物学的特徴に目を向けること。

**The biggest gains came from turning simple observations (how a division looks, how nuclei scale with density, how tissue drifts) into small, targeted models.**  
最大のスコア向上は、単純な観察(分裂がどのように見えるか、細胞核が密度によってどうスケールするか、組織がどうドリフトするか)を、焦点を絞った小さなモデルへと具現化することからもたらされました。

**Validate the way the test set differs.**  
テストセットの差異の現れ方に合わせて検証を設計すること。

**New embryos meant cross-embryo splits for the detector and out-of-fold stacking for everything downstream.**  
未知の胚がテスト対象となるため、検出器には胚間分割バリデーションを採用し、下流のすべての処理にはOut-of-foldによるスタッキングを用いました。

**The division term is worth fighting for.**  
分裂の項はとことん追求する価値があること。

**A learned, per-node division cost inside the ILP was the largest single gain, and it only works with labels taken strictly from annotated tracks.**  
ILP内部における学習ベースのノードごとの分裂コストが単一で最大の改善であり、これはアノテーション済みトラックから厳格に抽出したラベルがあって初めて機能します。

**Use a second model as a check, not as a replacement.**  
2つ目のモデルは置き換え用ではなく、確認・検証用として使うこと。

**Both the pseudo-label refinement and the track-support gate keep a track only when an independent model agrees with it.**  
疑似ラベルの洗練とトラック支持ゲートの双方が、独立したモデルの合意が得られた場合にのみトラックを保持するという仕組みを取っています。

**Be careful with changes that alter how many nodes you output.**  
出力するノード数を変化させるような変更には慎重になること。

**They transferred worst from local validation to the leaderboard, especially on sparse movies (the track-support gates in 5.3 and section 7).**  
それらはローカル検証からリーダーボードへの相関が最も悪く、特に疎な動画においてその傾向が顕著でした(5.3およびセクション7のトラック支持ゲート)。

---

**Author [BarryZhou](https://www.kaggle.com/songqizhou) songqizhou**  
著者 [BarryZhou](https://www.kaggle.com/songqizhou) songqizhou

**Share**  
共有

**Project Links**  
プロジェクトリンク

**[Biohub 4th Place Solution 7 days ago · 8 upvotes](https://kaggle.com/code/songqizhou/biohub-4th-place-solution)**  
[Biohub 4位解法 7日前 · 8 upvotes](https://kaggle.com/code/songqizhou/biohub-4th-place-solution)

**Kaggle Notebook**  
Kaggleノートブック

**Public**  
公開

**Citation**  
引用

**DOI (Digital Object Identifier) https://doi.org/10.34740/kaggle/w/115972**  
DOI (デジタルオブジェクト識別子) https://doi.org/10.34740/kaggle/w/115972
