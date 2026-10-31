<img src="image.png" width="800" />

## Abstract
- biohub 2位解法のまとめ

## 概要
先日、Biohub細胞トラッキングコンペが完了しました。
Private LB 1353位/3800人という大変残念な結果となってしまいました。初参加ということで雰囲気を味わうということを目的にしてたんですが、実際がっつり頑張ったのに、銅メダルにも届かないという結果は、すごく悔しいです。
しかし、
Kaggleは勝者の解法は公開されるという素晴らしい決まり事がありまして、もうこれは徹底的に解析して自分のスキルにするしかないなと。

## biohub 2位の解法まとめ
2位の方だけでなく、上位陣の人たちはモデルを複数駆使してて、また学習にも工夫が随所にみられるました。そして公開されるのは学習後モデルそのものでして、学習に使用したコードやデータ群までは公開されないんですね。ただ上位陣はそれぞれ自身の提出コードの解説ページを公開してくれてて、そこで結構情報を得られたりします。

- 解説のURL
https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/2nd-place-solution
- ソースコード
https://www.kaggle.com/code/soheilayati/biohub-2nd-place-notebook

## プロジェクト構成

このプロジェクトで使用ファイル（JSON, py, 重みファイル `.pt` / `.npz` / `.txt` 等）群とディレクトリの役割を機能カテゴリ別に整理。
**input**: /kaggle/input/datasets/soheilayati

---

### 1. 共通・データ・環境 (`.py` / `.zarr` / `.whl`)

| 相対パス | ファイル | 役割・機能 |
| :--- | :--- | :--- |
| `biohub-v343-hard-dim-code/` | `biohub/` パッケージ全般<br>(`pipeline/`, `tracker/`, `detector/` 等) | **メインパイプライン実装:** 検出、グラフ解法(Solver)、コンセンサス、モーション推定などの全コード本体。 |
| `input/test/` | `*.zarr` (4 データセット) | **入力テストボリューム:** 推論対象の 3D 顕微鏡タイムラプス画像データセット。 |
| `biohub-cell-tracking-wheels/`<br>`input/support_pack/wheels/` | `zarr-*.whl`<br>`tracksdata-*.whl` 等 | **オフライン配布ホイール:** インターネット遮断環境下で必要なライブラリ群をインストールするための wheel 群。 |

---

### 2. 検出器 (Detector) 重みファイル (`.pt`)

3D画像ボリュームから細胞の存在ヒートマップやオフセット・フローを抽出する深層学習モデル群。
- train/ の 3D 画像と正解の細胞中心座標を入力とし、3D U-Net などのニューラルネットワークを PyTorch で、GPU 学習 させて生成。
- 過学習を防ぎ汎化性能を高めるため、データを 5 つに分けた交差検証 (5-Fold CV) や、フレーム数を変えたモデル (3フレーム/5フレーム) をそれぞれ学習させて保存。

| 相対パス | ファイル | 役割・機能 |
| :--- | :--- | :--- |
| `biohub-cell-tracking-weights-v25-consensus/` | `fold0.ep05.pt` 〜 `fold4.ep05.pt`<br>`v24.fold0.ep05.pt` 〜 `v24.fold4.ep05.pt` | **主幹検出器 (V21 & V24):** 5分割交差検証(5-fold)による主検出モデル計10個。ヒートマップアンサンブルの基盤。 |
| `biohub-seed1-detector/` | `full.seed1.ep02.pt` | **独立シード検出器:** アンサンブルのヒートマップ多様性を向上させるための別シードモデル。 |
| `biohub-v35-strong-detectors/` | `strong.fold0.ep05.pt` 〜 `strong.fold4.ep05.pt` | **Stage-6 強検出器:** 高感度・高精度な第6ステージ用の5-fold検出モデル。 |
| `biohub-v41-probe-weights/` | `v39.fold0.ep05.pt` 〜 `v39.fold4.ep05.pt` | **V39 検出器 (5-frame入力):** 5フレーム時系列を入力として捉える高精度検出モデル。 |
| `biohub-v57-deployment-weights/` | `v38.fold0.ep05.pt` 〜 `v38.fold4.ep05.pt` | **V38 検出器 (残差検出):** V39 と組み合わせてリカバリ・補完を行う検出モデル。 |
| `biohub-v389-structured-head/` | `residual.pt`<br>`v389_structured_manifest.json` | **残差補正モデル:** V230木構造対応付けと併用し、画像残差から座標精度を高めるモデル。 |

---

### 3. トラッカー・グラフ判定モデル (`.npz` / `.json` / `.txt`)

検出された細胞候補点同士を時系列で結び、細胞分裂や枝刈りを判定する機械学習モデル群。
- 検出器が見つけた細胞候補点から、「細胞間のユークリッド距離」「動きの滑らかさ」「輝度の一致度」など何百種類の特徴量を計算して表データ (DataFrame) を作る。
- 正解のトラッキング線（エッジ）を教師データとして、LightGBM やロジスティック回帰 を学習させてツリー構造や重みをシリアライズする。

| 相対パス | ファイル | 役割・機能 |
| :--- | :--- | :--- |
| `biohub-cell-tracking-weights-v25-consensus/` | `count_model_v1.json` | **細胞数推定モデル:** 画像統計量からフレーム内の細胞数を推定し、探索予算(budget)を決定する。 |
| 同上 | `edge_model_b125.npz` | **エッジ結合モデル (GBM):** 時間軸方向の細胞間の接続確率を予測する。 |
| 同上 | `pair_model_v0.npz`<br>`pair_stack_faithful.npz` | **ペアモデル / ペアスタック:** 2点間の同一細胞判定および分裂判定。 |
| 同上 | `pair_allimg_lr.npz`<br>`pair_full_lr.npz`<br>`pair_hardneg_lr.npz`<br>`pair_highres_lr.npz` | **多様なペア判定器 (LR):** 全画像版、難例(Hard Negative)版、高解像度版などのロジスティック回帰スタック。 |
| 同上 | `fork_confirmation_lr.npz`<br>`edge_tracklet_lr_c0p003.npz` | **分岐確定 / トラックレット結合:** 細胞分裂(fork)の最終確認および小トラック(tracklet)の結合判定。 |
| 同上 | `component_current_lgb_l15_fold*.txt`<br>`edge_tracklet_lgb_l7_fold*.txt`<br>`edge_postselect_lgb_l31_fold*.txt`<br>(各 `.json` メタデータ) | **LightGBM ブースター群:** 連結成分評価、トラックレット評価、最終エッジ後処理選択用の勾配ブースティング決定木群。 |
| `biohub-v57-deployment-weights/` | `fold0.txt` 〜 `fold4.txt`<br>`v57_deployment_manifest.json` | **V57 リスク評価モデル (GBM):** V39 との置き換え時にノード除去のリスク度合いをスコアリングする。 |
| `biohub-v343-hard-dim-code/v343_models/` | `full.dim.txt`<br>`full.cell.txt`<br>`v343_dim_scaffold_manifest.json` | **特化足場モデル (Dim Specialist):** 暗い細胞や微弱な領域の検出足場を補強する LightGBM モデル。 |
| `biohub-v343-hard-dim-code/v230_models/` | `tree.txt` | **木構造対応付けモデル (V230):** 過去フレームと未来フレームの細胞対応付けツリー定義。 |

---

### 4. 後処理・保護モジュール (動的読み込み Python + マニフェスト JSON)

各バージョンごとの保護・改善ロジックを担う独立したモジュール群。`importlib` を通じて動的に `.py` がロードされ、同時に `.json` で設定・検証が実行される。
- ここは AI の自動学習というよりも、開発者の職人芸（エラー分析と最適化） 。
 - 途中段階の予測結果を可視化し、「暗い細胞が途切れやすい」「端っこにノイズの枝が残る」といった課題を見つけ、それを救うアルゴリズム（V394, V399, V454, V507, V533）を設計している。
- そのパラメータ（hole_um=4.5, min_contrast=0.1 など）をグリッドサーチ等で探索し、最も CV スコアが高くなった数値をマニフェスト (.json) に固定。

| 相対パス | ファイル | 役割・機能 |
| :--- | :--- | :--- |
| `biohub-v394-component-protector/` | `v394_component_protector.py`<br>`v394_component_manifest.json` | **連結成分保護:** 短すぎるトラックやノイズ成分をクラスタ別特性に応じてプルーニング(枝刈り)・保護する。 |
| `biohub-v399-tail-protector/` | `v399_tail_protector.py`<br>`v399_tail_manifest.json` | **終端テイル保護:** トラック終端の不確実な余分な枝や偽陽性テイルをトリミングする。 |
| `biohub-v454-current-graph-gap-admission/` | `v454_gap_admission.py`<br>`v454_manifest.json` | **ギャップ補完許可判定:** 途切れた細胞トラック間の接続(ギャップ)を最終グラフに許可・追加する。 |
| `biohub-v507-localization-fork/` | `v507_candidate.py`<br>`v507_manifest.json` | **局所化・分岐修正:** 細胞分裂(fork)判定の修正と、細胞重心座標の高精度化を行う。 |
| `biohub-v533-addback-edge/` | `v533_addback_edge.py`<br>`v533_manifest.json` | **最終エッジ補完 & 境界トリム:** コントラスト閾値に基づくエッジの再追加(add-back)および画像端(xy edge)の不要ノード除去。 |

---

# Biohub V533 シーケンス図 (セル単位)

ノートブックのコードセル(`# %%`)ごとに分割したシーケンス図。5〜8が処理の本体。ここを理解するとほぼ網羅できたといっても過言ではない。

## 1. (cell 1: line 7〜133) パス探索とマニフェスト読み込み
要は、ローカル環境とKaggle環境とではパスが違うので、それを吸収している処理。

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant Walk as walk()
    participant CodeRoot as find_code_root()
    participant TestRoot as find_test_root()
    participant WeightsRoot as find_weights_root()
    participant UniqueRoot as find_unique_root()
    participant JsonLoad as json.loads()

    Script->>CodeRoot: find_code_root()
    CodeRoot->>Walk: walk(ROOTS)
    Walk-->>CodeRoot: ディレクトリ巡回
    CodeRoot-->>Script: CODE_ROOT

    Script->>TestRoot: find_test_root()
    TestRoot->>Walk: walk(ROOTS)
    Walk-->>TestRoot: ディレクトリ巡回
    TestRoot-->>Script: TEST_ROOT

    Script->>WeightsRoot: find_weights_root()
    WeightsRoot->>Walk: walk(ROOTS)
    Walk-->>WeightsRoot: ディレクトリ巡回
    WeightsRoot-->>Script: WEIGHTS_ROOT

    Script->>UniqueRoot: find_unique_root(filename)
    UniqueRoot->>Walk: walk(ROOTS)
    Walk-->>UniqueRoot: ディレクトリ巡回
    UniqueRoot-->>Script: 各バージョンRoot (V39, V57, V343, ...)

    Script->>JsonLoad: json.loads(manifest_path.read_text())
    JsonLoad-->>Script: 各種マニフェスト辞書
```

---

## 2. (cell 2: line 134〜164) 依存関係 (zarr) のオフライン導入
入力データ(zarr)の読み込み準備(zarr操作ライブラリのインストールとか)。

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant ImportMod as importlib.import_module()
    participant Walk as walk()
    participant PipInstall as subprocess.run()

    Script->>ImportMod: importlib.import_module('zarr')
    alt すでにインストール済み
        ImportMod-->>Script: モジュール読み込み成功
    else 未インストール (ModuleNotFoundError)
        Script->>Walk: walk(ROOTS)
        Walk-->>Script: wheel_dir (zarr-*.whl 所在)
        Script->>PipInstall: subprocess.run(pip install --no-index --find-links=...)
        PipInstall-->>Script: インストール完了
    end
```

---

## 3. (cell 3: line 165〜185) 環境記録とデバイス選定
実行環境のチェックとか(GPU載ってないならエラーにしてる)。

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant ModImport as __import__()
    participant SelectDevice as biohub.device.select_device()

    loop 各主要ライブラリ ('numpy', 'scipy', 'torch', etc.)
        Script->>ModImport: __import__(m)
        ModImport-->>Script: バージョン確認
    end

    Script->>SelectDevice: select_device(allow_cpu=False)
    SelectDevice-->>Script: SPEC (device='cuda', capability, vram_gb, amp_dtype)
```

---

## 4. (cell 4: line 186〜535) モデルアンサンブルのロードと検証
cell 4 では、30 個以上 のモデルを同時にロードしてアンサンブルを構築する。

- このセルで実際にロードしている「アンサンブル」の内訳
  - **主幹検出器チーム(10個)**: 
    データを 5 分割して学習させた 5 人のモデル (fold0 〜 fold4) × 2 世代分 (V21 と V24) = 計 10 個
  - **強力な専門検出器チーム(5個)**:
    別の設定で極限まで鍛えた Stage-6 強力モデル 5 個
  - **長時間の動きを見るチーム(5個)**:
    5 フレームの時系列を捉える V39 モデル 5 個
  - **決定木・トラッカーチーム(数十個)**:
    細胞同士を繋ぐか判定する LightGBM のツリー群 (BoosterEnsemble)

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant CountModelLoad as CountModel.load()
    participant EdgeModelLoad as EdgeModel.load()
    participant LoadModels as load_models()
    participant ShaCheck as hashlib.sha256()
    participant PairModelLoad as PairModel.load() / PairStack.load()
    participant BoosterLoad as BoosterEnsemble.load() / Booster()

    Script->>CountModelLoad: CountModel.load('count_model_v1.json')
    CountModelLoad-->>Script: COUNT_MODEL

    Script->>EdgeModelLoad: EdgeModel.load('edge_model_b125.npz')
    EdgeModelLoad-->>Script: EDGE_MODEL

    Script->>ShaCheck: sha256(extra_path.read_bytes())
    ShaCheck-->>Script: ハッシュ一致検証

    Script->>LoadModels: load_models(paths, device)
    LoadModels-->>Script: MODELS (V21/V24 consensus), SEED_MODELS

    Script->>LoadModels: load_models(strong_paths, device)
    LoadModels-->>Script: STRONG_MODELS (Stage-6)

    Script->>LoadModels: load_models(V39/V38 paths, device)
    LoadModels-->>Script: V39_MODELS, V38_MODELS

    Script->>PairModelLoad: PairModel.load(), PairStack.load()
    PairModelLoad-->>Script: PAIR_MODEL, PAIR_STACK, ALLIMG_MODEL, etc.

    Script->>BoosterLoad: BoosterEnsemble.load(), Booster()
    BoosterLoad-->>Script: COMP7_MODEL, V343_DIM_MODEL, V343_CELL_MODEL
```

---

## 5. (cell 5: line 541〜669) 処理A: ボリューム取得・細胞検出

- **volumes_from_zarr()** ...処理の一番最初で、ロバスト正規化、等方化ダウンサンプリング(1, 4, 4)、float16 化を実施。
  (このコンペはz,y,xで[1.625μm, 0.40625μm, 0.40625μm]の解像度の違いがある)。
- **image_quantiles()** ... Dataset毎の統計情報::四分位数を取得している。取得したデータは、予測見積もりnode数に使用している。
- **v394.acquisition_cluster()** ... "acq_00:通常コントラストの標準的な撮影群" or "acq_01:低コントラスト・暗めの撮影群"を判別している。
- **detect_original_and_rotation()** ... メインの細胞検出器(V21モデル群)を実行する際に、「通常画像」と「180度回転させた画像（TTA）」の2通りで推論を行い、2種類の候補テーブルを同時に生成する関数。

```mermaid
%%{init: {"sequence": {"noteAlign": "left"}}}%%
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant VolZarr as volumes_from_zarr()
    participant Quantiles as image_quantiles()
    participant Cluster as v394.acquisition_cluster()
    participant DetectRot as detect_original_and_rotation()
    participant PeakRet as retain_original_peaks()
    participant Scaffold as specialist_scaffold()
    participant DetectBank as detect_bank()→detect()

    Script->>VolZarr: volumes_from_zarr(path)
    VolZarr-->>Script: shared_volumes
    Note over Script: shared_volumes:np.ndarray[shape=(100,Z,Y,X)]<br/>生データ→正規化→ダウンサンプリング(1:4:4のブロック平均→float16化)

    Script->>Quantiles: image_quantiles(path)
    Quantiles-->>Script: shared_quantiles
    Note over Script: shared_quantiles:主催者提供の分位点データ(明暗,コントラスト高低の判断基準)

    Script->>Cluster: v394.acquisition_cluster(path, router)
    Cluster-->>Script: v394_cluster
    Note over Script: v394_cluster:Dataset単位で、acq_00,acq_01にクラス分け

    Script->>DetectRot: detect_original_and_rotation(V21_MODELS, shared_volumes)
    DetectRot-->>Script: c21_count, c21, original_flow
    Note over Script: 検出本体(t-1,t,t+1)。<br/>c21_count:DataFrame...正面Heatmap※<br/>c21:DataFrame...正面/回転TTAアンサンブルHeatmap※<br/>original_flow:np.ndarray(shape=[100,3,Z,Y,X,float32])...Dataframe毎のFlowベクトル

    Script->>PeakRet: retain_original_peaks(c21, c21_count)
    PeakRet-->>Script: c21, peak_retention_audit
    Note over Script: 正面/回転TTAアンサンブル(前処理)で消した正面検出(c21_count)を救済する処理<br/>c21:DataFrame...元c21+救済後の検出結果<br/>peak_retention_audit:dict...救済件数などの監査情報※

    rect rgb(240, 240, 260)
    %% 動画ごとの細胞密度に合わせて、探索・保持する細胞候補の上限枠(キャパシティ)を自動調整している
    Script-->>Script: specialist_features = features_from(path.stem, shared_quantiles, c21_count)
    Note over Script: Dataset(=1動画)の特徴量を集計<br/>specialist_features: dict...Dataset(=1動画)の特徴量

    Script-->>Script: specialist_per_frame = float(COUNT_MODEL.predict_per_frame(specialist_features)[0])
    Note over Script: 1フレームあたりの細胞数(予測)を取得

    Script-->>Script: specialist_budget = max(1, int(round(specialist_per_frame * BUDGET_MULT)))
    Note over Script: 上限細胞数(budget)の算出

    Script-->>Script: specialist_cap = max(1, int(round(specialist_budget * SUPERSET_MULT)))
    Note over Script: 細胞数の上限キャパの算出(約2倍)
    end

    Script->>Scaffold: specialist_scaffold(shared_volumes, c21, V343_DIM_MODEL, V343_CELL_MODEL, specialist_cap, max_rank_shift=150, dim_region_threshold=None)
    Scaffold-->>Script: c21, v343_union, dim_specialist_audit,  
    Note over Script: 前処理で抽出した検出細胞群(c21)から、“暗いけど細胞の可能性のあるもの”を専門モデルで発掘し、トラックが途切れないための『足場候補』として候補リストに安全に追加・合流させる処理。<br/>scaffoldは足場という意味。specialistは特化したという意味。

    Note over Script: c21完成、続いて c24, c_stage6, c39, c38 を順次検出

    Script->>DetectBank: detect(V24, shared_volumes)
    DetectBank-->>Script: c24

    Script->>DetectBank: detect(Stage6, shared_volumes)
    DetectBank-->>Script: c_stage6

    Script->>DetectBank: detect(V39, shared_volumes)
    DetectBank-->>Script: c39

    Script->>DetectBank: detect(V38, shared_volumes)
    DetectBank-->>Script: c38

    Note over Script: 処理A完了 (全5系統の検出候補が出揃う)<br/>output: c21(+v343_union), c24, c_stage6, c39, c38
```

### ※c21_count:DataFrame(正面Heatmap)、c21:DataFrame(正面/回転TTAアンサンブルHeatmap)の列説明
|No|列名|説明|
|---|---|---|
|1|t|フレーム番号|
|2,3,4|z,y,x|検出中心座標(voxel)|
|5|score|検出信頼度(Heatmapピーク値)|
|6|rank_in_frame|同フレーム内scoreの順位|
|7,8,9|flow_z,y,x|次フレームへの移動量(Flow予測)|
|10|flow_sigma_um|当該Flow誤差の標準偏差σ[μm]（大:自信なし,小:自信あり）|
|11|near_um|最も近い近傍細胞までの距離(μm)|
|12|density_10um|半径10μm以内の局所細胞密度|
|13|peek_with_um|Heatmapピーク幅(μm)|
|14|boundary_um|画像の境界までの距離(μm)|
|15|emb_0～emb_47|Cell Embedding特徴量(48次元)。ReID、tracking,細胞リンク判定に使用する特徴ベクトル|

### ※peak_retention_audit:dict...救済件数などの監査情報の列説明
|No|列名|説明|
|---|---|---|
|1|heat_rows|元々のx21行数|
|2|recovered_original_rows|救済した行数|
|3|rows|最終行数|
|4|heat_fields_preserved|c21情報が改変されてないことの確認|
|5|original_fields_preserved|c21_count情報が改変されてないことの確認|
|6|maximum_frame_rows|frame毎の最大検出数|

### ※specialist_features: dict...Dataset(=1動画)分の特徴量のキー説明
NMS ... Non-Maximum Suppression(非極大値抑制)。密集した重複予測の中から「最もスコアが高い代表点だけを残し、周囲の重複した候補を間引く」 アルゴリズム。
|No|列名|説明|
|---|---|---|
|1,2|dataset, embryo|メタデータ|
|3,4,5|nms3.5_per_frame, nms5.0_per_frame, nms7.0_per_frame|NMS後候補数/フレーム|
|6,7,8|nms3.5_strong_per_frame, nms5.0_strong_per_frame, nms7.0_strong_per_frame|NMS後高スコア候補数/フレーム|
|9|pool_per_frame|全候補数/フレーム|
|10,11,12,13|score_p10, score_p50, score_p90, score_p99|スコア分位点|
|14,15,12,17,18,19,20,21|imq_0.0, imq_0.001, imq_0.01, imq_0.1, imq_0.9, imq_0.99, imq_0.999, imq_1.0|画像分位点統計量|
|22|n_frames|フレーム数|

### 5.1 v394.acquisition_cluster()のシーケンス

- メタデータ抽出（ステップ 1〜5）: 生ボリューム全体をスキャンせずに、Zarr の属性メタデータ（.attrs）から寸法・解像度スケール・パーセンタイル輝度を読み込む。
- 特徴量作成（ステップ 6）: コントラスト比やダイナミックレンジなど、撮影条件を特徴づける数値を計算して辞書にまとめます。
- 標準化（ステップ 7〜11）: 学習時に保存された平均・標準偏差(scaler_mean, scaler_scale)を使って標準化する。
- 最近傍分類（ステップ 12〜14）: あらかじめ学習・決定されている 2 つのクラスタ中心(canonical_centers)との距離を測り、近い方のインデックス(0 または 1)を選ぶ。
- 返却（ステップ 15〜16）: "acq_00" または "acq_01" の文字列として呼び出し元へ戻します。この情報は、後段の4つの後処理フェーズで使用される。

#### &emsp;&emsp; **あらかじめ学習・決定されている 2 つのクラスタ中心(canonical_centers)** とは ... 
&emsp;&emsp;&emsp;[右記](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/2nd-place-solution)に記載があり。``I grouped movies by image measurements, including brightness and contrast, using standardized features and K-means clustering.``
&emsp;&emsp;GTデータは使わず、trainデータから20特徴量を抜き出して、教師なし学習の「K-means クラスタリング（k=2）」で生成したもの。
&emsp;&emsp;trainデータ(3D動画輝度情報)を使わなかったのは、「情報量が多すぎる」のとDataset毎のコントラストや輝度情報などの 「撮影環境(カメラの写り具合)の違い」を学習させたかったためと思われ。
&emsp;&emsp;もし、trainデータを直接使用すると、「この動画は細胞が左にある」「こっちは右にある」という 細胞の位置のズレばかり を見てしまいかねない。
&emsp;&emsp;K-means によって求まった 2つのグループの重心(中心点)を、そのまま canonical_centers として .npz ファイルに保存している。

```mermaid
sequenceDiagram
    autonumber
    actor Caller as 呼び出し元<br/>(Main Script)
    participant Func as acquisition_cluster()
    participant Zarr as Zarr Storage<br/>(root.attrs & array)
    participant Router as Router Dict<br/>(V394_ROUTER)

    Caller->>Func: acquisition_cluster(path, router)
    activate Func

    Note over Func,Zarr: 1. メタデータと配列情報の取得
    Func->>Zarr: zarr.open_group(path, mode="r")
    Zarr-->>Func: root (Zarr Group)
    Func->>Zarr: root.attrs["multiscales"] から<br/>ボクセルスケール[1.625 μm,0.40625μm,0.40625μm]取得
    Func->>Zarr: array.shape から画像寸法<br/>(frames=100, depth=64, height=256, width=256) 取得
    Func->>Zarr: root.attrs["image_statistics"]["quantiles"] から<br/>各パーセンタイル輝度値(q) 取得

    Note over Func: 2. 特徴量辞書 values の構築<br/>・画像寸法、スケール、ビット深度<br/>・log1p(各パーセンタイル)<br/>・レンジ幅 (q99 - q01, q999 - q001)<br/>・コントラスト比 (q90/q10, q999/q99)

    Note over Func,Router: 3. 特徴量ベクトル化と標準化
    Func->>Router: router["feature_names"] を参照
    Router-->>Func: names = 特徴量名のリスト
    Note over Func: 特徴量名のリスト={<br/>--- 幾何・解像度情報 (8個) ---<br/>'frames', 'depth', 'height', 'width', 'scale_z', 'scale_y', 'scale_x', 'dtype_bits',<br/>--- 輝度パーセンタイル (8個) ---<br/>'log1p_q0p0', 'log1p_q0p001', 'log1p_q0p01', 'log1p_q0p1','log1p_q0p9', 'log1p_q0p99', 'log1p_q0p999', 'log1p_q1p0',<br/>--- レンジ・コントラスト比 (4個) ---<br/>'log1p_range_q99_q01', 'log1p_range_q999_q001','contrast_q90_q10', 'tail_q999_q99'" }
    Func->>Func: 辞書から特徴ベクトルを作成 (vector)
    Note over Func: vectorに、"2."で構築した特徴量辞書 valuesから、names の順番通りに数値を抜き出して、長さ20の1次元配列を格納。
    Func->>Router: router["scaler_mean"], router["scaler_scale"] を参照
    Router-->>Func: 平均値・スケール値
    Func->>Func: scaled = (vector - mean) / scale
    Note over Func: 標準化(Standardization): scaled = (値 - 平均) / 標準偏差
    Note over Func,Router: 4. 代表クラスタ中心との照合 (最近傍探索)
    Func->>Router: router["canonical_centers"] を参照
    Router-->>Func: 代表クラスタ中心 (2点)
    Func->>Func: ユークリッド距離の計算<br/>norm(canonical_centers - scaled, axis=1)
    Func->>Func: 最も近い中心のインデックス取得<br/>label = argmin(...)  # 0 または 1

    Note over Func: 5. クラスタID文字列の生成<br/>f"acq_{label:02d}" -> "acq_00" または "acq_01"

    Func-->>Caller: "acq_00" または "acq_01" を返却
    deactivate Func
```

### 5.2 detect_original_and_rotation()のシーケンス
&emsp;&emsp;・引数1 models: fold0.ep05.pt ～ fold4.ep05.pt
    &emsp;&emsp;&emsp;&emsp;・ **fold[0-4].ep05.pt** ... input:前後3フレームの3D画像、output:1.中心ヒートマップ(heatmap)、2.サブボクセル・オフセット(offset)、3.後退運動ベクトルと
    &emsp;&emsp;&emsp;&emsp;&emsp;不確実性(flow, flow_logvar)、4.画像特徴記述子(features)のモデル。一言で言うと、「前後3フレームの立体画像を入力として受け取り、細胞の中心位
    &emsp;&emsp;&emsp;&emsp;&emsp;置・サブピクセル補正量・前フレームからの移動量・細胞の特徴ベクトルを一括予測する、弱シグナル細胞に強い3D U-Net 検出器」 を、5分割交差検
    &emsp;&emsp;&emsp;&emsp;&emsp;証（5-Fold）で学習させている。
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;出力1.**中心ヒートマップ(heatmap)** ... 細胞の中心が存在する確率(0〜1）
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;出力2.**サブボクセル・オフセット(offset)** ... 粗いグリッド(1.625μm)の中での微小な位置ズレ量($\Delta z, \Delta y, \Delta x$)。元の高解像度座標に復元するために使用。
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;出力3.**後退運動ベクトルと不確実性(flow, flow_logvar)** ... 当該細胞が「前フレーム($t-1$)でどこから動いてきたか」という移動ベクトル(オプティカルフロー)と、
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;その予測の自信のなさ(分散)
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;出力4.**画像特徴記述子(features)** ... 細胞の見た目の特徴を表す埋め込みベクトル。フレーム間で「同じ細胞同士」を結びつける照合に使用。
&emsp;&emsp;・引数2 vols: shared_volumes ... zarrの本体。形状:(T, Z, Y, X) = (100, 64, 64, 64)float16型。すでに「正規化」と「等方化ダウンサンプリング(1.625 μm 立方体グリッド)」済。
&emsp;&emsp;・引数3 device: SPEC.device ... torch.device("cuda")。要はGPU指定。
&emsp;&emsp;・引数4 amp_dtype: SPEC.amp_dtype ... torch.float16。
&emsp;&emsp;・引数5 batch: バッチサイズ(整数値 8)。100フレームある動画を、「1回に8フレームずつまとめて」GPU に送って推論する。
&emsp;&emsp;・引数6 return_flow=True: 真偽値フラグ。モデルが出力する 「細胞の移動ベクトル(flow: 前フレームへの動きの向きと大きさ)」を戻り値に含めるかどうか。True に設定することで、
&emsp;&emsp;&emsp;関数の3番目の戻り値として original_flowを受け取る。このベクトルは、後段で「途切れた細胞の軌跡を繋ぐ(Gap closure)」際に、細胞がどこから移動してきたかを辿るために再利用する。

- 時系列スタックの作成（ステップ2〜3）:
指定したバッチの時刻 ts に対して、前後1フレームを含む時系列ボリューム（3フレーム分）を GPU に転送、入力テンソル x を構築。
- 通常推論と回転推論の累積加算（ステップ4〜12）:
同じバッチ x に対して、通常画像と 180度回転画像(torch.flip)の両方をモデルに入力。
得られた出力から、通常ヒートマップ(heat)、元画像と回転画像をTTA平均化したヒートマップ(rotated_heat)、移動ベクトル(flow)、不確実性(flow_logvar)、サブボクセルオフセット(offset)、および細胞識別特徴量(features)をそれぞれ取り出してモデル間で累積加算(+=)。
- モデルアンサンブルの平均化(ステップ 13〜18):
累積加算した各テンソルをモデル数 $k$ で除算し、モデル間のアンサンブル平均値を算出。
- 2系統の候補テーブル化(ステップ 19〜28):
zip(rows, (heat, rotated_heat)) により _append_rows() を2回呼び出す。
内部ではピーク検出・座標補正・細胞識別特徴量(48次元)のサンプリング等(ステップ 20〜25)を行い、通常ヒートマップ由来の候補(c21_count 相当: ステップ 26)と、回転TTAヒートマップ由来の候補(c21 相当: ステップ 27〜28)の2系統の DataFrame を同時に生成する。
- 一括返却(ステップ 29):
c21_count、c21、および途切れた軌跡の接続に使う運動ベクトル original_flow の3点をまとめて呼び出し元に返却。

```mermaid
sequenceDiagram
    autonumber
    actor Caller as 呼び出し元<br/>(2nd-place-notebook.py)
    participant Func as detect_original_and_rotation()
    participant Stacks as _stacks()<br/>(コンテキスト構築)
    participant GPU as PyTorch / GPU<br/>(V21 Models)
    participant Append as _append_rows()<br/>(ピーク抽出・テーブル化)

    Caller->>Func: detect_original_and_rotation(models, vols, device, return_flow=True)
    activate Func

    loop バッチ単位の処理 (batch=8)
        Func->>Stacks: _stacks(vols, ts, context=3)
        Stacks-->>Func: 前後フレームを重ねた3D入力テンソル(x[8, 3, 64, 64, 64])

        loop 各モデルごとの推論 (for model in models)
            Note over Func,GPU: 1. 通常と180度回転の2通りで推論<br/>with torch.autocast("cuda", dtype=amp_dtype,enabled=amp_dtype is not None and device.type == "cuda"):の効果(自動混合精度を有効)範囲内。
            Func->>GPU: original = model(x)
            Func->>GPU: rotated = model(torch.flip(x, dims=(-2, -1)))
            GPU-->>Func: 各モデルの出力辞書 (heatmap, flow, offset, features)

            Note over Func: 2. 戻り値整形
            Func->>Func: 戻り値1(heat)の累積加算: 通常ヒートマップの正規化(0-1: from sigmoid())
            Func->>Func: 戻り値2(rotated_heat)※の累積加算:回転ヒートマップではなく、「元画像と回転画像をTTAアンサンブル(平均化)したヒートマップ」 が格納される。
            Func->>Func: 戻り値3(flow)の累積加算
            Func->>Func: 戻り値4(flow_logvar)の累積加算
            Func->>Func: 戻り値5(offset)の累積加算
            Func->>Func: 戻り値6(features)の累積加算
            Note over Func: heat:(8, Z, Y, X):通常ヒートマップ。<br/> rotated_heat:(8, Z, Y, X):元画像と回転画像をTTAアンサンブル(平均化)したヒートマップ。<br/>flow:(8, 3, Z, Y, X):前後のフレーム間で細胞が移動する「3次元移動量ベクトル(dz, dy, dx)」単位は[μm]。<br/>flow_logvar:(8, Z, Y, X):動きの不確実性(分散の対数)<br/>offset:(8, 3, Z, Y, X):3次元ベクトル(Z, Y, X)方向の移動量・微小ズレ<br/>features:(8, n_emb, Z, Y, X):細胞識別特徴量(n_embは48チャンネル)
        end

        Note over Func: 3. モデル数 k=5 で除算してアンサンブル平均化
        Func->>Func: 戻り値1(heat)の累積加算 / k
        Func->>Func: 戻り値2(rotated_heat)の累積加算 / k
        Func->>Func: 戻り値3(flow)の累積加算 / k
        Func->>Func: 戻り値4(flow_logvar)の累積加算 / k
        Func->>Func: 戻り値5(offset)の累積加算 / k
        Func->>Func: 戻り値6(features)の累積加算 / k

        opt return_flow == True
            Note over Func: flow_batches に flow を退避
        end

        Note over Func,Append: 4. ピーク検出と候補抽出 (2系列)
        Func->>Append: _append_rows(heat, ...)<br/> 「1行＝1個の細胞」とした検出テーブル(DataFrame)を生成
        Append->>Append: ステップ1：細胞中心のピーク検出(L16〜23)
        Append->>Append: ステップ2：座標の精密補正(サブボクセル化)(29〜32)
        Append->>Append: ステップ3：動きベクトルと不確実性(標準偏差)の取得(L26〜27)
        Append->>Append: ステップ4：周囲の環境・幾何特徴の算出(L34〜38)
        Append->>Append: ステップ5：ピーク位置のボクセルから、48次元の 細胞識別特徴量(Embedding) を抜き出し(L43〜45)
        Append->>Append: ステップ6：DataFrame化(L47〜64)
        Note over Append: t, z, y, x ... 時間と精密化された3次元空間座標<br/>score, rank_in_frame	... 確信度スコアと、そのフレーム内での順位<br/>flow_z, flow_y, flow_x ... 次のフレームへの移動量予測ベクトル（$\mu\text{m}$）<br/>flow_sigma_um ... 動き予測の不確実性（標準偏差）<br/>near_um, density_10um ... 周囲の混雑度・最寄りの細胞距離<br/>peak_width_um, boundary_um	... 細胞サイズ・視野境界までの距離<br/>emb_0 〜 emb_47 ... 細胞識別特徴量(48次元ベクトル)
        Append-->>Func: rows[0] に候補追加(後の c21_count)
        Func->>Append: _append_rows(rotated_heat, ...)
        Append-->>Func: rows[1] に候補追加(後の c21)
    end
    Func-->>Caller: (c21_count, c21, original_flow) を返却
    deactivate Func

```

※**sigmoid** ... シグモイド関数(活性化関数)は、勾配消失問題があるが、それは学習時かつ中間層で使用する時に表面化する。今回は推論時かつ最終層で使用するだけなので勾配計算が発生せず問題ない。
※**戻り値2(rotated_heat)の算出** ...
  &emsp;・probability = 0.5 * (torch.sigmoid(original["heatmap"].float()) + torch.flip(torch.sigmoid(rotated["heatmap"].float()), dims=(-2, -1)))
  &emsp;&emsp;→ 元画像と回転画像をシグモイド化 → TTAアンサンブル(平均化)する。
  &emsp;・averaged_logit = torch.logit(probability.clamp(1e-6, 1.-1e-6))
  &emsp;&emsp;→ 確率($0 \sim 1$)を、元のモデル出力と同じ形式である Logit(未正規化スコア：$-\infty \sim +\infty$)に逆変換。Logit(ロジット)関数は、シグモイド関数の逆関数で、数式は以下の通り：
    $$\text{logit}(p) = \log\left(\frac{p}{1 - p}\right)$$
  &emsp;&emsp;もし確率 $p$ が完全に 0.0 や 1.0 になると：
  &emsp;&emsp;&emsp;$p = 0 \implies \log(0) \to -\infty$
  &emsp;&emsp;&emsp;$p = 1 \implies \frac{1}{0} \to +\infty$（ゼロ除算）
  &emsp;&emsp;となり、オーバーフローや数値計算エラー(NaN や Inf)が発生するため、確率値を $0.000001 \sim 0.999999$ の範囲に強制的に収める(クリッピングする)処理をしている。
  &emsp;・rh = torch.sigmoid(averaged_logit[:, 0].float()).cpu().numpy()
  &emsp;&emsp;→ averaged_logitをシグモイド化。sigmoid化 → logitで戻し → またsigmoid化してこの流れは無駄になってる。

### 5.3 retain_original_peaks()のシーケンス
**概要**: TTA平均化で取りこぼしてしまった細胞を、元画像の候補から3次元空間上で探し出して“救済・合流”させる。3次元の空間幾何計算(KD-Tree)や重複排除(NMS)を使用。

**引数**:
| **引数名** | **データ型** | **中身（中身の正体）** | 
|---|---|---|
| c21 | DataFrame | 「TTA回転アンサンブル」で平均化したヒートマップから検出した細胞候補テーブル(ノイズが少なしメインとなるベース) | 
| c21_count | DataFrame | 「元画像単体(TTAなし)」のヒートマップから検出した細胞候補テーブル(TTAで消えてしまった細胞をここから探して救済する) | 

**処理概要**:
- 空き枠の確認(ステップ 4): 1フレームあたりの上限枠(デフォルト3000件)に対して、TTA版(c21)の検出数に余裕があるか確認します。
- KD-Tree による重複判定(ステップ 5〜8): 元画像側(c21_count)の候補のうち、TTA側の既存細胞から $3.0\ \mu\text{m}$ より離れているものだけ を「見逃し細胞」として抽出します。
- NMS 間引き(ステップ 9〜11): 救済候補同士が近すぎる場合に共倒れしないよう NMS をかけ、空き枠数(slots)の分だけスコア上位を採用します。
- コンテキスト再計算(ステップ 14〜16): 細胞が追加されて周囲の配置が変わるため、最寄り細胞距離(near_um)や混雑度(density_10um)を再計算して整合性を保ちます。
- 返却(ステップ 20): 合成・救済された最終テーブル(result $\to$ 次の c21)と、何件救済されたかを記録した監査辞書(peak_retention_audit)を返します。

```mermaid
sequenceDiagram
    autonumber
    actor Caller as 呼び出し元<br/>(2nd-place-notebook.py)
    participant Func as retain_original_peaks()<br/>(heat_peak_retention.py)
    participant KDTree as cKDTree<br/>(3次元空間インデックス)
    participant NMS as select_frame()<br/>(NMS重複排除)
    participant NMSSub as NMS重複排除本体
    participant Feat as _neighbour_features()<br/>(近傍特徴量の再計算)

    Caller->>Func: retain_original_peaks(c21, c21_count)
    activate Func

    Note over Func: 入力バリデーション(カラム一致・非空チェック)

    loop 各フレームごとの処理 (for t in sorted_times)
        Note over Func: 当該時刻 t の候補インデックス(s)を取得<br/>hi: c21側(TTA), oi: c21_count側(Original)
        Func->>Func: 空き枠の確認: slots = TOP_K_PER_FRAME - len(hi)

        alt slots > 0 かつ len(oi) > 0 (空き枠があり、元画像候補が存在)
            Note over Func,KDTree: 1. 空間的重複の判定(3.0μmチェック)
            Func->>KDTree: cKDTree(c21の3D実座標 * SCALE_ARR[1.625, 0.40625, 0.40625])
            Func->>KDTree: query(c21_countの3D実座標 * SCALE_ARR)
            KDTree-->>Func: 最寄りのc21細胞までの距離 (distance)
            Func->>Func: distance > 3.0μm の候補のみ抽出 (available)

            opt available な候補が存在する場合
                Note over Func,NMS: 2. 救済候補内のNMSとスロット割当
                Func->>NMS: select_frame(座標, スコア, MATCH_UM=3.0)
                NMS->>NMS: idx = np.arange(len(zyx), スコア, 3μm)
                NMS->>NMSSub: nms(zyx[idx]*SCALE_ARR,scores[idx],3μm)
                NMSSub-->>NMS: 一番スコアの高いピークのidxを採用してその近傍を消す
                NMS-->>Func: 近接重複を除去した上位インデックス (keep)
                Func->>Func: 空き枠分を採用: selected = oi[available[keep]][:slots]
            end
        end

        alt selected が存在する場合 (救済セルがある)
            Note over Func: 3. テーブル結合とソート
            Func->>Func: pd.concat([c21の行, 救済されたc21_countの行])
            Func->>Func: score の降順でソート (sort_values)

            Note over Func,Feat: 4. 周囲環境特徴量の再計算
            Func->>Feat: _neighbour_features(全セルの3D座標, スコア)
            Feat->>Feat: cKDTree(全セルの3D座標).query_ball_point(全セルの3D座標, 10μm)<br>各細胞を中心とする「半径 10μm の球」の中に入っている細胞の数を一括カウント<br/>そのまま数えると**「自分自身」も1個としてカウントされてしまう**ため、- 1 する。
            Feat->>Feat: cKDTree(全セルの3D座標).query(全セルの3D座標, k=min(8, n))<br>各細胞から距離が近い順に最大8個、その距離 d と相手の番号 j を取得。
            Feat->>Feat: Loop<br/>最近傍距離 near_um の計算
            Feat-->>Func: 更新された near_um, density_10um<br/>幾何学的な環境特徴量(混雑度と最近傍距離)を返却
            Func->>Func: rank_in_frame を 0..N-1 で再採番

            Note over Func: 5. 座標・特徴量フィールドの非改変チェック
            Func->>Func: assert(元の座標やemb_0..47が破壊されていないこと)
            Func->>Func: pieces に合成フレームを追加
        else selected が空の場合 (救済セルなし)
            Func->>Func: pieces に c21 の行をそのまま追加
        end
    end

    Note over Func: 6. 全フレームの結合と最終監査
    Func->>Func: result = pd.concat(pieces, ignore_index=True)
    Func->>Func: 総行数の整合性検証 (len(result) == len(c21) + len(additions))

    Func-->>Caller: (result, peak_retention_audit) を返却
    deactivate Func
```

- **配列とスカラーの比較(NumPyのブロードキャスト)** ... NumPyは、1次元配列とスカラーを == で比較すると、ブロードキャスト機能によって配列の全要素に対してスカラー t との一致判定が自動的に行われます。
    ```python
    heat_times = np.array([0, 0, 1, 2, 1])
    t = 1  # スカラー値
    print(heat_times == t)
    # 出力: array([False, False, True, False, True])
    ```

### 5.4 specialist_scaffold()でやっていること
**概要**: c21(検出済全細胞群)の低スコア候補から、“暗いけど細胞として確か”を専用モデルで発掘(救済)し、トラックが途切れないための『足場候補』として候補リストに安全に追加・合流させる。
※専用モデルの作り方は、v343_models\full.json にレシピあり。

```mermaid
sequenceDiagram
    autonumber
    actor Caller as 呼び出し元 (notebook)
    participant SS as specialist_scaffold()
    participant FC as raw_flow_chain_state()
    participant IF as candidate_image_features()
    participant MF as candidate_model_features()
    participant DM as dim_model.predict()
    participant CM as cell_model.predict()
    participant BS as bounded_selection_score()
    participant DU as dim_union_superset()

    Caller->>+SS: specialist_scaffold(volume, candidate, cap, ...)
    
    rect rgb(240, 245, 255)
        Note over SS, MF: 1. 特徴量抽出
        SS->>+FC: raw_flow_chain_state(candidate)
        FC-->>-SS: state (Flow残差・追従性)
        SS->>+IF: candidate_image_features(volume, candidate)
        IF-->>-SS: image (周辺輝度特徴)
        SS->>+MF: candidate_model_features(candidate, state, image)
        MF-->>-SS: features (統合特徴量)
    end

    rect rgb(245, 255, 245)
        Note over SS, CM: 2. 専用モデルによる予測
        SS->>+DM: predict(features)
        DM-->>-SS: dim_probability (暗い領域確率)
        SS->>+CM: predict(features)
        CM-->>-SS: cell_probability (細胞確率)
    end

    rect rgb(255, 250, 240)
        Note over SS, BS: 3. スコア計算(暗い細胞のリランキング)
        SS->>+BS: bounded_selection_score(candidate, dim_weight, cell_prob, ...)
        BS-->>-SS: selection_score (暗い細胞を加味したスコア)
    end

    rect rgb(255, 245, 255)
        Note over SS, DU: 4. 候補の選別と重複排除(足場の合体)
        SS->>+DU: dim_union_superset(out(=c21), cap)
        Note over DU: ・control (元の通常上位候補) 抽出<br/>・treatment (専門スコア上位) 抽出<br/>・差分から重複排除 (NMS) して added 決定<br/>・union = concatenate([control, added])
        DU-->>-SS: control, replacement, added, union
    end

    Note over SS: 5. 安全装置の適用<br/>out(=c21).loc[added, "solver_selection_score"] = -1.0e6

    SS-->>-Caller: out(=c21), union(v343_union), audit
```


**処理概要**:
- ① 画像と時系列の局所特徴を抽出
    各候補点について、3Dボリューム画像から「局所的な明るさ・コントラスト」を取得し、さらに前後のフレームの動きベクトル(Flow)から「前後の点とどの程度きれいに追従しているか(残差)」を計算します。

- ② 2つの専用LightGBMで評価
    V343_DIM_MODEL: 「ここは暗い細胞が存在しやすい領域か？」を判定。
    V343_CELL_MODEL: 「ノイズではなく本物の細胞らしい特徴を持っているか？」を判定。

- ③ 通常の候補と「暗い細胞」を合体（Union）
    c21はそのまま、専用モデルによって高く評価された「暗い細胞候補」の中から、既存候補と被っていない(一定距離以上離れている)ものを追加候補(Added)として抽出→候補プール(v343_union)に追加。

- ④ 安全策：追加ノードのスコアを「-100万点」にする
``out.loc[added, "solver_selection_score"] = np.float32(-1.0e6)``
初期スコアをあえて極小値(-1.0e6)に設定。単体ではスコアが極小なので、存在するだけとなり採用されない(ノイズによる過剰検出ペナルティを防ぐ)。後段のトラッキングソルバー(Solver)で「前後のフレームの確実な細胞と強いエッジで繋がっている」と確認された場合のみ、前後のサポート(接続ボーナス点)を受けてトラックの欠損を埋めるため、足場候補から採用される。

- まとめ
specialist_scaffold() は、 「暗い細胞を、専用モデルでトラックが途切れるのを防ぐ中継ノード(Scaffold)を発掘し、過剰検出リスクをゼロに抑えた状態で後段のトラッカーに託す処理」 を実行する。


### 5.5 c21, c24, c_stage6, c39, c38の検出モデルの違い
この4つのモデル群は、それぞれ **「入力する時間幅」「学習方針」「後段のグラフ統合（アンサンブル）での役割」** が異なる。

|No| モデル | 時間窓 | 特徴 | 主なミッション |
|---| :--- | :---: | :--- | :--- |
|1| c21(基準) | 3フレーム | 基準となる基本検出器 | 全体のベースグラフを作る |
|2| c24 | 3フレーム | V21の別系統モデル | V21が見落としたトラックの補完 |
|3| c_stage6| 3フレーム | 高精度・強検出器 | V24との合意（Consensus）で確実なエッジを形成 |
|4| c39 | 5フレーム| 広い時間視野を持つモデル | リスク判定に基づく高品質トラックへの「入れ替え」 |
|5| c38 | 5フレーム| 5フレームの残差モデル | 最終的な残差補完 ＆ 細胞分裂の多数決投票 |


### 1. c21(基準)
細胞検出のベース。

### 2. c24(`V24_MODELS`)
* **何のモデルか**:
  主軸のc21と同じ構造を持つ **3フレーム入力（$t-1, t, t+1$）の 3D Temporal U-Net**（5-fold CV モデル）。
* **どういう学習をさせたか**:
  c21とはデータ拡張、サンプリング条件、または損失関数のバランスを少し変えて訓練された別バリエーション別系統）のモデル。
* **狙い(役割)**:
  - **「見落としの補完」**: V21が検出できなかった完全な細胞の軌跡（full-length track）を、後から安全に追加合成（`add_full_length_segments`）するため。
  - **「多様性の確保」**: 同じ3フレームモデルでも、V21と微妙に異なる予測を出させることで、アンサンブルの基礎を生成。

### 3. c_stage6(`STRONG_MODELS`)
* **何のモデルか**:
  単体性能を極限まで高めた **「強検出器（Strong Detector）」**（5-fold CV モデル）。
* **どういう学習をさせたか**:
  バックボーン設定や最適化された学習スケジュールで、検出ミスや位置ズレ（偽陽性・偽陰性）を極小化するように学習した高精度モデル。
* **狙い(役割)**:
  - 「高信頼なエッジの合意(Consensus)」: 後段で `add_stage6_v24_consensus_edges` という処理を行い、**「V24とStage6の両方が『この細胞同士は繋がっている』と合意した確実なエッジだけを採用する」** というジャッジに使用する。

### 3. c39(`V39_MODELS[route]`)
* **何のモデルか**:
  通常の3フレームではなく、5フレーム（$t-2, t-1, t, t+1, t+2$）を同時に入力する広域時系列 3D U-Net。
* **どういう学習をさせたか**:
  時間窓を前後に広げ、急激な移動や長い時間軸での細胞の連続性を捉えられるように5フレームを入力チャネルとして学習。
* **狙い（役割）**:
  - 「カウント中立な軌跡の入れ替え(Exchange / Recovery)」: 5フレームの広い視野を持つV39が作ったトラックと、既存のトラックを比較します。
  - 機械学習モデル（`V57_BOOSTERS`）でリスクを判定し、「既存のトラックよりもc39のトラックの方が確からしい」と判断された部分を、細胞の総数を増やさずに（ペナルティを避けつつ)高品質なトラックへと差し替えるために使用。

### 4. c38 (`V38_MODELS[route]`)
* **何のモデルか**:
  V39と同様に 5フレーム入力(`in_frames == 5`) を持つ、残差(Residual)補完用の時系列検出器。
* **どういう学習をさせたか**:
  V39とは異なるシード値や学習設定で訓練された、もう一つの5フレーム時系列モデルです。
* **狙い（役割）**:
  - **「最後の取りこぼし回収」**: V39の入れ替えを行ってもなお残った欠損部分に対し、最終段で残差トラックを追加合成（`add_full_length_segments`）する。
  - **「細胞分裂（Division）の多数決投票」**: 直後の処理で、細胞が2つに分裂するポイントを **4者合議（V24, Stage6, V39, V38 の投票 `fork_votes`）** で決定する際の投票役(Witness)。

---

## 6. (cell 5: line 670〜694) 処理B: 5系統の並列追跡 (トラッキング)

出揃った5つの細胞検出候補テーブル（`c21, c24, c_stage6, c39, c38`）を入力とし、`ThreadPoolExecutor`（3ワーカー）を用いて各モデル系統のトラッキング処理（`run_chain`）をCPUで並列実行する。
各系統で時系列エッジ確率の計算とソルバー最適化（`track_dataset_learned_v8`）を行い、5つの独立した細胞移動軌跡グラフ（`v21.graph, v24.graph, stage6.graph, v39.graph, v38.graph`）を生成・回収する。

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant RunChain as run_chain() (track_dataset_learned_v8)

    Note over Script: input: c21(+v343_union), c24, c_stage6, c39, c38

    par 5系統の並列トラッキング (ThreadPoolExecutor: 3 workers)
        Script->>RunChain: run_chain('v21_track', candidate_override=c21, superset_override=v343_union, ...)
        RunChain-->>Script: v21
    and
        Script->>RunChain: run_chain('v24_track', candidate_override=c24)
        RunChain-->>Script: v24
    and
        Script->>RunChain: run_chain('stage6_track', candidate_override=c_stage6)
        RunChain-->>Script: stage6
    and
        Script->>RunChain: run_chain('v39_track', candidate_override=c39)
        RunChain-->>Script: v39
    and
        Script->>RunChain: run_chain('v38_track', candidate_override=c38)
        RunChain-->>Script: v38
    end

    Note over Script: 処理B完了<br/>output: v21, v24, stage6, v39, v38 (各軌跡グラフ)
```

### 戻り値(メンバ構造とサイズ(次元))
このフェーズ(処理B)では、アンサンブルは行っていない。アンサンブルの元となるデータを生成している。

`v21`, `v24`, `stage6`, `v39`, `v38` ... NumPy配列やテンソルではなく、トラッキング結果を保持する Python のデータクラス(dataclass)オブジェクト。
- v21_track    ──> v21.graph （全体の基準ベース）
- v24_track    ──> v24.graph （取りこぼし補完用）
- stage6_track ──> stage6.graph （高信頼な強モデル）
- v39_track    ──> v39.graph （5フレーム広域モデル）
- v38_track    ──> v38.graph （5フレーム残差モデル）

#### 6.1. 型（Type）

* **型名**: **`biohub.pipeline.learned.TrackResult`**
  （追跡処理 `track_dataset_learned_v8` の実行結果オブジェクト）

#### 6.2. 内部のデータ構造（Attributes）
メンバ一覧

| 属性名 | 型 | 説明 |
| :--- | :--- | :--- |
| **`.graph`** | **`DatasetGraph`** | **追跡結果の本体（細胞ノードとエッジのグラフ）** |
| `.n_candidates` | `int` | 元の候補点の総数（例: 数万〜数十万点） |
| `.budget_per_frame`| `int` | 1フレームあたりに採択した目標細胞数（`budget`） |
| `.predicted_per_frame`| `float` | モデルが予測した1フレームあたりの平均細胞数 |
| `.n_selected` | `int` | 最終的にグラフに採択された細胞ノードの総数 |
| `.n_forks` | `int` | 検出された細胞分裂（1親→2子への分岐）の個数 |
| `.n_edgeless` | `int` | 前後と繋がらなかった孤立ノードの個数 |

直後の処理（`add_full_length_segments(v21.graph, v24.graph)` など）では、この中の **`.graph`** が使用される。

#### 6.3. 主データである `.graph`（`DatasetGraph`）の構造と Shape（サイズ）

`.graph` は **`biohub.io.submission.DatasetGraph`** 型で、以下の3つの主要データを持つ。

##### ① `graph.nodes`（細胞の位置情報）
* **型**: `dict[int, tuple[int, int, int, int]]`
* **構造**: `{ノードID: (t, z, y, x)}`
* **実質的な Shape**: **`(N, 4)`**（$N$ は採択された細胞の総数）
  - $N$ の規模：通常 1動画あたり数千〜数万個程度
  - 4次元の内訳：`t`（フレーム番号: 0〜99）, `z`（深度: 0〜63）, `y`（高さ: 0〜63）, `x`（幅: 0〜63）

##### ② `graph.edges`（細胞同士の繋がり・軌跡）
* **型**: `list[tuple[int, int, float]]`
* **構造**: `[(親ノードID, 子ノードID, 信頼度スコア), ...]`
* **実質的な Shape**: **`(E, 3)`**（$E$ は結ばれたエッジの総数）
  - $E$ の規模：ほとんどの細胞が前後のフレームと1対1で繋がるため、$E \approx N$（数千〜数万本）

##### ③ `graph.shape`（ボリューム寸法）
* **型**: `tuple[int, int, int, int]`
* **値**: **`(T, Z, Y, X)`**（例: `(100, 64, 64, 64)`）
  - 画像領域のバウンダリチェック（枠外はみ出し防止）に使用。

---

## 7. (cell 5: line 695〜789) 処理C: グラフ統合・コンセンサス・ギャップ補完

### 処理C-1: グラフ統合・コンセンサス・セグメント交換 (line 695〜739)

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant AddSeg as add_full_length_segments()
    participant AddStage6 as add_stage6_v24_consensus_edges()
    participant Exchange as apply_exchange()
    participant AddConsensus as add_consensus_edges()

    Script->>AddSeg: add_full_length_segments(v21.graph, v24.graph)
    AddSeg-->>Script: phase19_raw, p19_audit
    Note over Script: ベース側(v21.graph)で欠落している細胞軌跡のセグメント(軌跡の断片)を補完・追加する<br/>v21.graph+v24.graph → phase19_raw<br/>phase19_raw: DatasetGraph※, p19_audit:dict

    Script->>AddSeg: add_full_length_segments(phase19_raw, stage6.graph)
    AddSeg-->>Script: stage_raw, stage_audit
    Note over Script: ベース側(phase19_raw)で欠落している細胞軌跡のセグメント(軌跡の断片)を補完・追加する<br/>phase19_raw+stage6.graph → stage_raw<br/>stage_raw: DatasetGraph※, stage_audit:dict

    Script->>AddStage6: add_stage6_v24_consensus_edges(stage_raw, stage6.graph, v24.graph)
    AddStage6-->>Script: v37_raw, v37_edge_audit, v37_edge_actions
    Note over Script: 途切れているedgeの隙間を、Stage6とV24の『両モデルが一致して認めたエッジ』で安全に架橋(修復)する処理<br/>stage_raw+stage6.graph+v24.graph → v37_raw<br/>v37_raw: DatasetGraph ... 全実データ<br/>v37_edge_audit:dict ... 処理結果と整合性を記録した監査情報。<br/>v37_edge_actions: dictのリスト ... 実際に追加された各エッジの実データ。

    %% ---------------------------------------
    %% - v37(ベース) ⇔ v39 セグメント交換処理 -
    %% ---------------------------------------
    Note over Script,Exchange: v37(ベース) ⇔ v39 セグメント交換処理
    rect rgb(240, 255, 250)
    Script->>AddSeg: full_length_segment_actions(v37_raw, v39.graph)
    AddSeg-->>Script: v39_actions
    Note over Script: v37_raw+v39.graphの計画策定。<br/>※v39_actionsは追加予定リストで実データではない。<br/>v37_raw+v39.graph → v39_actions<br/>v39_actions: list[dict], stage_audit:dict

    Script->>Exchange: build_removal_features(v37_raw, phase19_raw, v21.graph, v24.graph, stage6.graph, v39.graph, v39_actions, {'v21': c21, 'v24': c24, 'stage6': c_stage6, 'v39': c39})<br/>v37_raw:stage_raw+stage6.graph+v24.graph, phase19_raw:v21.graph+v24.graph,v39_actions:v37_raw+v39.graph
    Exchange-->>Script: features
    Note over Script: ベースグラフ(v37_raw)の検出数を維持しつつ、v37 ⇔ v39 セグメント交換のためのv37削除候補ノードの特徴量を作成する<br/>features : DataFrame(shape=N, 65)※

    Script->>Exchange: predict_bagged_risk(features, V57_BOOSTERS[route])
    Exchange-->>Script: risk:
    Note over Script: 削除候補(features)から『削除すべき度合い(リスク確率)』を予測・平均化する。<br/>LightGBM(fold[0-4].txt)をアンサンブル平均化している。<br/>※bagged(バギングされた): 機械学習の基本手法である「Bagging(Bootstrap Aggregating)」に由来。<br/>内部で「複数のモデルの予測結果を平均化(集約)して安定した1つの予測にする」処理をしててその手法を指す。<br/>risk:numpy.ndarray[shape=(N,)] ... 入力 features の行数と全く同じ。各要素に、対応するノードの「削除リスク確率(0.0〜1.0 の実数値)」が格納される。

    Script->>Exchange: select_removals(features, risk)
    Exchange-->>Script: selected
    Note over Script: 各フレームで追加予定の枠数(quota)に合わせて、優先順位に従って実際に削除するノードを最終選定する<br/>selected:DataFrame(shape=(M, 66)) ... Mは次で求めるexpected_removalsと同数。66はfeaturesの65+p_risk列

    Script->>Script: removed = set(selected.node_id.astype(int))
    Note over Script: selected.node_idをset型に。

    Script->>Script: expected_removals = sum(len(json.loads(str(action['path_v24_nodes']))) for action in v39_actions)
    Note over Script: expected_removals:int ... 削除予定数 

    Script->>Exchange: apply_exchange(v37_raw, v39.graph, v39_actions, removed)
    Exchange-->>Script: v51_raw
    Note over Script: 交換実行関数<br/>v51_raw: DatasetGraph
    end

    Script->>AddConsensus: add_consensus_edges(v51_raw, witnesses)
    AddConsensus-->>Script: v52_raw※, edge_actions※, edge_census※
    Note over Script: 複数のモデル間で意見が一致した(合意の取れた)高信頼なエッジ(追跡線)を補完・追加する(エッジ修復)

    Script->>AddSeg: add_full_length_segments(v52_raw, v38.graph)
    AddSeg-->>Script: after_v38_raw, v38_audit
    Note over Script: ベース側(v52_raw)に追加<br/>after_v38_raw: v52_raw + v38.graph<br/>v38_audit:dict ... 監査情報
```

### 7.1 ※DatasetGraph型(biohub.io.submission.DatasetGraph)
このコンペ主催者が提供してくれてるソースコードの中で使用する型。

**メンバ**:
- **phase19_raw.shape**: tuple[int, int, int, int]=[100,64,256,256]
    - データセットの時空間ボリュームサイズ (T, Z, Y, X)
- **phase19_raw.nodes**: dict[int, tuple[int, int, int, int]]
    - キー: ノードID(int)
    - 値: 各ノードの時空間座標 (t, z, y, x)
    - ノード数は len(phase19_raw.nodes) (ベースのノード数 + 今回追加されたノード数)
- **phase19_raw.edges**: list[tuple[int, int, float]]
    - (source_id, target_id, confidence) ... edgeリスト
    - エッジ数は len(phase19_raw.edges) (ベースのエッジ数 + 今回追加されたエッジ数)
- **phase19_raw.dataset**: str
  - データセット識別子 ... ``44b6_0113de3b``とか

### 7.2 ※p19_audit(,stage_audit)の型: dict
追加処理の結果と整合性を記録した 9つのキーを持つ監査情報辞書

**メンバ**:
- ``dataset``: str ... データセット名(例:``44b6_0113de3b``)
- ``segments``: int ... 追加されたセグメントの総数
- ``bridges``: int ... 既存ノード間を繋いだブリッジ数
- ``tails``: int ... 末端から伸びたテイル数
- ``heads``: int ... 先頭前に遡るヘッド数
- ``added_nodes``: int ... 新規追加されたノード数
- ``added_edges``: int ... 新規追加されたエッジ数
- ``original_nodes_preserved``: bool (True)
- ``original_edges_preserved``: bool (True)
- ``division_forks_preserved``: bool (True)

### 7.3 add_stage6_v24_consensus_edges()のシーケンス
このモデルの解法(2位解法)はアンサンブルをいろんな局面で使い込んでるところが特徴。
そしてアンサンブルは、使い方を間違うとスコアダウンになるので、正しい使い方を理解すべくこのシーケンスを追ってみた。
#### 7.3.1. 処理概要
既存のグラフ(stage_raw)のノードは一切増やさず、途切れてしまっている既存ノード同士の1フレームの隙間に対し、Stage-6 と V24 の両モデルが「ここは繋がっている」と一致（AND合意）したエッジ(線)だけを安全に追加する。

#### 7.3.2. 戻り値の型と shape

|No| 戻り値 | 型 | shape / 構造 | 内容 |
|---| :--- | :--- | :--- | :--- |
|1| **`v37_raw`** | `DatasetGraph` | 合意エッジが追加された新しいグラフ。<br>・`nodes`: `stage_raw.nodes` と同じ(ノードは不変)。<br>・`edges`: `stage_raw.edges + selected`。<br>・`shape`: `(100, 64, 256, 256)` |
|2| **`v37_edge_audit`** | `dict` | 処理結果と整合性を記録した監査情報。<br>・`'selected_edges'`: 追加されたエッジ数<br>・`'raw_agreed_edges'`: 両者合意した候補数<br>・`'original_nodes_preserved'`: `True` 等 |
|3| **`v37_edge_actions`** |リスト | 実際に追加された各エッジの実データ。<br>各要素: `{'dataset': str, 'source': int, 'target': int, 'source_t': int}` |

### 7.3.3. 直前で c24 と stage6 を追加したのに、なぜさらにアンサンブルしてるのか？

直前の2行とこの処理では、「追加している対象」と「目的」が全く異なるため。

#### 7.3.3.(1) 直前の2行（`add_full_length_segments`）がやったこと
* 追加したもの: 「新しいノード」
* c24 と Stage-6 が細胞と認識した新ノードをグラフに追加した。

#### 7.3.3.(2) この処理（`add_stage6_v24_consensus_edges`）がやること
* 追加するもの: ノード追加はしない。既存ノード間を繋ぐエッジのみを追加
* グラフ内のノード(stage_raw)は1個も追加しない。

#### 7.3.3.(3) なぜこの順番で行うのか？
直前の2つのセグメント追加によって、グラフ空間上には
* c21 由来のノード
* c24 由来の追加ノード
* Stage-6 由来の追加ノード
が存在することになる。
その状態だと、「c21の軌跡の末端」と「Stage-6の先端」のように、異なるモデルが持ち寄ったnode同士が、時空間上で1フレームの隙間（Gap）を挟んで向かい合っている状態が起こりえる。

```text
[stage-6、c24ノード追加後]
c21のedge: (t=10) ●───● (t=11:末端) ???(隙間)
Stage6のedge:                                (t=12: 先端) ●───● (t=13)
[add_stage6_v24_consensus_edges 後]
両モデル合意:  (t=10) ●───● (t=11) ═══════════> (t=12) ●───● (t=13)
                                  ↑ 合意エッジで1本に結合！
```
上記の分断された隙間に対して、Stage-6 と V24 の両者に **「お前たち2人とも、ここが繋がっていると認めるか？」** と問い合わせ、2つのモデルが両方とも「繋がっている」と証言した箇所だけをエッジで架橋（Stitch）している。
これによって、バラバラだった軌跡の断片が1本のトラックへと統合することができるようになるため、このタイミングで実行する。

### 7.4. `add_stage6_v24_consensus_edges()` のシーケンス図

```mermaid
sequenceDiagram
    autonumber
    actor Caller as 2nd-place-notebook.py
    participant Consensus as add_stage6_v24_consensus_edges
    participant Mapper as _mapped_ordinary_edges
    participant Spatial as match_sequence (3D近傍マッチング)
    participant Validator as validate_graph

    Caller->>Consensus: 呼び出し (stage_raw, stage6.graph, v24.graph)
    
    rect rgb(240, 245, 255)
        note over Consensus, Spatial: Step 1. 各モデルのノードをベース座標へマッピング
        Consensus->>Mapper: stage6 のエッジ抽出依頼
        Mapper->>Spatial: 3Dユークリッド距離でベースノードと照合 (MAP_UM=3.0)
        Spatial-->>Mapper: 対応ノード対応表
        Mapper-->>Consensus: stage6_edges (ベースノードID空間でのエッジ集合)

        Consensus->>Mapper: v24 のエッジ抽出依頼
        Mapper->>Spatial: 3Dユークリッド距離でベースノードと照合
        Spatial-->>Mapper: 対応ノード対応表
        Mapper-->>Consensus: v24_edges (ベースノードID空間でのエッジ集合)
    end

    rect rgb(255, 248, 240)
        note over Consensus: Step 2. 両モデルのAND合意とフィルタリング
        Consensus->>Consensus: agreed = stage6_edges & v24_edges (論理積AND)
        Consensus->>Consensus: 途切れた端点のみ抽出 (outdegree[left]==0 かつ indegree[right]==0)
        Consensus->>Consensus: 1対1の一意な接続のみ選択 (selected)
    end

    rect rgb(240, 255, 240)
        note over Consensus, Validator: Step 3. グラフ構築と不変性の検証
        Consensus->>Consensus: 新しい DatasetGraph 作成 (既存ノード不変 + selectedエッジ追加)
        Consensus->>Validator: トポロジー検証 (既存ノード・エッジ・分裂が壊れていないか)
        Validator-->>Consensus: 検証パス確認
    end

    Consensus-->>Caller: 返却 (v37_raw, v37_edge_audit, v37_edge_actions)
```

### 7.5. features : DataFrame(shape=N, 65)※の詳細
#### 7.5.1. 全65列の内訳一覧

戻り値``features``の `pandas.DataFrame` に含まれる全65列の内訳は以下の通り

##### 7.5.2.(1) メタデータ・識別情報（4列）
モデル学習・推論の特徴量（`V55_FEATURES`）からは除外され、管理や後続の選択処理（`select_removals`）で使用される。

| No. | カラム名 | 型 | 説明 |
| :--- | :--- | :--- | :--- |
| 1 | `dataset` | object | データセット名 |
| 2 | `node_id` | int64 | ノードID（v37_raw 内の一意識別子） |
| 3 | `t` | int16 | タイムフレーム番号（時刻 $t$） |
| 4 | `frame_quota` | int16 | そのフレームにおいて交換（削除）が必要なノード数 |

##### 7.5.2.(2) グラフ位相・由来・モデル支持特徴量（21列）
トポロジー情報や、過去の各モデル（v21, v24, stage6, v39）との一致度合いを表す特徴量です。

| No. | カラム名 | 説明 |
| :--- | :--- | :--- |
| 5 | `origin_v21` | ノードが v21 由来かどうかの真偽値 |
| 6 | `origin_v24` | ノードが v24 由来かどうかの真偽値 |
| 7 | `origin_stage6` | ノードが stage6 由来かどうかの真偽値 |
| 8 | `v21_supported` | v21 のグラフに空間的に一致するノードが存在するか（bool） |
| 9 | `v24_supported` | v24 のグラフに空間的に一致するノードが存在するか（bool） |
| 10 | `stage6_supported` | stage6 のグラフに空間的に一致するノードが存在するか（bool） |
| 11 | `v39_supported` | v39 のグラフに空間的に一致するノードが存在するか（bool） |
| 12 | `supporting_banks` | このノードを支持しているモデルバンクの合計数（0〜4） |
| 13 | `v21_distance_um` | v21 の最近傍ノードとの空間距離（μm） |
| 14 | `v24_distance_um` | v24 の最近傍ノードとの空間距離（μm） |
| 15 | `stage6_distance_um` | stage6 の最近傍ノードとの空間距離（μm） |
| 16 | `v39_distance_um` | v39 の最近傍ノードとの空間距離（μm） |
| 17 | `indegree` | ノードの入次数（親ノード数） |
| 18 | `outdegree` | ノードの出次数（子ノード数） |
| 19 | `incident_degree` | ノードの総次数（入次数 + 出次数） |
| 20 | `history` | 過去方向へ何フレーム連続してトラックが続いているか |
| 21 | `future` | 未来方向へ何フレーム連続してトラックが続いているか |
| 22 | `track_support` | min(history, future) ... トラックの安定度 |
| 23 | `track_span` | トラック全体の継続フレーム長（history + future - 1） |
| 24 | `incoming_step_um` | 親ノードからの移動距離（μm） |
| 25 | `outgoing_step_um` | 子ノードへの移動距離（μm） |

##### 7.5.2.(3) 各モデルバンク（v21, v24, stage6, v39）の検出候補スカラー特徴量（32列）
各モデルの検出候補テーブル（`candidates`）から、該当位置・時刻の検出スコアや形態学的数値をマッピングしたもの（4モデル × 8種類 = 32列）。

| No. | カラム名 | 参照モデル | スカラー項目の説明 |
| :--- | :--- | :--- | :--- |
| 26〜33 | `v21_candidate_{suffix}` | v21 | 下記の8種類（距離、ランク、スコア、フローσ、近傍距離、密度、幅、境界） |
| 34〜41 | `v24_candidate_{suffix}` | v24 | 同上 |
| 42〜49 | `stage6_candidate_{suffix}` | stage6 | 同上 |
| 50〜57 | `v39_candidate_{suffix}` | v39 | 同上 |

※ 各モデル共通の8種類の suffix：
1. `distance_um`: 候補検出点との距離（μm）
2. `rank_fraction`: フレーム内でのスコア相対順位（0〜1）
3. `score`: 検出モデルの生スコア
4. `flow_sigma_um`: オプティカルフローの分散
5. `near_um`: 最も近い他の細胞候補との距離
6. `density_10um`: 半径10μm以内の局所細胞密度
7. `peak_width_um`: 検出シグナルのピーク幅
8. `boundary_um`: 画像境界からの距離

##### 7.5.2.(4) 由来モデル（Origin）の検出候補スカラー特徴量（8列）
そのノードの生みの親となったモデル（v21 / v24 / stage6 のいずれか）における、上記と同じ8種類のスカラー値です。

| No. | カラム名 | 説明 |
| :--- | :--- | :--- |
| 58 | `origin_candidate_distance_um` | 由来モデルでの最近傍距離 |
| 59 | `origin_candidate_rank_fraction` | 由来モデルでのスコア順位率 |
| 60 | `origin_candidate_score` | 由来モデルでの検出スコア |
| 61 | `origin_candidate_flow_sigma_um` | 由来モデルでのフロー分散 |
| 62 | `origin_candidate_near_um` | 由来モデルでの最近接他細胞距離 |
| 63 | `origin_candidate_density_10um` | 由来モデルでの局所密度 |
| 64 | `origin_candidate_peak_width_um` | 由来モデルでのピーク幅 |
| 65 | `origin_candidate_boundary_um` | 由来モデルでの境界距離 |

 総列数: 65列

### 7.6. add_consensus_edges()処理概要
#### 7.6.1. `add_consensus_edges()` は何をしているのか？
「複数のモデル間で意見が一致した(合意の取れた)高信頼なエッジ（追跡線）を補完・追加する(エッジ修復)」。

##### 概要
前段の処理(ノード交換など)により、途切れてしまったトラックや未接続の端点を、3つのモデル（`stage6`, `v24`, `expanded`=v39）の出力を見比べて、以下の合意条件(Consensus) を満たす接続線だけを新しく追加する：

1. **多数決の支持**: 2つ以上のモデルファミリーが「このノードとこのノードは繋がっている」と支持している。
2. **端点同士の接続**: 始点側のノードにまだ先がなく(出次数 0)、終点側のノードにまだ手前がない(入次数 0)。
3. **時間整合性**: 時間差がちょうど 1 フレーム差（$t \rightarrow t+1$）。
4. **1対1の接続Mutual Unique）**: 競合する接続候補が他に存在しない。

※ **ノード数（細胞数）は一切増やさず**、途切れていた細胞同士をつなぐ「エッジ（線）」のみを追加する。

#### 7.6.2. 各戻り値の型と shape

戻り値は `(v52_raw, edge_actions, edge_census)` のタプル。

##### 7.6.2.① `v52_raw`
* **型**: **`biohub.io.submission.DatasetGraph`**(追跡グラフデータクラス)
* **shape / 構造**:
  * **`.shape` 属性**: `tuple[int, int, int, int]`（元画像の4次元ボクセルサイズ `(T, Z, Y, X)`）
  * **ノード数 (`len(v52_raw.nodes)`)**: $N$
    * `v51_raw` と完全に同数（ノードは1つも増減しない）。
  * **エッジ数 (`len(v52_raw.edges)`)**: $E + K$
    * 合意によって新しく採択されたエッジの本数（$K$ 本）だけ増加。

##### 7.6.2.② `edge_actions`
* **型**: **`list[dict]`**（辞書のリスト）
* **要素数（shape）**: **長さ $K$**（新しく追加されたエッジの本数）
* **各要素の辞書の中身**:
  追加された各エッジの具体的な情報が格納されている。
```python
  {
      "dataset": str,     # データセット名
      "source": int,      # 接続の始点ノードID
      "target": int,      # 接続の終点ノードID
      "families": str     # 支持したモデル系統（例: "crossbank_solver+strong_detector"）
  }
```

##### 7.6.2.③ `edge_census`
* **型**: **`dict`**（辞書型）
* **要素数（shape）**: **6 個のキーを持つ辞書**
* **中身（監査・集計ログ）**:
  各モデルから何本のエッジが候補になり、最終的に何本追加されたかの統計サマリー。
```python
  {
      "dataset": str,                # データセット名
      "mapped_stage6_edges": int,    # stage6 から提案されたエッジ数
      "mapped_v24_edges": int,       # v24 から提案されたエッジ数
      "mapped_v25_edges": int,       # v25 から提案されたエッジ数 (あれば)
      "mapped_expanded_edges": int,  # expanded(v39) から提案されたエッジ数
      "selected_mutual_unique": int  # 実際に採択・追加されたエッジ数 (= K)
  }
```

### 処理C-2: 細胞分裂検出・ギャップ補完・座標補正 (line 741〜789)

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant ForkVotes as fork_votes()
    participant SelectActions as select_actions()
    participant Materialize as materialize()
    participant TempMotion as sample_temporal_motion()
    participant ProposeGaps as propose_gaps()
    participant MatGaps as materialize_gaps()
    participant Corresp as apply_correspondence()
    participant MotionSub as apply_motion_substitution()

    Note over Script,Materialize: 細胞分裂(Division / 分岐)を検出・確定する処理
    rect rgb(255, 240, 230)
    %% いまここ fork_votes()
    Script->>ForkVotes: fork_votes(after_v38_raw, division_witnesses)※
    ForkVotes-->>Script: division_votes, division_mapping
    Note over Script: 細胞分裂(Fork: 1つの親ノードから2つの子ノードへの分岐)について、4つのモデルから『どこで分裂が起きたか』の投票(Votes)を集計する<br/>division_votes: dict, division_mapping: dict

    Script->>SelectActions: select_actions(after_v38_raw, division_votes, 'root_consensus2', guard_hops=2)※
    SelectActions-->>Script: division_actions, division_conflicts
    Note over Script: 投票結果(division_votes)を評価し、ポリシー('root_consensus2': 2モデル以上の合意)を満たす分裂アクションを選定し、他の分裂との競合を解消して確定する<br/>division_actions: list[dict], division_conflicts: dict

    Script->>Materialize: materialize(after_v38_raw, division_actions)
    Materialize-->>Script: final_raw (分裂コンセンサス後)
    Note over Script: 採択された分裂アクションをグラフに正式反映(1親→2子の分岐エッジを追加)<br/>※ここまで細胞nodesは一つも追加しない<br/>final_raw: DatasetGraph...細胞分裂edges追加を含む全edges, division_audit:dict...監査結果
    end

    Note over Script,MotionSub: 最終後処理

    Script->>TempMotion: sample_temporal_motion(V21_MODELS, shared_volumes, ...)※
    TempMotion-->>Script: gap_backward, gap_forward
    Note over Script: 現在のグラフ(final_raw)内の全細胞ノードについて、flow(3次元移動量)から、過去方向(backward)と未来方向(forward)の移動ベクトルをサンプリング(抽出)する<br/>gap_backward:numpy.ndarray[shape:(N:全細胞ノードの総数,len(final_raw.nodes), 3:過去方向の移動ベクトル(Δz, Δy, Δx)]<br/>gap_forward:numpy.ndarray[shape:(N, 3)]

    Script->>ProposeGaps: propose_gaps(final_raw, ...)※
    ProposeGaps-->>Script: gap_actions, gap_selection
    Note over Script: 追跡途中で一時的に細胞を見失って途切れてしまったトラックの隙間(0〜2フレームの欠落ギャップ)を、前後のオプティカルフロー(双方向の動き予測)と検出候補(c21)を用いて橋渡し・再接続(Gap Closure)するための計画を策定する<br/>gap_actions:list[dict], gap_selection:dict

    Script->>MatGaps: materialize_gaps(final_raw, gap_actions)
    MatGaps-->>Script: final_raw (ギャップ補完後)
    Note over Script: 前段の propose_gaps() で計画されたギャップ補完(gap_actions)に従って、中間に抜けていた細胞ノードを新設し、前後のトラックを繋ぐエッジを追加。途切れを解消した最新のグラフを実体化・検証する<br/>final_raw:DatasetGraph, gap_raw_audit:dict

    Script->>Corresp: apply_correspondence(final_raw, final_out, ..., V230_MODELS)
    Corresp-->>Script: final_raw, final_out
    Note over Script: 細胞の3次元画像パッチ(外見・テクスチャ)を深層学習で比較し、追跡エッジの誤接続を解除、もしくは、より確度の高い接続へと繋ぎ直す『最終画像マッチング(対応付け: Correspondence)処理』

    Script->>MotionSub: apply_motion_substitution(final_raw, final_out, ...)
    MotionSub-->>Script: final_raw, final_out
    Note over Script: トラックの繋がり(エッジ)やノードIDは一切変えずに、前後の移動予測と矛盾している不自然な中間細胞ノードの位置座標(外れ値)を、より物理的に自然な検出候補(c21)の位置へとすり替える『座標の精密置換（Substitution）処理

```
### 7.7. fork_votes() ... 2位解法での細胞分裂判定
「細胞分裂(分岐)」は、この関数(fork_votes)から。
とはいえ、各モデル（v24, stage6, v39, v38）は、それぞれ単体でトラッキングの段階で細胞分裂を予測しており、それを使用して投票を集計する。

戻り値:
| 変数名 | 型 | shape / 要素数 | 役割・内容 |
| :--- | :--- | :--- | :--- |
| **`division_votes`** | `dict[tuple, set]` | $F$ 個のキー<br>キー: 3要素タプル<br>値: サイズ1〜4の集合 | 「どの分裂トリプル `(親, 子1, 子2)` に、どのモデルが投票したか」の集計表 |
| **`division_mapping`** | `dict[str, dict]` | 4モデル分のキー<br>値: 4統計値の辞書 | 各モデルのノード照合数やマッピング成功数の監査・統計ログ |


#### 7.7.1 `fork_votes()` のシーケンス図

```mermaid
sequenceDiagram
    autonumber
    actor Caller as 呼び出し元(Main Script)
    participant FV as fork_votes()
    participant Matcher as match_sequence()<br/>(空間座標照合)
    participant Witness as 目撃者モデル(v24, stage6, v39, v38)
    participant VoteBox as 投票箱(votes)

    Caller->>FV: fork_votes(after_v38_raw, division_witnesses)
    Note over FV: ベースグラフ(after_v38_raw)の全ノード座標(t, z, y, x)を取得

    loop 4つのモデル(v24, stage6, v39, v38)ごとに繰り返し
        FV->>Witness: 各モデルのノードID・座標を取得
        Witness-->>FV: donor_ids, donor_coords

        Note over FV,Matcher: 【Step 1. 空間座標マッチング】
        FV->>Matcher: match_sequence(同フレーム & 距離 <= 3.0μm)
        Matcher-->>FV: mapping (対応インデックス)
        Note over FV: 物理スケール換算で 3.0μm を超えるペアを除外<br/>1対1のノード対応辞書 pairs を作成 (witnessノード → baseノード)

        Note over FV,Witness: 【Step 2. 分裂候補(Fork)の抽出と検証】
        FV->>Witness: children(witness) でエッジ接続を取得
        Witness-->>FV: 各親ノードに対する子ノード一覧 (donor_children)

        loop モデル内の各親ノードについて判定
            Note over FV: ① 子ノードがちょうど2個か？ (len == 2)<br/>② 親と子2個がすべてベースグラフに対応付けできたか？<br/>③ ベース側で時間がちょうど +1 フレームか？<br/>(t_child == t_parent + 1)
            alt すべての条件を満たす場合
                Note over FV: 分裂トリプルを作成: triple = (p, 子a, 子b)
                FV->>VoteBox: votes[triple].add(モデル名)
                Note over VoteBox: この分裂トリプルに「1票」を投じる！
            else 条件を満たさない場合
                Note over FV: スキップ (投票しない)
            end
        end

        Note over FV: そのモデルの監査統計(audit)を記録<br/>(マッピング成功数、半径外拒絶数、分裂数)
    end

    FV-->>Caller: division_votes, division_mapping (audit)
```

#### 7.7.2 処理の核心ポイント（なぜこうしているのか？）

##### 7.7.2.1. 幾何学的な座標照合(3.0 μm マッチング)
* 各モデル（`v24`, `stage6`, `v39`, `v38`）はそれぞれ独立して推論を行っているため、**同じ細胞であってもノードID番号がバラバラ**になっている。
* そのため、まず「同じ時刻 $t$」かつ「空間距離 3.0 μm 以内」という条件で最近傍探索を行い、**「モデルAのノード10番は、ベースグラフのノード50番のことだな」** と ID辞書 `pairs` を作成。

##### 7.7.2.2. 分裂トリプル `(p, a, b)` の厳格な判定
モデル内で分裂（1親 $\rightarrow$ 2子）が見つかっても、そのまま採用するのではなく、以下の条件をすべてクリアしたものだけをベースグラフの分裂候補とする：
* **完全対応**: 親 $p$ だけでなく、2つの娘細胞 $a, b$ もベースグラフ上に存在すること。
* **時間整合性**: 親が時刻 $t$ なら、2つの子は両方とも**きっかり時刻 $t+1$** に存在すること。
* **正規化**: 子の順序（$a, b$ か $b, a$ か）で別の分裂と判定されないよう、`(p, min(a,b), max(a,b))` とソートして一意化。

##### 7.7.2.3. 投票（Vote）の集積
* 満たしたトリプルに対して、`votes[triple].add(bank)` で投票する。
* 例えば、
  * `v24` が「(10, 20, 21) は分裂だ」と判定 $\rightarrow$ 1票
  * `stage6` も「(10, 20, 21) は分裂だ」と判定 $\rightarrow$ 2票
  * `v39` も「(10, 20, 21) は分裂だ」と判定 $\rightarrow$ 3票
* これで、**「どの細胞が、どの2つの娘細胞に分裂したか」について、各モデルからの支持票が集計**される。

次段の `select_actions(..., 'root_consensus2')` で、「2票以上」集まった分裂を正式採用する。

### 7.8 select_actions()でやっていること
fork_votes() で集めた投票（division_votes）の中には、「1票しか集まらなかった怪しい分裂」や「複数の分裂候補で同じ細胞が取り合いになっている競合」が含まれる。select_actions でふるい落とす：
**ふるい落としのステップ**
1. 得票数の足切り:
    ポリシー 'root_consensus2' に従い、2票以上集まった候補のみを通過させる（1票のみは却下）。
2. グラフ構造の適合チェック:
    親ノードが現在「一本道(出次数が1)」であり、分裂候補の片方の子（first）とすでに繋がっていることを確認。もう片方の子（second）を「新しく追加する2番目の娘細胞」として認定。
3. 安全ガード(guard_hops):
    直前の処理で変更された不安定な箇所の近く（2ホップ以内）にあるノードは除外する。
4. 競合の解消（Conflict Resolution）:
    複数の分裂提案で同じノード(親や子)が重複して使われていないかをカウントし、競合・衝突している提案は安全のためにすべて除外。

#### 7.8.2. 戻り値の型と shape
`(division_actions, division_conflicts)` のタプル
##### 7.8.2.① `division_actions`
* **型**: **`list[dict]`**（辞書のリスト）
* **shape / 要素数**: **長さ $K$**（最終的に採択された分裂アクションの件数）
* **各要素の辞書の中身**:
  グラフに分裂を追加するための具体的な指示書データ（6つのキー）
  ```python
  {
      "parent": int,             # 分裂する親ノードID
      "first": int,              # すでに親と繋がっている1番目の娘ノードID
      "second": int,             # 新しく親と接続する2番目の娘ノードID
      "old_parent": int | None,  # second に元親がいた場合そのID (通常は None)
      "votes": list[str],        # 投票したモデル名のリスト (例: ['stage6', 'v24'])
      "participants": list[int]  # この分裂に関与する全ノードIDのリスト
  }
  ```

##### 7.8.2.② `division_conflicts`
* **型**: **`dict[str, int]`**（辞書型）
* **shape / 要素数**: **2個のキーを持つ辞書**
* **中身（監査・ログ）**:
  審査と競合解消の集計結果
  ```python
  {
      "qualified_before_conflicts": int,  # 競合チェック前に条件をクリアしていた提案数
      "conflict_rejections": int          # ノード重複(競合)で弾かれた提案数
  }
  ```

次の処理の `materialize(after_v38_raw, division_actions)` で、この `division_actions` 使って各 `(parent, second)` への新しいエッジが追加、正式な細胞分裂グラフ（`final_raw`）が生成される。

### 7.9 sample_temporal_motion()でやっていること

一言で言うと、**「現在のグラフ(final_raw)内に存在するすべての細胞ノードについて、深層学習モデルが予測したオプティカルフロー(3次元移動量)から、過去方向(backward)と未来方向(forward)の移動ベクトルをサンプリング(抽出)する処理」**。

#### 7.9.1 処理の流れ
1. **全細胞の座標を取得**:
   * `final_raw` に存在する全 $N$ 個の細胞の時空間座標 `(t, z, y, x)` を取り出す。
2. **2方向の動き（Temporal Motion）を抽出**:
   * **過去方向（`gap_backward`）**:
     時刻 $t$ の細胞が、直前のフレーム $t-1$ からどう動いてきたかを表す 3D 移動ベクトル。
   * **未来方向（`gap_forward`）**:
     入力画像を時間軸で反転（`torch.flip`）させてモデル（`V21_MODELS`）に通し、時刻 $t$ から直後のフレーム $t+1$ へどう動くかを表す 3D 移動ベクトル。
3. **連続値グリッドからのサンプリング**:
   * オプティカルフローの空間マップから、各細胞の座標位置における移動量 `(Δz, Δy, Δx)` を補間サンプリング。

#### 7.9.2 何のために使うのか？
直後の `propose_gaps()`で使用する。「途切れてしまったトラックの先端」から、このオプティカルフローによる移動ベクトルを前後に伸ばすことで、**「時間的に数フレーム飛んでしまった細胞同士」を物理的に正しく繋ぎ直す(ギャップ補完: Gap Closure)** ための補佐情報になる。

#### 7.9.3 戻り値の型と shape
関数は `(gap_backward, gap_forward)` のタプルを返却する。
##### 7.9.3.① `gap_backward`(過去方向の移動ベクトル)
* **型**: **`numpy.ndarray`**（要素は `float32`）
* **shape**: **`(N, 3)`**
  * **$N$**: 入力された全細胞ノードの総数（`len(final_raw.nodes)`）
  * **$3$**: 各細胞の 3次元移動ベクトル `(Δz, Δy, Δx)`

##### 7.9.3.② `gap_forward`(未来方向の移動ベクトル)
* **型**: **`numpy.ndarray`**（要素は `float32`）
* **shape**: **`(N, 3)`**
  * **$N$**: 入力された全細胞ノードの総数（`len(final_raw.nodes)`）
  * **$3$**: 各細胞の 3次元移動ベクトル `(Δz, Δy, Δx)`

### 7.10 `propose_gaps()` は何をしているのか？

追跡途中で一時的に細胞を見失って途切れてしまったトラックの隙間(0〜2フレームの欠落ギャップ)を、前後のオプティカルフロー(双方向の動き予測)と検出候補(c21)を用いて橋渡し・再接続(Gap Closure)するための計画を策定している。

```text
【ギャップ補完のイメージ(1フレーム欠落の例)】
トラック前半 (終端) : (t=10) ──> (t=11)
                                   ：  (オプティカルフローで予測)
中間ノード (補完)   :              [t=12] (c21の未利用ピークから補間ノードを挿入)
                                   ：
トラック後半 (始端) :              (t=13) ──> (t=14)
```

#### 7.10.1 具体的な処理の流れ
1. **途切れた端点の抽出**:
   * **終端(`terminals`)**: トラックの末端（出次数が0）で、手前に3フレーム以上安定して続いているノード。
   * **始端(`roots`)**: トラックの先頭（入次数が0）で、後方に3フレーム以上安定して続いているノード。
2. **双方向の動き検証(Bidirectional Motion Check)**:
   * **欠落 0 フレーム(直結できるギャップ)**:
     終端の未来予測位置（$pos + forward$）と、次フレームの始端の過去予測位置（$pos + backward$）が、互いに距離 3.0 μm 以内で一致しているかを検証。
   * **欠落 1〜2 フレーム(中間に細胞が抜けているギャップ)**:
     未採用だった検出候補テーブル(`c21`)の中から、前後のオプティカルフローの予測軌道上に存在する「未利用の細胞ピーク」を発見し、中間に挿入するノードとして割り当てる。
3. **低コスト順の選定と競合解消(Greedy Selection)**:
   * 移動ベクトルのズレ(平均二乗誤差 $\mu\text{m}^2$)が最も小さい順にソートし、ノードの取り合い(競合)がないように 1対1 で最適な接続アクションを決定。

#### 7.10.2. 戻り値の型と shape
`(gap_actions, gap_selection)` のタプルを返却。

#### 7.10.2.① `gap_actions`(採択されたギャップ補完アクションリスト)
* **型**: **`list[dict]`**（辞書のリスト）
* **shape / 要素数**: **長さ $G$**（採択されたギャップ接続の総数）
* **各要素の辞書の中身**:
  トラックを繋ぐための具体的な指示書データ（5つのキー）
  ```python
  {
      "parent": int,                           # 途切れていた前半トラックの終端ノードID
      "child": int,                            # 途切れていた後半トラックの始端ノードID
      "mean_squared_residual_um2": float,      # 移動ベクトルの平均二乗残差コスト (μm^2)
      "candidate_rows": list[int],             # 補間された中間細胞候補の行番号リスト
      "nodes": list[tuple[int, int, int, int]] # 途中に新規挿入する細胞座標 [(t, z, y, x), ...]
  }
  ```
  ※ `nodes` の長さは、0フレーム欠落なら `0`(直結)、1フレーム欠落なら `1`(ノード1個挿入)、2フレーム欠落なら `2`(ノード2個挿入)となります。

#### 7.10.2.② `gap_selection`（監査・集計統計）
* **型**: **`dict`**（辞書型）
* **shape / 要素数**: **4個のキーを持つ辞書**
* **中身**:
  ```python
  {
      "proposals": int,                   # 検討されたギャップ接続候補の総数
      "eligible_terminals": int,          # 接続候補になり得た終端ノード数
      "eligible_roots": int,              # 接続候補になり得た始端ノード数
      "selected_by_missing_cells": dict   # 欠落フレーム数別の採択内訳 (例: {0: 18, 1: 10, 2: 4})
  }
  ```

次行の`materialize_gaps(final_raw, gap_actions)`によって、この `gap_actions` に記載された中間ノードや接続エッジがグラフへ挿入され、途切れていたトラックが1本に繋がる。

---

## 8. (cell 5: line 790〜887) 処理D: グラフ確定と多段プロテクター後処理

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant ValidGraph as validate_graph()
    participant FinalizeCV as finalize_cv_graph()
    participant PruneComp as v394.prune_components()
    participant TrimTails as v399.trim_terminal_tails()
    participant GapAdmit as v454.apply_gap_admission()
    participant Prep507 as v507.prepare_actions()
    participant Apply507 as v507.apply_prepared()
    participant Apply533 as v533.apply()

    Script->>ValidGraph: validate_graph(final_raw / final_out)
    ValidGraph-->>Script: raw_valid,out_valid:整合結果データ(not use)
    Note over Script: (final_raw,final_out): DatasetGraphの整合性チェック

    Script->>FinalizeCV: finalize_cv_graph(final_raw, final_out)
    FinalizeCV-->>Script: v389_graph
    Note over Script: 引数の整合性チェックとDatasetGraphインスタンスを生成・返却する

    Script->>PruneComp: v394.prune_components(v389_graph, shared_volumes, ...)※
    PruneComp-->>Script: v394_graph, v394_audit
    Note over Script: 孤立ノードをグラフから刈り取り(削除)する。推論(LightGBM,Booster)を使用

    Script->>TrimTails: v399.trim_terminal_tails(v394_graph, shared_volumes, ...)※
    TrimTails-->>Script: v399_graph
    Note over Script: 末端のゴミ(tails:過剰追跡)を除去

    Script->>GapAdmit: v454.apply_gap_admission(v399_graph, ...)※
    GapAdmit-->>Script: v454_graph
    Note over Script: トラッキンググラフにおける 1フレーム欠損(時間ギャップ: t -> t+2)を検出・補間し、新しいノードとエッジを追加する後処理(V454 Gap Admission) を行う。

    Script->>Prep507: v507.prepare_actions(v454_graph, c21, ...)※
    Prep507-->>Script: v507_prepared
    Note over Script: 後処理実行の方針を決定する。

    Script->>Apply507: v507.apply_prepared(v454_graph, v507_prepared)※
    Apply507-->>Script: v507_graph
    Note over Script: 直前の v507.prepare_actions()に基づいて、入力グラフに対して実際に座標補正と分岐エッジ削除を適用して新しいグラフを生成・検証する

    Script->>Apply533: v533.apply(v507_graph, c21, v394_cluster, path)※
    Apply533-->>Script: v533_graph (最終グラフを graphs リストへ追加)
    Note over Script: 全体の最終段処理(ルールベースで整形)

```

### 8.1 v394.prune_components()でやっていること
短〜中規模の連結成分(トラック)を機械学習モデル(LightGBM Booster)で真偽判定し、誤検出と判定されたノイズ成分およびエッジを持たない孤立ノードをグラフから刈り取り(削除)する

#### 8.1.1 引数

| 引数名 | 型 (`type`) | ノートブックでの実引数例 | 役割・内容 |
| :--- | :--- | :--- | :--- |
| **`graph`** | `DatasetGraph` | `v389_graph`(final_rawと同じ) | **枝刈り対象の入力グラフ**。<br>これまでのパイプラインで構築されたノード座標とエッジ情報を持つ。 |
| **`volume`** | `np.ndarray` | `shared_volumes` | **顕微鏡画像の4Dボリュームデータ** `(T, Z, Y, X)`。<br>ノード周辺の $3 \times 3 \times 3$ 局所画像パッチをサンプリングし、輝度や前後フレーム間の相互相関(NCC)を計算するために使用。 |
| **`witnesses`** | `dict[str, DatasetGraph]` | v24: copy_cv_coordinates(v24.graph)<br/>stage6: copy_cv_coordinates(stage6.graph)<br/>v39: copy_cv_coordinates(v39.graph)<br/>v38: copy_cv_coordinates(v38.graph) | **他モデル・他ステージの予測グラフ群(証人)**。<br>対象グラフのエッジが他手法でも共通して支持されているか(アンサンブル支持数)を照合するために使用。 |
| **`model`** | `lightgbm.Booster` | `V394_MODEL` | **学習済み LightGBM モデル** (`component_protector.txt`)。<br>抽出された特徴量から、各短コンポーネントを保持すべき確率を推論する。 |
| **`metadata`** | `dict` | `V394_METADATA` | **モデルの設定辞書** (`metadata.json`)。<br>対象とする最大ノード数 (`max_component_nodes`)、特徴量名リスト (`feature_columns`)、クラスタ別閾値 (`protect_thresholds`) を含む。 |
| **`cluster`** | `str` | `v394_cluster` | **撮影条件クラスタ名** (例: `'acq_01'`)。<br>画像の解像度や輝度統計量から分類されたクラスタIDで、クラスタ別の確率閾値の選択や特徴量フラグ (`is_acq01`) に使用。 |

#### 8.1.1 具体的な処理手順
1. **連結成分の抽出 (Union-Find)**:
   - エッジ情報に基づきノード間の連結成分を分解し、各成分のノード数(`sizes`)や出次数を算出。
2. **判定対象候補の絞り込み**:
   - ノード数が2以上かつ `metadata["max_component_nodes"]` 以下の成分を抽出。
   - ただし、**分裂(出次数2)を含む成分(`fork_roots`)は誤削除を避けるため保護対象として除外**する。
3. **特徴量の抽出**:
   絞り込んだ候補成分ごとに、以下の特徴量を計算:
   - **トポロジー・位置**: ノード数、エッジ数、時間スパン、時間・空間の正規化座標。
   - **近傍距離**: 同時刻フレーム内での最近傍ノードまでの空間距離(`cKDTree` による `nearest_um`)。
   - **画像輝度パッチ**: 3D画像ボリューム(`volume`)からノード周辺の $3 \times 3 \times 3$ パッチをサンプリングし、平均輝度や標準偏差を算出。
   - **運動・連続性**: フレーム間変位距離(`edge_um`)、前後フレームのパッチ間の正規化相互相関(NCC: `edge_ncc`)、進行方向の変化(コサイン類似度 `turn`)、加速度(`acceleration_um`)。
   - **他モデルによる支持(Witnesses)**: 他の予測グラフ群(`witnesses`: v24, stage6, v39, v38 など)とシーケンスマッチングを行い、他モデルでも共通して予測されているエッジかどうか(`witnesses`)。
4. **LightGBM による保護判定**:
   - 特徴量行列を `model.predict()` に入力し、各コンポーネントを保持すべき確率(`probability`)を算出。
   - データセットのクラスタごとに設定された閾値(`metadata["protect_thresholds"][cluster]`)未満の成分を削除対象(`remove_roots`)にする。
5. **枝刈りとグラフ再構築**:
   - エッジが存在しない孤立ノード(`edgeless_nodes`)および削除対象と判定された成分のノード・エッジを除去した、新しい DatasetGraph を構築。
6. **バリデーションと監査情報の生成**:
   - 新グラフに対して validate_graph()を実行、枝刈りの実行結果(削除数など)を追記した監査メタデータを作成。

#### 8.1.2. 戻り値の型と各要素の type
戻り値は、DatasetGraph, dict。
##### 8.1.2.第1要素: `result`
- **型 (`type`)**: **`DatasetGraph`**
- **内容**: ノイズ成分や孤立ノードが削除された後の新しいDatasetGraph。

##### 8.1.2.第2要素: `audit`
- **型 (`type`)**: **`dict`** (実質的には `dict[str, Union[str, int, float, None]]`)
- **内容**: `validate_graph(result)` のサマリーと、以下の枝刈り監査キーが追加された辞書。

| キー名 | 値の型 (`type`) | 内容 |
| :--- | :--- | :--- |
| `"dataset"` | `str` | データセット名 |
| `"nodes"` | `int` | 枝刈り後の総ノード数 |
| `"edges"` | `int` | 枝刈り後の総エッジ数 |
| `"forks"` | `int` | 分裂ノード数 |
| `"births"` | `int` | 新規発生ノード数 |
| `"deaths"` | `int` | 追跡終了ノード数 |
| `"mean_disp_um"` | `float` | 平均変位量(μm) |
| `"p99_disp_um"` | `float` | 変位量の99パーセンタイル値(μm) |
| `"acquisition_cluster"` | `str` | 撮影条件クラスタ名(例: `'acq_01'`) |
| `"threshold"` | `float` | 判定に使用された確率閾値 |
| `"candidates"` | `int` | 判定対象となった候補コンポーネント数 |
| `"removed_components"` | `int` | 閾値未満で削除されたコンポーネント数 |
| `"edgeless_nodes"` | `int` | 削除されたエッジなし孤立ノード数 |
| `"removed_nodes"` | `int` | 削除された合計ノード数(孤立ノード＋削除コンポーネント) |
| `"probability_min"` | `float \| None` | 予測確率の最小値(候補がない場合は `None`) |
| `"probability_max"` | `float \| None` | 予測確率の最大値(候補がない場合は `None`) |


### 8.2. `v399.trim_terminal_tails()` でやっていること
十分な長さを持つ細胞軌跡(直鎖トラック)の両端(先頭・末尾)について、追跡の誤りや過剰延伸によるノイズ端点ノードを、機械学習モデル(LightGBM)を用いて外側から連続している部分だけトリミング(刈り込み)する。

**背景**
細胞追跡では、細胞が視野から消える直後や現れる直前、あるいは見失った後に、近傍のノイズや背景のムラを誤って1〜数フレーム余計に繋げてしまう(過剰追跡)ケースが起こりえる。この関数はその**末端のゴミ(tails)**を除去する。

#### 8.2.1 具体的な処理の流れ
1. **直鎖トラック(Simple Paths)の抽出[54〜101行目]**:
   - 途中に分岐(`outdegree > 1`)や合流(`indegree > 1`)を含まない、1本の直線状のトラックで、かつ一定以上の長さ(`minimum_component_nodes`)を持つものを抽出。
2. **端点刈り取り候補アクションの作成[126〜139行目]**:
   - 各トラックの始点側(`side=0`)と終点側(`side=1`)から、トラックの中央に向かって深さ `depth` (1 〜 `maximum_tail_depth`=5) のノードを削除候補アクションとしてリストアップする。
3. **特徴量の抽出**:
   - **端点と隣接点の画像特徴**: パッチ平均輝度、標準偏差、輝度比率(`patch_mean_ratio`)、エッジ間のパッチ正規化相互相関(NCC: `edge_ncc`)。
   - **空間・時間位置**: フレーム境界への近さ(`temporal_boundary_fraction`)、同時刻の最近傍ノード距離(`terminal_nearest_um`)。
   - **運動特性**: 端部の移動距離(`edge_um`)、進行方向の曲がり角(`first_turn_cosine`)、加速度(`first_acceleration_um`)。
   - **トラック全体の文脈**: トラック長、時間スパン、他モデル(`witnesses`)による支持数。
4. **LightGBM による推論**:
   - 各端点ノードについて「真の細胞(保持すべき)」である確率 `probability` を推論する。
5. **外側からの連続プレフィックスのみ削除[317〜322行目]**:
   - 端点(外側)から順に見ていき、確率が閾値 `threshold` 未満であれば削除リストに追加するが、**途中で一度でも閾値以上の信頼できるノードに達したら探索を即座に打ち切る(`break`)**。
   - これにより、トラックの途中に穴を開けることなく、「端っこに連続してくっついていた低信頼ノード」だけを綺麗に刈り取る。
6. **グラフ再構築と検証**:
   - 削除ノード・エッジを除去した新しい DatasetGraph を作成し、validate_graph()で整合性を確認する。

#### 8.2.2. 戻り値の型と各要素の type
DatasetGraphと、dict。

##### 8.2.2.第1要素: `result`
- **型 (`type`)**: **`biohub.io.submission.DatasetGraph`**
- **内容**: 信頼度の低い末端ノード・エッジがトリミングされた後の新しいグラフ。

##### 8.2.2.第2要素: `audit`
- **型 (`type`)**: **`dict`**
- **内容**: `validate_graph(result)` の出力に、末端トリミングの実行統計を追加した辞書。

| キー名 | 値の型 (`type`) | 内容 |
| :--- | :--- | :--- |
| `"dataset"` | `str` | データセット名 |
| `"nodes"` | `int` | 処理後の総ノード数 |
| `"edges"` | `int` | 処理後の総エッジ数 |
| `"forks"` | `int` | 分裂ノード数 |
| `"births"` | `int` | 新規発生ノード数 |
| `"deaths"` | `int` | 追跡終了ノード数 |
| `"mean_disp_um"` | `float` | 平均移動距離(μm) |
| `"p99_disp_um"` | `float` | 移動距離の99パーセンタイル値(μm) |
| `"acquisition_cluster"` | `str` | 撮影条件クラスタ名(例: `'acq_01'`) |
| `"threshold"` | `float` | 使用された確率閾値 |
| `"candidate_components"` | `int` | 評価対象となった直鎖トラック数 |
| `"candidate_actions"` | `int` | 評価した端点候補アクション数 |
| `"removed_actions"` | `int` | 削除されたアクション数 |
| `"removed_nodes"` | `int` | 削除された末端ノード数 |
| `"removed_edges"` | `int` | 削除されたエッジ数 |
| `"probability_min"` | `float` | 予測確率の最小値 |
| `"probability_max"` | `float` | 予測確率の最大値 |
| `"witness_matching_used"` | `bool` | 他モデルによる照合特徴量を使用したか |
| `"new_forward_passes"` | `int` | (追加フォワードパス数、`0`) |

### 8.3. `v454.apply_gap_admission()` でやっていること
トラッキンググラフにおける**1フレーム欠損(時間ギャップ: $t \rightarrow t+2$)を検出・補間し、新しいノードとエッジを追加する後処理(V454 Gap Admission)**をする。

#### 8.3.1 引数

| 引数名 | ノートブックでの実引数 | 型 | shape / 形式 | 説明 |
| :--- | :--- | :--- | :--- | :--- |
| **`graph`** | `v399_graph` | `DatasetGraph` | オブジェクト | 処理対象となる現在のトラッキンググラフ(直前の `v399.trim_terminal_tails` でノイズ末端が除去されたグラフ)。内部に `nodes`, `edges`, `shape` を保持。 |
| **`motion_ids`** | `gap_ids` | `np.ndarray` | `(N,)` (dtype: `int64`) | モーションベクトルが計算されている基準ノードのID一覧`final_raw.nodes`のキー配列。 |
| **`backward`** | `gap_backward` | `np.ndarray` | `(N, 3)` (dtype: `float64`) | `motion_ids` の各ノードに対する**過去方向($t-1$)への推定移動ベクトル** $(\Delta z, \Delta y, \Delta x)$ [$\mu\text{m}$単位]。 |
| **`forward`** | `gap_forward` | `np.ndarray` | `(N, 3)` (dtype: `float64`) | `motion_ids` の各ノードに対する**未来方向($t+1$)への推定移動ベクトル** $(\Delta z, \Delta y, \Delta x)$ [$\mu\text{m}$単位]。 |
| **`zarr_path`** | `path` | `pathlib.Path` | パスオブジェクト | データセットのZarr画像ディレクトリへのパス。候補点周辺のパッチ画像を切り出してコントラスト等の生画像特徴量を計算するために使用。 |
| **`metadata`** | `V454_METADATA` | `dict` | 辞書 | 事前学習済みのロジスティック回帰モデルおよび前処理パラメータ`load_assets() でロードする。 |

##### 補足: `metadata` 辞書に含まれる主な内容
- `features`: 使用する8種類の特徴量名リスト
- `threshold`: 採択判定の確率閾値(`0.65`)
- `imputer_statistics`: 欠損値補完用の中央値配列(shape: `(8,)`)
- `scaler_mean`: 特徴量正規化用の平均配列(shape: `(8,)`)
- `scaler_scale`: 特徴量正規化用のスケール配列(shape: `(8,)`)
- `logistic_coef`: ロジスティック回帰の重み配列(shape: `(8,)`)
- `logistic_intercept`: ロジスティック回帰のバイアス値(スカラー `float`)
- 

#### 8.3.2 主な処理フロー
1. **候補(Proposal)の抽出** `_proposal_rows()`
   - トラッキングが途切れている終端ノード(terminal: 出次数0, 時刻 $t$)と、新しく始まっているノード(root: 入次数0, 時刻 $t+2$)を探索。
   - terminalの順方向モーション予測(`forward`)と、rootの逆方向モーション予測(`backward`)から中間時刻 $t+1$ の位置を推測。
   - 予測位置の一致度(`agreement_um` $\le 3.0\,\mu\text{m}$)が高く、既存セルとも重複しない(新規性がある)中間点を補間候補として抽出。
2. **生画像特徴量の抽出** `_patch_stats()`
   - Zarrボリュームから候補座標 $(t+1, z, y, x)$ 周辺のパッチを切り出し、コントラスト(`raw_center_contrast`, `raw_peak_contrast`)、SNR(`raw_peak_snr`)、中心からのピークズレ(`raw_peak_offset`)を算出。
3. **採択確率のスコアリング** `_probabilities()`
   - 幾何特徴量と画像特徴量(計8次元)を標準化し、事前学習済みのロジスティック回帰モデルで採択確率(`selection_probability`)を計算。
4. **ノード・エッジの追加** [L184-L204]
   - 閾値(`threshold`=0.65)以上の候補から最もスコアの良い1件を選択。
   - 新規ノード `new_id` を $(t+1, z, y, x)$ に追加し、`parent -> new_id` および `new_id -> child` の2本のエッジ(confidence=0.30)を接続してバリデーション(`validate_graph()`を実行。

#### 8.3.3. 戻り値の型とshape

DatasetGraphとdict。

##### 8.3.3.第1要素: `result` (型: `DatasetGraph`)

| フィールド | 型 | 構造 / 要素数(shape相当) | 内容 |
| :--- | :--- | :--- | :--- |
| `dataset` | `str` | スカラー | データセット識別子名(例: `"acq_01"`) |
| `nodes` | `dict[int, tuple[int, int, int, int]]` | 長さ $N$ (採択時: 元ノード数 $+1$)、値タプルは `(t, z, y, x)` の長さ4 | ノードIDから4次元座標へのマッピング |
| `edges` | `list[tuple[int, int, float]]` | 長さ $E$ (採択時: 元エッジ数 $+2$)、各タプルは `(parent, child, conf)` の長さ3 | 接続エッジのリスト |
| `shape` | `tuple[int, int, int, int] \| None` | 長さ4: `(T, Z, Y, X)` | 画像ボリュームの全体形状 |

##### 8.3.3.第2要素: `audit` (型: `dict`)
処理結果の監査・メタ情報辞書。

- **候補なし / 閾値未満で採択なしの場合**:
  - `proposals` (`int`): 抽出された候補数
  - `above_threshold` (`int`): 閾値以上の候補数(0)
  - `selected` (`int`): 採択数(0)
- **1件採択された場合**:
  - `proposals`: 候補数 (`int`)
  - `above_threshold`: 閾値超え候補数 (`int`)
  - `selected`: `1` (`int`)
  - `selection_probability`: 採択確率 (`float`)
  - `parent`: 親ノードID (`int`)
  - `child`: 子ノードID (`int`)
  - `new_node_id`: 新規ノードID (`int`)
  - `coordinate_tzyx`: `[t, z, y, x]` (要素数4の `list[int]`)
  - `agreement_um`: 一致度距離 (`float`)
  - `endpoint_distance_um`: 端点間距離 (`float`)
  - `features`: 計算に使用した8種類の特徴量 (`dict[str, float]`)

### 8.4. `v507.prepare_actions()` がしていること
トラッキンググラフに対する2つの後処理アクション、**①ノード座標の微小補正(Localization / ゲーティング付き位置修正)** と **②誤った細胞分裂分岐の剪定(Fork Pruning / 分岐エッジ削除)** の適用対象を事前決定(準備・ロック)する。ここで決定されたアクション辞書に基づいて、直後の `v507.apply_prepared() によってグラフへ適用される。

#### 8.4.1. 引数
| 引数名 | 実引数 | 型 | shape / 形式 | 説明 |
| :--- | :--- | :--- | :--- | :--- |
| **`graph`** | `v454_graph` | `DatasetGraph` | オブジェクト | 処理対象のグラフ(直前の `v454.apply_gap_admission` 後のグラフ)。 |
| **`candidates`** | `c21` | `pd.DataFrame` | 行数 $M$, カラム数多数 | 検出器が出力した全候補点データフレーム(`t`, `z`, `y`, `x`, `score`, `rank_in_frame` など)。 |
| **`cluster`** | `v394_cluster` | `str` | スカラー | 取得条件クラスター名(例: `"acq_01"` かそれ以外か)。 |
| **`motion_ids`** | `gap_ids` | `np.ndarray` | `(N,)` (dtype: `int64`) | モーションサンプリングが行われた基準ノードID一覧。 |
| **`motion_nodes`** | `gap_nodes` | `np.ndarray` | `(N, 4)` (dtype: `int64`) | `motion_ids` 各ノードの `(t, z, y, x)` 座標配列。 |
| **`motion_backward`** | `gap_backward` | `np.ndarray` | `(N, 3)` (dtype: `float64`) | 過去方向への推定移動量 $(\Delta z, \Delta y, \Delta x)$ [$\mu\text{m}$単位]。 |
| **`motion_forward`** | `gap_forward` | `np.ndarray` | `(N, 3)` (dtype: `float64`) | 未来方向への推定移動量 $(\Delta z, \Delta y, \Delta x)$ [$\mu\text{m}$単位]。 |
| **`assets`** | `V507_ASSETS` | `dict` | 辞書 | `v507.load_assets()` でロードされたメタデータとモデル群(`localization` 3軸分, `gate`, `fork`)。 |

#### 8.4.2 主な処理の流れ
1. **グラフの表形式化** `_tables()`
   - 入力グラフの `nodes` と `edges` を DataFrame に変換。
2. **座標補正アクションの選定** `_localization_actions()`
   - ノードの前後速度・中点残差や、近傍検出候補点(`candidates`)とのオフセット・検出スコア等から特徴量を抽出。
   - 3軸別($z, y, x$)の学習済み回帰木モデルで変位量(residual)を予測し、ゲートモデル(gate model)で補正の信頼度を評価。
   - ゲート値が閾値(`localization_gate_threshold_um`)以上の高信頼ノードのみ、最大移動量キャップ(`localization_norm_cap_um`)をかけて補正後の $(z, y, x)$ 整数座標を決定。
3. **分岐剪定アクションの選定** `_fork_action()
   - 親ノード1つに対して子ノードが2つある分岐点(`fork`)をすべて抽出。
   - 2本のアームの幾何距離、速度残差、オプティカルフローモーション予測残差(`forward`/`backward`)などの特徴量を抽出。
   - 分岐剪定モデルで「誤った余剰分岐である確率」を推論し、閾値(`fork_probability_threshold`)を超えた最も確信度の高い分岐から、モーション不整合が大きい方の子へのエッジ $(parent \rightarrow child)$ を削除対象として最大1件選定。

#### 8.4.3. 戻り値の型とshape

dictと dict。

##### 8.4.3.第1要素: `prepared` (型: `dict`)

| キー | 型 | 構造 / 要素数(shape相当) | 説明 |
| :--- | :--- | :--- | :--- |
| `"localization"` | `dict[int, tuple[int, int, int]]` | 長さ $K$ ($0 \le K \le |V|$)、値は長さ3のタプル | 補正対象のノードID $\rightarrow$ 新しい座標 `(z, y, x)` の辞書。 |
| `"fork"` | `dict \| None` | 辞書または `None` | 剪定対象となる分岐エッジ情報。<br>選定時: `{"parent": int, "child": int, "probability": float}`<br>非該当時: `None` |

##### 8.4.3.第2要素: `audit` (型: `dict`)
実行時のログ・評価指標辞書。

- **`audit["localization"]`** (`dict`):
  - `gate_threshold_um` (`float`): ゲート閾値
  - `selected` (`int`): ゲートを通過したノード数
  - `moved` (`int`): 実際に座標値が変化するノード数
  - `gate_min`, `gate_max` (`float`): ゲートスコアの最小・最大値
- **`audit["fork"]`** (`dict`):
  - `candidates` (`int`): 評価した分岐の総数
  - `above_threshold` (`int`): 確率閾値を超えた分岐数
  - `selected` (`int`): 採択数(0 または 1)
  - 採択時は `parent`, `child`, `probability` が追加されます。


### 8.5. `v507.apply_prepared()` がしていること

直前の `v507.prepare_actions()`で決定・事前準備されたアクション辞書(`prepared`)を元に、**入力グラフに対して実際に座標補正と分岐エッジ削除を適用して新しいグラフを生成・検証する**。

#### 8.5.1. 引数

| 引数名 | 実引数 | 型 | shape / 形式 | 説明 |
| :--- | :--- | :--- | :--- | :--- |
| **`graph`** | `v454_graph` | `DatasetGraph` | オブジェクト | アクション適用対象となる現在のグラフ。 |
| **`prepared`** | `v507_prepared` | `dict` | 辞書 | `v507.prepare_actions()`で作成されたアクション辞書。<br>・`"localization"`: `dict[int, tuple[int, int, int]]` (ノードID $\rightarrow$ 新座標)<br>・`"fork"`: `dict \| None` (`{"parent": int, "child": int, "probability": float}`) |


#### 8.5.2 具体的な処理の流れ
1. **ノード座標の更新(Localizationの適用)** [L407-L416]
   - `prepared["localization"]` に登録されているノードIDと新座標 `(z, y, x)` を走査。
   - グラフ内に存在するノードについて、時刻 $t$ は維持したまま空間座標 $(z, y, x)$ を予測された補正座標に書き換え。
   - 実際に座標値が変化した件数を `moved` としてカウントする。
2. **分岐エッジの削除(Fork Pruningの適用)** [L417-L425]
   - `prepared["fork"]` が指定されている場合、対象の $(parent \rightarrow child)$ エッジを探索し、該当する1本のエッジをグラフから削除(剪定)。
3. **グラフの再構築とバリデーション** [L426-L436]
   - 更新された `nodes` と `edges` を用いて新しい `DatasetGraph` インスタンスを生成。
   - `validate_graph()` で重複IDや座標の有効範囲、分岐数などを検証し、監査辞書(`audit`)とともに返却する。

#### 8.5.3. 戻り値の型とshape
DatasetGraphとdict。

##### 8.5.3.第1要素: `result` (型: `DatasetGraph`)
アクション適用後の新しいトラッキンググラフオブジェクト。

| フィールド | 型 | 構造 / 要素数(shape相当) | 内容 |
| :--- | :--- | :--- | :--- |
| `dataset` | `str` | スカラー | データセット識別子名(例: `"acq_01"`) |
| `nodes` | `dict[int, tuple[int, int, int, int]]` | 長さ $N$ (ノード数は不変)、値タプルは `(t, z, y, x)` | 座標補正が適用されたノード辞書 |
| `edges` | `list[tuple[int, int, float]]` | 長さ $E$ (fork削除が適用された場合は $-1$) | 剪定後のエッジリスト |
| `shape` | `tuple[int, int, int, int] \| None` | 長さ4: `(T, Z, Y, X)` | 画像ボリュームの全体形状 |

##### 8.5.3.第2要素: `audit` (型: `dict`)
アクション適用結果の監査情報辞書(すべてスカラーの整数値)。

| キー | 型 | 説明 |
| :--- | :--- | :--- |
| `"localization_actions"` | `int` | 準備されていた位置補正アクション総数 |
| `"localization_moved"` | `int` | 実際に座標値が書き換わって移動したノード数 |
| `"localization_nodes_removed_by_tail"` | `int` | 補正対象だったがグラフ上に存在しなかったノード数 |
| `"fork_selected"` | `int` | 分岐削除アクションが選択されていたか(1 または 0) |
| `"fork_applied"` | `int` | 実際に該当分岐エッジが削除されたか(1 または 0) |
| `"fork_skipped"` | `int` | 削除対象だったがスキップされたか(1 または 0) |
| `"nodes"` | `int` | 適用後の総ノード数 |
| `"edges"` | `int` | 適用後の総エッジ数 |
| `"components"` | `int` | グラフ内の連結成分数 (`nodes - edges`) |
| `"forks"` | `int` | グラフ内に残存している細胞分裂分岐の総数 |

### 8.6. `v533.apply()` がしていること
全体の最終段処理として、**①未検出チェーンの復活(Contrast-Gated Add-Back)**と、**②視野境界アーティファクトの削除(XY Edge Trim)**を実施。機械学習モデルは使わず、NumPy/SciPy/Zarrによる幾何・画像演算のみ動作。

#### 8.6.1 引数

| 引数名 | ノートブックでの実引数 | 型 | shape / 形式 | 説明 |
| :--- | :--- | :--- | :--- | :--- |
| **`graph`** | `v507_graph` | `DatasetGraph` | オブジェクト | 処理対象となるグラフ(V507適用後のグラフ)。 |
| **`candidates`** | `c21` | `pd.DataFrame` | 行数 $M$, カラム数多数 | 検出器が出力したプライマリ検出候補データフレーム(`t`, `z`, `y`, `x`, `score`, `rank_in_frame` 等)。ステップAの未検出セル探索に使用。 |
| **`cluster`** | `v394_cluster` | `str` | スカラー | データセットの取得条件クラスター名(`"acq_01"` の場合のみステップAが有効化)。 |
| **`zarr_path`** | `path` | `pathlib.Path` | パスオブジェクト | データセットのZarrディレクトリパス。画像有効領域バウンディングボックスの計算や、ステップAのコントラスト計測に使用。 |
| **`p`** | *(省略: None)* | `dict` | 辞書(省略時は既定パラメータ) | ハイパーパラメータ辞書。省略時はモジュール内の `PARAMS` 辞書が使用される(主な値: `edge_px: 1.5`, `min_nodes: 9`, `min_contrast: 0.1` など)。 |


#### 8.6.2 具体的な処理の流れ
1. **ステップA: コントラスト判定付きチェーン追加(Add-Back)** (※`cluster == "acq_01"` のみ実行)
   - 既存グラフから $4.5\,\mu\text{m}$ 以上離れた未検出候補点(ホール)を検出器の候補プール(`candidates`)から抽出。
   - 近傍点同士を時間フレーム間で繋いでチェーンを形成(長さ $\ge 3$)し、既存トラックの端点への接続も探索。
   - 厳格な採択基準(ノード数 $\ge 9$、近傍距離 $\le 8\,\mu\text{m}$、実画像での局所コントラスト中央値 $> 0.1$)をすべて満たすチェーンのみを新規ノードおよびエッジとしてグラフに追加。
2. **ステップB: 視野境界ノード・エッジの剪定(XY Edge Trim)** (※全データセットで実行)
   - Zarr生画像の有効データ領域(ゼロパディング領域を除くバウンディングボックス)をフレームごとに計算。
   - $y$ 軸または $x$ 軸のデータ有効境界から 1.5 px(`edge_px`) 以内の端に存在するノードを、境界アーティファクトとして接続エッジごと削除。
3. **最終グラフの生成と検証** [L291-L296]
   - 最終的なノードとエッジで `DatasetGraph`を構築、`validate_graph() で形式・制約をチェックして返却。

#### 8.6.3. 戻り値の型とshape
DatasetGraphとdict。

##### 8.6.3.第1要素: `out` (型: `DatasetGraph`)
コンペ提出の最終出力となるトラッキンググラフ。

| フィールド | 型 | 構造 / 要素数(shape相当) | 内容 |
| :--- | :--- | :--- | :--- |
| `dataset` | `str` | スカラー | データセット識別子名(例: `"acq_01"`) |
| `nodes` | `dict[int, tuple[int, int, int, int]]` | 長さ $N_{final}$、値タプルは `(t, z, y, x)` | 最終ノード辞書(未検出チェーンの追加、境界ノードの削除が反映) |
| `edges` | `list[tuple[int, int, float]]` | 長さ $E_{final}$、各エッジは `(parent, child, conf)` | 最終エッジリスト |
| `shape` | `tuple[int, int, int, int] \| None` | 長さ4: `(T, Z, Y, X)` | 画像ボリュームの全体形状 |

##### 8.6.3.第2要素: `audit` (型: `dict`)
処理の詳細ログ辞書。

| キー | 型 | 説明 |
| :--- | :--- | :--- |
| `"version"` | `str` | `"v533_addback_edge_v1"` |
| `"cluster"` | `str` | クラスター名 |
| `"addback_applied"` | `bool` | ステップA(add-back)が実行されたか(`cluster == "acq_01"`) |
| `"proposals"` | `int` | 構築されたチェーン候補総数 |
| `"length_near_pass"` | `int` | 長さ・近傍条件を通過した候補数 |
| `"selected"` | `int` | コントラスト閾値をクリアして採択された候補数 |
| `"kept"` | `int` | 競合解消後に残ったチェーン数 |
| `"added_nodes"` | `int` | ステップAで新たに追加されたノード数 |
| `"added_edges"` | `int` | ステップAで新たに追加されたエッジ数 |
| `"trimmed_nodes"` | `int` | ステップBで視野端境界として削除されたノード数 |
| `"trimmed_edges"` | `int` | 境界ノード削除に伴って削除されたエッジ数 |
| `"nodes"` | `int` | 最終的な総ノード数 |
| `"edges"` | `int` | 最終的な総エッジ数 |
| `"forks"` | `int` | 最終的な分岐数 |
| `"seconds_addback"` | `float` | ステップAの所要時間(秒) |
| `"seconds_total"` | `float` | V533全体の所要時間(秒) |

---

## 9. (cell 6: line 891〜911) サブミッションの検証・出力
submission.csvを生成する処理。大したことはやってない。

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant BuildSub as build_submission()
    participant WriteSub as write_submission()
    participant FileSys as Path.replace() / write_text()

    Script->>BuildSub: build_submission(graphs)
    BuildSub-->>Script: df (DataFrame), summary

    Script->>WriteSub: write_submission(df, temporary_output)
    WriteSub-->>Script: 一時ファイル (.tmp.csv) 書き込み完了

    Script->>FileSys: temporary_output.replace(OUT)
    FileSys-->>Script: submission.csv 確定置換
    Script->>FileSys: audit_path.write_text(audit_json)
    FileSys-->>Script: v533_deployment_audit.json 保存完了
```
## 処理解説完了
ここまでで、処理概要のまとめは完了。
2位の解法のポイントはアンサンブルな気がしてます。アンサンブルは使い方次第で、スコアダウンするのですが、バギング、ブースティング、スタッキングを巧みに使っているのがハイスコアのポイントかと。
なので、アンサンブルをどんな風に使っているかをまとめてみました。

---

### アンサンブルまとめ
「検出」「トラック結合」「細胞分裂（fork）」「後処理」のすべての階層にわたって、バギング・ブースティング・スタッキング・多数決合意形成(Consensus)で、多段統合されている。
単なる「モデルの平均化」にとどまらず、**ニューラルネット $\rightarrow$ GBDT（ブースティング） $\rightarrow$ メタ線形モデル（スタッキング） $\rightarrow$ 異種トラッカー間の多数決合意** という階層的なアンサンブル構造。

#### ディレクトリ別仕分け（全12個）

| ディレクトリ名 | 含まれるアンサンブル・モデルの役割 |
| :--- | :--- |
| **`biohub-cell-tracking-weights-v25-consensus`** | 5-fold U-Netのバギング、5-fold GBDT（エッジ/ペア）、多段ロジスティック回帰スタッキング群 |
| **`biohub-v35-strong-detectors`** | 5-fold U-Netバギング（独立トラッカー `stage6` 生成用） |
| **`biohub-v41-probe-weights`** | 5フレーム時間窓 U-Net（`v39` トラッカー生成用） |
| **`biohub-v57-deployment-weights`** | `v38` トラッカー用U-Net、リスク評価用の5-fold LightGBMバギング（`V57_BOOSTERS`） |
| **`biohub-v343-hard-dim-code`** | 暗部検出LightGBM（`full.dim.txt`）、トラッカー多数決投票ロジック（`v169`） |
| **`biohub-v389-structured-head`** | GBDT `tree.txt` と結合するニューラル残差モデル（`residual.pt`） |
| **`biohub-v394-component-protector`** | 孤立成分保護LightGBM（4トラッカーの合意を特徴量に使用） |
| **`biohub-v399-tail-protector`** | 末端ノイズトリミング用LightGBM（`tail_protector.txt`） |
| **`biohub-v454-current-graph-gap-admission`** | モーションと画像特徴量を統合するロジスティック回帰（1フレーム欠損補完） |
| **`biohub-v507-localization-fork`** | 3軸 HistGradientBoosting 回帰木＋ゲート決定木（座標補正・分岐剪定） |
| **`biohub-v533-addback-edge`** | 未検出チェーン復活と視野端境界トリミング（※ルールベースのためMLモデルなし） |

#### 1. バギング（Bagging / Fold Averaging）
※バギング(Bagging) ...「別々に学習させた複数のモデルの予測を、並列に多数決や平均でまとめるアンサンブル手法」 のこと。
各フォールドで独立に学習した同一アーキテクチャのモデル群を並行実行し、予測値を平均化して分散を低減させています。

| 適用箇所 | 使用ファイル（`datasets/soheilayati` 配下） | 処理内容 |
| :--- | :--- | :--- |
| **細胞検出ヒートマップのバギング** | `biohub-cell-tracking-weights-v25-consensus/`<br>・`fold0.ep05.pt` 〜 `fold4.ep05.pt` ... `V21_MODELS`<br>・`v24.fold0.ep05.pt` 〜 `v24.fold4.ep05.pt` ... `V24_MODELS`<br>`biohub-v35-strong-detectors/`<br>・`strong.fold0.ep05.pt` 〜 `strong.fold4.ep05.pt` ... `STRONG_MODELS` | 5つの U-Net モデルの予測ヒートマップおよびベクトル場を単純相加平均（5モデル平均＋180度回転TTA）。 |
| **トラックレット（小トラック）結合のバギング** | `biohub-cell-tracking-weights-v25-consensus/`<br>・`edge_tracklet_lgb_l7_fold0.txt` 〜 `fold4.txt`<br>・`edge_tracklet_lgb_l15_fold0.txt` 〜 `fold4.txt`<br>・`edge_postselect_lgb_l31_fold0.txt` 〜 `fold4.txt` | それぞれ5つの LightGBM ブースター(テキスト形式)を読み込み、`BoosterEnsemble.predict_p()` で5モデルの確率を平均化。 |
| **誤検出ノード削除リスクのバギング** | `biohub-v57-deployment-weights/`<br>・`fold0.txt` 〜 `fold4.txt` ... `V57_BOOSTERS` | 5つの LightGBM ブースターによる平均確率を `predict_bagged_risk()` で推論し、トラック交換時の削除リスクを安定判定。 |

---

#### 2. ブースティング（Boosting / GBDT）
※Boost...強化する。弱いモデルを鍛え上げて強いモデルへとブーストするモデルのこと。バギングが「みんなで一斉に解いて多数決をとる（並列）」のに対し、ブースティングは「前の人のミスを次の人が集中的に直す（直列リレー）」となっている。
弱学習器（決定木）を逐次的に追加して誤差を補正していく。テーブル特徴量（幾何距離、速度、スコア等）を扱うモデルはほぼすべて GBDT(LightGBM または HistGradientBoosting)で構成。

| 適用箇所 | 使用ファイル（`datasets/soheilayati` 配下） | 処理内容 |
| :--- | :--- | :--- |
| **暗部細胞（Dim Region）の局所補正** | `biohub-v343-hard-dim-code/`<br>・`full.dim.txt`  ... `V343_DIM_MODEL`<br>・`full.cell.txt` ... `V343_CELL_MODEL`| 顕微鏡の暗い領域で見落とされた候補をサルベージ(救助)し、候補ランクを再重み付けする LightGBM ブースター。 |
| **トラック接続エッジ / 分裂ペア判定** | `biohub-cell-tracking-weights-v25-consensus/`<br>・`edge_model_b125.npz` ... `EDGE_MODEL`<br>・`pair_model_v0.npz` ... `PAIR_MODEL`<br>・`edge_model_sv11_ctx.npz` ... `CTX_MODEL`| 時間的連続性や姉妹細胞ペアの尤度をスコアリングする 5-fold 勾配ブースティング決定木。 |
| **グラフ後処理の保護・剪定** | `biohub-v394-component-protector`<br>・`component_protector.txt`<br>`biohub-v399-tail-protector`<br>・`tail_protector.txt` | 孤立した偽トラックの削除判定、およびトラック終端のノイズ尾部（tails）の剪定を行う LightGBM モデル。 |
| **ゲート付き座標補正 & 分岐剪定** | `biohub-v507-localization-fork/`<br>・`localization_axis0.npz` 〜 `axis2.npz`<br>・`localization_gate.npz`<br>・`fork_pruner.npz` | 3軸方向の座標補正量を回帰木（HistGradientBoosting）で予測。さらにゲートモデル（決定木）が高確信度を出したノードのみを動かすゲート付きブースティング。 |

#### 3. スタッキング（Stacking / 階層型メタモデル）
※stack ...積み重ねる、積み上げる という意。専門家A（画像が得意）と専門家B（数値計算が得意）にそれぞれ意見を出させ、最後に『審査員長（メタモデル）』が『このパターンのときは専門家Bの意見を重視しよう』と判断して決める方法。
1次モデル(GBDTや検出器)の予測出力やマージンを特徴量として上位のメタモデル(主にロジスティック回帰)へ渡し、確率を再学習・較正している。

| 適用箇所 | 使用ファイル | 処理内容 |
| :--- | :--- | :--- |
| **グループ較正（勝者総取りスタッキング）** | `biohub-cell-tracking-weights-v25-consensus/`<br>・`edge_model_b125.npz` 内のキャリブレータパラメータ | 1位候補エッジのロジット、2位候補との差分(マージン)、子ノード特徴量を入力とし、上位のロジスティック回帰(`rescore_argmax`)で接続確率を較正。 |
| **時間的分裂スタック（Temporal Division Stack）** | `biohub-cell-tracking-weights-v25-consensus/`<br>・`pair_stack_faithful.npz` ... `PAIR_STACK` | 1次モデル `PAIR_MODEL` の予測ロジット（`base_pair_logit`）を入力特徴量に含め、時間・空間特徴量とともに 5-fold のロジスティック回帰(`PairStack`) でメタ判定。 |
| **難例・高解像度特化スタッキング群** | `biohub-cell-tracking-weights-v25-consensus/`<br>・`pair_allimg_lr.npz`<br>・`pair_full_lr.npz`<br>・`pair_hardneg_lr.npz`<br>・`pair_highres_lr.npz`<br>・`fork_confirmation_lr.npz` | ハードネガティブ（誤接続しやすいペア）や高解像度特徴量で個別に訓練されたロジスティック回帰スタッキングモデル群を多段に適用。 |
| **欠損ギャップ採択スタック** | `biohub-v454-current-graph-gap-admission/`<br>・`metadata.json` ... `V454_METADATA`| 前方・後方モーションベクトルの一致度距離と、生画像パッチから抽出したコントラスト・SNR特徴量を結合し、正規化＋ロジスティック回帰で採択確率を算出。 |

#### 4. 合意形成（Multi-Witness Consensus / 多数決アンサンブル）

**「設定の異なる複数の独立トラッカー（Witnesses）を並行して構築し、グラフ同士の合意を取る」** という、グラフレベルのアンサンブル。

1. **4系統の独立トラッカーの並行生成**:
   - `v24`: 3フレーム窓ベースライン検出器によるトラッカー
   - `stage6`: `biohub-v35-strong-detectors`（強力検出器）によるトラッカー
   - `v39`: `biohub-v41-probe-weights`（5フレーム窓検出器）によるトラッカー
   - `v38`: `biohub-v57-deployment-weights`（長期追跡モデル）によるトラッカー
2. **エッジの合意追加（Consensus Edges）**:
   - 複数のウィットネスグラフ間で共通して支持される高信頼エッジ（multi-witness edges）を抽出し、ベースグラフへ注入。
3. **細胞分裂(Fork)の多数決投票**:
   - 細胞分裂 $(parent \rightarrow child_1, child_2)$ に対して各トラッカーが1票ずつ投票し、**2つ以上のトラッカーが一致して支持した場合のみ分裂を採用（`root_consensus2`）**。


#### 各フェーズで使用されているバギングまとめ
2位解法における**「検出」「トラック結合」「細胞分裂」「後処理」**の全4フェーズにわたる処理フローと、各ステップで実行されているアンサンブル手法をまとめてみた。

```mermaid
flowchart TD
    %% スタイル定義
    classDef bagging fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef boosting fill:#e8f8f5,stroke:#26a69a,stroke-width:2px,color:#004d40;
    classDef stacking fill:#fff3e0,stroke:#fb8c00,stroke-width:2px,color:#e65100;
    classDef consensus fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#4a148c;
    classDef data fill:#f5f5f5,stroke:#757575,stroke-width:2px,color:#212121;

    RAW["3D顕微鏡生画像 (.zarr)"]:::data

    %% ----------------------------------------------------
    %% フェーズ1: 検出
    %% ----------------------------------------------------
    subgraph P1 ["1. 検出フェーズ (Detection)"]
        D1["5-fold U-Net 単純平均 ＋ 180°TTA<br><b>[バギング]</b> (V21_MODELS)"]:::bagging
        D2["暗部領域の細胞サルベージ<br><b>[ブースティング]</b> (V343_DIM_MODEL / LightGBM)"]:::boosting
        RAW --> D1
        D1 --> D2
    end

    %% ----------------------------------------------------
    %% フェーズ2: トラック結合
    %% ----------------------------------------------------
    subgraph P2 ["2. トラック結合フェーズ (Tracking & Edge Consensus)"]
        T1["4系統の独立トラッカー並行生成<br><b>[マルチウィットネス生成]</b> (v24, stage6, v39, v38)"]:::consensus
        T2["5-fold GBDT によるエッジ尤度推論<br><b>[ブースティング]</b> (EDGE_MODEL)"]:::boosting
        T3["勝者総取りのグループ較正<br><b>[スタッキング]</b> (rescore_argmax / ロジスティック回帰)"]:::stacking
        T4["トラックレット結合スコアリング<br><b>[バギング]</b> (TRACK_MODEL / 5-fold LightGBM)"]:::bagging
        T5["複数トラッカー共通の高信頼エッジ注入<br><b>[多数決合意]</b> (add_consensus_edges)"]:::consensus
        T6["ノード削除リスクの評価<br><b>[バギング]</b> (predict_bagged_risk / 5-fold LightGBM)"]:::bagging

        D2 --> T1
        D2 --> T2
        T2 --> T3
        T3 --> T4
        T1 --> T5
        T4 --> T5
        T5 --> T6
    end

    %% ----------------------------------------------------
    %% フェーズ3: 細胞分裂
    %% ----------------------------------------------------
    subgraph P3 ["3. 細胞分裂フェーズ (Cell Division / Fork Consensus)"]
        F1["5-fold GBDT による姉妹ペア尤度推論<br><b>[ブースティング]</b> (PAIR_MODEL)"]:::boosting
        F2["GBDTロジットを入力とする多段メタモデル<br><b>[スタッキング]</b> (PairStack / 5-fold ロジスティック回帰)"]:::stacking
        F3["分裂確定スコアリング<br><b>[スタッキング]</b> (CONFIRM_MODEL / ロジスティック回帰)"]:::stacking
        F4["4トラッカーによる分裂多数決投票<br><b>[多数決合意]</b> (fork_votes / root_consensus2)"]:::consensus

        T6 --> F1
        F1 --> F2
        F2 --> F3
        T1 -.->|4つのトラッカーグラフ| F4
        F3 --> F4
    end

    %% ----------------------------------------------------
    %% フェーズ4: 後処理
    %% ----------------------------------------------------
    subgraph P4 ["4. 後処理フェーズ (Post-Processing & Refinement)"]
        PP1["4トラッカー合意に基づく孤立成分保護<br><b>[ブースティング ＋ 合意]</b> (V394 / LightGBM)"]:::boosting
        PP2["トラック終端のノイズ尾部トリミング<br><b>[ブースティング]</b> (V399 / LightGBM)"]:::boosting
        PP3["1フレーム欠損ギャップの補間採択<br><b>[スタッキング]</b> (V454 / ロジスティック回帰)"]:::stacking
        PP4["3軸微小移動 & 余剰分裂の剪定<br><b>[ゲート付きブースティング]</b> (V507 / HistGradientBoosting)"]:::boosting
        PP5["未検出チェーン追加 & 視野端境界トリミング<br><b>[ルールベース]</b> (V533)"]:::data

        F4 --> PP1
        PP1 --> PP2
        PP2 --> PP3
        PP3 --> PP4
        PP4 --> PP5
    end

    OUT["最終提出ファイル (submission.csv)"]:::data
    PP5 --> OUT
```

##### 各ステップの詳細とアンサンブル対照表

###### 1. 検出フェーズ (Detection)
画像から細胞の位置 $(t, z, y, x)$ を検出し、候補点プールを構築するフェーズ。

| ステップ | アンサンブル種別 | 使用モデル・ファイル | 処理内容と目的 |
| :--- | :---: | :--- | :--- |
| **ヒートマップ生成** | 🔵 **バギング** | `V21_MODELS`<br>`paired_rotation.py`| 5つの U-Net（5-fold）の予測値を**単純平均**し、さらに180度回転画像（TTA）との平均をとることで、ノイズに頑健なヒートマップを生成。 |
| **暗部領域の補正** | 🟢 **ブースティング** | `V343_DIM_MODEL`<br>`dim_scaffold.py`| 顕微鏡の暗い領域（dim region）で見落とされがちな細胞を、専用に訓練された **LightGBM** でスコア再重み付け・サルベージ。 |

###### 2. トラック結合フェーズ (Tracking & Edge Consensus)
細胞同士を時間方向に結びつけ、個々の細胞の軌跡（トラック）を形成するフェーズ。

| ステップ | アンサンブル種別 | 使用モデル・ファイル | 処理内容と目的 |
| :--- | :---: | :--- | :--- |
| **独立トラッカー生成** | 🟣 **合意用モデル群** | `v24`, `stage6`, `v39`, `v38`<br>`2nd-place-notebook.py` | 異なる時間窓や検出器（`STRONG_MODELS` など）で学習した **4系統のトラッカー** を並行実行し、合意形成用の証人グラフを用意。 |
| **エッジ尤度推論** | 🟢 **ブースティング** | `EDGE_MODEL`<br>`edge_model_b125.npz`| 5-fold の **勾配ブースティング木（GBDT）** で、細胞間の移動ベクトル整合性・距離からエッジ接続スコアを推論。 |
| **グループ較正** | 🟠 **スタッキング** | `rescore_argmax`<br>`edge_model.py` | 1位エッジのロジット、2位とのマージンを入力とし、上位の **ロジスティック回帰** で勝者総取りの確率を再学習・較正。 |
| **トラックレット結合** | 🔵 **バギング** | `TRACK_MODEL`<br>`edge_tracklet_lgb_l7/l15`| 小トラック同士を接続する際、葉数違い（leaves 7, 15）の **5-fold LightGBM** の予測確率を平均化。 |
| **エッジ合意注入** | 🟣 **多数決合意** | `add_consensus_edges`<br>`v55_deploy.py`| 単一モデルの接続に依存せず、**複数の独立トラッカー（witnesses）が共通して支持した高信頼エッジ** をベースグラフに追加。 |
| **削除リスク評価** | 🔵 **バギング** | `predict_bagged_risk`<br>`V57_BOOSTERS`| トラック入れ替え時に誤検出ノードを削除する際、**5つの LightGBM ブースターの平均確率** でリスクを判定。 |

###### 3. 細胞分裂フェーズ (Cell Division / Fork Consensus)
1つの親細胞が2つの娘細胞に分裂する分岐（fork）を検出・確定するフェーズ。エラーペナルティが最も重いため、最も重厚なアンサンブルが組まれている。

| ステップ | アンサンブル種別 | 使用モデル・ファイル | 処理内容と目的 |
| :--- | :---: | :--- | :--- |
| **分裂ペア尤度推論** | 🟢 **ブースティング** | `PAIR_MODEL`<br>`pair_model_v0.npz` | 5-fold の **勾配ブースティング木（GBDT）** で、分裂候補となる2つの娘細胞ペアの尤度スコアを算出。 |
| **多段メタ推論** | 🟠 **スタッキング** | `PairStack`<br>`pair_stack_faithful.npz`等 | 1階の `PAIR_MODEL` の予測ロジット（`base_pair_logit`）を入力特徴量とし、幾何情報とともに 5-fold の **ロジスティック回帰** で確率をメタ学習。難例特化（HardNeg）や高解像度（HighRes）スタックも並行適用。 |
| **分裂多数決投票** | 🟣 **多数決合意** | `fork_votes`<br>`root_consensus2`| 4系統のトラッカーがそれぞれ細胞分裂候補に1票ずつ投票。**2つ以上のトラッカーが合意した場合のみ分裂を正式採用**。 |

###### 4. 後処理フェーズ (Post-Processing & Refinement)
完成したグラフからノイズを除去し、座標精度を極限まで高めるフェーズです。

| ステップ | アンサンブル種別 | 使用モデル・ファイル | 処理内容と目的 |
| :--- | :---: | :--- | :--- |
| **孤立成分保護 (V394)** | 🟢 **ブースティング**<br>＋ 🟣 **合意** | `V394_MODEL`<br>`v394_component_protector.py`| **LightGBM** で孤立成分を剪定。その際「他トラッカーが支持しているか（Witness一致度）」を特徴量として利用。 |
| **尾部トリミング (V399)** | 🟢 **ブースティング** | `V399_MODEL`<br>`v399_tail_protector.py`| トラックの端っこに生じるノイズのヒゲ（尾部）を **LightGBM** で検出して切り落とす。 |
| **欠損ギャップ採択 (V454)** | 🟠 **スタッキング** | `V454_METADATA`<br>`v454_gap_admission.py`| 前後モーション予測のズレと生画像のパッチ特徴量を **ロジスティック回帰** に通し、1フレーム途切れたギャップを補完。 |
| **座標補正 & 分岐剪定 (V507)** | 🟢 **ゲート付き<br>ブースティング** | `V507_ASSETS`<br>`v507_candidate.py`| 3軸の微小移動量を **HistGradientBoosting 回帰木** で予測。さらに **ゲート決定木** が高スコアを出した確実なノードのみ位置を微修正。 |
| **視野端処理 (V533)** | — (ルールベース) | `v533_addback_edge.py` v533_addback_edge.py | 有効データ境界ギリギリにあるアーティファクトノードを削除し、提出用フォーマットを完成。 |

## モデル生成レシピとアンサンブルのまとめ

### 1. 全体の学習データと分割設計（共通基盤）

すべてのモデルの学習基盤には、極めて限られたデータリソースを最大活用するための**時間軸分割戦略**と、**完全な OOF（Out-of-Fold）多段学習**が敷かれています。

#### ① 学習用データの正体（全4本の動画）
コンペティションで提供された教師データは、**2種類の胚（`44b6` と `6bba`）** にそれぞれ2つの撮影条件（`acq_00`, `acq_01`）が存在する、**合計4本の3Dタイムラプス動画**のみです。
- 胚1（`6bba`）: 正解ラベル全体の **84.6%** が集中している。
- 胚2（`44b6`）: 全体の **15.4%**（細胞全体の0.77%）しか注釈がない。

#### ② 本番モデルの分割戦略: 時間軸5分割（Temporal 5-fold）
胚が2つしかなく、正解ラベルの偏りも極端であるため、胚単位でデータを切り捨てるような分割は行っていません。
**2つの胚(全4動画)のデータを100%フル投入し、動画の「時間軸(タイムフレーム番号)」を5つの連続ブロックに分割(Temporal 5-fold)** して、Fold 0〜4 の 5つのモデルを学習させています。

これにより、全動画の貴重な注釈を余すことなく学習に使いつつ、時間方向の未知フレームに対する頑健性を確保しています。

#### ③ 完全な OOF（Out-of-Fold）による多段学習
後段のモデル（GBDTやスタッキングモデル）を学習させる際、**「前段のモデルが学習時に一度も見たことがない OOF 予測値」** を特徴量として使用しています。これにより、訓練時の過学習スコアに騙されず、テストデータと全く同じ分布で上位モデルを学習させています。

### 2. 各フェーズの詳細と生成レシピ（モデル別詳細）

#### ◆フェーズ1: 検出 (Detection)
画像から細胞の中心座標 $(t, z, y, x)$ を検出し、候補点プールを構築するフェーズ。

##### ① 3D U-Net モデル群（`V21`, `V24`, `STRONG`, `V39`, `V38`）
- **役割**: 画像から細胞中心のヒートマップと移動ベクトル(flow)を予測。本番推論では 5-fold の予測値を **単純平均(バギング)** し、180度回転画像（TTA）とも平均化。
- **モデル構造**:
  - 3D Temporal U-Net(`TemporalUNet3d`)。
  - バッチ統計の破綻を防ぐため BatchNorm ではなく **GroupNorm** を採用。
  - ボトルネックに時間混合層（`TemporalMix`）を配置。
- **入力データ**:
  - 3フレーム（`v39`/`v38` は 5フレーム）の連続スタック $[t-1, t, t+1]$。
  - 空間を $1.625\,\mu\text{m}$ の等方グリッドにダウンサンプリング(全体が $64 \times 64 \times 64$ に収まるためパッチ分割不要で丸ごと入力)。
- **学習損失関数の秘密(最大のブレイクスルー)**:
  コンペの注釈は非常に疎（スパース）で、全細胞の 1% 未満しか注釈がありません。そこで領域を3分割して学習させています(`targets.py:L11-L32`)
  1. **Positive（正例）**: 注釈中心から半径 $\sigma = 2.0\,\mu\text{m}$ のガウシアン球。複数細胞の重なりは sum ではなく **max** で合成（細胞分裂時の中心の「くびれ」を消さないため）。
  2. **Background（真の背景）**: 正規化輝度が極めて暗い領域（$\tau < 0.02$）のみを背景として学習。
  3. **Unknown（未注釈領域）**: それ以外の全領域は **「損失（Loss）を一切計算しない（No Loss）」**。
     - ※公式ベースラインは未注釈領域を「背景」として扱っていたため、約99%の実細胞を背景として誤学習していました。これを「損失無視」にしたことで検出精度が大幅に向上す。

##### ② 暗部領域スペシャリスト（`V343_DIM_MODEL`, `V343_CELL_MODEL`）
- **役割**: 顕微鏡の暗い領域（dim region）で見落とされやすい細胞をサルベージ。
- **アルゴリズム**: **LightGBM（ブースティング）**
- **レシピ**: 訓練画像の局所輝度・コントラスト統計量と U-Net の初期スコアから、暗部の微弱な正解細胞をスコア再重み付けして上位候補へ引き上げる。

#### ◆フェーズ2: トラック結合 (Tracking)
検出された細胞同士を時間方向に線（エッジ）で繋ぎ、トラックを形成するフェーズ。

##### ① 4系統の独立トラッカー並行生成（マルチウィットネス生成）
- **役割**: 合意形成の証人となる 4つのトラッカー（`v24`, `stage6`, `v39`, `v38`）を並行実行。
- **レシピ**: それぞれ異なる時間窓（3フレーム窓 / 5フレーム窓）や独立検出器（`STRONG_MODELS`）で学習したモデルを使用。

##### ② エッジ尤度推論（`EDGE_MODEL`）
- **アルゴリズム**: **5-fold GBDT（ブースティング）**
- **レシピ**: 前段の U-Net が OOF で出力した候補点ペアから、移動距離、速度ベクトル、オプティカルフロー残差、近傍候補とのマージンを特徴量にしてエッジ接続確率を学習。

##### ③ グループ較正（`rescore_argmax`）
- **アルゴリズム**: **ロジスティック回帰(スタッキング)**
- **レシピ**: 1位候補エッジのロジットと、2位候補との差分（マージン）を入力特徴量とし、上位のロジスティック回帰で勝者総取りの確率を較正(`edge_model.py:L212-L242`)。

##### ④ トラックレット結合 & 削除リスク(`TRACK_MODEL`, `V57_BOOSTERS`)
- **アルゴリズム**: **5-fold LightGBM（バギング）**
- **レシピ**: 葉数違い（leaves 7, 15）のモデル群の予測値を平均化して小トラックを接続。また、グラフ入れ替え時のノード削除リスクを 5つの LightGBM の平均確率(`predict_bagged_risk`)で安全に判定。

##### ⑤ エッジ合意形成（`add_consensus_edges`）
- **アルゴリズム**: **多数決合意（Consensus）**
- **レシピ**: 4つの独立トラッカーが共通して認めた高信頼エッジのみをベースグラフへ注入(`v55_deploy.py:L300-L350`)。

#### ◆フェーズ3: 細胞分裂 (Cell Division)
1つの親細胞が2つの娘細胞に分かれる「分裂（Fork）」を判定。

##### ① 分裂ペア尤度推論（`PAIR_MODEL`）
- **アルゴリズム**: **5-fold GBDT（ブースティング）**
- **レシピ**: 親セルと娘セル2つのトリプル $(parent, child_1, child_2)$ について、娘細胞間の対称性、距離、コサイン類似度から分裂確率を評価(`pairs.py:L24-L34`)。

##### ② 多段スタッキングメタモデル群（`PairStack`, `ALLIMG`, `HARDNEG`, `HIGHRES` 等）
- **アルゴリズム**: **5-fold ロジスティック回帰（スタッキング）**
- **レシピ**:
  - 1階の `PAIR_MODEL` の予測ロジット（`base_pair_logit`）を入力特徴量として渡す。
  - `pair_hardneg_lr`: 誤接続しやすい近傍の紛らわしい細胞（ハードネガティブ）に特化して学習。
  - `pair_highres_lr`: 高解像度生画像から切り出したパッチ統計量を追加して学習。
  - 推論時は scikit-learn を使わず、パラメータ（`coef`, `intercept`, `mean`, `scale`）から NumPy のみで線形計算(`PairStack:L348-L383`)。

##### ③ 分裂多数決投票（`root_consensus2`）
- **アルゴリズム**: **多数決合意（Consensus）**
- **レシピ**: 4つの独立トラッカーが分裂トリプルに1票ずつ投票し、**2つ以上のトラッカーが一致して支持した場合のみ分裂を正式採用** (`v169_division_consensus.py:L21-L60`)。

#### ◆フェーズ4: 後処理 (Post-Processing)
完成したグラフのノイズ除去と座標補正を行います。

##### ① V394 孤立成分の保護・剪定
- **アルゴリズム**: **ブースティング ＋ 合意（LightGBM）**
- **レシピ**: 63,116個の連結成分データを学習。特徴量に **「他の独立トラッカーが支持しているか（Witness一致度）」** を含め、偽の孤立トラックを削除(`v394_component_protector.py`)。

##### ② V399 尾部トリミング
- **アルゴリズム**: **ブースティング（LightGBM）**
- **レシピ**: 1,409,466個のトラック末端パスデータを学習。トラック終端の余分なノイズのヒゲを切り落とす(`v399_tail_protector.py`)。

##### ③ V454 ギャップ補完
- **アルゴリズム**: **スタッキング（ロジスティック回帰）**
- **レシピ**: 前後の移動予測残差と生画像パッチのコントラスト・SNR（計8次元）をロジスティック回帰に入力し、1フレーム途切れたトラックを補間接続(`v454_gap_admission.py`)。

##### ④ V507 座標補正 & 分岐剪定
- **アルゴリズム**: **ゲート付きブースティング（HistGradientBoosting）**
- **レシピ**: 追跡座標と真の座標の微小ズレ $(\Delta z, \Delta y, \Delta x)$ を学習した 3軸の回帰木（150本）と、適用可否を決めるゲート決定木（150本）。scikit-learn で学習した木構造を `.npz` にダンプし、推論時は自作の高速ループで評価(`v507_candidate.py`)。

##### ⑤ V533 視野端境界処理
- **アルゴリズム**: ルールベース(`v533_addback_edge.py`)
- **レシピ**: 有効データ境界から 1.5 px 以内にある境界アーティファクトノードを削除。

---

### まとめ

1. **学習データ**:
   提供された **2つの胚（全4動画）のすべてを学習データに投入し、時間軸で5分割（Temporal 5-fold）** して本番用の 5-fold モデルを作成した。
2. **アンサンブルの階層構造**:
   - **画像検出**: U-Netの **バギング（単純平均＋TTA）**
   - **候補の接続判定**: LightGBM等の **ブースティング**
   - **確率の較正・精緻化**: GBDTの出力を受け取る **スタッキング（ロジスティック回帰）**
   - **構造破壊の防止（エッジ・分裂）**: 4系統の独立トラッカーによる **多数決合意（Consensus）**

## 総括
ホントにたくさんのモデルをアンサンブルしてて、まとめたんだけど理解できてないです。
2位解法はムズい。
