18-⑤Kaggle実践3 Biohub細胞トラッキング：1位解法のまとめ

<img src="image.png" width="800" />

## Abstract
- biohub 1位解法のまとめ(Public 0.976 / Private 0.977)
- 「幾何学的追跡」から「ビジュアルトラッキング(視覚的追跡)」へのパラダイムシフトにより、CV・LB共に圧倒的スコアで優勝した1位チーム(sersasj)の完全解説。

## 概要
先日、Biohub細胞トラッキングコンペが完了しました。
で、惨敗だった僕の結果(1353位/3800人)はさておき、Kaggleの素晴らしいところは、勝者たちの解法が惜しみなく公開されること。
この学びの宝庫を徹底的に分析し、確実に自身のスキルへと昇華させるべく自分の血肉にしてやりますばい。

## biohub 1位の解法まとめ
1位の方だけでなく、上位陣の人たちはモデルを複数駆使してて、また学習にも工夫が随所にみられました。そして公開されるのは学習後モデルそのものでして、学習に使用したコードやデータ群までは公開されないんですね。ただ上位陣はそれぞれ自身の提出コードの解説ページを公開してくれてて、そこで結構情報を得られたりします。

- 解説のURL
https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/1st-place-solution
- ソースコード
https://www.kaggle.com/code/sersasj/biohub-1st-place-inference

---

## プロジェクト構成

1位解法のリポジトリ(`biohub-1st-place-reproduction`)および推論処理で使用されている主要スクリプト・重みファイル群の役割・構成一覧です。

```text
biohub-1st-place-reproduction/
├── biohub-1st-place-inference.py       # Kaggle推論エントリーポイント
├── cell_tracking/                      # メインパッケージ
│   ├── inference.py                    # 推論処理全体の統括(main/cli)
│   ├── common/                         # 共通処理(tracks, I/O, 正規化, 幾何計算)
│   ├── detector/                       # 検出器モジュール
│   │   ├── model.py                    # 3D U-Net + Cellposeヘッド定義
│   │   ├── inference.py                # 検出推論・TTA・ピーク抽出
│   │   └── resnet.py                   # 3D ResNetバックボーン
│   ├── soon/                           # ビジュアルトラッカー(Soon Net)
│   │   ├── model.py                    # SoonNet(CNN Tokenizer + EVA Transformer + FPN)
│   │   ├── crops.py                    # 局所3Dボリューム(3フレーム)切り出し
│   │   ├── scoring.py                  # 状態分類・占有マップ推論
│   │   └── checkpoints.py              # 重みロード
│   └── linker/                         # 学習型リンカー(Learned Linker)
│       ├── ensemble.py                 # EdgeEnsemble(5-fold平均)
│       ├── link.py                     # キャッシュからのエッジ推論・貪欲デコード
│       ├── graph.py                    # TrackGraph構築・後処理
│       ├── postprocess.py              # 線形フィッティング平滑化・短トラック除去
│       └── tiered_gate.py              # 高信頼度ゲーティング
└── weights/                            # 5-fold学習済みモデル重み群
    ├── det_fold0.ckpt ~ det_fold4.ckpt # 3D U-Net検出器重み(5-fold)
    ├── soon_fold0.pt ~ soon_fold4.pt   # Soon Net重み(5-fold)
    └── edge_fold0/ ~ edge_fold4/       # 学習型リンカー重み(edge.pt, config.json)
```

### 主要モデル重みファイル一覧

| モジュール | 重みファイル | パラメータ規模・入力 | 役割・機能 |
| :--- | :--- | :--- | :--- |
| **Detector** | `det_fold0.ckpt` 〜 `det_fold4.ckpt` | 約3.1M params<br>入力: 128×128×64 crop | **3D U-Net検出器(5-fold):** 3Dフロー(dz, dy, dx)と中心確率マップを予測。xy反転4x TTAと組み合わせて各フレーム20パスで高精度検出。 |
| **Soon Net** | `soon_fold0.pt` 〜 `soon_fold4.pt` | 約744K params (Tokenizer:236K, Transformer:508K)<br>入力: 3フレーム局所3D (16×64×64) | **ビジュアルトラッカー(5-fold):** 細胞の「もうすぐ分裂する」外見変化の3クラス分類と、FPNによる移動先・娘細胞の占有マップ(Occupancy map)を直接予測。 |
| **Linker** | `edge_fold0/edge.pt` 〜 `edge_fold4/edge.pt` | 軽量MLP/線形モデル<br>特徴量: ノード42次元, ペア23次元 | **学習型リンカー(5-fold):** 検出点ペアとSoon Net占有マップ・幾何特徴から接続スコア・分裂スコアを算出。貪欲デコードでトラックを構築。 |

---

## 1位解法の全体像：

多くの参加者(2位を含む上位陣も含む)は、検出後の細胞トラッキングにおいて「座標間のユークリッド距離」「カルマンフィルタによる移動速度・方向」「輝度の差」といった**幾何学的特徴量(幾何ヒューリスティクス)**に大きく依存していました。

しかし1位チームは、このアプローチの本質的な弱点を見抜いていました。

> **1位の洞察 (Key Insight):**
> 1. **幾何学的特徴量は細胞そのものを見ていない(blind):** 距離や速度だけでは、細胞が密集する領域での交差や、分裂直後の急激な挙動変化に追従できない。
> 2. **細胞は分裂する直前に見た目が変化する:** 生物学的に、細胞は分裂する前に丸まり、光の反射やテクスチャ、形状が明確に変化する。人間が見れば「もうすぐ分裂する」と容易に分かる。
> 3. **「点と点の距離」ではなく「画像そのものに基づく視覚的追跡(Visual Tracking)」へ:** 候補点同士の距離計算をメインにするのではなく、ニューラルネットワークに「細胞の次のフレームでの存在領域(占有マップ)」を画像から直接予測させる。

この発想の転換により、1位チームは **「3D U-Net検出器」→「Soon Net(視覚的追跡・分裂予測)」→「Learned Linker(学習型グラフ接続)」** という、高度に洗練された視覚的追跡システムを構築しました。

---

## 全体クラス図

1. **検出 (Detection)**
2. **視覚追跡 (Soon Net)**
3. **学習型リンカー (Learned Linker)**
4. **後処理 (Post Processing)**

```mermaid
classDiagram
    direction LR

    %% ==========================================
    %% 1. 検出 (Detection)
    %% ==========================================
    class UnetCellPoseModel["Detection<br/>UnetCellPoseModel"] {
        +UNet3D net
        +int pool
        +dict decode
    }
    class _UnetCellPoseNet["Detection<br/>«abstract» _UnetCellPoseNet"] {
        +Channels ch
        +Conv3d out_flow
        +Conv3d out_prob
    }
    class EnsembleTTANet["Detection<br/>EnsembleTTANet"] {
        +ModuleList nets
        +tuple flips
    }
    class UNet3D["Detection<br/>UNet3D"] {
        +forward(x)
        +forward_with_d1(x)
    }

    _UnetCellPoseNet <|-- UNet3D
    UnetCellPoseModel *-- _UnetCellPoseNet
    EnsembleTTANet *-- UNet3D

    %% ==========================================
    %% 2. 視覚追跡 (Soon Net)
    %% ==========================================
    class _TemporalSoonNet["Soon Net<br/>«abstract» _TemporalSoonNet"] {
        +int n_frames
        +SharedSpatialEncoder encoder
    }
    class SharedSpatialEncoder["Soon Net<br/>SharedSpatialEncoder"] {
        +forward(frames)
    }
    class SiameseSpaceTimeSoonNet["Soon Net<br/>SiameseSpaceTimeSoonNet"] {
        +forward(x, return_aux)
    }
    class EvaTransformerEncoder["Soon Net<br/>EvaTransformerEncoder"] {
        +forward(seq, rope)
    }
    class AuxFPN["Soon Net<br/>AuxFPN"] {
        +forward(skips, top)
    }

    _TemporalSoonNet <|-- SiameseSpaceTimeSoonNet
    _TemporalSoonNet *-- SharedSpatialEncoder
    SiameseSpaceTimeSoonNet *-- EvaTransformerEncoder
    SiameseSpaceTimeSoonNet *-- AuxFPN

    %% ==========================================
    %% 3. 学習型リンカー (Learned Linker)
    %% ==========================================
    class SimpleNodeTransformer["Linker<br/>SimpleNodeTransformer"] {
        +ModuleList blocks
        +encode_frames()
    }
    class CrossAttentionBlock["Linker<br/>CrossAttentionBlock"] {
        +MultiheadAttention cross_attn
        +forward(q, kv)
    }
    class ForkHeadMixin["Linker<br/>«mixin» ForkHeadMixin"] {
        +Sequential fork_mlp
        +forward_fork()
    }
    class UnetCellPoseEdgeTransformer["Linker<br/>UnetCellPoseEdgeTransformer"] {
        +Sequential pair_mlp
        +forward_pairs()
    }
    class EdgeEnsemble["Linker<br/>EdgeEnsemble"] {
        +list models
        +str shortlist
        +forward()
    }

    SimpleNodeTransformer <|-- UnetCellPoseEdgeTransformer
    ForkHeadMixin <|-- UnetCellPoseEdgeTransformer
    SimpleNodeTransformer *-- CrossAttentionBlock
    EdgeEnsemble *-- UnetCellPoseEdgeTransformer

    %% ==========================================
    %% 4. 後処理 (Post Processing)
    %% ==========================================
    class GraphPostProcessor["Post<br/>«module» GraphPostProcessor"] {
        +graph_from_pair_links()
        +tiered_temporal_gate()
        +postprocess_graph()
    }
    class TrackGraph["Post<br/>TrackGraph"] {
        +ndarray node_t, z, y, x
        +ndarray edges
    }

    GraphPostProcessor ..> TrackGraph

    %% ==========================================
    %% ステージ間の主軸フロー (左から右へ整列させる推進力)
    %% ==========================================
    UNet3D ..> SiameseSpaceTimeSoonNet : ① 検出座標 coords
    SiameseSpaceTimeSoonNet ..> UnetCellPoseEdgeTransformer : ② 分裂スコア・占有マップ
    EdgeEnsemble ..> GraphPostProcessor : ③ ペア・分岐候補
```

---

### 4フェーズの役割とデータの流れ

| フェーズ | 主要クラス / 関数 | 入力 | 出力 | 役割・ポイント |
|---|---|---|---|---|
| **1. 検出 (Detection)** | `UNet3D`<br>`EnsembleTTANet` | 生3Dボリューム画像 `(Z, Y, X)` | 細胞中心座標 `coords`, 形状特徴 `feats` | 3D U-NetによるFlow(重心ベクトル)と前景確率の予測。XY-TTAアンサンブルで頑健に検出。 |
| **2. 視覚追跡 (Soon Net)** | `SiameseSpaceTimeSoonNet`<br>`SharedSpatialEncoder`<br>`EvaTransformer` | 検出点周辺の3フレームCrop画像 | 分裂確率 `scores`, 占有マップ `occ, div` | 「どの細胞が直後に2分裂するか」を予測。3D RoPEアテンションとAuxFPNで時空間の幾何変化を捉える。 |
| **3. 学習型リンカー (Learned Linker)** | `UnetCellPoseEdgeTransformer`<br>`EdgeEnsemble`<br>`ForkHeadMixin` | ノード特徴 + Soon Netの占有マップ・分裂スコア | フレーム間ペア結合確率, 1対2フォーク分岐確率 | 前後フレームの細胞同士のクロスアテンション。多重継承された `ForkHeadMixin` で1親→2子の分岐を判定。 |
| **4. 後処理 (Post Processing)** | `GraphPostProcessor`<br>`TrackGraph` | 接続確率, 分岐候補 | 最終的な追跡グラフ `TrackGraph` (`submission.csv`) | `tiered_temporal_gate`(低信頼エッジの階層除去)、貪欲デコード、ギャップ補間(短期間の検出漏れ救済)、分裂不応期制約の適用。 |

この4分割の視点は、**アルゴリズムのパイプラインの段階（前処理・特徴抽出・推定・後処理）とコード上のクラス配置が1対1で完全に一致**するため、解法を説明・理解する上で最も美しい構成です。

---

## 4フェーズの詳細解説

1位解法の一連の処理は、以下の4つの処理段階(フェーズ)で構成されています。

```mermaid
flowchart TD
    subgraph Phase1["フェーズ1: 検出 (Detection)"]
        A[3Dタイムラプス画像] --> B["3D U-Net (3.1M params)<br/>MAE事前学習 + 疑似ラベル"]
        B --> C["5-fold × 4x TTA (xy反転)<br/>20パス平均化"]
        C --> D["細胞中心座標 & 3Dフロー抽出"]
    end

    subgraph Phase2["フェーズ2: 視覚追跡 (Soon Net)"]
        D --> E["各細胞の周囲 16×64×64 クロップ<br/>(t-1, t, t+1 の3フレーム)"]
        E --> F["Small CNN Tokenizer + EVA Transformer"]
        F --> G1["3クラス分類ヘッド<br/>(間期 / 分裂直前 / 分裂直後)"]
        F --> G2["FPNヘッド: 占有マップ<br/>(継続マップ & 分裂マップ)"]
    end

    subgraph Phase3["フェーズ3: 学習型リンカー (Learned Linker)"]
        D --> H["候補生成 (半径30μm, 近傍k=16)"]
        G1 --> I["特徴量結合<br/>ノード42次元 + ペア23次元<br/>(占有マップ適合度・幾何・強度)"]
        G2 --> I
        H --> I
        I --> J["EdgeEnsemble (5-fold MLP)<br/>Fork Configuration Loss"]
        J --> K["貪欲デコード (Greedy Decode)<br/>log p(opt) - log p(no child) > 0"]
    end

    subgraph Phase4["フェーズ4: 後処理 (Post Processing)"]
        K --> L["Tiered Temporal Gating<br/>(高信頼度トラックによる選別)"]
        L --> M["1フレーム ギャップ補完 (Gap Closing)"]
        M --> N["孤立ノード枝刈り & 3フレーム未満削除"]
        N --> O["分裂保護付き 線形フィッティング平滑化"]
        O --> P["最終提出データ (submission.csv)"]
    end
```

### フェーズ0. 概要シーケンス

本体コード (`cell_tracking/inference.py`) では、動画1本に対して **2つの処理関数** を実行。
1. **`build_caches()`**: 各フレームをループしながら、**【フェーズ1: 検出】** と **【フェーズ2: 視覚追跡】** を同時に実行して中間キャッシュを保存。
2. **`link_and_post()`**: 保存された全フレームの中間キャッシュを読み込み、動画全体に対して **【フェーズ3: グラフ接続】** と **【フェーズ4: 仕上げの後処理】** を実行。

```mermaid
sequenceDiagram
    autonumber
    participant Entry as biohub-1st-place-inference.py
    participant Main as cell_tracking/inference.py (main / run_worker)
    participant DetMod as cell_tracking/detector/inference.py
    participant SoonScore as cell_tracking/soon/scoring.py
    participant LinkMovie as cell_tracking/linker/link.py
    participant GraphMod as cell_tracking/linker/graph.py
    participant Tracks as cell_tracking/common/tracks.py

    Entry->>Main: cli() -> main()
    Main->>Main: load_stack(device) [5-foldモデル重み群をVRAMへロード]
    Note over Main: テスト動画群(.zarr)をサイズ順にキューへ登録しGPU並列スレッド起動

    loop 各動画 (movie) の処理 [run_worker]
        Note over Main: ━━━━━━━━ 1. キャッシュ構築処理: build_caches() ━━━━━━━━
        Main->>Main: build_caches(detector, soon, raw, dirs, movie, device)

        loop 各フレーム t = 0 .. n_t-1 (3Dボリューム vol)
            rect rgb(235, 245, 255)
            Note over Main, DetMod: 【フェーズ1: 検出 (Detection)】
            Main->>DetMod: extract_frame_cache(detector, vol) [4x反転TTA (計20パス)]
            DetMod-->>Main: 座標・幾何特徴・輝度キャッシュ保存 (frame_000x, intensity)
            end

            rect rgb(240, 255, 240)
            Note over Main, SoonScore: 【フェーズ2: 視覚追跡 (Soon Net)】
            Main->>SoonScore: score_soon(nvol, coords, prev, nxt) [Soon Net 5-fold]
            SoonScore-->>Main: 状態確率(3ch) & 移動先・分裂先の占有マップ(occ, div)保存
            end
        end

        Note over Main: ━━━━━━━━ 2. 接続・後処理: link_and_post() ━━━━━━━━
        Main->>Main: link_and_post(edge, cfg, dirs, movie, device)

        rect rgb(255, 250, 235)
        Note over Main, LinkMovie: 【フェーズ3: グラフ接続 (Learned Linker)】
        Main->>LinkMovie: link_movie_from_cache(edge, dirs["cache"], ...)
        Note over LinkMovie: 全フレームのキャッシュから近傍ペア探索 + 占有マップ照合<br/>EdgeEnsemble (5-fold) + 貪欲デコードでエッジ確定
        LinkMovie-->>Main: frames, links (接続リスト), link_probs
        end

        rect rgb(255, 240, 245)
        Note over Main, GraphMod: 【フェーズ4: 仕上げの後処理 (Post Processing)】
        Main->>GraphMod: graph_from_pair_links() & postprocess_graph()
        Note over GraphMod: 信頼度ゲート選別 + 1フレーム補間 + 孤立点削除 + 短トラック削除 + 分裂保護平滑化
        GraphMod-->>Main: g (完成したTrackGraph)
        end

        Main->>Tracks: graph_to_rows(胚番号, g)
        Tracks-->>Main: rows (動画ごとの提出行)
    end

    Note over Main: 全動画完了後
    Main->>Tracks: write_submission(all_rows, OUTPUT)
    Tracks-->>Entry: submission.csv 書き出し完了
```

##### 概要シーケンス 登場関数一覧

| No | 関数 / メソッド | 呼び出し元 | 引数 (型・shape・説明) | 戻り値 (型・shape・説明) | 主な処理内容と役割 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `cli()` | `Entry (biohub-1st-place-inference.py)` | なし (CLI引数は `sys.argv` 経由) | `None` | コマンドライン引数(データパス、重みパス、出力先等)をパースし、`main()` を起動。 |
| 2 | `main()` | `cli()` | なし | `None` | 推論処理全体の統括。重みロード、テスト動画キュー管理、GPU並列スレッド管理、最終CSV出力。 |
| 3 | `load_stack()` | `main()` | `device`: `str`<br>(例: `"cuda:0"`, `"cpu"`) | `tuple`: `(detector, soon, edge, cfg)`<br>- `detector`: 5-fold 3D U-Net<br>- `soon`: 5-fold Soon Net<br>- `edge`: 5-fold EdgeEnsemble<br>- `cfg`: `dict` 設定辞書 | 指定デバイス(GPU/CPU)へ5-foldの検出器、Soon Net、学習型リンカーの全重みをロード。 |
| 4 | `run_worker()` | `main()` (Thread) | `k`: `int`<br>(デバイス番号/スレッド番号) | `None` | GPUごとに立ち上がるワーカースレッド関数。キューから動画を順次取得し各処理を実行。 |
| 5 | `build_caches()` | `run_worker()` | - `detector`: 5-fold 3D U-Net<br>- `soon`: 5-fold Soon Net<br>- `raw`: `list[np.ndarray]`, shape: 各 `(Z, Y, X)` (全フレーム画像)<br>- `dirs`: `dict[str, str]` 一時ディレクトリ群<br>- `movie`: `str` 動画名<br>- `device`: `str` デバイス名 | `int`: `n_det`<br>(動画全体の総検出細胞数) | 全フレームをループし、【フェーズ1: 検出】と【フェーズ2: 視覚追跡】を実行してキャッシュ保存。 |
| 6 | `link_and_post()` | `run_worker()` | - `edge`: EdgeEnsemble<br>- `cfg`: `dict` 設定辞書<br>- `dirs`: `dict[str, str]`<br>- `movie`: `str` 動画名<br>- `device`: `str` | `TrackGraph | None`<br>(完成した時空間細胞トラックグラフ) | 全フレームのキャッシュを読み込み、【フェーズ3: 接続】と【フェーズ4: 後処理】を一括実行。 |
| 7 | `graph_to_rows()` | `run_worker()` | - `name`: `str` 動画名<br>- `g`: `TrackGraph` オブジェクト | `list[tuple]`, shape: `(N_points, 6)`<br>(各行: `(name, t, id, z, y, x)`) | 完成した TrackGraph を提出用フォーマットの行データに変換。 |
| 8 | `write_submission()` | `main()` | - `rows`: `list[tuple]`, shape: `(全行数, 6)`<br>- `path`: `str` 出力先ファイルパス | `None` (ディスク書き出し: `submission.csv`) | 全動画の行データを集約し、最終的な `submission.csv` を出力。 |
---



### モデル読み込みシーケンス `main()` → `load_stack()`
`main()` → `load_stack()` 　で、各モデル（Detector / Soon Net / Linker）がロードされるまでの処理フロー。

```mermaid
sequenceDiagram
    autonumber
    actor User as 実行環境 (CLI / main)
    participant Inf as cell_tracking/inference.py
    participant DetInf as cell_tracking/detector/inference.py
    participant DetMod as cell_tracking/detector/model.py
    participant SoonCkpt as cell_tracking/soon/checkpoints.py
    participant SoonMod as cell_tracking/soon/model.py
    participant LinkLink as cell_tracking/linker/link.py
    participant LinkMod as cell_tracking/linker/model.py
    participant LinkEns as cell_tracking/linker/ensemble.py

    User ->> Inf: main() 起動
    Note over Inf: GPU/CPUデバイス判定<br/>devices = ['cuda:0', ...]

    loop 各利用可能デバイス (device)
        Inf ->> Inf: load_stack(device)
        activate Inf

        %% ========================================================
        %% 1. Detector ロード (5-fold)
        %% ========================================================
        rect rgb(240, 248, 255)
            Note over Inf, DetMod: 1. 検出モデルのロード (Detector Ensemble)
            Inf ->> DetInf: load_unetcellpose_ensemble(DETECTORS, device, tta="xy", expected_pool=2)
            activate DetInf
            loop 5-fold (det_fold0 ~ 4.ckpt)
                DetInf ->> DetInf: load_unetcellpose(ckpt_path)
                DetInf ->> DetMod: build_backbone("plain", in_ch, pool_schedule, ...)
                activate DetMod
                DetMod -->> DetInf: UNet3D インスタンス
                deactivate DetMod
                DetInf ->> DetInf: state_dict 読込 & UnetCellPoseModel 作成
            end
            DetInf ->> DetInf: EnsembleTTANet(nets, ch, flips) 生成
            DetInf -->> Inf: detector (EnsembleTTANet)
            deactivate DetInf
        end

        %% ========================================================
        %% 2. Soon Net ロード (5-fold)
        %% ========================================================
        rect rgb(255, 250, 240)
            Note over Inf, SoonMod: 2. 視覚追跡モデルのロード (Soon Net)
            loop 5-fold (soon_fold0 ~ 4.pt)
                Inf ->> SoonCkpt: load_soon_crop(weight_path, device)
                activate SoonCkpt
                Note over SoonCkpt: チェックポイントのハイパーパラメータ検証
                SoonCkpt ->> SoonMod: SiameseSpaceTimeSoonNet(num_registers, dropout=0.0)
                activate SoonMod
                SoonMod -->> SoonCkpt: net インスタンス
                deactivate SoonMod
                SoonCkpt ->> SoonCkpt: net.load_state_dict() & eval()
                SoonCkpt -->> Inf: (net, meta)
                deactivate SoonCkpt
                Inf ->> SoonCkpt: deploy_context(meta)
                SoonCkpt -->> Inf: context (フレーム時間幅: 2)
                Inf ->> Inf: soon リストに {net, crop_size, xy_pool, context} を格納
            end
            Note over Inf: 全foldの crop_size と xy_pool の一致を検証
        end

        %% ========================================================
        %% 3. Linker ロード (5-fold) & Ensemble
        %% ========================================================
        rect rgb(240, 255, 240)
            Note over Inf, LinkEns: 3. 学習型リンカーのロード (Linker Ensemble)
            loop 5-fold (edge_fold0 ~ 4/edge.pt)
                Inf ->> LinkLink: load_edge_checkpoint(weight_path, device)
                activate LinkLink
                Note over LinkLink: リンカー構成・入力次元の検証
                LinkLink ->> LinkMod: UnetCellPoseEdgeTransformer(feat_dim, hidden_dim, ...)
                activate LinkMod
                LinkMod -->> LinkLink: model インスタンス
                deactivate LinkMod
                LinkLink ->> LinkLink: model.load_state_dict() & eval()
                LinkLink -->> Inf: (model, config)
                deactivate LinkLink
            end
            Inf ->> LinkEns: EdgeEnsemble(models, shortlist="consensus")
            activate LinkEns
            LinkEns -->> Inf: edge (EdgeEnsemble)
            deactivate LinkEns
            Note over Inf: 5-fold間の config (feat_dim, occupancy等) の整合性検証
        end

        Inf -->> Inf: return (detector, soon, edge, cfg)
        deactivate Inf
    end

    Note over Inf: stacks (各デバイス分のモデル一式) を保持して推論ループへ
```

---

### シーケンスの要点

1. **`inference.py` がオーケストレーター**
   `main()` から GPU/CPU ごとに `load_stack(device)` を呼び出し、各ステージのモジュールから 5-fold 分の重みを並行して読み込む。

2. **1. Detector (`detector/inference.py`, `detector/model.py`)**
   - 5つのチェックポイント (`det_fold0.ckpt`〜`4`) をループして `UNet3D` を構築・初期化。
   - 最後に `EnsembleTTANet` クラスでラップし、XY方向のテスト時データ拡張(TTA: Flip-XY)と5-fold平均を1つのモジュールとして扱えるように集約。

3. **2. Soon Net (`soon/checkpoints.py`, `soon/model.py`)**
   - `load_soon_crop()` が厳格なメタデータチェック(CNNバックボーン、RoPEベース、SwiGLU設定等)を行い、`SiameseSpaceTimeSoonNet` に重みを流し込み。
   - `deploy_context()` で前後の参照フレーム数(`context=2`)を取り出し、推論時用のクロップ設定辞書を組み立て。

4. **3. Linker (`linker/link.py`, `linker/model.py`, `linker/ensemble.py`)**
   - `load_edge_checkpoint()` が `UnetCellPoseEdgeTransformer` を初期化して各foldをロード。
   - 読み込まれた5つのモデルを `EdgeEnsemble` に渡し、「consensus(合意)モード」によるエッジ候補絞り込みと平均出力を行えるように束ねる。

---

### フェーズ1. 検出 (Detection)

#### 1. アーキテクチャと特徴
- **vanilla 3D U-Net + Cellposeスタイルヘッド:** 128×128×64のクロップサイズで学習。
- **異方性プーリング (Anisotropic Pooling):** 顕微鏡画像のZ軸方向の解像度がXY平面と異なるため、ダウンサンプリングを `(1,2,2), (1,2,2), (2,2,2)` と異方的に設計。
- **4チャンネル出力:** 各ボクセルに対して「中心確率(1ch)」と「3D方向オフセット・フロー(dz, dy, dxの3ch)」を出力。
- **Deep Supervision:** 1/2スケール、1/4スケールの中間層でもフローと確率マップの損失を計算し、勾配消失を防ぎながら多重解像度の特徴を学習。

#### 2. スパースな正解データへの対策 (MAE事前学習 & 疑似ラベリング)
- **Masked Autoencoder (MAE) 事前学習:** ドイツがん研究センター(MIC-DKFZ)の `nnssl` フレームワークを活用。大量の未ラベル3D顕微鏡画像から自己教師あり学習で堅牢な3D視覚表現を獲得。
- **2段階ファインチューニング & 疑似ラベル (Pseudo Labeling):**
  - Round 1学習後、5-fold平均 + 4x flip TTAを用いて全195本の動画と150個の外部クロップに対して高信頼度な検出結果を擬似ラベル化。
  - 公開のtwoPassトラッカーで長さ3以上のトラックのみを抽出・精緻化し、スパースだった正解データを大幅に増強。
- **推論時TTA:** 本番推論では、5フォールド × 4倍反転TTA (xy反転) = **1フレームあたり計20回のフォワードパス**を実行して平均化。

#### フェーズ1 ``検出(Detection)``シーケンス

```mermaid
sequenceDiagram
    autonumber
    participant Main as cell_tracking/inference.py<br/>(build_caches)
    participant Crops as cell_tracking/soon/<br/>crops.py
    participant LinkCache as cell_tracking/linker/<br/>cache.py
    participant DetMod as cell_tracking/detector/<br/>inference.py
    participant UNet as cell_tracking/detector/<br/>model.py
    participant LinkFeat as cell_tracking/linker/<br/>features.py
    participant IntSidecar as cell_tracking/linker/<br/>intensity.py

    Main->>Crops: native_crop_size(crop_size, xy_pool=1) 16,64,64

    Note over Main: build_caches() 内のフレームループ
    loop 各フレーム t = 0 .. n_t-1 (3Dボリューム vol)
        Main->>LinkCache: extract_frame_cache(detector, vol, device, frame_at)
        LinkCache->>DetMod: decode_instances(fsm:UnetCellPoseModel, vol[64,256,256], frame_at, ...)
        loop 4方向反転 TTA (xy反転)
            DetMod->>UNet: fsm.net(t) → forward(t) [5-foldアンサンブル]
            UNet-->>DetMod: 64x128x128 3Dフローベクトル群 (dz, dy, dx) + 中心確率 (prob)
        end
        Note over DetMod: 20パス平均化 & flow_follow_decode()<br/>(フロー追跡により細胞IDマップ inst を生成)
        DetMod-->>LinkCache: inst:各細胞のID番号index<br/>pred:64x256x256 3Dフローベクトル群(dz, dy, dx) ＋ 確率
        LinkCache->>LinkFeat: instance_node_features(pred, inst, ...) ... 細胞中心計算処理
        LinkFeat-->>LinkCache: coords(細胞中心座標), feats(幾何特徴[42次元]), vols(体積)
        Note over LinkFeat,LinkCache: coordsは細胞中心座標(例:492個), featsは幾何特徴[42次元](例:492個), volsは体積でかつボクセル数
        LinkCache-->>Main: coords※, feats, vols
        Main->>LinkCache: write_frame_cache(dirs["cache"], coords, feats, vols, t)<br/>npzファイル保存 ... coords(中心座標),feats(幾何特徴量42次元),volumes(体積),フレーム番号
        Note over LinkCache: .../cache/<胚番号>/frame_0000.npz<br/>.../cache/<胚番号>/frame_0001.npz<br/>...<br/>.../cache/<胚番号>/frame_0099.npz
        Main->>IntSidecar: write_intensity_sidecar(dirs["intensity"], vol, coords, vols, t)<br/>intensityは輝度,sidecarは補助データの意<br/>npzファイル保存...輝度の(平均,最大値,標準偏差,合計),(時間経過退色や露光ブレの)正規化輝度
        Note over IntSidecar: .../intensity/<胚番号>/frame_0000_intensity.npz<br/>.../intensity/<胚番号>/frame_0001_intensity.npz<br/>...<br/>.../intensity/<胚番号>/frame_0099_intensity.npz
    end
```

##### フェーズ1 登場関数一覧

| No | 関数 / メソッド | 呼び出し元 | 引数 (型・shape・説明) | 戻り値 (型・shape・説明) | 主な処理内容と役割 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `build_caches()` | `Main`<br/>(`run_worker`) | - `detector`: 5-fold 3D U-Net<br>- `soon`: 5-fold Soon Net<br>- `raw`: `list[np.ndarray]`, shape: 各 `(Z, Y, X)` (全フレーム画像)<br>- `dirs`: `dict[str, str]` (一時cache保存先dict)<br>- `movie`: `str` (ムービー名)<br>- `device`: `str` (デバイス名) | `int`: `n_det`<br>(全フレームの総検出細胞数) | 各フレームの3D画像ループを回し、フェーズ1の細胞検出とフェーズ2のSoon/占有マップのキャッシュ保存を統括。 |
| 2 | `native_crop_size()` | `build_caches()`<br/>(`Main`) | - `crop_size`: `tuple[int, int, int]`(クロップサイズ(16, 64, 64))<br>- `xy_pool`: `int = 1` (XYプーリング係数) | `tuple[int, int, int]`<br>(クロップサイズ `(16, 64, 64)`) | プーリング係数を考慮し、元画像(ボクセル)解像度における3Dクロップ切り出しサイズを算出。<br/>※voxel単位だと $(16, 64, 64)$ 、現実空間だと「$26 \mu\text{m} × 26 \mu\text{m} × 26 \mu\text{m}$ の真四角な立方体」|
| 3 | `extract_frame_cache()` | `build_caches()`<br/>(`Main`) | - `detector`: `UnetCellPoseModel` (3D U-Netモデル)<br>- `vol`: `np.ndarray`, shape: `(Z, Y, X)` (3Dボリューム画像)<br>- `device`: `str` (実行デバイス)<br>- `frame_at`: `callable` (前後フレーム取得関数)<br>- `include_geom`: `bool = True`, `include_time`: `bool = True`, `include_appearance`: `bool = False` | `tuple`: `(coords, feats, vols)`<br>- `coords`: `np.ndarray`, shape: `(N, 3)` (細胞中心座標、float64)<br>- `feats`: `np.ndarray`, shape: `(N, 42)` (幾何・位置特徴量、float32)<br>- `vols`: `np.ndarray`, shape: `(N,)` (細胞体積、float64) | 1フレームの3D画像から細胞インスタンスをデコードし、各細胞の中心座標・幾何特徴量・体積を抽出して返す。 |
| 4 | `decode_instances()` | `extract_frame_cache()`<br/>(`LinkCache`) | - `fsm`: `UnetCellPoseModel`<br>- `vol`: `np.ndarray`, shape: `(Z, Y, X)`<br>- `device`: `str`, `frame_at`: `callable`<br>- `_return_pred`: `bool = True` | `tuple`: `(inst, pred)`<br>- `inst`: `np.ndarray`, shape: `(Z, Y, X)` (int32、各細胞のインスタンスIDマップ)<br>- `pred`: `np.ndarray`, shape: `(4, Z, Y, X)` (3Dフローdz, dy, dx + 中心確率prob) | 3D U-Netによる推論(4方向反転TTA、計20パス)を実行・平均化し、`flow_follow_decode()` で細胞IDマップを生成。 |
| 5 | `UNet3D.forward()` | `decode_instances()`<br/>(`DetMod`) | `crops`: `torch.Tensor`, shape: `(B, 1, 64, 128, 128)` (3Dクロップ画像テンソル) | `torch.Tensor`, shape: `(B, 4, 64, 128, 128)`<br>(ch0..2: 3Dフローdz, dy, dx、ch3: 中心確率prob) | 3D U-Netモデル本体。立体画像クロップから細胞中心へ向かう3Dフローベクトルと中心確率マップを出力。 |
| 6 | `flow_follow_decode()` | `decode_instances()`<br/>(`DetMod`) | - `flow`: `np.ndarray`, shape: `(3, Z, Y, X)` (3Dフロー)<br>- `prob_logit`: `np.ndarray`, shape: `(Z, Y, X)` (中心確率)<br>- `fg_thresh`: `float`, `step_um`: `float`, `min_size`: `int` | `np.ndarray`, shape: `(Z, Y, X)`<br>(int32、各細胞のインスタンスIDラベルマップ `inst`) | 予測された3Dフローベクトルに沿ってオイラー積分で追跡し、同じ中心点(シンク)へ収束したボクセル群を1つの細胞としてラベリング。 |
| 7 | `instance_node_features()` | `extract_frame_cache()`<br/>(`LinkCache`) | - `instances`: `np.ndarray`, shape: `(Z, Y, X)` (細胞IDマップ)<br>- `d1`: `np.ndarray | None` (アピアランス特徴マップ)<br>- `prob_logit`: `np.ndarray`, shape: `(Z, Y, X)` (中心確率)<br>- `pool`: `int`, `include_geom`: `bool`, `include_time`: `bool` | `tuple`: `(ids, coords_native, feats, volumes)`<br>- `coords_native`: `(N, 3)` (細胞中心座標)<br>- `feats`: `(N, 42)` (幾何特徴量・位置エンコーディング)<br>- `volumes`: `(N,)` (細胞体積・ボクセル数) | セグメンテーションされた細胞インスタンス領域から、中心座標(重心)、幾何特徴(共分散・主軸比・中心確率)、ボクセル体積を高速集計。 |
| 8 | `write_frame_cache()` | `build_caches()`<br/>(`Main`) | - `path`: `str` 出力先パス (`.../frame_000x.npz`)<br>- `coords`: `np.ndarray`, shape: `(N, 3)`<br>- `feats`: `np.ndarray`, shape: `(N, 42)`<br>- `vols`: `np.ndarray`, shape: `(N,)`<br>- `frame`: `int` フレーム番号 $t$ | `None`<br>(ディスク書き出し: `frame_000x.npz`) | 検出された細胞の中心座標、幾何特徴量、体積データをzip圧縮 (`np.savez_compressed`) してディスクへキャッシュ保存。 |
| 9 | `write_intensity_sidecar()` | `build_caches()`<br/>(`Main`) | - `path`: `str` 出力先パス (`.../frame_000x_intensity.npz`)<br>- `vol`: `np.ndarray`, shape: `(Z, Y, X)` (3D生画像)<br>- `coords`: `np.ndarray`, shape: `(N, 3)` (中心座標)<br>- `volumes`: `np.ndarray`, shape: `(N,)`<br>- `frame`: `int` フレーム番号 $t$ | `Path`<br>(ディスク書き出し: `frame_000x_intensity.npz`) | 3D生画像から各細胞の局所蛍光輝度(平均、最大値、標準偏差、合計)および正規化輝度特徴量を抽出し、補助データとして保存。 |
---

### フェーズ2: 分裂直前 / ビジュアルトラッキング (Soon Net)

1位解法の特徴、独自設計モデル **`Soon Net`** 。

#### 1. モチベーション
親細胞と2つの娘細胞の対応付けに誤判定が頻発してた。
その対応策として、「目視すれば分裂の全長はすぐに分かる」という気づきを起点として、細胞が分裂直前に示す「見た目の予兆(球状化や輝度・コントラストの変化)」を捉えるために追加したフェーズ。

#### 2. アーキテクチャ
- **入力:** 時点 $t-1, t, t+1$ における対象細胞周辺の局所3Dボリューム (サイズ: 16×64×64)。
- **小型CNNトークナイザー (Small CNN Tokenizer, 236K params):** 局所ボリュームを過学習なくコンパクトな特徴トークン列へと変換。
- **Transformerバックボーン (EVA Blocks, 508K params):** 時間軸方向 ($t-1 \leftrightarrow t \leftrightarrow t+1$) のコンテキストを統合。
- **2系統の出力ヘッド:**
  1. **状態分類ヘッド (Classification Head):**
     - 「間期 (interphase)」「もうすぐ分裂 (soon-to-divide)」「分裂直後 (just divided)」の3クラス確率を出力。
  2. **FPNヘッド (Occupancy Map Head):**
     - $t-1, t, t+1$ の各フレームにおける「継続マップ(continuation map)」と「分裂マップ(division map)」の2種類の3D占有マップを予測。
     - **継続マップ:** 通常細胞について「次のフレームで細胞がどの座標に移動するか」を球体ガウス分布で点灯。
     - **分裂マップ:** 母細胞について「次のフレームで2つの娘細胞がそれぞれどこに出現するか」、娘細胞について「前フレームで母細胞がどこにいたか」を点灯。

```text
【Soon Netの予測する占有マップの挙動】
通常継続時:
  フレーム t-1 [  ●  ] ──→ フレーム t [   ●   ] ──→ フレーム t+1 [    ●   ] (移動先を点灯)

細胞分裂時:
  フレーム t (母細胞) [   ● (soon)   ] ──→ フレーム t+1 (分裂マップ) [  ● (娘A)   ● (娘B)  ] (両娘の位置を点灯)
```

この占有マップにより、「リンカーが次にどこへ線を結べばよいか」の答えそのものを画像から直接得られるようにした。

#### フェーズ2 ``分裂直前(Soon Net)``シーケンス

```mermaid
sequenceDiagram
    autonumber
    participant Main as cell_tracking/inference.py
    participant SoonScore as cell_tracking/soon/scoring.py
    participant SoonModel as cell_tracking/soon/model.py
    participant SoonCache as cell_tracking/linker/soon_cache.py
    participant OccMod as cell_tracking/linker/occupancy.py

    Note over Main: build_caches() 内のフレームループ (フェーズ1の直後)
    loop 各フレーム t = 0 .. n_t-1
        Note over Main: raw:[0,0,0]～[63,255,255]輝度スケール
        Main->>Main: 生データ(現・前・次フレーム(nvol, prev, nxt))=NormWindow::window(t)
        rect rgb(255, 250, 240)
            Main->>Main: score_soon(soon:soonモデル, nvol, coords:細胞中心座標s, prev, nxt, device)
            Main->>SoonScore: score_points_per_net(nets=soon::net(s), vol=nvol, coords, crops={16×64×64})
            Note over SoonScore: Soon Net(5-foldアンサンブル)実行<br/>分裂確率と前後フレームでの移動先予測マップを算出
            SoonScore->>SoonScore: _crop_tensor(vol=row, zyx=coords, crop_size={16×64×64}, prev=prev, nxt=nxt, device=device)
            Note over SoonScore: _crop_tensor()※下記参照
            SoonScore->>SoonModel: _forward(net,xb,...)→net(xb, return_aux=True)→<br/>　　　　　　   　　　　SiameseSpaceTimeSoonNet::forward(crops)[5-fold Soon Netアンサンブル]
            Note over SoonModel: CNN Tokenizer※ ... self.encoder = SharedSpatialEncoder(...)<br/>Transformer※ ... self.transformer = EvaTransformerEncoder(...)<br/>FPNヘッド推論※ ... self.aux_fpn = AuxFPN(...)
            SoonModel-->>SoonScore: 3クラス状態確率, 継続占有マップ(occ), 分裂占有マップ(div)
            SoonScore-->>Main: scores(5-fold平均), occ, div ... score_soon()の結果
        end
        Main->>SoonCache: write_soon_frame(dirs["soon"], scores)
        Note over SoonCache: .../soon/<胚番号>/frame_0000.npy<br/>.../soon/<胚番号>/frame_0001.npy<br/>...<br/>.../soon/<胚番号>/frame_0099.npy
        opt t >= 1 (1フレーム遅延 flush)
            Main->>Main: flush(t - 1, coords)
            Main->>OccMod: write_occupancy(dirs["occ"], occ) [継続マップ]
            Note over OccMod: .../occ/<胚番号>/frame_0000_occupancy.npz<br/>.../occ/<胚番号>/frame_0001_occupancy.npz<br/>...<br/>.../occ/<胚番号>/frame_0099_occupancy.npz
            Main->>OccMod: write_occupancy(dirs["occ_div"], div) [分裂マップ]
            Note over OccMod: .../occ_div/<胚番号>/frame_0000_occupancy.npz<br/>.../occ_div/<胚番号>/frame_0001_occupancy.npz<br/>...<br/>.../occ_div/<胚番号>/frame_0099_occupancy.npz
            Note over Main,OccMod: ↑ 3Dマップテンソル(occ, div)[t-1]をメモリ解放
        end
    end
```

##### フェーズ2 関数一覧

| No | 関数 / メソッド | 呼び出し元 | 引数 (型・shape・説明) | 戻り値 (型・shape・説明) | 主な処理内容と役割 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `NormWindow.window()` | `Main` | - `t`: `int` フレーム番号<br>- `context`: `int` 前後参照数(通常 1 または 2) | - `tuple`: `(vol, prev, nxt)`<br/>- `vol`: `np.ndarray`, shape: `(Z, Y, X)` <br/>- `prev`: `np.ndarray` | - `None`, shape: `(Z, Y, X)`<br>- `nxt`: `np.ndarray | None`, shape: `(Z, Y, X)` | 前後フレームをキャッシュしながら、現フレーム `t` とその前後フレームの正規化3D画像を取得。 |
| 2 | `score_soon()` | `Main` | - `soon`: `list[dict]` (5-fold Soon Netモデルリスト)<br>- `nvol`: `np.ndarray`, shape: `(Z, Y, X)` (現フレーム画像)<br>- `coords`: `np.ndarray`, shape: `(N, 3)` (細胞中心座標)<br>- `prev`, `nxt`: `np.ndarray | - `None`, shape: `(Z, Y, X)` (前後フレーム画像)<br>- `device`: `str` | - `tuple`: `(mean, occ, div)`<br>- `mean`: `np.ndarray`, shape: `(N, 3)` (3クラス状態確率、5-fold平均)<br>- `occ`: `np.ndarray`, shape: `(N, 3, 16, 64, 64)` (継続占有マップ)<br>- `div`: `np.ndarray`, shape: `(N, 3, 16, 64, 64)` (分裂占有マップ) | 検出座標リストに対して 5-fold Soon Net の推論を一括呼び出し、アンサンブル平均確率スコアおよび時空間占有マップ(occ, div)を算出。 |
| 3 | `score_points_per_net()` | `score_soon()` | - `nets`: `list[SoonNet]` (アンサンブル用Soon Netリスト)<br>- `vol`: `np.ndarray`, shape: `(Z, Y, X)`<br>- `coords`: `np.ndarray`, shape: `(N, 3)`<br>- `crop_size`: `tuple[int, int, int]` = `(16, 64, 64)`<br>- `prev`, `nxt`: `np.ndarray | - `None`, shape: `(Z, Y, X)`<br>- `device`: `str` | - `tuple`: `(scores, occ, div)`<br>- `scores`: `list[np.ndarray]`, 各shape: `(N, 3)`<br>- `occ`: `np.ndarray`, shape: `(N, 3, 16, 64, 64)`<br>- `div`: `np.ndarray`, shape: `(N, 3, 16, 64, 64)` | `_crop_tensor()` で切り出した局所パッチをバッチ化して `_forward()` で各Foldモデルへ入力し、3クラス分類スコアと占有マップ(occ, div)を収集・集約。 |
| 4 | `_crop_tensor()` | `score_points_per_net()` | - `vol`: `np.ndarray`, shape: `(Z, Y, X)` (現フレーム画像)<br>- `zyx`: `np.ndarray`, shape: `(N, 3)` (細胞中心座標)<br>- `crop_size`: `tuple[int, int, int]` = `(16, 64, 64)`<br>- `prev`, `nxt`: `np.ndarray | - `None`, shape: `(Z, Y, X)`<br>- `device`: `str | `torch.device` | `crops`: `torch.Tensor`, shape: `(N, 3, 4, 16, 64, 64)`<br>(N: 細胞数, 3フレーム: prev/curr/nxt, 4ch: 輝度+ガウシアン核3種) | 生画像から各細胞座標を中心とした 16×64×64 の局所3Dボリュームを切り出し、前後フレームおよびガウシアン核と結合してモデル入力テンソルを構築。 |
| 5 | `_forward()` / `SiameseSpaceTimeSoonNet.forward()` | `score_points_per_net()` | - `net`: `SoonNet` (SiameseSpaceTimeSoonNet)<br>- `xb`: `torch.Tensor`, shape: `(B, 3, 4, 16, 64, 64)`<br>- `return_aux`: `bool` = True | - `tuple`: `(logits, aux)`<br>- `logits`: `torch.Tensor`, shape: `(B, 3)` (3クラス状態ロジット)<br>- `aux`: `torch.Tensor`, shape: `(B, 6, 16, 64, 64)` (継続3ch + 分裂3chの時空間占有マップ予測) | CNN Tokenizer (SharedSpatialEncoder) + 時空間Transformer (EvaTransformerEncoder) + FPNヘッド (AuxFPN) を実行、3クラス分類ロジットと継続・分裂の時空間占有マップを出力。 |
| 6 | `write_soon_frame()` | `Main` | - `dir_path`: `str | - `Path` (`dirs["soon"]`)<br>- `frame_idx`: `int` フレーム番号 `t`<br>- `scores`: `np.ndarray`, shape: `(N, 3)` (float32, 5-fold平均) | `None` (ディスク出力: `.../soon/<動画名>/frame_000x.npy`) | 各細胞の「分裂直前(soon)」等の3クラス状態確率スコアをフレーム単位の `.npy` 形式でディスクにキャッシュ保存。 |
| 7 | `flush()` | `Main` | - `t`: `int` 対象フレーム番号 (`t - 1`)<br>- `coords`: `np.ndarray`, shape: `(N, 3)` (次フレームの細胞中心座標) | `None` | 1フレーム遅延させて前フレームの移動・分裂の対応付けを確定し、`write_occupancy()` を呼び出して占有マップを出力後、不要となった3Dマップテンソル `(occ, div)[t-1]` のメモリを解放。 |
| 8 | `write_occupancy()` | `flush()` | - `dir_path`: `str | - `Path` (`dirs["occ"]` または `dirs["occ_div"]`)<br>- `movie`: `str` 動画名<br>- `t`: `int` フレーム番号 (`t - 1`)<br>- `crop_size`: `tuple[int, int, int]` = `(16, 64, 64)`<br>- `coords`: `np.ndarray`, shape: `(N, 3)`<br>- `maps`: `np.ndarray`, shape: `(N, 3, 16, 64, 64)` (occ または div)<br>- `dest_next`, `dest_prev`: `np.ndarray`, shape: `(N, 3)` | `None` (ディスク出力: `.../occ/<動画名>/frame_000x_occupancy.npz` 等) | 継続マップ (occ) および分裂マップ (div) の予測値を、細胞リンカーが高速読み込みできる `.npz` 形式のサイドカーファイルとしてディスクに保存。 |

##### ◆_crop_tensor()でやっていること。
「生画像 nvol（64×256×256）から、検出された細胞の座標 coords を中心に、周辺 16×64×64 を切り取る」処理を実行している。
イメージは下記。

```
1. 生データ raw (画面全体: 64×256×256)
   ┌──────────────────────────────┐
   │                              │
   │          (z, y, x)           │
   │             ● ←ここ！        │
   │                              │
   └──────────────────────────────┘
        │
        ├─→ 【フェーズ1: Detector】
        │   生画像を見て、細胞の中心座標 (z, y, x) を検出する。
        │
        ▼
2. 【フェーズ2: Soon Net】への入力
   Detectorが見つけてくれた座標 (z, y, x) を目印にして、
   生画像 raw からその周辺「16×64×64」だけをサイコロ状にくり抜く！
   
   ┌──────┐
   │  ●   │ ← くり抜いた生画像パッチ (サイズ: 16×64×64)
   └──────┘
        │
        ▼
   Soon Net (CNN Tokenizer + Transformer) に入力！
```
![](copilot__crop_tensor()でやっていること.png)

##### ◆アテンション(Attention / 注意機構) とは？
※Transformerの説明で出現する単語。
「自分以外のどのパーツに注目（注意）すべきか」を計算する重み付けの仕組みのこと。

* 文章系の例:
  `"The animal didn't cross the street because it was too tired."`
  AIが `"it"` という単語を読むとき、「`"it"` は `"street"` のことか？ それとも `"animal"` のことか？」を判断するために、文章中の全単語との関連度を計算します。当然 `"animal"` に強い**アテンション（注目）**が向く。

* 細胞動画系の例:
  時刻 $t$（現在）にいる細胞のトークンが、時刻 $t+1$（未来）の全トークンを見渡して、
  「次フレームのどの場所に、自分と同じ細胞（または分裂した娘細胞）がいるか？」を探すために、**次フレームの該当箇所へ強いアテンション(注意の矢印)を向ける。**

```mermaid
graph LR
    subgraph 前フレーム t-1
        A1["左上に細胞A"]
    end
    subgraph 現在フレーム t
        B1["中央に細胞A"]
    end
    subgraph 次フレーム t+1
        C1["右下に細胞A"]
        C2["遠くの細胞B"]
    end

    B1 -.->|"強いアテンション! (関連大)"| A1
    B1 ==>|"強いアテンション! (ここに移動した)"| C1
    B1 -.->|"アテンションほぼゼロ (無関係)"| C2
```

##### ◆CNN Tokenizer とは
ここ(1位解法)だけの方言。「CNNを使ってTransformer向けの入力を整形する方法全般」を指している。

###### 1. 実装上のクラス名は `SharedSpatialEncoder`。

* **定義場所:** [cell_tracking/soon/encoder.py 5行目]
* ソースコードの冒頭（docstring）に、役割の記載あり：
  ```python
  class SharedSpatialEncoder(nn.Module):
      """3D CNN extracting localized spatial tokens."""
      # （和訳: 局所的な空間トークンを抽出する3D CNN）
  ```

###### 2. 実体はモデルファイルの一部。

独立した別ファイルではなく、**Soon Net のモデル重みファイル（`soon_fold0.pt` 〜 `soon_fold4.pt`）の中に丸ごと一体として格納** されている。
モデル全体(`SiameseSpaceTimeSoonNet`)の部品として、以下のように組み込まれる：
```python
# cell_tracking/soon/model.py 34行目
self.encoder = SharedSpatialEncoder(...)  # ← これが「CNN Tokenizer」の実体
self.transformer = EvaTransformerEncoder(...) # ← これが「Transformer」の実体
```

###### 3. なぜ「CNN Tokenizer(トークナイザー)」と呼ぶのか？
Vision Transformer(画像向けTransformer)の世界では、**「生画像をTransformerが読める形式（トークンの列）に変換・小分けする処理」のことを「トークン化（Tokenize）」** と呼ぶ。
この処理を担当しているのが、一般的なViT(直線の線形射影)ではなく **「小型の3D CNN(畳み込みニューラルネット)」** であるため、作者や論文の解説で **「CNN Tokenizer」**（または **CNN Encoder**）という通称で呼んでいる。

###### まとめ

| 項目 | 実態 |
| :--- | :--- |
| **CNN Tokenizer とは** | 画像をトークン（$4 \times 8 \times 8$）に変換する**部品の通称・役割名** |
| **実装クラス名** | **`SharedSpatialEncoder`**（`cell_tracking/soon/encoder.py`） |
| **モデルファイル** | **`soon_fold0.pt` などの中**に、Transformerと一緒に1つのモデルとして保存されている |

<br/>

##### ◆Transformer(トランスフォーマー) とは
ChatGPTなどの大規模言語モデルのベースにもなっている、現代AIの標準的なニューラルネットワーク構造のこと。一言でいうと、**「並べられたデータ（トークン）同士の『つながり・関係性』を計算する頭脳」** です。

###### 1. CNN と Transformer の違い

| 項目 | CNN(畳み込み) | Transformer |
| :--- | :--- | :--- |
| **見え方** | 虫眼鏡(局所的) | ドローン(大域的・全体) |
| **得意なこと** | すぐ隣のピクセルを見て、輪郭やテクスチャを抽出する | 遠く離れた場所や、**過去・未来のデータ同士の関連性** を直接見通す |
| **仕組み** | 決まった枠（$3 \times 3$ など）をスライドさせる | **アテンション（注意機構）** で「どれとどれが関係あるか？」を直接計算する |

###### 2. このコード(Soon Net)で Transformer がやっていること

cell_tracking/soon/model.py 118行目 の `self.transformer` が実体：

```python
self.transformer = EvaTransformerEncoder(...)
```

1. **入力されるもの:**
   CNN Tokenizer が作った、過去 $(t-1)$・現在 $(t)$・未来 $(t+1)$ の **3フレーム分のトークン(計768個のブロック)** 。
2. **Transformer の役割:**
   全トークン同士を総当たりでアテンション(見比べ)させます：
   * 「現在のフレームにいるこの細胞は、**過去 $(t-1)$ のどのブロックから来たのか？**」
   * 「**未来 $(t+1)$ のどのブロックへ動いていくのか？**」
   * 「形がくびれてきているから、**もうすぐ分裂する関係性にあるか？**」
3. **出力:**
   時間軸と空間軸のつながり（ストーリー）を理解した、より高度な特徴量を出力する。

###### まとめ
* **CNN Tokenizer:** 画像をサイコロ状の小ブロック（トークン）に切り分ける「目（下ごしらえ）」
* **Transformer:** そのブロックたちを時間（過去・現在・未来）をまたいで見比べ、「細胞がどう動いて、分裂するか？」という文脈を読み解く「頭脳（思考部分）」

<br/>

##### ◆FPNヘッド推論とは

1. ヘッド(Head / 出力部)とは？
   AIモデルの「最終的な出口（役割ごとの専門の口）」のこと。
   胴体部分(バックボーン: CNN + Transformer)が理解した内容を受け取って、最後に具体的な答えを出力するパーツのこと。

2. FPN(Feature Pyramid Network / 特徴ピラミッド)とは？
   モザイク画のように粗くなってしまった特徴を、ピラミッドを逆再生するように**段階的に解像度を元の細かさに復元(拡大)していくネットワーク構造** のこと。

3. 推論(Inference)とは？
   学習済みのモデルにテスト画像を入力して、**実際に答え（計算結果）を出すこと**。

###### - Soon Net の「2つのヘッド」について
実は、Soon Net には目的の異なる **2つのヘッド(出口)** がついている：

```
             ┌─── 【① 分類ヘッド (Classification Head)】
             │     → 「この細胞は分裂直前か？」などの確率(ただの3つの数字)を出力。
[Transformer]│　　　　この出力がメイン出力。
             │
             └─── 【② FPNヘッド(AuxFPN Head)】 ★これが「FPNヘッド推論」！
                   → 「前後のフレームで、この細胞はどこに移動するか？」という立体的な【3Dヒートマップ画像】を出力
                   この出力がサブ出力(Auxiliary:補助)。
```

###### - FPNヘッドは何をしているのか？
Transformer が思考し終わった直後のデータは、CNNで圧縮されているため **$4 \times 8 \times 8$ という超モザイク（粗いブロック）** になっているため、このままでは、「右下に動くぞ」という大まかな方角は分かっても、**「具体的にどの座標（ピクセル）に動いたか？」という精密な位置がボヤけて分からない。**
そこで、** FPNヘッド(`AuxFPN`)** を使用する：

1. **モザイクを拡大する:**
   $4 \times 8 \times 8$ の粗いデータを、段階的に $16 \times 32 \times 32$ へと3D拡大する。
2. **途中のクッキリした輪郭(スキップ)を合体させる:**
   拡大する際、CNNの初期段階で保存しておいた「元の細かい輪郭情報」を横から足し合わせます。
3. **高精細なマップを出力する:**
   ボヤけることなく、「次フレームで細胞が着地する場所」がピシッと尖った **高精細な3D確率マップ(占有マップ `occ` と 分裂マップ `div`)** を出力する。

###### コードでの該当箇所
cell_tracking/soon/model.py 153行目にある `self.aux_fpn` が、「FPNヘッド」。

```python
# model.py 153行目
self.aux_fpn = AuxFPN(...)  # ← これが FPNヘッドの実体
```

そして、推論時に 207〜212行目の `_aux_decode()`を通して、`occ`（移動先マップ）と `div`（分裂娘細胞マップ）が計算される。

###### まとめ

**「FPNヘッド推論」** とは、単なる「分裂確率の数字」を出すだけでなく、**「FPN構造を使って解像度を復元し、『細胞が次フレームのどの3D空間座標に移動（占有）するか』を表す立体ヒートマップを計算・出力する処理」** のこと。

---
### フェーズ3: 学習型リンカー(Learned Linker)

#### 1. 候補抽出と特徴量設計
- 探索半径 `support_radius_um = 30.0μm`、最近傍 `k = 16` の候補細胞ペアを列挙。
- 以下の豊富な特徴量を統合：
  - **ノード特徴量 (42次元):** 検出スコア、細胞体積・主軸比率、Soon Netの3クラス確率(soon scores)など。
  - **ペア特徴量 (23次元):** 幾何学的変位ベクトル、局所強度(intensity)の一致度、**Soon Netが予測した占有マップとの一致度スコア(occupancy score)**、局所オプティカルフロー適合度。

#### 2. 損失関数と学習の工夫
- **アノテーション済みエッジのみのNLL損失 (NLL on annotated edges only):**
  - データセットの正解ラベルはスパースであり、正解がついていないエッジが大量に存在します。通常通りに学習させると「未アノテーションの真のエッジ」を「陰性例(繋いではいけない)」として誤学習してしまいます。1位解法では、アノテーションが存在するエッジのみで負の対数尤度(NLL)を計算し、未アノテーション部をペナルティから除外しました。
- **分岐構成損失 (Fork Configuration Loss):**
  - 親ノードに対して「分裂して子A, Bを持つ」「継続して子Aのみを持つ」「子なし」という排他的な構造全体に対してソフトマックス損失を適用。
  - 分裂イベントはデータ全体の数%と稀少なため、**学習時に分裂を8倍オーバーサンプリング**。
- **タイムステップ拡張 (Time-step Augmentation):**
  - 学習データの25%で1フレーム飛ばし $(t, t+2)$ のペアを意図的に提示し、検出漏れによるギャップ耐性をモデルに学習させました。

#### 3. 貪欲デコード (Greedy Decode)
- 複雑なILP(整数線形計画法)や外部ソルバーは使わず、高速で破綻のない貪欲デコードを採用。
- 各親細胞と各選択肢について、**「子を持たない場合との対数オッズ比」** を算出：
  $$\Delta = \log p(	ext{option}) - \log p(	ext{no child})$$
- $\Delta > 0$ となる候補のみを残し、スコア降順にソート。
- 「1つの親は最大1つの選択肢(継続または分裂)」「1つの子は最大1つの親」という制約を満たすように上から順に確定。
- 親がつかなかった子細胞は新規トラックとして開始。

#### フェーズ3 学習型リンカー(Learned Linker)シーケンス

```mermaid
sequenceDiagram
    autonumber
    participant Main as cell_tracking/inference.py
    participant LinkMovie as cell_tracking/linker/link.py
    participant LinkCache as cell_tracking/linker/cache.py
    participant OccMod as cell_tracking/linker/occupancy.py
    participant EdgeEns as cell_tracking/linker/ensemble.py
    participant Model as cell_tracking/linker/model.py
    participant Decode as cell_tracking/linker/decode.py

    Note over Main: run_worker() から呼び出し
    Main->>Main: link_and_post(edge:EdgeEnsemble, cfg, dirs, movie:胚名称, device)
    Note over Main: 【フェーズ3: グラフ接続(Learned Linker)】
    Main->>LinkMovie: link_movie_from_cache(edge:EdgeEnsemble, dirs["cache"], movie:胚名称, ...)
    LinkMovie->>LinkCache: 全フレームの座標・幾何特徴・輝度キャッシュ読込
    LinkMovie->>OccMod: Soonスコア & 継続・分裂占有マップ読込
    Note over LinkMovie: 半径30μm・近傍k=16で候補ペア抽出<br/>特徴量生成 (ノード42次元, ペア23次元)
    LinkMovie->>EdgeEns: _decode_pair()→_pair_logits()→_run()→model()→EdgeEnsemble::__call__()
    EdgeEns->>Model: m()→UnetCellPoseEdgeTransformer::forward(features) [5-fold 各モデルを推論]
    Model-->>EdgeEns: 各foldの未正規化スコア群(<br/>- logits: ペア接続ロジット(t の細胞と t+1 の細胞の全ペアに対する1対1接続スコア)<br/>- config_logits:分裂設定ロジット(各親細胞の全選択肢(消失・単独接続・2分裂)に対する対数確率)<br/>- kind_logits:分類ロジット(各親細胞が「分裂しない(0)」か「分裂する(1)」かの2値分類スコア)<br/>- config_child1,config_child2:娘細胞インデックス(各選択肢が選択された場合の、第1娘細胞のIDと第2娘細胞のID) )
    EdgeEns-->>LinkMovie: 5-fold平均化されたロジット群 (辞書形式)
    LinkMovie ->> Decode: decode_fork_configurations(fork_np)
    Note over Decode: 貪欲デコード (Greedy Decode) 実行
    Decode -->> LinkMovie: - links: list[tuple[int, int]]例:[(0, 1), (0, 2), (3, 4)]※(親細胞ID, 子細胞ID) のペア<br/>- link_probs: list[float]例:[0.85, 0.85, 0.94]※各エッジに対応する確率(0.0〜1.0)
    Note over LinkMovie: ① ロジットを Softmax して「接続確率」「分裂確率」に変換<br/>② log p(option) - log p(no child) > 0 を降順ソート<br/>③ 各親最大1選択肢、各子最大1親でエッジ確定
    LinkMovie-->>Main: frames, links (接続リスト), link_probs (確定した確率)
```

##### フェーズ3 学習型リンカー(Learned Linker)関数一覧

| No | 関数 / メソッド | 呼び出し元 | 引数 (型・shape・意味) | 戻り値 (型・shape・意味) | 主な処理内容と役割 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `link_and_post()` | `Main` | - `edge`: `EdgeEnsemble`<br>- `cfg`: `dict` 設定辞書<br>- `dirs`: `dict[str, str]` 一時ディレクトリ群<br>- `movie`: `str` 胚名称<br>- `device`: `str` 実行デバイス | `TrackGraph` | `None`<br>(追跡グラフオブジェクト、またはフレーム数不足時None) | 保存済みキャッシュから `link_movie_from_cache()` を呼んでエッジ接続を行い、続けてゲート処理や平滑化などの後処理を適用して最終グラフを返す。 |
| 2 | `link_movie_from_cache()` | `link_and_post()` | - `model`: `EdgeEnsemble`<br>- `cache_dir`: `Path`｜`str`<br>- `movie`: `str` 胚名称<br>- `support_radius_um`: `float` = 30.0<br>- `support_k`: `int` = 16<br>- `soon_dir`, `intensity_dir`, `occupancy_dir`, `occupancy_division_dir`: `Path`｜`str`｜`None` | - `tuple`: `(frames, links, None, link_probs)`<br>- `frames`: `list[np.ndarray]`, 各shape: `(N_t, 3)` (細胞座標)<br>- `links`: `list[list[tuple[int, int]]]`, 長さ: `T-1` (各フレームペアの `(親idx, 子idx)` 確定エッジリスト)<br>- `None`: 未使用スロット<br>- `link_probs`: `list[list[float]]`, 長さ: `T-1` (各確定エッジの確率値 `0.0〜1.0` のリスト) | 全フレームの座標・幾何特徴・輝度・SOON・占有キャッシュを読み込み、探索半径30μm・近傍k=16で候補ペアを抽出してエッジ推論を行い、貪欲デコードでエッジと確率を確定。 |
| 3 | `EdgeEnsemble.__call__()`<br>(内部で `UnetCellPoseEdgeTransformer.forward()`) | `link_movie_from_cache()`<br>(内部の `_decode_pair()` → `_pair_logits()`) | - `*args`: `feat_t, feat_t1, coords_t, coords_t1, volumes_t, volumes_t1` (テンソル群)<br>- `**kwargs`: `soon_t, intensity_t, occ_next, pair_mask, return_fork=True` 等 | - `dict[str, Tensor]` (5-fold平均化された出力辞書)<br>- `logits`: `(N_t, N_t1)` ペア接続ロジット<br>- `config_logits`: `(N_t, N_configs)` 分裂設定対数確率<br>- `kind_logits`: `(N_t, 2)` 分裂有無分類ロジット<br>- `config_child1`, `config_child2`: `(N_t, N_configs)` 娘細胞ID候補 | 5-fold の各エッジモデル (`m.forward()`) を順次実行し、1対1ペア接続ロジットおよび細胞分裂(fork)設定ロジット等の出力テンソルを平均化して1つの辞書にまとめて返す。 |
| 4 | `decode_fork_configurations()`<br>(補助: `resolve_refractory_by_prob()`) | `link_movie_from_cache()` | - `config_output`: `dict[str, np.ndarray]` (`fork_output` のNumPy辞書: `config_logits, config_mask, config_kind, config_child1, config_child2`)<br>- `threshold`: `float` = 0.0<br>- `max_children`: `int` = 2 | - `tuple[list[tuple[int, int]], list[float]]`<br>- `links`: `list[tuple[int, int]]`, 長さ: `E`(`(親idx, 子idx)`の採択エッジリスト)<br>- `link_probs`: `list[float]`, 長さ: `E`(各採択エッジに対応する確率値 `0.0〜1.0`) | 各親細胞の `config_logits` を Softmax して確率化し、空事象との対数オッズ比 `margin = log p(option) - log p(empty)` を降順ソート。親1つに最大1選択肢(単独または2分裂)、子1つに最大1親の制約を満たすよう貪欲(Greedy)にエッジを確定。 |
---

### フェーズ4: 後処理 (Post Processing)

複雑なヒューリスティクスを極力排除し、シンプルかつ確実なルールのみで構成：

1. **Tiered Temporal Gate (段階的信頼度ゲート):**
   - 信頼度スコア `TIER_CONF = 0.3` 未満かつ長さ `TIER_MIN_LEN = 3` 未満の不確実な短エッジを排除。
2. **1フレーム ギャップ補完 (Gap Closing):**
   - 1フレームの一時的な検出漏れを距離閾値(4.0μm)・ドリフト閾値(30.0μm)内で補完。
3. **孤立ノードの枝刈り & 長さフィルタ:**
   - どこにも繋がっていない孤立ノードを削除。全体の長さが3フレーム未満のトラックを除去。
4. **分裂保護付き 線形フィッティング平滑化 (Line-fit Smoothing):**
   - 細胞のトラッキング座標の揺らぎを局所線形回帰で滑らかに補正。
   - ただし、分裂分岐点(`stop_at_forks=True`)では娘細胞の座標が母細胞側へと引きずられて潰れないよう平滑化を停止。
5. **分裂不応期ルール (Division Refractory Rule):**
   - 生まれたばかりの娘細胞が、直後のフレームで再び分裂することを生物学的に禁止。
   - 競合が発生した場合は、予測信頼度が高い分裂を優先して採用。

#### フェーズ4 後処理(Post Processing)シーケン図

```mermaid
sequenceDiagram
    autonumber
    participant Entry as biohub-1st-place-inference.py
    participant Main as cell_tracking/inference.py
    participant GraphMod as cell_tracking/linker/graph.py
    participant ScoreMod as cell_tracking/linker/scoring.py
    participant TierGate as cell_tracking/linker/tiered_gate.py
    participant PostMod as cell_tracking/linker/postprocess.py
    participant Tracks as cell_tracking/common/tracks.py

    Note over Main: link_and_post() 内でグラフ構築・後処理
    Main->>GraphMod: graph_from_pair_links(B, frames, links)
    GraphMod-->>Main: TrackGraph型への詰め直し<br/>g(TrackGraph初期グラフ)
    Main->>ScoreMod: link_confidence(frames, links, link_probs)<br/>frames:各フレームに含まれる細胞一覧のリスト(要素数len(f)が各フレームの細胞数)<br/>隣接フレーム間(t と t+1)で紐付けられた細胞ペア(i, j) のリスト<br/>各ペア(i, j)が同一細胞である予測確率
    Note over ScoreMod: 時系列全体の各細胞(ノード)について、接続されているリンク(前後のフレームとの結びつき)の中で最も高い予測確率(確信度)を算出する
    ScoreMod-->>Main: conf (リンク確信度)

    Main->>TierGate: tiered_temporal_gate(g, conf, conf_thresh=0.3, min_len=3)
    Note over TierGate: 信頼度0.3未満かつ長さ3未満の低信頼エッジを排除
    TierGate-->>Main: g (選別済みグラフ)

    Main->>GraphMod: postprocess_graph(B, frames, g, linefit=True, max_gap=1, ...)
    GraphMod->>Tracks: close_gaps(frames, g, max_gap=1, gap_dist_um=4.0)
    Note over Tracks: 1フレームの一時的検出漏れを補間
    GraphMod->>Tracks: prune_isolated(g)
    Note over Tracks: 孤立ノード削除
    GraphMod->>PostMod: filter_short_tracks(g, min_track_len=3)
    Note over PostMod: 3フレーム未満の短トラック削除
    GraphMod->>PostMod: linefit_smooth(stop_at_forks=True)
    Note over PostMod: 局所線形回帰平滑化 (分裂ノードは保護)
    GraphMod-->>Main: g (後処理完了TrackGraph)

    Note over Main: run_worker() へ戻り行データ変換
    Main->>Tracks: graph_to_rows(胚番号, g)
    Tracks-->>Main: rows
    Note over Main: main() で全動画完了後に提出ファイル出力
    Main->>Tracks: write_submission(all_rows, OUTPUT)
    Tracks-->>Entry: submission.csv 書き出し完了
```

##### フェーズ4 登場関数一覧

| No | 関数 / メソッド | 呼び出し元 | 引数 (型・shape・説明) | 戻り値 (型・shape・説明) | 主な処理内容と役割 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `graph_from_pair_links()` | `link_and_post()` | - `B`: モジュールオブジェクト<br>- `frames`: `list[np.ndarray]`, 各shape: `(N_t, 3)`<br>- `pair_links`: `list[list[tuple[int, int]]]` | - `TrackGraph`<br>- `node_ids`: `np.ndarray`, shape: `(N_total,)`<br>- `node_t, z, y, x`: `np.ndarray`, shape: `(N_total,)`<br>- `edges`: `np.ndarray`, shape: `(E_total, 2)` | フレームごとのインデックス接続リストから、全体を表現する `TrackGraph` データ構造を構築。 |
| 2 | `link_confidence()` | `link_and_post()` | - `frames`: `list[np.ndarray]`<br>- `links`: `list[list[tuple[int, int]]]`<br>- `link_probs`: `list[np.ndarray]`, 各shape: `(L_t,)` | `dict[tuple[int, int], float]`<br>(エッジ `(node_u, node_v)` をキー、確信度スコアを値とする辞書) | 各接続エッジの推論確率からリンクの総合確信度スコアを算出。 |
| 3 | `tiered_temporal_gate()` | `link_and_post()` | - `g`: `TrackGraph`<br>- `conf`: `dict[tuple[int, int], float]`<br>- `conf_thresh`: `float` = 0.3<br>- `min_len`: `int` = 3 | `TrackGraph`<br>(低信頼かつ短期間の接続エッジが削除されたグラフ) | 確信度が低く、かつ持続期間が3フレーム未満の信頼性の低い短エッジを排除。 |
| 4 | `postprocess_graph()` | `link_and_post()` | - `B`: モジュールオブジェクト<br>- `frames`: `list[np.ndarray]`<br>- `g`: `TrackGraph`<br>- `linefit`: `bool` = True<br>- `max_gap`: `int` = 1<br>- `gap_dist_um`: `float` = 4.0<br>- `gap_drift_um`: `float` = 30.0 | `TrackGraph`<br>(ギャップ補完・平滑化後の最終完成グラフ) | ギャップ補完、孤立点削除、短トラック削除、線形平滑化の後処理チェーンを一括統括。 |
| 5 | `close_gaps()` | `postprocess_graph()` | - `frames`: `list[np.ndarray]`<br>- `g`: `TrackGraph`<br>- `max_gap`: `int` = 1<br>- `gap_dist_um`: `float` = 4.0<br>- `drift_radius_um`: `float` = 30.0 | `TrackGraph`<br>(補間エッジが追加されたグラフ) | 1フレームの一時的な検出漏れを距離閾値内で自動補間。 |
| 6 | `prune_isolated()` | `postprocess_graph()` | `g`: `TrackGraph` | `TrackGraph`<br>(孤立ノードが削除されたグラフ) | どこにも繋がっていない孤立ノードをグラフから削除。 |
| 7 | `filter_short_tracks()` | `postprocess_graph()` | - `g`: `TrackGraph`<br>- `min_track_len`: `int` = 3<br>- `keep_division_components`: `bool` = False | `TrackGraph`<br>(短トラックが除外されたグラフ) | 全体の長さが3フレーム未満の微小なノイズトラックを削除。 |
| 8 | `linefit_smooth()` | `postprocess_graph()` | - `g`: `TrackGraph`<br>- `window`: `int` = 2<br>- `weight`: `float` = 0.8<br>- `stop_at_forks`: `bool` = True | `TrackGraph`<br>(座標が平滑化されたグラフ) | 局所線形回帰で座標のブレを平滑化(分裂分岐点では母娘が混ざらないよう平滑化を停止)。 |
---

## ソースコード (`inference.py`) の推論フロー詳解

エントリーポイントとなる `biohub-1st-place-inference.py` から呼び出される `cell_tracking/inference.py` を起点とし、前述の「推論処理の全体概要シーケンス図」に示した通り、各動画に対して `build_caches()` (フェーズ1・2) と `link_and_post()` (フェーズ3・4) が順に実行されます。

ここでは、各ファイル(オブジェクト)の役割分担および主要パラメータを整理します。

### 各ファイル (オブジェクト) の役割と分担

| フェーズ | ファイルパス (オブジェクト) | 主な関数 / クラス | 担当する具体的な処理 |
| :--- | :--- | :--- | :--- |
| **全体統括** | `cell_tracking/inference.py` | `cli()`, `main()`, `run_worker()`, `build_caches()`, `link_and_post()` | 各種重みのロード、動画ごとのワーカースレッド管理、フェーズ1〜4の実行制御。 |
| **フェーズ1** | `cell_tracking/detector/inference.py` | `load_unetcellpose_ensemble()`, `extract_frame_cache()` | 3D U-Netによる立体画像推論、4方向反転TTA(計20パス)、細胞の中心座標・3Dフロー・体積の抽出。 |
| **フェーズ1〜2** | `cell_tracking/linker/cache.py`<br>`cell_tracking/linker/intensity.py`<br>`cell_tracking/linker/occupancy.py` | `write_frame_cache()`, `write_intensity_sidecar()`, `write_occupancy_from_maps()` | 検出座標、細胞輝度、Soon Netが予測した占有マップをフレームごとの中間キャッシュとして保存。 |
| **フェーズ2** | `cell_tracking/soon/scoring.py`<br>`cell_tracking/soon/model.py` | `score_points_per_net()`, `SoonNet.forward()` | 細胞周囲の3フレーム局所画像から、「分裂直前か」の3クラス分類と、次フレームの移動先・分裂先の占有マップを生成。 |
| **フェーズ3** | `cell_tracking/linker/link.py`<br>`cell_tracking/linker/ensemble.py` | `link_movie_from_cache()`, `EdgeEnsemble` | 近傍候補ペアの探索、Soon Net占有マップ適合度の結合、5-fold平均スコアの算出、貪欲デコードによるエッジ確定。 |
| **フェーズ4** | `cell_tracking/linker/tiered_gate.py` | `tiered_temporal_gate()` | 信頼度の低い孤立した短い接続エッジを選別して排除。 |
| **フェーズ4** | `cell_tracking/linker/graph.py`<br>`cell_tracking/linker/postprocess.py` | `postprocess_graph()`, `linefit_smooth()`, `filter_short_tracks()` | ギャップ補完、孤立ノード削除、3フレーム未満トラックの除去、分裂保護付き線形フィッティング平滑化。 |
| **出力** | `cell_tracking/common/tracks.py` | `graph_to_rows()`, `write_submission()` | 完成したトラックグラフから提出用フォーマットの行データを生成し、`submission.csv` へ書き出し。 |

---

## 2位解法との比較：

| 項目 | 1位解法(sersasj) | 2位解法(soheilayati) |
| :--- | :--- | :--- |
| **追跡の基本思想** | **ビジュアルトラッキング (視覚的追跡)**<br>細胞の見た目変化・FPN占有マップから直接追跡 | **幾何ヒューリスティクス + 特徴量工学**<br>座標間距離・速度・カルマンフィルタに基づく |
| **分裂(Division)の捉え方** | **Soon Netによる事前検知**<br>「間期・直前・直後」の3クラス分類 + 娘細胞占有マップ | **ペア特徴量 + 分裂スタッキングモデル**<br>距離・幾何特徴からLightGBM/ペアモデルで判定 |
| **アンサンブル手法** | **厳密なOOF 5-fold平均**<br>(検出器 5-fold × Soon Net 5-fold × Linker 5-fold) | **多層メタスタッキング & 多数決合意**<br>(4系統トラッカー + 多数のGBDT/線形分類器) |
| **デコード方式** | **シンプルな貪欲デコード (Greedy Decode)**<br>対数オッズ比による高速・堅牢な逐次決定 | **多段制約ソルバー & 競合調停**<br>(Stage-6リランキング、エッジ合意、フォーク投票) |
| **システム複雑度** | **コンパクトかつ洗練されたDNN設計**<br>(総パラメータ数は数M程度でコードも簡潔) | **極めて複雑な処理構成**<br>(多数の重みファイルと何百行もの特徴量抽出) |

## 1位解法の特徴: 勝利の要因は学習にあり
「データの追加・拡張」と「Soon Net（分裂・状態推定）」も特徴ではありますが、やはり一番大きな特徴は学習にあると思います。スパース(疎)な学習データに対し、疑似正解データを生成して学習データを大幅に増強していますがそういった独自の学習データ強化策と思います。

### 1. 検出器（Detector）の学習：MAE事前学習とCellpose風フロー

多くの参加者は単に3D U-Netで「細胞中心のガウスヒートマップ」を予測させましたが、1位解法は以下のように学習させています。

* **自己教師あり事前学習（MAE: Masked Autoencoder）**:
  ラベルが付いていない膨大な3D顕微鏡ボリュームデータを用い、ボクセルのマスク復元タスクによってバックボーンの表現力を事前に底上げしました。
* **Cellpose風の「中心への3Dベクトルフロー」学習**:
  単なるヒートマップではなく、各ボクセルから「細胞中心へ向かう3Dベクトル」を学習させました。これにより、密集した細胞同士の境界を綺麗に分離でき、副産物として**細胞ごとの「体積（Volume）」や「幾何形状（楕円体パラメータ）」を精密に抽出**して後のリンカーに渡せるようにしました。

---

### 2. Soon Net の学習：マルチタスク学習とOOFの徹底

Soon Net は単に「次フレームの移動先」を当てるモデルではありません。

* **3クラスの細胞周期フェーズ分類**:
  前後3フレーム（$t-1, t, t+1$）の $16 \times 64 \times 64$ の3Dローカルクロップから、「間期（通常状態）」「まもなく分裂（soon-to-divide）」「分裂直後（just divided）」の3状態を学習。
* **占有マップ（Occupancy Map）の同時学習**:
  「細胞が次のフレームでどのボクセル領域を占めるか」というヒートマップ（Occupancy）と、「分裂後の娘細胞がどこに出現するか（Division Occupancy）」を同時に学習させています。
* **OOF（Out-of-Fold）によるリーク防止**:
  5-Fold交差検証でSoon Netを学習させ、リンカーの学習時には**モデルが見たことのないOOF予測値**を特徴量として入力することで、推論時とのギャップ（過学習）を徹底的に防いでいます。

---

### 3. リンカー（Edge Transformer）の学習：極度の不均衡と欠損への対策

リンカー（細胞同士・分裂の紐付けモデル）には、以下のような際立った工夫があります。

* **分裂イベントの8倍オーバーサンプリング (`oversample_divisions: 8`)**:
  全データ中、細胞分裂（1つの親 $\rightarrow$ 2つの娘）はごく稀にしか起きません。普通に学習させるとモデルが「分裂しない」方向にバイアスされるため、**分裂イベントを含むペアを8倍の頻度でオーバーサンプリング**して学習させています。
* **1フレーム飛ばし（Gap）の混同学習 (`gaps: [1, 2]`, `gap_weights: [0.75, 0.25]`)**:
  学習時、隣接フレーム（$t \rightarrow t+1$）だけでなく、**1フレーム飛ばした（$t \rightarrow t+2$）ペアを25%の割合で混ぜて学習**させています。これにより、検出器が一時的に1フレーム見失っても、リンカーが途切れることなく結びつけられる耐性を学習段階で獲得させています。
* **生物学的制約の損失関数への反映 (`fork_vol_conserv: true`)**:
  「分裂した2つの娘細胞の体積の合計は、親細胞の体積とおおよそ一致する（体積保存則）」などの物理的特徴を損失関数や特徴量に反映させ、不自然なマッチングを抑制。
* **EMA（指数移動平均: 0.9999）による重みの平滑化 (`ema_decay: 0.9999`)**:
  エポックごとのモデルの揺らぎを抑え、極めて汎化性能の高い重みを獲得しています。

---

### まとめ

1位解法の学習は、単に「データを増やした」だけにとどまらず、
1. **「データ観察から得た知見（細胞分裂の予兆や見た目）」をタスク（Soon Net / Occupancy）として定式化**
2. **「極端な不均衡（分裂の少なさ）」を8倍オーバーサンプリングで解消**
3. **「検出漏れ」を想定して最初からGap混同学習を実施**

というように、**問題固有のボトルネックを学習段階で先回りして潰す設計**が徹底されている点が最大の特徴です。