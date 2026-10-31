# biohub_infer — Biohub - Cell Tracking During Development の推論パッケージ

最終スタック（submission 060: private 0.96918 / public 0.97832）の推論専用パッケージ。テスト動画（`<sid>.zarr`）を
コンペの node / edge CSV に変換する。学習済みモデルを読むだけで、学習コードは含まない。

## データセット

| データセット | 中身 |
|---|---|
| `ren4yu/biohub-inference` | このパッケージ（`biohub_infer/`）とオフライン install 用の `wheels/` |
| `ren4yu/biohub-weights` | TensorRT engine（検出器・分裂モデル）、matcher / flow の checkpoint、較正器、`md5sums.txt` |

engine は **Tesla T4 + TensorRT 10.9.0.34** 用にビルドしたもの。Kaggle では T4 x2 accelerator を使う（GPU 1 枚に
worker 1 プロセス）。

## Kaggle での実行

2 つのデータセットとコンペデータを notebook に attach し（T4 x2、internet off）、`notebook/inference.ipynb` を
実行する。中身は「image に無い依存を `wheels/` から入れる」→「`python -m biohub_infer --test-dir ... --weights ...
--out submission.csv`（`PYTHONPATH` = コードデータセット）」の 2 段だけ。

隠しテスト 166 本は T4 x2 で約 11〜12 h（12 h 制限ぎりぎり）。

## パイプライン

GPU で動画ごと（`biohub_infer/infer/clip.py`）:

1. **検出**: 3 メンバー（EfficientNetV2-L 2.5D U-Net 0.3、3D SegResNet 0.3、EfficientNet-B7 2.5D U-Net 0.4）、
   各メンバーは 5 fold engine の平均。Gaussian blur σ 0.7、NMS (3, 7, 7)、閾値 0.2。
2. **flow と対応付け**: dense 3D flow（`ssl_flow`）、matcher（5 fold の logit 平均）、本番 Hungarian、second round、
   LP wiring、半径 12 µm の top-5 候補。
3. **分裂スコア**: div3f の parent / pre / post / native parent（各 5 fold、rot4 TTA）を encoder / decoder engine で。
   半径 16 µm の追加候補グラフ。

その後 CPU で（`biohub_infer/lineage/build.py`）:

4. **較正**: logistic / LightGBM / XGBoost の較正器で候補を edge / division の確率にする。
5. **lineage LP**: テストのブロックごと（sorted 位置 [0,106) と [106,D)）に pooled LP、120 s × 3 round。
6. **後処理**: gap closing（3 フレームまで）、長さ 6 未満の成分を削除、affine 座標補正。

## 検証

Kaggle（T4 x2）でこの 2 データセットだけを attach し、公開テスト 4 本で `submission.csv` が submission 060 と
バイト一致（md5 `04e2696254fb488a9b2033edacf277ad`）、upstream の中間キャッシュ（graphs / confidence /
div_scores / division_graphs）もすべて一致することを確認した。

## 構成

- `biohub_infer/` — パッケージ本体（各ディレクトリの役割は README.md の Layout を参照）
- `notebook/build_notebook.py` — Kaggle notebook（`inference.ipynb` + `kernel-metadata.json`）を書き出す
- `stage_datasets.py` — 2 つのデータセットのディレクトリを組み立てる

`div797_lineage` などのモジュール名は、抽出元の研究リポジトリの実験番号をそのまま残している（結果を遡れるように）。
