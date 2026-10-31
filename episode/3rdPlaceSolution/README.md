# biohub_infer — inference for Biohub - Cell Tracking During Development

Inference-only package of our final stack (submission 060: private 0.96918 / public 0.97832). It turns the test
movies (`<sid>.zarr`) into the competition's node / edge CSV using trained models only (no training code).

## Datasets

| Dataset | Contents |
|---|---|
| `ren4yu/biohub-inference` | this package (`biohub_infer/`) and `wheels/` for the offline installs |
| `ren4yu/biohub-weights` | TensorRT engines (detectors, division models), matcher / flow checkpoints, calibrators, `md5sums.txt` |

The engines are built for a **Tesla T4 with TensorRT 10.9.0.34**. On Kaggle, use the T4 x2 accelerator; the run
uses one worker per GPU.

## Run on Kaggle

Attach the two datasets and the competition data to a notebook (T4 x2, internet off), then run
`notebook/inference.ipynb`. In short:

```python
import os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                "--find-links=/kaggle/input/datasets/ren4yu/biohub-inference/wheels", ...])   # what the image lacks
subprocess.run([sys.executable, "-m", "biohub_infer",
                "--test-dir", "/kaggle/input/competitions/biohub-cell-tracking-during-development/test",
                "--weights", "/kaggle/input/datasets/ren4yu/biohub-weights",
                "--work-dir", "/kaggle/working/work", "--out", "/kaggle/working/submission.csv"],
               env=dict(os.environ, PYTHONPATH="/kaggle/input/datasets/ren4yu/biohub-inference"), check=True)
```

All 166 hidden-test movies take about 11-12 h on T4 x2 (close to the 12 h limit).

## Pipeline

Per movie on the GPU (`biohub_infer/infer/clip.py`):

1. **Detection**: three detectors (EfficientNetV2-L 2.5D U-Net 0.3, 3D SegResNet 0.3, EfficientNet-B7 2.5D U-Net 0.4),
   each the mean of its five fold engines; Gaussian blur σ 0.7, NMS (3, 7, 7), threshold 0.2.
2. **Flow and matching**: dense 3D flow (`ssl_flow`), cell matcher (five folds, logit mean), production Hungarian
   cell, second round, LP wiring, radius-12 top-5 candidates.
3. **Division scores**: div3f parent / pre / post / native-parent models (five folds each, rot4 TTA) as
   encoder / decoder engines; radius-16 extra candidate graph.

Then on the CPU (`biohub_infer/lineage/build.py`):

4. **Calibration**: logistic / LightGBM / XGBoost calibrators turn candidates into edge and division probabilities.
5. **Lineage LP**: one pooled LP per test block (sorted positions [0,106) and [106,D)), three rounds of 120 s.
6. **Post-processing**: gap closing (up to 3 frames), components shorter than 6 removed, affine coordinate refinement.

## Layout

```
biohub_infer/
  __main__.py        CLI: preflight (files, md5, calibrator), GPU workers, downstream, CSV validation
  config.py          constants of the stack and the weight file names
  trt.py             TensorRT runners (detector engines, div3f encoder / decoder engines)
  infer/             stages 1-3: clip.py (per-movie driver), common.py (detector TTA / blur / peaks, matcher
                     loading), linking.py + second_round.py + lp_wiring.py (matching and candidate graph),
                     div797_parent*.py (flow-warped native-parent division scores)
  lineage/           stages 4-6: build.py (driver), div688 / div797 / div846_lineage.py (calibrated predictions
                     and the division-aware LP), run_metric_event_pipeline.py (LP rounds), gap_close.py,
                     affine_refinement.py
  core/              matcher network, calibration features, LP problem, coordinate refinement, zarr access
  div3f/, ssl_flow/  division-model geometry helpers, the dense flow network
  submission.py      CSV writing / validation
notebook/            build_notebook.py writes the Kaggle notebook (inference.ipynb + kernel-metadata.json)
stage_datasets.py    builds the two dataset directories
```

Module names such as `div797_lineage` keep the experiment ids of the research repository the code was
extracted from, so results can be traced back to it.

## Verification

On Kaggle (T4 x2) with only these two datasets attached, the four visible test movies give a `submission.csv`
byte-identical to submission 060 (md5 `04e2696254fb488a9b2033edacf277ad`), and so do every upstream cache
(graphs / confidence / division scores / candidate graphs).
