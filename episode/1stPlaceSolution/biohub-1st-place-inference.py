"""Biohub 1st place: inference from the released cell_tracking package and shipped weights."""
import glob
import os
import sys

hits = glob.glob("biohub-1st-place-reproduction/cell_tracking/inference.py", recursive=True)
if not hits:
    raise FileNotFoundError("reproduction dataset not attached: %s" % sorted(glob.glob("../input*")))
ROOT = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, ROOT)

test = [d for d in glob.glob("../input/test", recursive=True) if glob.glob(d + "/*.zarr")]
if not test:
    raise FileNotFoundError("no test movies under ../input")

from cell_tracking.inference import cli

sys.argv = ["infer", "--data-dir", test[0], "--weights-dir", ROOT + "/weights",
            "--work-dir", "/tmp/work", "--output", "working/submission.csv"]
cli()
