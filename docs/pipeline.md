# Complete Solar pipeline

The completed native run trained on 558 observations for 24 epochs, froze its
threshold using eight development observations, evaluated 141 holdout
observations and inferred all 180 test observations. Actual model weights and
native results are included in `model/` and `outputs/`. The readable source
notebook preserves the complete process without fabricated execution outputs.
There is also an explicit from-scratch reproduction mode. The source does not
download data, install packages, upload a CSV, or submit a competition form.

## Run the included completed model without retraining

For a released repository, put `dataset.py`, `evaluation.py`,
`unet_candidate.py`, `pipeline.py`, and `run_pipeline.py` at the repository root.
The three actual model files are in `model/` and the completed development
receipt is in `outputs/development-results.json`. The readable, self-contained
`solar_complete_pipeline.ipynb` defaults to resume mode and prefers those
repository paths when present; edit its input path to the mounted official
competition data. The notebook contains the complete source and hash checks.
Its source cells write only to `/kaggle/working/solar_final_source`.

From the repository root, this CLI runs the same post-training path:

```sh
python run_pipeline.py --data-root /path/to/MAGFiLO_1.0_Kaggle_2026 --model-dir model --output-dir reproduced-outputs --development-report outputs/development-results.json --device cuda --delivery-zip solar-reproduced-delivery.zip
```

Use `--device cpu --no-display` for a noninteractive CPU environment. Required
libraries must already be installed: PyTorch, NumPy, SciPy, OpenCV, Matplotlib,
IPython for displayed figures, and genuine pycocotools for ground truth and
compressed RLE. No approximate COCO decoder or pretrained weight fallback is
provided. `runtime-manifest.json` records actual distribution versions, the
OpenCV runtime and full build information, Python, CUDA build/devices and source
hashes from the environment that executes inference.
Use a separate reproduction output directory to preserve the included native
result files and their recorded identities.

To reproduce training explicitly, set notebook `MODE = "train"` with separate
artifact paths, or use CLI `--train --checkpoint-dir retrained-checkpoints`
with `--model-dir retrained-model --output-dir retrained-outputs` and omit the
old model's development report. This trains 558 observations
for 24 epochs, batch size 4, with the original reviewed implementation and no
external weights. Each completed epoch saves a model checkpoint; optimizer
state is not saved, so interrupted-training continuation is not promised.
Seeds are fixed, but cross-platform GPU bitwise determinism is not promised.

## Evaluation contract

- The observation filename split is fixed at seed 2026: 566 initial training
  observations and 141 holdout observations. The same eight evenly spaced
  names from the 566 are development data, leaving exactly 558 training names.
  All annotators of one observation remain together.
- A completed model bundle must contain exactly those 558 names and every
  finite training-loss record for epochs 1–24. Its development receipt must
  agree with the model's configuration and completed training summary.
- Development chooses once among predefined thresholds 0.45, 0.65 and 0.80;
  ties use the first threshold in that order. The resulting freeze records
  model, split, source and receipt hashes before holdout or test inference.
  Incompatible existing freezes or resume journals fail instead of being
  replaced. The notebook/CLI reuse a compatible development receipt when
  available and compute development only when no receipt exists.
- Holdout evaluation follows the visible organizer self-evaluation v6:
  strict IoU greater than 0.5, all qualifying pairs, separate annotator-image
  masks, and globally accumulated overlap/TP/FP/FN before PQ division.
  Annotation sets with no GT filament rows are excluded from that official
  GT-row universe and recorded in `zero-gt-universe.json` and score receipts.
  GT unions are used only for training and figure display.
- Uncertainty uses 1,000 observation-cluster bootstrap resamples, seed 2026,
  of the saved per-observation counts. All annotators stay inside their
  observation cluster. The 95% percentile interval is exploratory; a filename
  hash split does not establish temporal independence or guaranteed coverage.
  Bootstrap does not rerun inference or choose a threshold.
- Test inference requires all 180 original 2048×2048 grayscale JPEGs. Every
  filament ID is unique, uses an unchanged filename stem before the first
  underscore, and has a nonempty suffix. Every actual written CSV row is
  reread, decoded with genuine pycocotools and compared with a hash of the
  original inferred binary mask. The coverage manifest accounts for every
  image, including those with no detections.
- A zero-detection image receives no fabricated filament row. The visible
  offline evaluator supports this policy; actual server acceptance is still
  pending server processing. The actual Kaggle prediction upload is pending
  scoring; the final entry form is not yet complete.

## Figures, attribution and delivery

The first three sorted holdout observations are chosen before their scores
are computed. Their three-panel figures show grayscale pixels, GT union
boundaries and prediction union boundaries, with actual per-observation scores.
They are displayed through Matplotlib/IPython and saved as both PNG and PDF in
`outputs/figures/` in the included native run, so ordinary notebook output
export can retrieve them. Outside-image attribution for the fixed examples:

MAGFiLO / NSO GONG H-alpha; CC BY-NC 4.0; noncommercial competition report.

Dataset attribution: MAGFiLO / NSO GONG H-alpha observations, under the
competition's CC BY-NC 4.0 terms, used for this noncommercial competition
report. See the [official data page](https://www.kaggle.com/competitions/filament-segmentation-2026/data).
Derived comparison figures include this attribution. Do not redistribute the
original JPEG collection or annotation JSON with this source repository.

After successful inference, the notebook creates a delivery ZIP in its Kaggle
working directory; the CLI creates one only when `--delivery-zip` is supplied.
The archive allowlist contains only:

- `model/weights.pt`, `model/config.json`, `model/training-manifest.json`;
- actual CSV, JSON/JSONL result receipts and the six derived figure files
  under `outputs/`;
- a `delivery-manifest.json` with each file's actual size and SHA-256.

The ZIP stays outside its two source roots. It never recursively includes raw
competition input, credentials, unrelated files or private audit directories.
Archive integrity is checked and its actual bytes, SHA-256 and file count are
printed for normal notebook output download. This packaging step does not
publish the archive.

## Source and synthetic checks

The notebook embeds the complete original implementation with its own required
hashes. Its source cells normalize CRLF line endings to LF for writefile
execution; this changes bytes without changing the algorithms. No external
build receipt is needed to execute the notebook.

The new contract checks are in `tests/test_pipeline_contracts.py`. Run
`python -m unittest discover -s tests -p test_pipeline_contracts.py -v` from the
repository root.

Synthetic contract checks cover partition leakage, dev-only
freeze/tie/replacement rules, training-receipt matching, zero-GT universe,
filename/CSV structure, journal identity, test coverage, observation bootstrap,
and ZIP allowlist/integrity behavior. These checks establish contracts rather
than real-data accuracy.

Actual native execution records 141-observation holdout PQ 0.318381864185715,
all 180 test images and 1,661 compressed-RLE mask roundtrips in the included
output artifacts. The source notebook remains unexecuted and includes no
fabricated outputs or scores; actual results are in `outputs/`. Final server
scoring remains pending. The completed four-page report is
[`report.pdf`](report.pdf).
