# Data, figures and trained-weight handling

Original implementation and synthetic test code use the repository's MIT
license. That grant does not relicense the competition dataset, derived
comparison figures, or the included trained weights.

## Dataset and derived figures

The original competition JPEG collection and annotation JSON are not included.
Obtain them from the
[official competition](https://www.kaggle.com/competitions/filament-segmentation-2026)
and preserve the dataset notices. The
[MAGFiLO dataset paper](https://doi.org/10.1038/s41597-024-03876-y) supplies the
source reference.

The small comparison panels in `outputs/figures/` are derived report examples
showing downsampled H-alpha pixels, GT union boundaries and prediction union
boundaries. Their attribution is:

MAGFiLO / NSO GONG H-alpha; CC BY-NC 4.0; noncommercial competition report.

The dataset's [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
terms include attribution, a license link, indication of changes and
noncommercial use; do not imply endorsement. The
[legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)
controls over this short explanation. Competition conditions and other rights
may also apply.

## Included trained weights

`model/weights.pt` is the actual original U-Net trained from scratch on the
558-observation training partition. Configuration and training provenance are
in `model/config.json`, `model/training-manifest.json` and the native result
manifests. No external pretrained weights were used.

These dataset-trained weights are **excluded from the MIT code license**.
The project's weight-use policy is noncommercial research and competition
reproduction only. Preserve this notice, dataset attribution and provenance
with any permitted redistribution, and comply with the applicable dataset and
competition conditions. This separate project policy does not assert that
all trained models automatically inherit a particular copyright license or
that the code license grants rights in the training data.

No third-party library source is bundled; installed dependencies keep their
respective licenses. Publication of the final package remains subject to the
final artifact and licensing review.
