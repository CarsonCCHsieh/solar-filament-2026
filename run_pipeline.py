"""Local CLI: resume defaults; --train explicitly reproduces training."""
from pathlib import Path
import argparse
import sys

_source_root = Path(__file__).resolve().parent
if not (_source_root / "dataset.py").is_file():
    _source_root = _source_root.parent
sys.path.insert(0, str(_source_root))
from pipeline import run_posttraining, train_from_scratch, create_delivery_zip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--development-report")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--delivery-zip")
    args = parser.parse_args()
    data = Path(args.data_root)
    annotation = data / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    image_root, test_root = data / "train" / "train_images", data / "test" / "test_images"
    if args.train:
        if args.checkpoint_dir is None:
            parser.error("--train requires explicit --checkpoint-dir")
        train_from_scratch(annotation, image_root, args.model_dir, args.checkpoint_dir, args.device)
    run_posttraining(annotation, image_root, test_root, args.model_dir, args.output_dir,
        args.development_report, args.device, display_figures=not args.no_display)
    if args.delivery_zip:
        create_delivery_zip(args.model_dir, args.output_dir, args.delivery_zip)


if __name__ == "__main__":
    main()
