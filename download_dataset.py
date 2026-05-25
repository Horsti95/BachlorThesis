"""
Download and validate the BOAS dataset from OpenNeuro.

Downloads the Bitbrain Open Access Sleep (BOAS) dataset (ds005555)
and validates that the structure is compatible with this pipeline.

Usage:
    python download_dataset.py                          # download to ./data/BOAS
    python download_dataset.py --target-dir /path/to/dir
    python download_dataset.py --validate-only ./data/BOAS
    python download_dataset.py --validate-only ./data/BOAS --test-load

Author: Lennart Gorzel
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_ID = "ds005555"
DEFAULT_TARGET = Path("./data/BOAS")
OPENNEURO_URL = "https://openneuro.org/datasets/ds005555"
EXPECTED_SUBJECTS = 128
REQUIRED_PSG_GLOB = "*_acq-psg_eeg.edf"
EXPECTED_CHANNELS = ["PSG_F3", "PSG_F4", "PSG_C3", "PSG_C4", "PSG_O1", "PSG_O2"]


def install_openneuro_cli():
    print("Installing openneuro-py ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "openneuro-py"],
        stdout=subprocess.DEVNULL,
    )
    print("  Installed successfully.")


def download_with_openneuro(target: Path):
    try:
        import openneuro  # noqa: F401
    except ImportError:
        install_openneuro_cli()

    print(f"\nDownloading BOAS dataset ({DATASET_ID}) to: {target}")
    print("This may take a while depending on your connection (~10-20 GB).\n")

    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "openneuro",
        "download", "--dataset", DATASET_ID, "--target-dir", str(target),
    ]
    subprocess.check_call(cmd)
    print("\nDownload complete.")


def download_with_aws(target: Path):
    aws = shutil.which("aws")
    if not aws:
        return False

    print(f"\nDownloading BOAS dataset via AWS CLI to: {target}")
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        aws, "s3", "sync", "--no-sign-request",
        f"s3://openneuro.org/{DATASET_ID}", str(target),
    ]
    subprocess.check_call(cmd)
    print("\nDownload complete.")
    return True


def normalize_event_files(data_path: Path):
    """Rename *_events.tsv to *_events.txt for pipeline compatibility.

    OpenNeuro distributes BIDS-standard .tsv files, but the pipeline
    expects .txt.  The content is identical (tab-separated), only the
    extension differs.
    """
    renamed = 0
    for tsv in data_path.rglob("*_events.tsv"):
        txt = tsv.with_suffix(".txt")
        if not txt.exists():
            tsv.rename(txt)
            renamed += 1
    if renamed:
        print(f"  Renamed {renamed} event files (.tsv -> .txt) for pipeline compatibility.")


def find_events_file(subject_dir: Path):
    eeg_dir = subject_dir / "eeg"
    for ext in ("*.txt", "*.tsv"):
        matches = list(eeg_dir.glob(f"*_events{ext[1:]}"))
        if matches:
            return matches[0]
    # Try both globs properly
    for glob_pat in ["*_events.txt", "*_events.tsv"]:
        matches = list(eeg_dir.glob(glob_pat))
        if matches:
            return matches[0]
    return None


def validate(data_path: Path, test_load: bool = False):
    print(f"\nValidating dataset at: {data_path}\n")
    errors = []
    warnings = []

    if not data_path.exists():
        print(f"  FAIL: Directory does not exist: {data_path}")
        return False

    subject_dirs = sorted(
        [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    )

    if not subject_dirs:
        print(f"  FAIL: No sub-* directories found in {data_path}")
        return False

    print(f"  Found {len(subject_dirs)} subject directories")
    if len(subject_dirs) < EXPECTED_SUBJECTS:
        warnings.append(
            f"Expected {EXPECTED_SUBJECTS} subjects, found {len(subject_dirs)}"
        )

    missing_eeg_dir = []
    missing_psg = []
    missing_events = []
    tsv_events = []
    ok_count = 0

    for sd in subject_dirs:
        eeg_dir = sd / "eeg"
        if not eeg_dir.exists():
            missing_eeg_dir.append(sd.name)
            continue

        psg_files = list(eeg_dir.glob(REQUIRED_PSG_GLOB))
        if not psg_files:
            missing_psg.append(sd.name)

        events_file = find_events_file(sd)
        if events_file is None:
            missing_events.append(sd.name)
        elif events_file.suffix == ".tsv":
            tsv_events.append(sd.name)

        if psg_files and events_file is not None:
            ok_count += 1

    print(f"  Complete subjects (PSG + events): {ok_count}/{len(subject_dirs)}")

    if missing_eeg_dir:
        errors.append(f"Missing eeg/ subdirectory: {missing_eeg_dir[:5]}{'...' if len(missing_eeg_dir) > 5 else ''}")
    if missing_psg:
        errors.append(f"Missing PSG EDF file: {missing_psg[:5]}{'...' if len(missing_psg) > 5 else ''}")
    if missing_events:
        errors.append(f"Missing events file: {missing_events[:5]}{'...' if len(missing_events) > 5 else ''}")
    if tsv_events:
        errors.append(
            f"{len(tsv_events)} subjects have .tsv event files instead of .txt. "
            f"Run: python download_dataset.py --normalize {data_path}"
        )

    # Show sample subject structure
    sample = subject_dirs[0]
    sample_eeg = sample / "eeg"
    if sample_eeg.exists():
        print(f"\n  Sample structure ({sample.name}/):")
        for f in sorted(sample_eeg.iterdir()):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    eeg/{f.name}  ({size_mb:.1f} MB)")

    # Test-load one subject with the actual pipeline loader
    if test_load and ok_count > 0 and not tsv_events:
        print("\n  Test-loading first subject with BOASDataLoader ...")
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from data_loader_boas import BOASDataLoader

            loader = BOASDataLoader(
                base_path=str(data_path),
                target_channels=EXPECTED_CHANNELS,
                target_sfreq=128.0,
                epoch_duration=30.0,
                use_human_labels=True,
            )
            subjects = loader.list_subjects()
            raw, annotations, metadata = loader.load_subject(
                subjects[0], apply_channel_selection=True, apply_resampling=False
            )
            validation = loader.validate_subject_data(raw, annotations, metadata)
            summary = validation.get("_summary", {})

            print(f"    Subject:   {metadata.subject_id}")
            print(f"    Channels:  {metadata.channels}")
            print(f"    Sfreq:     {metadata.sampling_rate} Hz")
            print(f"    Epochs:    {metadata.n_valid_epochs}/{metadata.n_epochs} valid")
            print(f"    Stages:    {sorted(annotations['stage'].unique().tolist())}")
            print(f"    Checks:    {summary.get('passed', '?')}/{summary.get('total', '?')} passed")

            if not summary.get("all_passed"):
                for issue in summary.get("issues", []):
                    warnings.append(f"Validation: {issue}")
            else:
                print("    Pipeline compatibility: OK")

        except Exception as e:
            errors.append(f"Test-load failed: {e}")
    elif test_load and tsv_events:
        print("\n  Skipping test-load: run --normalize first to fix .tsv extensions.")

    # Summary
    print("\n" + "-" * 50)
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")
    if not errors:
        print("  RESULT: Dataset is valid and compatible with the pipeline.")
        print(f"\n  To run the pipeline:")
        print(f"    python run_full_pipeline.py --quick --data-path \"{data_path}\"")
    else:
        print("  RESULT: Dataset has issues — see errors above.")
    print("-" * 50)

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Download and validate the BOAS dataset from OpenNeuro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python download_dataset.py                              # download + validate
  python download_dataset.py --target-dir ./my_data       # custom location
  python download_dataset.py --validate-only ./data/BOAS  # skip download
  python download_dataset.py --validate-only ./data/BOAS --test-load
  python download_dataset.py --normalize ./data/BOAS      # fix .tsv -> .txt

Dataset: {OPENNEURO_URL}
        """,
    )
    parser.add_argument(
        "--target-dir", type=str, default=str(DEFAULT_TARGET),
        help=f"Where to download the dataset (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--validate-only", type=str, default=None, metavar="PATH",
        help="Skip download, only validate an existing dataset directory",
    )
    parser.add_argument(
        "--normalize", type=str, default=None, metavar="PATH",
        help="Rename .tsv event files to .txt for pipeline compatibility",
    )
    parser.add_argument(
        "--test-load", action="store_true",
        help="After validation, test-load one subject with BOASDataLoader",
    )

    args = parser.parse_args()

    if args.normalize:
        normalize_event_files(Path(args.normalize))
        ok = validate(Path(args.normalize), test_load=args.test_load)
        sys.exit(0 if ok else 1)

    if args.validate_only:
        ok = validate(Path(args.validate_only), test_load=args.test_load)
        sys.exit(0 if ok else 1)

    target = Path(args.target_dir)

    # Download
    try:
        download_with_openneuro(target)
    except Exception as e:
        print(f"\nopenneuro-py download failed: {e}")
        print("Trying AWS CLI fallback ...")
        if not download_with_aws(target):
            print(
                f"\nAutomatic download failed. Please download manually from:\n"
                f"  {OPENNEURO_URL}\n\n"
                f"Then place the dataset in: {target}\n"
                f"And re-run: python download_dataset.py --validate-only {target} --test-load"
            )
            sys.exit(1)

    # Normalize .tsv -> .txt after download
    normalize_event_files(target)

    # Validate
    ok = validate(target, test_load=args.test_load)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
