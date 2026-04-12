import argparse
from pathlib import Path
import multiprocessing
import os

import cv2
import numpy as np
# Import the specific metric function from scikit-image
from skimage.metrics import structural_similarity
from tqdm import tqdm


# ---------- helpers ----------
def load_video(path: str):
    """Return cv2.VideoCapture and its metadata (frame_count, w, h)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open {path}")
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, frame_cnt, w, h


# ---------- core ----------
def compute_average_ssim(
    ref_cap,
    gen_cap,
    *,
    expected_len: int,
    expected_wh: tuple[int, int],
):
    """Compute average SSIM for two open cv2.VideoCapture objects."""
    # sanity checks
    w, h = expected_wh
    assert int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == w
    assert int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == h
    assert int(gen_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == w
    assert int(gen_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == h

    total_ssim = 0.0
    processed_frames = 0

    while True:
        # Read one frame from each video
        ret1, frame1 = ref_cap.read()
        ret2, frame2 = gen_cap.read()

        # If either video has ended, break the loop
        if not ret1 or not ret2:
            break

        processed_frames += 1

        # --- SSIM CALCULATION ---
        ssim_value = structural_similarity(frame1,
                                           frame2,
                                           channel_axis=2,
                                           data_range=255)
        total_ssim += ssim_value

    if processed_frames != expected_len:
        raise RuntimeError(
            f"Expected {expected_len} frames but processed {processed_frames}")

    if processed_frames == 0:
        return 0.0

    return total_ssim / processed_frames


def evaluate_pair(pair: tuple[Path, Path]):
    """
    Loads a pair of videos, computes SSIM, and returns the result.
    The result is a tuple of (status, value), where status is one of
    'ok', 'skip', or 'error'.
    """
    ref_path, gen_path = pair
    ref_cap = None
    gen_cap = None
    try:
        ref_cap, n_frames, w, h = load_video(str(ref_path))
        gen_cap, n_frames2, w2, h2 = load_video(str(gen_path))

        if n_frames != n_frames2 or (w, h) != (w2, h2):
            return 'skip', ref_path.name

        score = compute_average_ssim(
            ref_cap,
            gen_cap,
            expected_len=n_frames,
            expected_wh=(w, h),
        )
        return 'ok', score

    except Exception as e:
        return 'error', (ref_path.name, e)

    finally:
        if ref_cap:
            ref_cap.release()
        if gen_cap:
            gen_cap.release()


def ssim_dir(ref_root: Path, gen_root: Path, workers: int):
    """
    Computes and prints the average SSIM between videos in two directories.
    Returns a dictionary with scores, skips, and errors.
    """
    # collect valid pairs
    pairs: list[tuple[Path, Path]] = []
    for ref_path in ref_root.rglob("*.mp4"):
        gen_path = gen_root / ref_path.relative_to(ref_root)
        if gen_path.exists():
            pairs.append((ref_path, gen_path))

    if not pairs:
        raise FileNotFoundError(
            f"No matching video pairs found in {ref_root} and {gen_root}.")

    scores = []
    errors = []
    mismatches = []
    with multiprocessing.Pool(processes=workers) as pool:
        with tqdm(total=len(pairs), desc="Evaluating SSIM") as pbar:
            for status, result in pool.imap_unordered(evaluate_pair, pairs):
                if status == 'ok':
                    scores.append(result)
                elif status == 'skip':
                    mismatches.append(result)
                elif status == 'error':
                    errors.append(result)
                pbar.update()

    return {"scores": scores, "mismatches": mismatches, "errors": errors}


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Compute average SSIM between two sets of videos.")
    parser.add_argument("--ref-dir",
                        type=Path,
                        required=True,
                        help="Directory with reference videos.")
    parser.add_argument("--gen-dir",
                        type=Path,
                        help="Directory with generated videos.")
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=os.cpu_count(),
        help=
        "Number of worker processes to use. Defaults to all available CPUs.",
    )
    args = parser.parse_args()

    print(f"Using {args.workers} worker processes.")
    try:
        results = ssim_dir(args.ref_dir, args.gen_dir, args.workers)
    except FileNotFoundError as e:
        print(e)
        return

    scores = results["scores"]
    mismatches = results["mismatches"]
    errors = results["errors"]

    # --- Summary ---
    if mismatches:
        print("\n--- Skipped Videos ---")
        for name in mismatches:
            print(f"Skipping {name} due to property mismatch.")
    if errors:
        print("\n--- Errors ---")
        for name, e in errors:
            print(f"❌ Error processing {name}: {e}")

    print("\n--- Summary ---")
    if scores:
        print(f"Pairs evaluated: {len(scores)}")
        average_ssim = sum(scores) / len(scores)
        print(f"Overall average SSIM: {average_ssim:.6f}")
    else:
        print("No successful evaluations.")


if __name__ == "__main__":
    main()
