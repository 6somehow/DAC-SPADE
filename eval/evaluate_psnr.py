import argparse
from pathlib import Path
from multiprocessing import Pool
import os

import cv2
import numpy as np
# Import the specific metric function from scikit-image
from skimage.metrics import peak_signal_noise_ratio
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
def compute_average_psnr(
    ref_cap,
    gen_cap,
    *,
    expected_len: int,
    expected_wh: tuple[int, int],
):
    """Compute average PSNR for two open cv2.VideoCapture objects."""
    # sanity checks
    w, h = expected_wh
    assert int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == w
    assert int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == h
    assert int(gen_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == w
    assert int(gen_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == h

    total_psnr = 0.0
    finite_psnr_frames = 0
    processed_frames = 0

    while True:
        # Read one frame from each video
        ret1, frame1 = ref_cap.read()
        ret2, frame2 = gen_cap.read()

        # If either video has ended, break the loop
        if not ret1 or not ret2:
            break

        processed_frames += 1

        # Calculate PSNR using the scikit-image function.
        psnr_value = peak_signal_noise_ratio(frame1, frame2, data_range=255)

        if psnr_value != float('inf'):
            total_psnr += psnr_value
            finite_psnr_frames += 1

    if processed_frames != expected_len:
        raise RuntimeError(
            f"Expected {expected_len} frames but processed {processed_frames}")

    if finite_psnr_frames == 0:
        # This happens if all frames were identical (or no frames processed)
        return float('inf') if processed_frames > 0 else 0.0

    return total_psnr / finite_psnr_frames


# ---------- worker ----------
def process_pair(pair: tuple[Path, Path]):
    """
    Processes a single pair of videos to compute PSNR.
    Returns a tuple of (status, data).
    status can be 'success', 'mismatch', or 'error'.
    """
    ref_path, gen_path = pair
    ref_cap = None
    gen_cap = None
    try:
        ref_cap, n_frames, w, h = load_video(str(ref_path))
        gen_cap, n_frames2, w2, h2 = load_video(str(gen_path))

        if n_frames != n_frames2 or (w, h) != (w2, h2):
            return "mismatch", ref_path.name

        score = compute_average_psnr(
            ref_cap,
            gen_cap,
            expected_len=n_frames,
            expected_wh=(w, h),
        )
        return "success", score

    except Exception as e:
        return "error", (ref_path.name, e)

    finally:
        if ref_cap:
            ref_cap.release()
        if gen_cap:
            gen_cap.release()


def psnr_dir(ref_root: Path, gen_root: Path, workers: int):
    """
    Computes the average PSNR between videos in two directories.
    Returns a dictionary with scores, mismatches, and errors.
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
    mismatches = []
    errors = []
    # Use multiprocessing to parallelize the evaluation
    with Pool(processes=workers) as pool:
        results = list(
            tqdm(pool.imap_unordered(process_pair, pairs),
                 total=len(pairs),
                 desc="Evaluating PSNR"))

    for status, data in results:
        if status == "success":
            scores.append(data)
        elif status == "mismatch":
            mismatches.append(data)
        elif status == "error":
            name, e = data
            errors.append((name, e))

    return {"scores": scores, "mismatches": mismatches, "errors": errors}


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Compute average PSNR between two sets of videos.")
    parser.add_argument(
        "--ref-dir",
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

    print(f"Using {args.workers} worker processes for PSNR evaluation.")
    try:
        results = psnr_dir(args.ref_dir, args.gen_dir, args.workers)
    except FileNotFoundError as e:
        print(e)
        return

    scores = results["scores"]
    mismatches = results["mismatches"]
    errors = results["errors"]

    if mismatches:
        print("\n--- Skipped Videos ---")
        for name in mismatches:
            print(f"Skipping {name} due to property mismatch.")
    if errors:
        print("\n--- Errors ---")
        for name, e in errors:
            print(f"❌ Error processing {name}: {e}")

    # --- Summary ---
    finite_scores = [s for s in scores if s != float('inf')]
    inf_count = len(scores) - len(finite_scores)

    print("\n--- Summary ---")
    if scores:
        print(f"Pairs evaluated: {len(scores)}")
        if inf_count > 0:
            print(f"  ({inf_count} pairs were identical, PSNR = inf)")

        if finite_scores:
            average_psnr = sum(finite_scores) / len(finite_scores)
            print(
                f"Overall average PSNR (for {len(finite_scores)} pairs with finite scores): {average_psnr:.4f} dB"
            )
        elif inf_count > 0:
            print("All successfully evaluated pairs were identical.")
    else:
        print("No successful evaluations.")


if __name__ == "__main__":
    main()
