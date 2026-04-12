import argparse
import os
from pathlib import Path
import multiprocessing

import cv2
import lpips
import torch
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


def frame_to_tensor(frame, device):
    """Convert BGR uint8 frame -> RGB float32 tensor in [-1, 1] (B,C,H,W)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return lpips.im2tensor(rgb).to(device)


# ---------- core ----------
def compute_average_lpips(
    ref_cap,
    gen_cap,
    loss_fn,
    device,
    *,
    expected_len: int,
    expected_wh: tuple[int, int],
    batch_size: int = 1,
):
    """Compute average LPIPS for two open cv2.VideoCapture objects."""
    # sanity checks
    w, h = expected_wh
    assert int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == w
    assert int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == h
    assert int(gen_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == w
    assert int(gen_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == h

    total = 0.0
    processed = 0
    ref_frames_cache = []
    gen_frames_cache = []

    def process_batch():
        nonlocal total, processed
        if not ref_frames_cache:
            return

        ref_batch = torch.cat(ref_frames_cache, dim=0)
        gen_batch = torch.cat(gen_frames_cache, dim=0)
        with torch.no_grad():
            # loss_fn returns [N,1,1,1], sum over N
            losses = loss_fn(ref_batch, gen_batch)
            total += losses.sum().item()

        processed += len(ref_frames_cache)
        ref_frames_cache.clear()
        gen_frames_cache.clear()

    while True:
        ok1, frm1 = ref_cap.read()
        ok2, frm2 = gen_cap.read()
        if not ok1 or not ok2:
            break
        ref_frames_cache.append(frame_to_tensor(frm1, device))
        gen_frames_cache.append(frame_to_tensor(frm2, device))
        if len(ref_frames_cache) >= batch_size:
            process_batch()

    process_batch()  # process remaining frames

    if processed != expected_len:
        raise RuntimeError(
            f"Expected {expected_len} frames but processed {processed}")
    return total / processed


def evaluate_pair_lpips(pair: tuple[Path, Path]):
    """
    Processes a single pair of videos to compute LPIPS on CPU.
    Returns a tuple of (status, data).
    status can be 'success', 'mismatch', or 'error'.
    """
    ref_path, gen_path = pair
    ref_cap = None
    gen_cap = None
    try:
        # Each worker process has its own model instance on the CPU
        device = torch.device("cpu")
        loss_fn = lpips.LPIPS(net="alex").to(device)

        ref_cap, n_frames, w, h = load_video(str(ref_path))
        gen_cap, n_frames2, w2, h2 = load_video(str(gen_path))

        if n_frames != n_frames2 or (w, h) != (w2, h2):
            return "mismatch", ref_path.name

        score = compute_average_lpips(
            ref_cap,
            gen_cap,
            loss_fn,
            device,
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


def evaluate_pair_lpips_cuda(
    pair: tuple[Path, Path],
    loss_fn: lpips.LPIPS,
    device: torch.device,
    batch_size: int,
):
    """
    Processes a single pair of videos to compute LPIPS on GPU with batching.
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

        score = compute_average_lpips(
            ref_cap,
            gen_cap,
            loss_fn,
            device,
            expected_len=n_frames,
            expected_wh=(w, h),
            batch_size=batch_size,
        )
        return "success", score

    except Exception as e:
        return "error", (ref_path.name, e)

    finally:
        if ref_cap:
            ref_cap.release()
        if gen_cap:
            gen_cap.release()


def lpips_dir(
    ref_root: Path,
    gen_root: Path,
    workers: int,
    *,
    use_gpu: bool,
    batch_size: int,
):
    """
    Computes the average LPIPS between videos in two directories.
    If use_gpu is False, it uses multiprocessing on CPU.
    If use_gpu is True, it uses a single GPU with batching.
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
    errors = []
    mismatches = []

    if not use_gpu:
        with multiprocessing.Pool(processes=workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(evaluate_pair_lpips, pairs),
                    total=len(pairs),
                    desc="Evaluating LPIPS (CPU)",
                ))
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available, but --use-gpu was specified.")
        device = torch.device("cuda")
        loss_fn = lpips.LPIPS(net="alex").to(device)
        results = []
        for pair in tqdm(pairs, desc="Evaluating LPIPS (GPU)"):
            results.append(
                evaluate_pair_lpips_cuda(pair, loss_fn, device, batch_size))

    for status, data in results:
        if status == "success":
            scores.append(data)
        elif status == "mismatch":
            mismatches.append(data)
        elif status == "error":
            name, e = data
            errors.append((name, e))

    return {"scores": scores, "mismatches": mismatches, "errors": errors}


def lpips_dir_cuda(ref_root: Path, gen_root: Path, gpus: list[int]):
    """
    Computes the average LPIPS between videos in two directories
    using multiprocessing on specified GPUs.
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

    # Assign pairs to GPUs
    tasks = []
    for i, pair in enumerate(pairs):
        gpu_id = gpus[i % len(gpus)]
        tasks.append((pair, gpu_id))

    scores = []
    errors = []
    mismatches = []
    # Use 'spawn' to avoid CUDA initialization issues in forked processes
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=len(gpus)) as pool:
        results = list(
            tqdm(pool.imap_unordered(evaluate_pair_lpips_cuda, tasks),
                 total=len(tasks),
                 desc="Evaluating LPIPS on GPUs"))

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
        description="Compute average LPIPS between two sets of videos.")
    parser.add_argument(
        "--ref-dir",
        type=Path,
        required=True,
        help="Directory with reference videos.")
    parser.add_argument("--gen-dir",
                        type=Path,
                        help="Directory with generated videos.")
    parser.add_argument("--use-gpu",
                        action="store_true",
                        help="Use GPU for evaluation.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for GPU evaluation.",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=os.cpu_count(),
        help=
        "Number of worker processes to use (CPU evaluation). Defaults to all available CPUs.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=
        "Comma-separated list of GPU IDs to use for evaluation (e.g., '0,1').")

    args = parser.parse_args()

    if args.use_gpu:
        print(
            f"Using GPU for LPIPS evaluation with batch size {args.batch_size}."
        )
    else:
        print(
            f"Using {args.workers} worker processes for LPIPS evaluation on CPU."
        )
    try:
        results = lpips_dir(
            args.ref_dir,
            args.gen_dir,
            args.workers,
            use_gpu=args.use_gpu,
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, RuntimeError) as e:
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

    if scores:
        print("\n--- Summary ---")
        print(f"Pairs evaluated: {len(scores)}")
        average_lpips = sum(scores) / len(scores)
        print(f"Overall average LPIPS: {average_lpips:.6f}")
    else:
        print("\nNo successful evaluations.")


if __name__ == "__main__":
    main()
