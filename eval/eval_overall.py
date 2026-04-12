import os
import argparse
import json

from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import sys

from evaluate_lpips import lpips_dir
from evaluate_psnr import psnr_dir
from evaluate_ssim import ssim_dir

dir_names = ('spade', 'spade-turbo', 'sparge', 'svg', 'svg2')
baseline_dir_name = 'origin'

vbench_prefix = 'vbench_'
model_dir_names = (vbench_prefix + 'i2v_wan21', vbench_prefix + 't2v_wan21',
                   vbench_prefix + 'i2v_wan22', vbench_prefix + 't2v_wan22',
                   vbench_prefix + 't2v_hyvideo')

eval_sim_func = (('lpips', lpips_dir), ('ssim', ssim_dir), ('psnr', psnr_dir))

vbench_dimension = {
    't2v':
    ('subject_consistency', 'background_consistency', 'motion_smoothness',
     'dynamic_degree', 'aesthetic_quality', 'imaging_quality'),
    'i2v':
    ('subject_consistency', 'background_consistency', 'motion_smoothness',
     'dynamic_degree', 'aesthetic_quality', 'imaging_quality')
}


def check_mp4_files_same(directory: str):
    root_path = Path(directory)
    baseline_path = root_path / baseline_dir_name

    if not baseline_path.exists():
        print(
            f"Warning: Baseline directory '{baseline_path}' not found. Skipping file check."
        )
        return

    baseline_model_dirs = [
        d for d in baseline_path.iterdir()
        if d.is_dir() and d.name in model_dir_names
    ]

    all_match = True
    for b_model_dir in baseline_model_dirs:
        baseline_files = {f.name for f in b_model_dir.glob('*.mp4')}

        for dir_name in dir_names:
            gen_dir = root_path / dir_name / b_model_dir.name
            if not gen_dir.exists():
                continue

            gen_files = {f.name for f in gen_dir.glob('*.mp4')}
            if baseline_files != gen_files:
                all_match = False
                print(
                    f"File mismatch detected for '{gen_dir}' compared to '{b_model_dir}':"
                )
                missing_files = baseline_files - gen_files
                extra_files = gen_files - baseline_files
                if missing_files:
                    print(f"  Missing files: {sorted(list(missing_files))}")
                if extra_files:
                    print(f"  Extra files: {sorted(list(extra_files))}")

    if all_match:
        print("MP4 file name check completed: All files match.")
    else:
        print("MP4 file name check completed: Mismatches found.")


def eval_vbench_item(my_vbench: VBench, eval_vbench_dim: str,
                     eval_input_video_dir_path: Path, json_name: str,
                     vbench_output_dir: Path):
    my_vbench.evaluate(videos_path=str(eval_input_video_dir_path),
                       name=json_name,
                       dimension_list=[eval_vbench_dim],
                       local=True,
                       read_frame=False,
                       mode='custom_input')

    with open(vbench_output_dir / (json_name + '_eval_results.json'),
              'r') as f:
        result = json.load(f)

    score = result[eval_vbench_dim][0]
    return score


def load_vbench(vbench_root: Optional[Path]):
    if vbench_root is not None:
        sys.path.insert(0, str(vbench_root))

    try:
        from vbench import VBench
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import VBench. Install VBench, pass --vbench-root, or set VBENCH_ROOT."
        ) from exc
    return VBench


def eval_vbench(root_dir: str, workers: int, vbench_json_path: Path,
                vbench_root: Optional[Path]):

    root_path = Path(root_dir)

    eval_vbench = {}
    vbench_cls = load_vbench(vbench_root)

    cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    vbench_output_dir = cur_dir / f'vbench_result_{datetime.now().strftime("%S%M%H_%d%m%Y")}'
    os.makedirs(vbench_output_dir, exist_ok=True)

    device = torch.device("cuda")

    for dir_name in (dir_names + (baseline_dir_name, )):
        dir_path = root_path / dir_name
        eval_vbench[dir_name] = {}

        if dir_path.exists():
            sub_dirs = [d.name for d in dir_path.iterdir() if d.is_dir()]
            for sub_dir in sub_dirs:
                eval_vbench[dir_name][sub_dir] = {}
                eval_input_video_dir_path = dir_path / sub_dir
                assert sub_dir in model_dir_names, f"{sub_dir} not in {model_dir_names}"

                if 'i2v' in sub_dir:
                    my_vbench = vbench_cls(device, str(vbench_json_path),
                                           vbench_output_dir)
                    dimensions = vbench_dimension['i2v']
                else:
                    my_vbench = vbench_cls(device, str(vbench_json_path),
                                           vbench_output_dir)
                    dimensions = vbench_dimension['t2v']

                for eval_vbench_dim in dimensions:
                    print(
                        f"Evaluating {eval_vbench_dim} for {sub_dir} in {dir_name}..."
                    )
                    name = f'{dir_name}_{sub_dir}_{eval_vbench_dim}'

                    eval_vbench[dir_name][sub_dir][
                        eval_vbench_dim] = eval_vbench_item(
                            my_vbench, eval_vbench_dim,
                            eval_input_video_dir_path, name, vbench_output_dir)

        else:
            raise FileNotFoundError(f'{dir_path} does not exist')
    return eval_vbench


def eval_sim(root_dir: str, workers: int, batch_size: int):

    root_path = Path(root_dir)
    origin_dir = root_path / baseline_dir_name

    eval_similarity = {}

    for dir_name in dir_names:
        dir_path = root_path / dir_name
        eval_similarity[dir_name] = {}

        if dir_path.exists():
            sub_dirs = [d.name for d in dir_path.iterdir() if d.is_dir()]
            for sub_dir in sub_dirs:
                eval_similarity[dir_name][sub_dir] = {}
                assert sub_dir in model_dir_names, f"{sub_dir} not in {model_dir_names}"
                for eval_func_name, eval_func_dir in eval_sim_func:
                    print(
                        f"Evaluating {eval_func_name} for {sub_dir} in {dir_name}..."
                    )
                    ref_dir = origin_dir / sub_dir
                    gen_dir = dir_path / sub_dir
                    if eval_func_name == 'lpips':
                        res = eval_func_dir(ref_dir,
                                            gen_dir,
                                            workers=workers,
                                            use_gpu=True,
                                            batch_size=batch_size)
                    else:
                        res = eval_func_dir(ref_dir, gen_dir, workers)
                    if res['errors']:
                        raise RuntimeError(
                            f"Errors during {eval_func_name} evaluation for {sub_dir}: {res['errors']}"
                        )
                    eval_similarity[dir_name][sub_dir][eval_func_name] = sum(
                        res['scores']) / len(res['scores'])

        else:
            raise FileNotFoundError(f'{dir_path} does not exist')
    return eval_similarity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, required=True)
    parser.add_argument(
        "--vbench-root",
        type=Path,
        default=os.environ.get("VBENCH_ROOT"),
        help="Optional path to a local VBench checkout to add to PYTHONPATH.",
    )
    parser.add_argument(
        "--vbench-json",
        type=Path,
        default=os.environ.get("VBENCH_JSON"),
        help="Path to VBench_full_info.json. Can also be set with VBENCH_JSON.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Batch size for GPU LPIPS evaluation.",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=os.cpu_count(),
        help=
        "Number of worker processes to use. Defaults to all available CPUs.",
    )
    args = parser.parse_args()
    if args.vbench_json is None:
        parser.error("--vbench-json is required unless VBENCH_JSON is set")

    current_time = datetime.now().strftime("%S%M%H_%d%m%Y")
    check_mp4_files_same(args.dir)
    sim_res = eval_sim(args.dir, args.workers, args.batch_size)

    with open(f'sim_{current_time}.json', 'w') as f:
        json.dump(sim_res, f, indent=4)

    vbench_res = eval_vbench(args.dir, args.workers, args.vbench_json,
                             args.vbench_root)

    with open(f'vbench_{current_time}.json', 'w') as f:
        json.dump(vbench_res, f, indent=4)
