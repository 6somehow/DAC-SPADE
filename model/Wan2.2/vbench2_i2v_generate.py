# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
import json
import math

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image
from shutil import copyfile

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import save_video, str2bool
from wan.configs.spade_patch import spade_engine, is_spade_engine

EXAMPLE_PROMPT = {
    "i2v-A14B": {
        "prompt":
        "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image": "examples/i2v_input.JPG",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Generate a image or video from a text prompt or image using Wan")
    parser.add_argument("--task",
                        type=str,
                        default="i2v-A14B",
                        choices=list(WAN_CONFIGS.keys()),
                        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help=
        "The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help=
        "How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument("--ckpt_dir",
                        type=str,
                        default=None,
                        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help=
        "Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument("--ulysses_size",
                        type=int,
                        default=1,
                        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument("--t5_fsdp",
                        action="store_true",
                        default=False,
                        help="Whether to use FSDP for T5.")
    parser.add_argument("--t5_cpu",
                        action="store_true",
                        default=False,
                        help="Whether to place T5 model on CPU.")
    parser.add_argument("--dit_fsdp",
                        action="store_true",
                        default=False,
                        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument("--src_video",
                        type=str,
                        default=None,
                        help="The file of the source video. Default None.")
    parser.add_argument("--src_mask",
                        type=str,
                        default=None,
                        help="The file of the source mask. Default None.")
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help=
        "The file list of the source reference images. Separated by ','. Default None."
    )
    parser.add_argument("--prompt",
                        type=str,
                        default=None,
                        help="The prompt to generate the image or video from.")
    parser.add_argument("--use_prompt_extend",
                        action="store_true",
                        default=False,
                        help="Whether to use prompt extend.")
    parser.add_argument("--prompt_extend_method",
                        type=str,
                        default="local_qwen",
                        choices=["dashscope", "local_qwen"],
                        help="The prompt extend method to use.")
    parser.add_argument("--prompt_extend_model",
                        type=str,
                        default=None,
                        help="The prompt extend model to use.")
    parser.add_argument("--prompt_extend_target_lang",
                        type=str,
                        default="zh",
                        choices=["zh", "en"],
                        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="[image to video] The image to generate the video from.")
    parser.add_argument(
        "--first_frame",
        type=str,
        default=None,
        help=
        "[first-last frame to video] The image (first frame) to generate the video from."
    )
    parser.add_argument(
        "--last_frame",
        type=str,
        default=None,
        help=
        "[first-last frame to video] The image (last frame) to generate the video from."
    )
    parser.add_argument("--sample_solver",
                        type=str,
                        default='unipc',
                        choices=['unipc', 'dpm++'],
                        help="The solver used to sample.")
    parser.add_argument("--sample_steps",
                        type=int,
                        default=None,
                        help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument("--sample_guide_scale",
                        type=float,
                        default=None,
                        help="Classifier free guidance scale.")
    parser.add_argument("--convert_model_dtype",
                        action="store_true",
                        default=False,
                        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--vbench_i2v_root",
        type=str,
        default=None,
        help=
        "Root directory for VBench I2V prompts. If specified, runs VBench I2V evaluation."
    )
    parser.add_argument("--vbench_output_dir",
                        type=str,
                        default="vbench2_i2v_wan22_outputs",
                        help="Output directory for VBench I2V videos.")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl",
                                init_method="env://",
                                rank=rank,
                                world_size=world_size)

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.vbench_i2v_root:
        assert "i2v" in args.task, "VBench I2V evaluation only supports I2V tasks."
        if rank == 0:
            logging.info("Running VBench I2V evaluation...")

        cfg = WAN_CONFIGS[args.task]
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        dimension_list = ["i2v_subject", "i2v_background", "camera_motion"]
        info_path = os.path.join(args.vbench_i2v_root,
                                 "vbench2_i2v_full_info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(
                f"VBench I2V info file not found at {info_path}")
        info_list = json.load(open(info_path, "r"))

        width, height = [int(x) for x in args.size.split('*')]
        common_divisor = math.gcd(width, height)
        aspect_ratio = f"{width//common_divisor}-{height//common_divisor}"
        image_folder = os.path.join(args.vbench_i2v_root,
                                    f"data/crop/{aspect_ratio}")
        if rank == 0:
            print('image_folder', image_folder)
        if not os.path.isdir(image_folder):
            raise FileNotFoundError(
                f"VBench I2V image folder not found at {image_folder}")

        os.makedirs(args.vbench_output_dir, exist_ok=True)
        copyfile('./wan/configs/spade_patch.py',
                 os.path.join(args.vbench_output_dir, 'spade_patch.py'))

        dimension_sample_len = 14
        for dimension in dimension_list:
            if rank == 0:
                logging.info(f"Processing dimension: {dimension}")

            inputs = [(os.path.join(image_folder,
                                    info["image_name"]), info["prompt_en"])
                      for info in info_list if dimension in info["dimension"]]

            for image_path, prompt in inputs[:dimension_sample_len]:
                if not os.path.exists(image_path):
                    logging.warning(
                        f"Image not found: {image_path}, skipping.")
                    continue

                img = Image.open(image_path).convert("RGB")
                sample_len = 1

                for index in range(sample_len):
                    current_seed = args.base_seed + index
                    if rank == 0:
                        logging.info(
                            f"Generating video {index+1}/{sample_len} for prompt: {prompt[:80]}..."
                        )
                    if is_spade_engine and spade_engine is not None:
                        spade_engine.reset()

                    prompt_for_filename = prompt.replace(" ", "_").replace(
                        "/", "_")
                    save_dir = os.path.join(args.vbench_output_dir, dimension)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(
                        save_dir, f'{prompt_for_filename[:180]}-{index}.mp4')

                    if os.path.exists(save_path):
                        logging.info(
                            f"Video {save_path} already exists, skip.")
                        continue

                    video = wan_i2v.generate(
                        prompt,
                        img,
                        max_area=MAX_AREA_CONFIGS[args.size],
                        frame_num=args.frame_num,
                        shift=args.sample_shift,
                        sample_solver=args.sample_solver,
                        sampling_steps=args.sample_steps,
                        guide_scale=args.sample_guide_scale,
                        seed=current_seed,
                        offload_model=args.offload_model)

                    logging.info(f"Saving generated video to {save_path}")
                    save_video(tensor=video[None],
                               save_file=save_path,
                               fps=cfg.sample_fps,
                               nrow=1,
                               normalize=True,
                               value_range=(-1, 1))
                    del video
        logging.info("Finished.")
        return


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
