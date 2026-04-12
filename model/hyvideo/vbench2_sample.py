import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime
import argparse
from shutil import copyfile

from hyvideo.config import parse_args
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.inference import HunyuanVideoSampler

from hyvideo.modules.spade_patch import spade_engine, is_spade_engine


def run_single(args, hunyuan_video_sampler, save_path):
    # Start sampling
    # TODO: batch inference check
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale)
    samples = outputs['samples']

    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(
                time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            cur_save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, cur_save_path, fps=24)
            logger.info(f'Sample save to: {cur_save_path}')


def run_vbench(args, hunyuan_video_sampler):
    logger.info("Running VBench evaluation...")

    dimension_list = [
        'Complex_Plot', 'Camera_Motion', 'Human_Anatomy', 'Human_Identity',
        'Human_Clothes', 'Diversity', 'Composition',
        'Dynamic_Spatial_Relationship', 'Dynamic_Attribute',
        'Motion_Order_Understanding', 'Human_Interaction', 'Complex_Landscape',
        'Motion_Rationality', 'Instance_Preservation', 'Mechanics',
        'Thermotics', 'Material', 'Multi-View_Consistency'
    ]

    # For HyVideo, use hunyuan_aug_prompt
    prompt_folder = os.path.join(
        args.vbench_root, 'VBench-2.0/prompts/prompt_aug/VBench2_aug_prompt')
    short_prompt_folder = os.path.join(args.vbench_root,
                                       'VBench-2.0/prompts/prompt')

    if not os.path.isdir(prompt_folder):
        logger.warning(
            f"VBench augmented prompt folder not found at {prompt_folder}. Trying the original prompt folder."
        )
        prompt_folder = short_prompt_folder

    if not os.path.isdir(short_prompt_folder):
        raise FileNotFoundError(
            f"VBench short prompt folder not found at {short_prompt_folder}. Please check --vbench_root."
        )

    any_dimension_sample_len = 1
    Human_Anatomy_sample_len = 1

    os.makedirs(args.vbench_output_dir, exist_ok=True)
    copyfile('hyvideo/modules/spade_patch.py',
             os.path.join(args.vbench_output_dir, 'spade_patch.py'))

    for dimension in dimension_list:
        if 'LOCAL_RANK' not in os.environ or int(
                os.environ['LOCAL_RANK']) == 0:
            logger.info(f"Processing dimension: {dimension}")

        short_prompt_path = os.path.join(short_prompt_folder,
                                         f'{dimension}.txt')
        with open(short_prompt_path, 'r', encoding='utf-8') as f:
            prompt_list_short = [p.strip() for p in f.readlines()]

        aug_prompt_path = os.path.join(prompt_folder, f'{dimension}.txt')
        try:
            with open(aug_prompt_path, 'r', encoding='utf-8') as f:
                prompt_list = [p.strip() for p in f.readlines()]
        except FileNotFoundError:
            logger.warning(
                f"Augmented prompt file not found for {dimension}, using short prompts instead."
            )
            prompt_list = prompt_list_short

        dimension_sample_len = any_dimension_sample_len
        if dimension == 'Human_Anatomy':
            dimension_sample_len = Human_Anatomy_sample_len

        for idx, prompt in enumerate(prompt_list[:dimension_sample_len]):
            if dimension == 'Diversity':
                iter_num = 1
            else:
                iter_num = 1

            for index in range(iter_num):
                current_seed = args.seed + index

                if 'LOCAL_RANK' not in os.environ or int(
                        os.environ['LOCAL_RANK']) == 0:
                    logger.info(
                        f"Generating video {index+1}/{iter_num} for prompt: {prompt_list_short[idx][:80]}..."
                    )
                if is_spade_engine:
                    spade_engine.reset()

                prompt_for_filename = prompt_list_short[idx]
                save_dir = os.path.join(args.vbench_output_dir, dimension)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(
                    save_dir,
                    f"{prompt_for_filename[:180].replace('/','')}-{index}.mp4")

                if os.path.exists(save_path):
                    logger.info(f"Video {save_path} already exists, skip.")
                    continue

                outputs = hunyuan_video_sampler.predict(
                    prompt=prompt,
                    height=args.video_size[0],
                    width=args.video_size[1],
                    video_length=args.video_length,
                    seed=current_seed,
                    negative_prompt=args.neg_prompt,
                    infer_steps=args.infer_steps,
                    guidance_scale=args.cfg_scale,
                    num_videos_per_prompt=1,
                    flow_shift=args.flow_shift,
                    batch_size=args.batch_size,
                    embedded_guidance_scale=args.embedded_cfg_scale)

                sample = outputs['samples'][0].unsqueeze(0)

                logger.info(f"Saving generated video to {save_path}")
                save_videos_grid(sample, save_path, fps=24)

            if os.path.exists(
                    os.path.join(args.vbench_output_dir, 'sparse_rate.pt')):
                for pt_file in ('sparse_rate.pt', 'record_attn.pt',
                                'record_sim.pt'):
                    copyfile(os.path.join(args.vbench_output_dir, pt_file),
                             os.path.join(save_dir, pt_file))


def main():
    args = parse_args()
    # if args.neg_prompt is None:
    #     args.neg_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path) and not args.vbench_root:
        os.makedirs(save_path)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
        models_root_path, args=args)

    # Get the updated args
    args = hunyuan_video_sampler.args

    if args.vbench_root:
        run_vbench(args, hunyuan_video_sampler)
    else:
        run_single(args, hunyuan_video_sampler, save_path)


if __name__ == "__main__":
    main()
