import argparse
import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)


def _get_language_model(model):
    if hasattr(model, "language_model"):
        return model.language_model

    nested_model = getattr(model, "model", None)
    if nested_model is not None and hasattr(nested_model, "language_model"):
        return nested_model.language_model

    raise AttributeError(
        "Could not find the language model inside LlavaForConditionalGeneration. "
        "Expected either `model.language_model` or `model.model.language_model`."
    )


def preprocess_text_encoder_tokenizer(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.input_dir)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.input_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    language_model = _get_language_model(model)
    language_model.save_pretrained(
        f"{args.output_dir}"
    )
    processor.tokenizer.save_pretrained(
        f"{args.output_dir}"
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="The path to the llava-llama-3-8b-v1_1-transformers.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="The output path of the llava-llama-3-8b-text-encoder-tokenizer."
        "if '', the parent dir of output will be the same as input dir.",
    )
    args = parser.parse_args()

    if len(args.output_dir) == 0:
        args.output_dir = "/".join(args.input_dir.split("/")[:-1])

    preprocess_text_encoder_tokenizer(args)
