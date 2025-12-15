# This file is derived from:
# https://github.com/ByteDance-Seed/FlexPrefill/blob/main/tests/test_llm.py
# - test vllm using llama model patched with xattention

# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import warnings
from argparse import ArgumentParser
from ast import literal_eval
from time import time
from vllm import LLM, SamplingParams

import torch

from xattn.src.patch import patch_model

warnings.filterwarnings("ignore")


def str_to_dict(cfg: str):
    cfg = [c.strip() for c in cfg.split(",") if c.strip() != ""]
    cfg = {c.split("=")[0]: c.split("=")[1] if "=" in c else True for c in cfg}
    for k, v in cfg.items():
        try:
            cfg[k] = literal_eval(v)
        except:
            cfg[k] = v
    return cfg


def get_args():
    # cli args
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="model/Llama-3.1-8B-Instruct",
        help="model for test, default to model/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="default",
        choices=[
            "default",
            "flash",
            "xattn",
        ],
        help="attention pattern, default to `default`",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="attention config, a string like 'cfg1=value1,cfg2=value2'",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=str,
        choices=["4k", "8k", "16k", "32k", "64k", "128k", "all"],
        default="all",
        help="select a test length, default to all",
    )
    parser.add_argument(
        "-n",
        "--max_new_tokens",
        type=int,
        default=64,
        help="max new tokens, default to 64",
    )
    args = parser.parse_args()
    return args


def test_llm():
    args = get_args()
    # load data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(current_dir, "example_data.json"), "r", encoding="utf-8"
    ) as f:
        data = json.load(f)
    # load model

    model = LLM(args.model, enable_chunked_prefill=False, max_num_seqs=1)
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    # replace attention
    attention_pattern = args.pattern
    attention_config = (
        str_to_dict(args.config)
        if args.config is not None
        else {}
    )
    patch_model(model, attention_pattern, attention_config)
    print(json.dumps(attention_config, ensure_ascii=False, indent=2))

    print_str = f"# Model Test: {args.model} #"
    print("#" * len(print_str))
    print(print_str)
    print("#" * len(print_str))

    # warmup triton kernel
    model.generate(
        prompts=[data[5]["prompt"]], sampling_params=sampling_params, use_tqdm=False
    )
    torch.cuda.empty_cache()

    len2idx = {
        "4k": 0,
        "8k": 1,
        "16k": 2,
        "32k": 3,
        "64k": 4,
        "128k": 5,
    }
    if args.length != "all":
        data = data[len2idx[args.length] : len2idx[args.length] + 1]

    for i, d in enumerate(data):
        # generate output
        torch.cuda.synchronize()
        time_start = time()
        outputs = model.generate(
            prompts=[d["prompt"]],
            sampling_params=sampling_params,
            use_tqdm=False
        )
        prompt_len = len(outputs[0].prompt_token_ids)
        output = outputs[0].outputs[0].text
        output_tokens = len(outputs[0].outputs[0].token_ids)
        torch.cuda.synchronize()
        time_end = time()

        print(f"Input {i}")
        print(f" - Prompt Length: {prompt_len}")
        print(f" - Output Length: {output_tokens}")
        print(f" - Total Seconds: {time_end - time_start}")
        print(f" - Model Output: {output}")
        print(f" - True Answer: {d['answer']}")
        print()


if __name__ == "__main__":
    test_llm()
