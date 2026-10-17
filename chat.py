#!/usr/bin/env python3
"""
Try a model on a prompt. Works with your merged model and, for comparison, the
official DeepSeek distill.

    python chat.py --model out/merged "What is 17 * 24?"
    python chat.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --template tokenizer "What is 17 * 24?"

--template chatml     the format train.py trains on (use this for your own model)
--template tokenizer  the model's built-in chat template (use this for the official distill)
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default="What is the sum of all positive divisors of 36?")
    ap.add_argument("--model", default="out/merged")
    ap.add_argument("--template", choices=["chatml", "tokenizer"], default="chatml")
    ap.add_argument("--max-new-tokens", type=int, default=2048,
                    help="raise this if the answer is cut off in the middle of its thinking")
    ap.add_argument("--temperature", type=float, default=0.6)  # DeepSeek recommends 0.5-0.7
    ap.add_argument("--top-p", type=float, default=0.95)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    if torch.cuda.is_available():
        # Older GPUs emulate bf16 slowly, so use fp16 on them.
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    try:  # newer transformers call this argument `dtype`
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, device_map="auto")
    except TypeError:  # older releases only know `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, device_map="auto")

    question = a.prompt + SUFFIX
    if a.template == "chatml":
        text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    else:
        text = tok.apply_chat_template([{"role": "user", "content": question}],
                                       tokenize=False, add_generation_prompt=True)
    # DeepSeek recommends forcing the response to start with "<think>\n" so the model reasons.
    if not text.rstrip().endswith("<think>"):
        text += "<think>\n"

    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    stop_ids = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id)
                if i is not None and i >= 0]
    out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=True,
                         temperature=a.temperature, top_p=a.top_p,
                         eos_token_id=stop_ids, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    print("<think>\n" + tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
