#!/usr/bin/env python3
"""
Reinforcement learning with GRPO (Weeks 6 and 7): the model writes several answers to each math problem,
rewards.py scores them, and training pushes the model toward the answers that scored above the average
of their group. It starts from the model that train.py produced and trains a new, small LoRA adapter.

    # Week 6: an easy task, to learn how it behaves (GSM8K, a short run)
    python grpo.py --model out/merged --problems gsm8k --n 2000 --max-steps 100 --out out-grpo-easy

    # Week 7: the real run (harder problems, longer)
    python grpo.py --model out/merged --problems math,gsm8k --n 3000 --max-steps 300 --out out-grpo

    # Free sessions end: run the same command again with --resume
    python grpo.py --model out/merged --problems gsm8k --n 2000 --max-steps 100 --out out-grpo-easy --resume

    # Then fold the adapter into the model and score it (same folder you ran grpo.py from)
    python train.py merge --adapter out-grpo/adapter --output out-grpo/merged
    python harness.py eval --model out-grpo/merged --name grpo --bench gsm8k,math500

Needs: pip install trl 'math-verify[antlr4_13_2]'
The training problems come from the GSM8K and MATH *train* sets; the test sets are never used for training.
Add --vllm to let vLLM write the answers (faster, but one more thing to install and to go wrong).
"""
import argparse
import dataclasses
import os
import random

import rewards
from harness import TRAIN_SOURCES, load_train_problems
from rewards import correctness_reward, format_reward, make_prompt
from train import LORA_TARGETS, last_checkpoint, load_causal_lm, precision, set_stop_tokens

# GRPOConfig options this script sets. If your TRL has not heard of one, you need a newer TRL.
NEEDED_CONFIG_FIELDS = ("num_generations", "max_completion_length", "temperature", "steps_per_generation")


def plan_batches(num_generations, per_device_batch, prompts_per_step):
    """Return gradient_accumulation_steps.

    One optimizer step covers `prompts_per_step` problems, each answered `num_generations` times. TRL needs
    the number of answers it writes at once to be a whole number of groups, which holds when it is exactly
    num_generations * prompts_per_step, so the micro-batch size must divide that.
    """
    if per_device_batch < 1:
        raise SystemExit("--per-device-batch must be at least 1.")
    if num_generations < 2:
        raise SystemExit("GRPO needs --num-generations of at least 2: it compares answers within a group.")
    answers = num_generations * prompts_per_step
    if answers % per_device_batch:
        raise SystemExit(f"--per-device-batch {per_device_batch} must divide --num-generations x "
                         f"--prompts-per-step = {answers}. Try --per-device-batch {largest_divisor_up_to(answers)}.")
    return answers // per_device_batch


def largest_divisor_up_to(total, wanted=4):
    """The largest divisor of `total` that is not above `wanted` (a suggestion for the error message)."""
    return max(d for d in range(1, wanted + 1) if total % d == 0)


def reward_self_check(problems):
    """Keep only problems whose known answer the checker accepts when it is given back as a \\boxed{} answer.

    A problem whose own answer fails this can never be rewarded, so it would only add noise.
    """
    texts = [f"Working.\n</think>\n\nThe answer is \\boxed{{{p['gold']}}}." for p in problems]
    ok = correctness_reward(texts, [p["gold"] for p in problems])
    return [p for p, r in zip(problems, ok) if r > 0]


def build_rows(sources, n, seed, count_tokens, max_prompt_tokens):
    """Prompt/answer rows for GRPOTrainer. Returns (rows, dropped for length, dropped by the self-check)."""
    problems = []
    for src in sources:
        problems += load_train_problems(src, n, seed)
    random.Random(seed).shuffle(problems)
    checked = reward_self_check(problems)
    rows, too_long = [], 0
    for p in checked:
        prompt = make_prompt(p["question"])
        if count_tokens(prompt) > max_prompt_tokens:
            too_long += 1  # long prompts could be cut by older TRL versions, so leave them out
            continue
        rows.append({"prompt": prompt, "answer": p["gold"]})
    return rows, too_long, len(problems) - len(checked)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="out/merged", help="the SFT model to improve (a folder or a Hub id)")
    ap.add_argument("--out", default="out-grpo")
    ap.add_argument("--problems", default="gsm8k", help=f"comma list from: {', '.join(TRAIN_SOURCES)}")
    ap.add_argument("--n", type=int, default=2000, help="problems taken from each source")
    ap.add_argument("--max-steps", type=int, default=100, help="optimizer steps (each uses --prompts-per-step problems)")
    ap.add_argument("--num-generations", type=int, default=8, help="answers written per problem (the group size)")
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--per-device-batch", type=int, default=2, help="answers processed at once; lower it if memory runs out")
    ap.add_argument("--max-completion-length", type=int, default=1024,
                    help="longest answer; one that hits it never closes </think> and earns no reward")
    ap.add_argument("--max-prompt-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8, help="sampling temperature while learning")
    ap.add_argument("--lr", type=float, default=2e-5, help="a starting guess for a LoRA adapter; watch the reward curve")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in --out")
    ap.add_argument("--vllm", action="store_true", help="write answers with vLLM (colocated on the same GPU)")
    ap.add_argument("--vllm-memory", type=float, default=0.3, help="share of GPU memory for vLLM, with --vllm")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if rewards.parse is None:
        raise SystemExit("Math-Verify is needed to check answers: pip install 'math-verify[antlr4_13_2]'")
    grad_accum = plan_batches(a.num_generations, a.per_device_batch, a.prompts_per_step)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    missing = [f for f in NEEDED_CONFIG_FIELDS if f not in {x.name for x in dataclasses.fields(GRPOConfig)}]
    if missing:
        raise SystemExit(f"This TRL does not know GRPOConfig options {missing}. Upgrade: pip install -U trl")
    set_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    sources = [s.strip() for s in a.problems.split(",") if s.strip()]
    rows, too_long, unverifiable = build_rows(
        sources, a.n, a.seed, lambda t: len(tok(t, add_special_tokens=False)["input_ids"]), a.max_prompt_tokens)
    print(f"{len(rows)} training problems from {', '.join(sources)} "
          f"(left out: {unverifiable} whose own answer the checker cannot read, {too_long} with a long prompt)")
    if not rows:
        raise SystemExit("No usable problems.")
    wanted = a.max_steps * a.prompts_per_step
    if wanted > len(rows):
        print(f"Note: {a.max_steps} steps need {wanted} problems but there are {len(rows)}; some will repeat.")

    dtype, use_bf16, use_fp16 = precision()
    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU found. GRPO will be extremely slow.")
    model = load_causal_lm(a.model, dtype=dtype)
    # <|im_end|> must be the end token, or TRL cannot tell where an answer stops
    set_stop_tokens(model, tok)
    model.enable_input_require_grads()

    config = dict(
        output_dir=os.path.join(a.out, "checkpoints"),
        per_device_train_batch_size=a.per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_generations=a.num_generations,
        max_completion_length=a.max_completion_length,
        temperature=a.temperature,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(0.05 * a.max_steps)),
        max_steps=a.max_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=a.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=a.seed,
    )
    if a.vllm:
        config.update(use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_memory)
    args = GRPOConfig(**config)

    lora = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0,  # no dropout: it would add noise to RL
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=model, processing_class=tok, args=args, train_dataset=Dataset.from_list(rows),
                          reward_funcs=[format_reward, correctness_reward], peft_config=lora)
    print(f"{a.max_steps} steps, {a.prompts_per_step} problems x {a.num_generations} answers per step "
          f"({a.per_device_batch} at a time, {grad_accum} accumulation steps). Watch the correctness reward mean "
          f"in the log: it should drift up.")

    resume = last_checkpoint(args.output_dir) if a.resume else None
    if a.resume:
        print(f"Resuming from {resume}" if resume else f"No checkpoint in {args.output_dir}; starting from scratch")
    trainer.train(resume_from_checkpoint=resume)

    adapter_dir = os.path.join(a.out, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"\nSaved the GRPO adapter to {adapter_dir}")
    print(f"Next: python train.py merge --adapter {adapter_dir} --output {os.path.join(a.out, 'merged')}")


if __name__ == "__main__":
    main()
