#!/usr/bin/env python3
"""
Small-scale reproduction of the DeepSeek-R1-Distill recipe, on a newer base model.

DeepSeek's recipe: start from a base model (Qwen2.5-Math-1.5B for the 1.5B distill)
and run plain supervised fine-tuning (SFT) on ~800k samples written by the full
DeepSeek-R1. No reinforcement-learning stage. This script does the same thing at
a much smaller scale: LoRA SFT on a public set of R1-written reasoning traces
(bespokelabs/Bespoke-Stratos-17k, ~16.7k rows).

The default base is Qwen/Qwen3-1.7B-Base (32K context, Apache-2.0, needs
transformers >= 4.51). Pick another with --base, for example Qwen/Qwen3-0.6B-Base
for quick tests on a free GPU.

Commands:
    python train.py inspect                    # check the data converts cleanly
    python train.py train --max-samples 200    # quick smoke test
    python train.py train                      # default run (2,000 samples, 2 epochs)
    python train.py train --resume             # carry on after a disconnected session
    python train.py merge                      # fold the LoRA adapter into the base model

Your own data (see harness.py gen): `--data FILE.jsonl` on `inspect` and `train` reads rows with the
fields question, thought and solution instead of Bespoke-Stratos. To mix the two:
    python train.py convert --n 2000 --out data/bespoke.jsonl
    cat data/bespoke.jsonl data/teacher.jsonl > data/mix.jsonl
    python train.py train --data data/mix.jsonl --max-samples 0
"""
import argparse
import itertools
import json
import math
import os
import random
import re
import statistics

DEFAULT_BASE = "Qwen/Qwen3-1.7B-Base"
DATASET = "bespokelabs/Bespoke-Stratos-17k"

# Bespoke-Stratos assistant messages look like:
#   <|begin_of_thought|> ...reasoning... <|end_of_thought|>
#   <|begin_of_solution|> ...final answer... <|end_of_solution|>
THOUGHT_RE = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.S)
SOLUTION_RE = re.compile(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.S)
USER_ROLES = {"user", "human"}
ASSISTANT_ROLES = {"assistant", "gpt", "model"}

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# --------------------------------------------------------------------------
# Data formatting (pure Python, no heavy imports)
# --------------------------------------------------------------------------
def convert_row(row):
    """Return (question, thought, solution), or None if the row has an unexpected shape."""
    conv = row.get("conversations") or []
    question = next((m["value"] for m in conv if m.get("from") in USER_ROLES), None)
    answer = next((m["value"] for m in conv if m.get("from") in ASSISTANT_ROLES), None)
    if not question or not answer:
        return None
    thought, solution = THOUGHT_RE.search(answer), SOLUTION_RE.search(answer)
    if not (thought and solution):
        return None
    return question.strip(), thought.group(1), solution.group(1)


def build_prompt(question):
    # ChatML (what Qwen tokenizers know), no system prompt, as DeepSeek recommends.
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def build_response(thought, solution):
    # R1-style output: reasoning inside <think> tags, then the final answer.
    return f"<think>\n{thought.strip()}\n</think>\n\n{solution.strip()}<|im_end|>"


def encode(question, thought, solution, tok):
    prompt_ids = tok(build_prompt(question), add_special_tokens=False)["input_ids"]
    resp_ids = tok(build_response(thought, solution), add_special_tokens=False)["input_ids"]
    return prompt_ids, resp_ids


def precision():
    """Pick (dtype, use_bf16, use_fp16) for the hardware we are on."""
    import torch

    if torch.cuda.is_available():
        # bf16 runs natively on Ampere (compute capability 8.0) and newer. Older GPUs such as
        # the Colab T4 can only emulate it slowly, so they train in fp16 instead.
        if torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16, True, False
        return torch.float16, False, True
    return torch.float32, False, False  # CPU / Apple silicon: slow, but works


def load_causal_lm(name, dtype=None, **kwargs):
    """AutoModelForCausalLM.from_pretrained that works across transformers versions.

    Newer releases call the precision argument `dtype`; older ones only know `torch_dtype`.
    """
    from transformers import AutoModelForCausalLM

    if dtype is None:
        return AutoModelForCausalLM.from_pretrained(name, **kwargs)
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kwargs)


def plan_steps(n_examples, batch_size, grad_accum, epochs, warmup_frac=0.03):
    """Return (optimizer steps in the whole run, warm-up steps).

    Warm-up is given in whole steps because every transformers version accepts that;
    the older `warmup_ratio` option is deprecated in transformers v5.
    """
    steps_per_epoch = math.ceil(n_examples / (batch_size * grad_accum))
    total = max(1, math.ceil(steps_per_epoch * epochs))
    return total, max(1, round(warmup_frac * total))


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def last_checkpoint(folder):
    """Newest `checkpoint-<step>` folder inside `folder`, or None. Used by --resume."""
    if not os.path.isdir(folder):
        return None
    found = [(int(m.group(1)), name) for name in os.listdir(folder) if (m := CHECKPOINT_RE.match(name))]
    return os.path.join(folder, max(found)[1]) if found else None


def read_jsonl_rows(path):
    """(question, thought, solution) triples from a JSONL file, as written by `harness.py gen` or
    `train.py convert`. A line that cannot be used gives None, so it is counted as a bad row."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                q, t, s = r["question"], r["thought"], r["solution"]
                ok = all(isinstance(x, str) and x.strip() for x in (q, t, s))
            except (json.JSONDecodeError, KeyError, TypeError):
                ok = False
            if not ok:
                rows.append(None)
                continue
            t = t.strip()
            if t.startswith("<think>"):  # the tag is added when the example is built
                t = t[len("<think>"):]
            rows.append(None if "</think>" in t else (q.strip(), t, s))
    return rows


def triples(data, seed):
    """Yield (question, thought, solution), or None for a row that cannot be used.
    From the public Bespoke-Stratos dataset, or from your own JSONL file if `data` is given."""
    if data:
        rows = read_jsonl_rows(data)
        random.Random(seed).shuffle(rows)
        yield from rows
    else:
        from datasets import load_dataset

        for row in load_dataset(DATASET, split="train").shuffle(seed=seed):
            yield convert_row(row)


def load_examples(tok, max_len, max_samples, seed, data=None):
    examples, bad_format, too_long, seen = [], 0, 0, 0
    for parts in triples(data, seed):
        seen += 1
        if parts is None:
            bad_format += 1
            continue
        prompt_ids, resp_ids = encode(*parts, tok)
        if len(prompt_ids) + len(resp_ids) > max_len:
            too_long += 1  # dropped rather than truncated, so the answer is never cut off
            continue
        examples.append({
            "input_ids": prompt_ids + resp_ids,
            # loss only on the response: the model learns to answer, not to echo the question
            "labels": [-100] * len(prompt_ids) + resp_ids,
        })
        if max_samples and len(examples) >= max_samples:
            break
    print(f"Scanned {seen} rows -> kept {len(examples)} | bad format: {bad_format} | "
          f"longer than {max_len} tokens: {too_long}")
    if not examples:
        raise SystemExit("No usable examples. Run `python train.py inspect` and/or raise --max-len.")
    return examples


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_inspect(a):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.base)
    lengths, bad, n = [], 0, 0
    sample = None
    for parts in itertools.islice(triples(a.data, a.seed), a.n):
        n += 1
        if parts is None:
            bad += 1
            continue
        p, r = encode(*parts, tok)
        lengths.append(len(p) + len(r))
        sample = sample or build_prompt(parts[0]) + build_response(parts[1], parts[2])
    source = a.data or DATASET
    print(f"Checked {n} rows of {source} with the {a.base} tokenizer: {len(lengths)} convert cleanly, {bad} do not.")
    if lengths:
        p90 = statistics.quantiles(lengths, n=10)[-1] if len(lengths) > 1 else lengths[0]
        print(f"Token lengths: median {int(statistics.median(lengths))}, "
              f"p90 {int(p90)}, max {max(lengths)}")
        for cap in (2048, 4096, 8192):
            share = sum(l <= cap for l in lengths) / len(lengths)
            print(f"  fits in {cap:>5} tokens: {share:.0%}")
        print("\n--- sample (start) ---\n" + sample[:700])
        print("\n--- sample (end) ---\n" + sample[-400:])


def cmd_train(a):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(a.seed)
    random.seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    context = AutoConfig.from_pretrained(a.base).max_position_embeddings
    max_len = min(a.max_len, context)
    print(f"Base model {a.base}: context window {context} tokens, training with --max-len {max_len}")
    rows = load_examples(tok, max_len, a.max_samples, a.seed, a.data)

    dtype, use_bf16, use_fp16 = precision()
    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU found. Training will be very slow. "
              "A free Colab/Kaggle GPU is recommended.")

    if a.load_in_4bit:  # QLoRA: for GPUs with < ~10 GB of memory (NVIDIA only)
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        model = load_causal_lm(a.base, dtype=dtype, quantization_config=bnb, device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = load_causal_lm(a.base, dtype=dtype)
    model.config.use_cache = False
    model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    for p in model.parameters():  # keep trainable LoRA weights in fp32 for stable updates
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    class Rows(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            return rows[i]

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), tok.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            labels[i, :L] = torch.tensor(b["labels"])
            mask[i, :L] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

    total_steps, warmup_steps = plan_steps(len(rows), a.batch_size, a.grad_accum, a.epochs)
    # Free GPU sessions get cut off, so by default write a resumable checkpoint every few steps.
    saving = ({"save_strategy": "steps", "save_steps": a.save_steps} if a.save_steps > 0
              else {"save_strategy": "epoch"})
    every = f"every {a.save_steps} steps" if a.save_steps > 0 else "at the end of each epoch"
    print(f"{len(rows)} examples -> about {total_steps} optimizer steps ({warmup_steps} warm-up), "
          f"checkpoint {every}")

    adapter_dir = os.path.join(a.out, "adapter")
    args = TrainingArguments(
        output_dir=os.path.join(a.out, "checkpoints"),
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=5,
        **saving,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        seed=a.seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=Rows(), data_collator=collate)
    resume = last_checkpoint(args.output_dir) if a.resume else None
    if a.resume:
        print(f"Resuming from {resume}" if resume else f"No checkpoint in {args.output_dir}; starting from scratch")
    trainer.train(resume_from_checkpoint=resume)

    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"\nSaved LoRA adapter to {adapter_dir}")
    print(f"Next: python train.py merge --adapter {adapter_dir} --output {os.path.join(a.out, 'merged')}")


def cmd_convert(a):
    """Write Bespoke-Stratos rows to a JSONL file, so they can be mixed with your own teacher data."""
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    kept = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for parts in triples(None, a.seed):
            if parts is None:
                continue
            q, t, s = parts
            f.write(json.dumps({"question": q, "thought": t.strip(), "solution": s.strip(),
                                "source": "bespoke-stratos"}, ensure_ascii=False) + "\n")
            kept += 1
            if a.n and kept >= a.n:
                break
    print(f"Wrote {kept} rows to {a.out}")


def set_stop_tokens(model, tok):
    """Make <|im_end|> the end-of-sequence token of the saved model.

    train.py teaches the model to finish every answer with <|im_end|>, but a Qwen3 base model
    declares <|endoftext|> as its end token, so generate() would keep going past the end of the answer.
    Returns True if the change was made.
    """
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is None or im_end < 0 or im_end == tok.unk_token_id:
        return False  # not a ChatML tokenizer: leave it alone
    known = model.generation_config.eos_token_id
    known = known if isinstance(known, list) else [known]
    ids = [im_end]
    for i in [tok.eos_token_id] + known:  # keep the old end tokens as well, in order, without repeats
        if i is not None and i not in ids:
            ids.append(i)
    tok.eos_token = "<|im_end|>"
    model.generation_config.eos_token_id = ids
    return True


def cmd_merge(a):
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoTokenizer

    # The adapter remembers which base model it was trained on, so --base is optional here.
    base_name = a.base or PeftConfig.from_pretrained(a.adapter).base_model_name_or_path
    print(f"Merging {a.adapter} into {base_name}")
    base = load_causal_lm(base_name, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(a.adapter)
    if set_stop_tokens(model, tok):
        print("End-of-sequence token set to <|im_end|> (generation also still stops at the old one)")
    model.save_pretrained(a.output, safe_serialization=True)
    tok.save_pretrained(a.output)
    print(f"Merged model saved to {a.output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="check data conversion and token-length stats")
    p.add_argument("--base", default=DEFAULT_BASE, help="base model whose tokenizer is used (default: %(default)s)")
    p.add_argument("--n", type=int, default=500, help="how many rows to check")
    p.add_argument("--data", default=None, help="your own JSONL file instead of Bespoke-Stratos")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("convert", help="write Bespoke-Stratos rows to JSONL (to mix with your own data)")
    p.add_argument("--n", type=int, default=2000, help="rows to write, 0 = all")
    p.add_argument("--out", default="data/bespoke.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("train", help="LoRA supervised fine-tuning")
    p.add_argument("--base", default=DEFAULT_BASE,
                   help="Hugging Face id (or local path) of the base model (default: %(default)s)")
    p.add_argument("--out", default="out")
    p.add_argument("--data", default=None,
                   help="your own JSONL file (question, thought, solution) instead of Bespoke-Stratos")
    p.add_argument("--max-samples", type=int, default=2000, help="0 = use every usable row")
    p.add_argument("--max-len", type=int, default=4096,
                   help="drop samples longer than this many tokens (capped at the base model's limit)")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--load-in-4bit", action="store_true", help="QLoRA, for small NVIDIA GPUs")
    p.add_argument("--save-steps", type=int, default=50,
                   help="write a checkpoint every this many optimizer steps, 0 = once per epoch (default: %(default)s)")
    p.add_argument("--resume", action="store_true",
                   help="continue from the newest checkpoint in --out (use the same options as the first run)")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("merge", help="merge the LoRA adapter into the base model")
    p.add_argument("--adapter", default="out/adapter")
    p.add_argument("--output", default="out/merged")
    p.add_argument("--base", default=None, help="base model to merge into (default: read from the adapter)")
    p.set_defaults(fn=cmd_merge)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
