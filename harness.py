#!/usr/bin/env python3
"""
The measuring tool for the plan: sample answers from a model, check them, report pass@1.

Score a model (Weeks 2 to 8):
    python harness.py eval --model Qwen/Qwen3-0.6B-Base --name base-0.6b
    python harness.py eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --template tokenizer --name official-1.5b
    python harness.py eval --model out/merged --name sft-0.6b --bench gsm8k,math500,aime24 --samples 8

Put every result you have side by side, and read some raw answers:
    python harness.py table results/*.json
    python harness.py peek results/sft-0.6b.samples.jsonl --wrong --n 5

Write your own training data with a free teacher (Week 5):
    python harness.py gen --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --out data/teacher.jsonl

How a score is made: every problem gets --samples answers at temperature 0.6. The part after
</think> is read by Math-Verify, which compares the final \\boxed{} answer with the known answer.
An answer that hits the token cap counts as wrong. pass@1 is the average share of correct answers.
Every model sees the same problems, because the subset is chosen with a fixed --seed.
"""
import argparse
import json
import math
import os
import random
import re
import statistics
import time
import zlib
from dataclasses import dataclass

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
THINK_END = "</think>"
EVAL_BENCHES = ("gsm8k", "math500", "aime24")
TRAIN_SOURCES = ("gsm8k", "math")


# --------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------
def gsm8k_gold(answer_field):
    """GSM8K ends each worked solution with '#### 72'; the number after it is the answer."""
    return answer_field.split("####")[-1].strip().replace(",", "")


def last_boxed(text):
    """The contents of the last \\boxed{...} in text (nested braces are handled), or None."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, out = 1, []
    for ch in text[start + len("\\boxed{"):]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
    return None


def pick_subset(items, n, seed):
    """The same n items for every model (fixed seed), so the scores can be compared."""
    if n <= 0 or n >= len(items):
        return list(items)
    keep = sorted(random.Random(seed).sample(range(len(items)), n))
    return [items[i] for i in keep]


def load_eval_problems(bench, n, seed):
    from datasets import load_dataset

    if bench == "gsm8k":
        rows = load_dataset("openai/gsm8k", "main", split="test")
        items = [{"id": f"gsm8k-{i}", "question": r["question"], "gold": gsm8k_gold(r["answer"])}
                 for i, r in enumerate(rows)]
    elif bench == "math500":
        rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [{"id": r.get("unique_id") or f"math500-{i}", "question": r["problem"], "gold": str(r["answer"])}
                 for i, r in enumerate(rows)]
    elif bench == "aime24":
        rows = load_dataset("HuggingFaceH4/aime_2024", split="train")
        items = [{"id": f"aime24-{r['id']}", "question": r["problem"], "gold": str(r["answer"])} for r in rows]
        n = 0  # there are only 30 problems, so always use all of them
    else:
        raise SystemExit(f"Unknown benchmark {bench!r}. Choose from: {', '.join(EVAL_BENCHES)}")
    return pick_subset(items, n, seed)


def load_train_problems(source, n, seed):
    """Training problems with a known answer (the test sets above are never used here)."""
    from datasets import load_dataset

    if source == "gsm8k":
        rows = load_dataset("openai/gsm8k", "main", split="train")
        items = [{"source": "gsm8k", "question": r["question"], "gold": gsm8k_gold(r["answer"])} for r in rows]
    elif source == "math":
        rows = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
        items = []
        for r in rows:
            gold = last_boxed(r["solution"])  # the answer is inside \boxed{} in the worked solution
            if gold:
                items.append({"source": "math", "question": r["problem"], "gold": gold})
    else:
        raise SystemExit(f"Unknown problem source {source!r}. Choose from: {', '.join(TRAIN_SOURCES)}")
    return pick_subset(items, n, seed)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
def build_prompt(question, template, tok):
    """The text the model continues. Same recipe as chat.py: DeepSeek recommends starting with <think>."""
    q = question + SUFFIX
    if template == "chatml":  # what train.py trains on: ChatML, no system prompt
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    else:  # the model's own template, e.g. for the official DeepSeek distill
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True)
    if not text.rstrip().endswith("<think>"):
        text += "<think>\n"
    return text


def stop_ids_for(tok, template):
    """Token ids that end an answer. ChatML models end with <|im_end|>."""
    ids = []
    if template == "chatml":
        i = tok.convert_tokens_to_ids("<|im_end|>")
        if i is not None and i >= 0 and i != tok.unk_token_id:
            ids.append(i)
    if tok.eos_token_id is not None and tok.eos_token_id not in ids:
        ids.append(tok.eos_token_id)
    return ids


# --------------------------------------------------------------------------
# Generation: vLLM (fast) or plain transformers (slow, always works)
# --------------------------------------------------------------------------
@dataclass
class Gen:
    text: str
    finished: bool  # False = the answer was cut off at the token cap
    tokens: int


def gpu_dtype_name():
    """bfloat16 on GPUs that run it natively (compute capability 8+), else float16 (for example a T4)."""
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return "bfloat16"
    return "float16"


def generate_vllm(a, model, tok, texts, stop_ids):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise SystemExit("vLLM is not installed. Try `pip install vllm` on a GPU machine, "
                         "or add `--engine hf` (slower, but it needs nothing extra).")
    llm = LLM(model=model, dtype=gpu_dtype_name(), seed=a.seed,
              max_model_len=a.max_model_len or a.max_tokens + 1024,
              gpu_memory_utilization=a.gpu_memory_utilization)
    params = SamplingParams(n=a.samples, temperature=a.temperature, top_p=a.top_p,
                            max_tokens=a.max_tokens, stop_token_ids=stop_ids)
    # Token ids go in directly, so a template that already starts with a BOS token is not given a second one.
    prompts = [{"prompt_token_ids": tok(t, add_special_tokens=False)["input_ids"]} for t in texts]
    results = llm.generate(prompts, params)
    return [[Gen(o.text, o.finish_reason == "stop", len(o.token_ids)) for o in r.outputs] for r in results]


def read_generated_ids(ids, stop_ids):
    """(ids before the first stop token, finished?). Padding after the stop token is dropped."""
    for i, t in enumerate(ids):
        if t in stop_ids:
            return ids[:i], True
    return ids, False


def generate_hf(a, model, tok, texts, stop_ids):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if gpu_dtype_name() == "bfloat16" else torch.float16
    try:  # newer transformers call this argument `dtype`, older ones `torch_dtype`
        lm = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, device_map="auto")
    except TypeError:
        lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype, device_map="auto")
    lm.eval()
    tok.padding_side = "left"  # generation continues from the right-hand end of each prompt
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    results = []
    for start in range(0, len(texts), a.batch_size):
        chunk = texts[start:start + a.batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(lm.device)
        out = lm.generate(**enc, max_new_tokens=a.max_tokens, do_sample=a.temperature > 0,
                          temperature=a.temperature, top_p=a.top_p, num_return_sequences=a.samples,
                          eos_token_id=stop_ids, pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:].tolist()
        for p in range(len(chunk)):  # the samples of one prompt sit next to each other
            group = []
            for ids in new[p * a.samples:(p + 1) * a.samples]:
                ids, finished = read_generated_ids(ids, stop_ids)
                group.append(Gen(tok.decode(ids, skip_special_tokens=True), finished, len(ids)))
            results.append(group)
        print(f"  generated {min(start + a.batch_size, len(texts))}/{len(texts)} prompts", flush=True)
    return results


def generate(a, model, tok, texts, stop_ids):
    """One list of `a.samples` Gen objects per prompt."""
    if a.engine == "vllm":
        return generate_vllm(a, model, tok, texts, stop_ids)
    return generate_hf(a, model, tok, texts, stop_ids)


# --------------------------------------------------------------------------
# Checking answers
# --------------------------------------------------------------------------
def require_math_verify():
    try:
        import math_verify  # noqa: F401
    except ImportError:
        raise SystemExit("Math-Verify is needed to check answers: pip install 'math-verify[antlr4_13_2]'")


def final_part(text):
    """What counts as the answer: the text after the last </think>, or everything if there is no tag."""
    return text.rsplit(THINK_END, 1)[-1] if THINK_END in text else text


def parse_gold(gold):
    from math_verify import parse

    try:
        return parse(f"${gold}$")  # Math-Verify reads LaTeX inside $...$
    except Exception:
        return []


def check_answer(text, finished, gold_parsed):
    """(correct, answered). An answer that hit the token cap never counts as correct."""
    from math_verify import parse, verify

    try:
        pred = parse(final_part(text))
    except Exception:
        pred = []
    answered = bool(pred)
    if not (finished and answered and gold_parsed):
        return False, answered
    try:
        return bool(verify(gold_parsed, pred)), True  # the order matters: gold first
    except Exception:
        return False, True


def make_checker():
    """check(text, finished, gold) -> (correct, answered), parsing each gold answer only once."""
    cache = {}

    def check(text, finished, gold):
        if gold not in cache:
            cache[gold] = parse_gold(gold)
        return check_answer(text, finished, cache[gold])

    return check


def percentile(sorted_values, q):
    if not sorted_values:
        return 0
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def summarize(per_problem):
    """per_problem: one list of sample dicts per problem. Returns the numbers that go in your table."""
    rates = [sum(s["correct"] for s in samples) / len(samples) for samples in per_problem]
    flat = [s for samples in per_problem for s in samples]
    n = len(rates)
    se = statistics.stdev(rates) / math.sqrt(n) if n > 1 else 0.0
    tokens = sorted(s["tokens"] for s in flat)

    def share(key):
        return sum(bool(s[key]) for s in flat) / len(flat)

    return {
        "n_problems": n,
        "samples": len(per_problem[0]),
        "pass1": sum(rates) / n,
        "ci95": 1.96 * se,  # about 95% of the time the true score is within this of pass1
        "avg_tokens": sum(tokens) / len(tokens),
        "p90_tokens": percentile(tokens, 0.9),
        "cut_off": 1 - share("finished"),
        "closes_think": share("closed"),
        "no_answer": 1 - share("answered"),
    }


def describe(bench, r):
    return (f"{bench}: pass@1 {100 * r['pass1']:.1f} ± {100 * r['ci95']:.1f}  "
            f"({r['n_problems']} problems x {r['samples']} answers) | "
            f"avg {r['avg_tokens']:.0f} tokens, p90 {r['p90_tokens']} | "
            f"cut off at the cap {100 * r['cut_off']:.1f}% | closes </think> {100 * r['closes_think']:.1f}% | "
            f"no answer found {100 * r['no_answer']:.1f}%")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_eval(a):
    require_math_verify()
    from transformers import AutoTokenizer

    benches = [b.strip() for b in a.bench.split(",") if b.strip()]
    tok = AutoTokenizer.from_pretrained(a.model)
    problems = {b: load_eval_problems(b, a.n, a.seed) for b in benches}
    flat = [(b, p) for b in benches for p in problems[b]]
    texts = [build_prompt(p["question"], a.template, tok) for _, p in flat]
    stop_ids = stop_ids_for(tok, a.template)
    print(f"Scoring {a.model}: {len(flat)} problems x {a.samples} answers, up to {a.max_tokens} tokens each, "
          f"engine {a.engine}, template {a.template}")
    print("First prompt, so you can check the format:\n" + "-" * 40 + f"\n{texts[0]}\n" + "-" * 40)

    t0 = time.time()
    gens = generate(a, a.model, tok, texts, stop_ids)
    print(f"Generation took {(time.time() - t0) / 60:.1f} minutes")

    check = make_checker()
    by_bench = {b: [] for b in benches}
    samples_out = []
    for (b, p), group in zip(flat, gens):
        records = []
        for k, g in enumerate(group):
            correct, answered = check(g.text, g.finished, p["gold"])
            rec = {"correct": correct, "answered": answered, "finished": g.finished,
                   "closed": THINK_END in g.text, "tokens": g.tokens}
            records.append(rec)
            samples_out.append({"bench": b, "id": p["id"], "sample": k, "gold": p["gold"], "text": g.text, **rec})
        by_bench[b].append(records)

    results = {b: summarize(v) for b, v in by_bench.items()}
    print()
    for b in benches:
        print(describe(b, results[b]))

    os.makedirs(a.out, exist_ok=True)
    name = a.name or a.model.strip("/").replace("/", "_")
    summary_path = os.path.join(a.out, f"{name}.json")
    samples_path = os.path.join(a.out, f"{name}.samples.jsonl")
    settings = {k: getattr(a, k) for k in ("template", "n", "samples", "max_tokens", "temperature", "top_p",
                                           "seed", "engine")}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"name": name, "model": a.model, "settings": settings, "results": results,
                   "date": time.strftime("%Y-%m-%d %H:%M")}, f, indent=2)
    with open(samples_path, "w", encoding="utf-8") as f:
        for s in samples_out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nSaved {summary_path} and {samples_path}")
    print(f"Read some answers by hand: python harness.py peek {samples_path} --wrong --n 5")


def cmd_table(a):
    runs = []
    for path in a.files:
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))
    if not runs:
        raise SystemExit("No result files given. Example: python harness.py table results/*.json")
    benches = [b for b in EVAL_BENCHES if any(b in r["results"] for r in runs)]
    print("| model | " + " | ".join(benches) + " | answers per problem | token cap |")
    print("|" + "---|" * (len(benches) + 3))
    for r in runs:
        cells = []
        for b in benches:
            res = r["results"].get(b)
            cells.append(f"{100 * res['pass1']:.1f} ± {100 * res['ci95']:.1f}" if res else "-")
        s = r["settings"]
        print(f"| {r['name']} | " + " | ".join(cells) + f" | {s['samples']} | {s['max_tokens']} |")
    kinds = {(r["settings"]["samples"], r["settings"]["max_tokens"], r["settings"]["n"],
              r["settings"]["temperature"], r["settings"]["seed"]) for r in runs}
    if len(kinds) > 1:
        print("\nWarning: these runs used different settings (answers per problem, token cap, problems, "
              "temperature or seed), so compare them with care.")


def cmd_peek(a):
    with open(a.file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if a.bench:
        rows = [r for r in rows if r["bench"] == a.bench]
    if a.wrong:
        rows = [r for r in rows if not r["correct"]]
    random.Random(a.seed).shuffle(rows)
    if not rows:
        print("Nothing to show with those filters.")
    for r in rows[:a.n]:
        text = r["text"]
        shown = text if len(text) <= 1400 else text[:800] + "\n   [... middle left out ...]\n" + text[-600:]
        status = "finished" if r["finished"] else "CUT OFF at the token cap"
        print("=" * 78)
        print(f"{r['bench']} {r['id']} sample {r['sample']} | gold: {r['gold']} | "
              f"correct: {r['correct']} | {r['tokens']} tokens | {status}")
        print(shown)


def norm_question(q):
    return re.sub(r"\W+", " ", q.lower()).strip()


def looks_repetitive(text, floor=0.25):
    """Heuristic: text that zlib squeezes below 25% of its size is looping, so we do not train on it."""
    raw = text.encode("utf-8")
    return len(raw) > 1000 and len(zlib.compress(raw)) / len(raw) < floor


def filter_traces(problems, gens, check, keep, max_trace_tokens, banned=frozenset()):
    """Turn raw teacher answers into training rows. Returns (rows, counts after each filter)."""
    counts = {"samples written by the teacher": 0}
    candidates = []  # (problem index, row)
    stages = ["finished with exactly one </think>", "final answer correct", "not repetitive",
              "short enough with a written solution"]
    for s in stages:
        counts[s] = 0
    for pi, (prob, group) in enumerate(zip(problems, gens)):
        for g in group:
            counts["samples written by the teacher"] += 1
            if not g.finished or g.text.count(THINK_END) != 1:
                continue
            counts[stages[0]] += 1
            correct, _ = check(g.text, True, prob["gold"])
            if not correct:
                continue
            counts[stages[1]] += 1
            thought, final = g.text.split(THINK_END)
            thought = thought.strip()
            if thought.startswith("<think>"):
                thought = thought[len("<think>"):].strip()
            if looks_repetitive(thought):
                continue
            counts[stages[2]] += 1
            if g.tokens > max_trace_tokens or not final.strip():
                continue
            counts[stages[3]] += 1
            candidates.append((pi, {"question": prob["question"], "thought": thought, "solution": final.strip(),
                                    "gold": prob["gold"], "source": prob["source"], "tokens": g.tokens}))

    taken, capped = {}, []
    for pi, row in candidates:
        if taken.get(pi, 0) < keep:
            taken[pi] = taken.get(pi, 0) + 1
            capped.append(row)
    counts[f"at most {keep} per problem"] = len(capped)

    seen, unique = set(), []
    for row in capped:
        key = (row["question"], row["thought"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    counts["after removing duplicates"] = len(unique)

    clean = [r for r in unique if norm_question(r["question"]) not in banned]
    counts["after removing overlap with the test sets"] = len(clean)
    return clean, counts


def cmd_gen(a):
    require_math_verify()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.teacher)
    problems = []
    for src in [s.strip() for s in a.problems.split(",") if s.strip()]:
        problems += load_train_problems(src, a.n, a.seed)
    banned = frozenset()
    if not a.no_decontaminate:
        banned = frozenset(norm_question(p["question"]) for b in EVAL_BENCHES
                           for p in load_eval_problems(b, 0, a.seed))
    texts = [build_prompt(p["question"], a.template, tok) for p in problems]
    stop_ids = stop_ids_for(tok, a.template)
    print(f"Teacher {a.teacher}: {len(problems)} problems x {a.samples} answers, up to {a.max_tokens} tokens each")

    t0 = time.time()
    gens = generate(a, a.teacher, tok, texts, stop_ids)
    print(f"Generation took {(time.time() - t0) / 60:.1f} minutes")

    rows, counts = filter_traces(problems, gens, make_checker(), a.keep, a.max_trace_tokens, banned)
    for r in rows:
        r["teacher"] = a.teacher
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    card_path = os.path.splitext(a.out)[0] + ".card.json"
    settings = {k: getattr(a, k) for k in ("teacher", "problems", "n", "samples", "max_tokens", "temperature",
                                           "top_p", "keep", "max_trace_tokens", "seed", "engine")}
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump({"settings": settings, "problems": len(problems), "counts": counts,
                   "date": time.strftime("%Y-%m-%d %H:%M")}, f, indent=2)
    print("\nWhat is left after each filter:")
    for k, v in counts.items():
        print(f"  {v:>7}  {k}")
    print(f"\nWrote {len(rows)} training rows to {a.out} and the counts to {card_path}")
    print(f"Next: python train.py inspect --data {a.out}")


# --------------------------------------------------------------------------
def add_generation_options(p, default_max_tokens):
    p.add_argument("--samples", type=int, default=4, help="answers sampled per problem")
    p.add_argument("--max-tokens", type=int, default=default_max_tokens,
                   help="longest answer allowed; one that hits this counts as wrong")
    p.add_argument("--temperature", type=float, default=0.6)  # DeepSeek recommends 0.5-0.7
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0, help="also picks which problems are used")
    p.add_argument("--engine", choices=["vllm", "hf"], default="vllm",
                   help="vllm is fast; hf (plain transformers) is slow but needs nothing extra")
    p.add_argument("--batch-size", type=int, default=2, help="--engine hf only: prompts per batch")
    p.add_argument("--max-model-len", type=int, default=None, help="vLLM only; default is --max-tokens + 1024")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM only")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("eval", help="score a model on math problems")
    p.add_argument("--model", required=True, help="Hugging Face id or a local folder such as out/merged")
    p.add_argument("--name", default=None, help="label for this run; results/<name>.json (default: from --model)")
    p.add_argument("--template", choices=["chatml", "tokenizer"], default="chatml",
                   help="chatml = what train.py trains on (also fine for base models); "
                        "tokenizer = the model's own template (use it for the official DeepSeek distill)")
    p.add_argument("--bench", default="gsm8k,math500", help=f"comma list from: {', '.join(EVAL_BENCHES)}")
    p.add_argument("--n", type=int, default=100,
                   help="problems per benchmark (0 = all); aime24 always uses all 30")
    p.add_argument("--out", default="results")
    add_generation_options(p, default_max_tokens=4096)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("table", help="print a markdown table of saved results")
    p.add_argument("files", nargs="*")
    p.set_defaults(fn=cmd_table)

    p = sub.add_parser("peek", help="print some raw answers from a samples file")
    p.add_argument("file")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--wrong", action="store_true", help="only answers that were marked wrong")
    p.add_argument("--bench", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_peek)

    p = sub.add_parser("gen", help="write training data with a teacher model (keeps only verified answers)")
    p.add_argument("--teacher", required=True, help="e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B "
                                                    "or Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--template", choices=["chatml", "tokenizer"], default="tokenizer")
    p.add_argument("--problems", default="gsm8k,math", help=f"comma list from: {', '.join(TRAIN_SOURCES)}")
    p.add_argument("--n", type=int, default=400, help="problems taken from each source")
    p.add_argument("--keep", type=int, default=2, help="correct answers kept per problem")
    p.add_argument("--max-trace-tokens", type=int, default=3500,
                   help="drop longer answers, so they fit train.py's --max-len 4096")
    p.add_argument("--no-decontaminate", action="store_true",
                   help="skip the check against the test problems (needs no extra downloads)")
    p.add_argument("--out", default="data/teacher.jsonl")
    add_generation_options(p, default_max_tokens=6000)
    p.set_defaults(fn=cmd_gen)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
