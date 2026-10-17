"""
Reward functions for GRPO (Weeks 6 and 7). Plain Python, so you can read them, test them and change them.

GRPO samples several answers per problem, scores each one with functions like these, and pushes the
model toward the answers that scored above the group's average. Use them with the official Unsloth GRPO
notebook, or with TRL's GRPOTrainer, which that notebook is built on:

    from rewards import make_prompt, gsm8k_gold, format_reward, correctness_reward

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.map(lambda r: {"prompt": make_prompt(r["question"]), "answer": gsm8k_gold(r["answer"])})
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, args=config, train_dataset=ds,
                          reward_funcs=[format_reward, correctness_reward])

The prompt is a plain string in the same ChatML format that train.py trained on, ending with the <think>
opener, so the model continues exactly as it learned. TRL passes every dataset column except `prompt` to the
reward functions by name, which is how `answer` reaches correctness_reward.

Run `python rewards.py` for a small demonstration.
"""
import re

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
THINK_END = "</think>"

try:  # Math-Verify: pip install 'math-verify[antlr4_13_2]'
    from math_verify import parse, verify
except ImportError:  # a plainer comparison is used instead (see same_answer)
    parse = verify = None


def make_prompt(question):
    """The text the model continues: ChatML, no system prompt, thinking already opened."""
    return f"<|im_start|>user\n{question}{SUFFIX}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def gsm8k_gold(answer_field):
    """GSM8K ends each worked solution with '#### 72'; the number after it is the answer."""
    return answer_field.split("####")[-1].strip().replace(",", "")


def completion_text(completion):
    """TRL passes a string for plain prompts, or a list of chat messages for chat prompts."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


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


def _number(s):
    try:
        return float(re.sub(r"[\s$,]|\\[,!;]|\\text\{[^}]*\}", "", s).rstrip("."))
    except ValueError:
        return None


_gold_cache = {}


def same_answer(final_text, gold):
    """Does the final answer in `final_text` match the known answer `gold`?"""
    gold = str(gold)
    if parse is not None:
        try:
            if gold not in _gold_cache:
                _gold_cache[gold] = parse(f"${gold}$")
            pred = parse(final_text)
            return bool(pred and _gold_cache[gold] and verify(_gold_cache[gold], pred))
        except Exception:
            return False
    # Without Math-Verify: compare the last \boxed{} answer as numbers, or as text.
    boxed = last_boxed(final_text)
    if boxed is None:
        return False
    a, b = _number(boxed), _number(gold)
    if a is not None and b is not None:
        return abs(a - b) < 1e-9
    return boxed.replace(" ", "") == gold.replace(" ", "")


def format_reward(completions, **kwargs):
    """0.5 for closing the thinking with exactly one </think>, and 0.5 more for a \\boxed{} answer after it.

    The prompt already ends with <think>, so a completion should not contain another opening tag.
    """
    rewards = []
    for c in completions:
        text = completion_text(c)
        r = 0.0
        if text.count(THINK_END) == 1 and "<think>" not in text:
            r += 0.5
            if "\\boxed{" in text.split(THINK_END)[1]:
                r += 0.5
        rewards.append(r)
    return rewards


def correctness_reward(completions, answer, **kwargs):
    """2.0 if the final answer after </think> is right, else 0. An answer cut off before </think> gets 0."""
    rewards = []
    for c, gold in zip(completions, answer):
        text = completion_text(c)
        right = THINK_END in text and same_answer(text.rsplit(THINK_END, 1)[-1], gold)
        rewards.append(2.0 if right else 0.0)
    return rewards


if __name__ == "__main__":
    done = "Add them up: 48 + 24 = 72.\n</think>\n\nNatalia sold \\boxed{72} clips."
    cases = {
        "right answer, right format": done,
        "wrong answer, right format": done.replace("72}", "70}"),
        "right number but never closed the thinking": "Add them up: 48 + 24 = 72. So \\boxed{72}",
        "closed the thinking, no \\boxed{}": "Add them up.\n</think>\n\nThe answer is 72.",
    }
    texts = list(cases.values())
    print("Math-Verify is", "installed" if parse else "not installed (using the plainer comparison)")
    print(f"{'case':<46}{'format':>8}{'correct':>9}")
    for name, f, c in zip(cases, format_reward(texts), correctness_reward(texts, ["72"] * len(texts))):
        print(f"{name:<46}{f:>8}{c:>9}")
