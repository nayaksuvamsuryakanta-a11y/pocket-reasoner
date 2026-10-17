import argparse
import itertools
import re
import sys
from pathlib import Path

# Configure terminal output before loading samples.
sys.stdout.reconfigure(encoding="utf-8")

# Check the two external packages without attempting installation.
try:
    import datasets
    from transformers import AutoTokenizer
except ModuleNotFoundError as error:
    package = error.name if error.name in {"datasets", "transformers"} else "transformers or datasets"
    print(f"Missing required package: {package}. This script does not install packages.")
    raise SystemExit(1)

# Add the project folder so the training data helpers can be imported.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from train import DEFAULT_BASE, DATASET, triples, build_prompt, build_response, encode

# Parse the sample count, seed, and output path.
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", type=Path, default=Path("tools/samples.txt"))
args = parser.parse_args()

# Load the tokenizer and select the first usable rows.
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE)
samples = itertools.islice((row for row in triples(None, args.seed) if row is not None), args.n)
output = [f"Dataset: {DATASET}"]

# Format each complete sample and count the requested markers.
for number, parts in enumerate(samples, start=1):
    question, thought, solution = parts
    prompt_ids, response_ids = encode(question, thought, solution, tokenizer)
    counts = {
        "Wait": len(re.findall(r"\bwait\b", thought, re.IGNORECASE)),
        "Let me": len(re.findall(r"\blet\s+me\b", thought, re.IGNORECASE)),
        "Alternatively": len(re.findall(r"\balternatively\b", thought, re.IGNORECASE)),
        "Hmm": len(re.findall(r"\bhmm\b", thought, re.IGNORECASE)),
        "double-check/double check": len(re.findall(r"\bdouble[- ]check\b", thought, re.IGNORECASE)),
    }
    output.extend([
        f"\n{'=' * 72}\nSample {number} | Total tokens: {len(prompt_ids) + len(response_ids)}\n{'=' * 72}",
        f"Question:\n{question}",
        f"Thought:\n{thought}",
        f"Marker counts: {counts}",
        f"Solution:\n{solution}",
    ])

# Print and save identical UTF-8 text.
text = "\n\n".join(output) + "\n"
print(text, end="")
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(text, encoding="utf-8")