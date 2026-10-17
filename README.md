# Training a small R1-style reasoning model

The code for the 8-week schedule ("8-Week Schedule: Build Your Small Reasoning Model"). It teaches a base
model to write long step-by-step reasoning, first by fine-tuning it on traces written by a stronger model
(SFT), then by reinforcement learning on problems with checkable answers (GRPO).

| File | What it does | Week |
|---|---|---|
| `train.py` | `inspect` / `convert` / `train` / `merge`: LoRA fine-tuning on reasoning traces | 3 to 5, 8 |
| `harness.py` | `eval` / `table` / `peek`: score a model; `gen`: write your own teacher data | 2 to 8, 5 |
| `rewards.py` | the reward functions GRPO uses (format, correctness) | 6, 7 |
| `grpo.py` | GRPO with TRL, starting from the SFT model | 6, 7 |
| `chat.py` | try one prompt on a model | any |

## What DeepSeek did

DeepSeek-R1-Distill-Qwen-1.5B is `Qwen/Qwen2.5-Math-1.5B` fine-tuned with plain supervised learning on about
800k samples written by the full DeepSeek-R1 (671B). There is no reinforcement-learning stage in the distill.
The model learns to imitate R1's long chain-of-thought. The schedule adds a GRPO stage afterwards, as in
DeepSeek's R1-Zero experiments, because it is the part that teaches the most.

## What this project does

The same recipe at hobby scale, on a newer base model:

| | DeepSeek | This project |
|---|---|---|
| Base model | Qwen2.5-Math-1.5B (4,096-token context) | `Qwen/Qwen3-1.7B-Base` (32,768-token context); change it with `--base` |
| Teacher data | ~800k R1 samples (not public) | `bespokelabs/Bespoke-Stratos-17k` (R1-written) and your own, from `harness.py gen` |
| Method | full fine-tune | LoRA (r=64, all linear layers), then GRPO with a LoRA adapter |
| Output format | `<think>…</think>` then answer | same |

Why not DeepSeek's own base? Its 4,096-token limit drops most R1 traces, which are long. Qwen3-1.7B-Base reads
32,768 tokens and is Apache-2.0 licensed. Use `--base Qwen/Qwen3-0.6B-Base` for cheap runs, or
`--base Qwen/Qwen2.5-Math-1.5B` if you want DeepSeek's exact starting point.

Expect a model that has *learned to reason in the R1 style*, not one that matches the official scores. For
reference, DeepSeek reports 83.9 on MATH-500 and 28.9 on AIME 2024 for the official 1.5B distill. A few
thousand samples with LoRA will land well below that. Scaling up the data is the main lever.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Qwen3 needs `transformers` 4.51 or newer. On Colab or Kaggle, if you see import errors, upgrade the libraries
together: `pip install -U transformers peft accelerate datasets trl`.

You want an NVIDIA GPU. Rules of thumb (estimates, not measurements):

- **Qwen3-0.6B-Base** fits a free Colab or Kaggle GPU.
- **Qwen3-1.7B-Base** with LoRA is comfortable on a 24 GB card at the default `--max-len 4096`. On a 16 GB card,
  lower `--max-len` (try 2048) or add `--load-in-4bit` (needs `pip install bitsandbytes`).

It also runs on CPU or Apple silicon, but very slowly.

## Run it

```bash
# Check the data converts and see how long the traces are
python train.py inspect

# Smoke test with the small model: 200 samples, 1 epoch, just to prove the pipeline works
python train.py train --base Qwen/Qwen3-0.6B-Base --max-samples 200 --epochs 1 --out out-smoke

# The real (small) run on Qwen3-1.7B-Base
python train.py train

# Merge the LoRA adapter into the base weights (the adapter remembers its base model)
python train.py merge

# Try it
python chat.py --model out/merged "What is the sum of all positive divisors of 36?"
```

Use a fresh `--out` folder for each base model or experiment so runs do not overwrite each other. To merge the
smoke test: `python train.py merge --adapter out-smoke/adapter --output out-smoke/merged`.

### Measuring (Week 2 onwards)

```bash
# Score a model: 100 problems from each benchmark, 4 answers each. Add --engine hf if vLLM is not installed.
python harness.py eval --model Qwen/Qwen3-0.6B-Base --name base-0.6b
python harness.py eval --model out/merged --name sft-0.6b --bench gsm8k,math500,aime24 --samples 8

# All results side by side, and a look at the raw answers
python harness.py table results/*.json
python harness.py peek results/sft-0.6b.samples.jsonl --wrong --n 5
```

### Your own teacher data (Week 5)

```bash
# A free teacher writes answers to GSM8K/MATH training problems; only verified answers are kept
python harness.py gen --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --out data/teacher.jsonl
python train.py inspect --data data/teacher.jsonl

# Mix it with Bespoke-Stratos, then train on the mix
python train.py convert --n 2000 --out data/bespoke.jsonl
cat data/bespoke.jsonl data/teacher.jsonl > data/mix.jsonl
python train.py train --data data/mix.jsonl --max-samples 0 --out out-mix
```

### Reinforcement learning (Weeks 6 and 7)

```bash
python grpo.py --model out/merged --problems gsm8k --n 2000 --max-steps 100 --out out-grpo-easy
python train.py merge --adapter out-grpo-easy/adapter --output out-grpo-easy/merged
python harness.py eval --model out-grpo-easy/merged --name grpo-easy
```

Read `rewards.py` first: it is 140 lines of plain Python, and `python rewards.py` shows what each reward gives
for four example answers. If the model closes `</think>` and boxes an answer, it earns 1.0 for format; a
correct final answer earns 2.0 more. Answers that run past `--max-completion-length` earn nothing, which is how
the model learns to finish.

### Export to Ollama (Week 8)

Ollama runs GGUF files, so the merged model is converted with llama.cpp. Do this where you have internet
(Colab or Kaggle), then download the `.gguf` file to the computer that runs Ollama.

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py out/merged --outfile tiny-reasoner-q8_0.gguf --outtype q8_0
```

`q8_0` is close to lossless and needs no build step (about 1.8 GB for the 1.7B model). For a smaller Q4_K_M
file you also need llama.cpp's `llama-quantize` (built from the same repository, see its build notes), then
`llama-quantize tiny-reasoner-f16.gguf tiny-reasoner-Q4_K_M.gguf Q4_K_M` on an `--outtype f16` file.

Next to the `.gguf` file, create a text file named `Modelfile`:

```
FROM ./tiny-reasoner-q8_0.gguf
TEMPLATE """<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
<think>
{{ .Response }}<|im_end|>
"""
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER num_ctx 8192
```

Then `ollama create tiny-reasoner -f Modelfile` and `ollama run tiny-reasoner`. The template is the ChatML format that
`train.py` trained on, with `<think>` already opened (so the answer starts inside the thinking, as in
training). Add the same ending to your questions: "Please reason step by step, and put your final answer within
\boxed{}."

## Running it for free

A free Kaggle or Colab GPU (about 15 GB of memory, usually a T4) is enough for the smoke test and for
Qwen3-0.6B-Base runs. A Qwen3-1.7B-Base run should also fit if you add `--load-in-4bit --max-len 2048` (an
estimate, not a measurement). Time the 200-sample smoke test first: a 2,000-sample run takes roughly ten times
as long.

- **Kaggle:** create a notebook, pick the T4 GPU as the accelerator (Kaggle has listed it as `GPU T4 x2`; the
  P100 will not run vLLM) and turn Internet on (Kaggle may ask you to verify your phone number first). In the
  first cell run `!pip install -q -U transformers peft datasets accelerate bitsandbytes trl "math-verify[antlr4_13_2]"`.
  Get the scripts into the notebook's folder by uploading them as a dataset or by pasting each into a cell that
  starts with `%%writefile train.py`. Then start with the smoke test:
  `!CUDA_VISIBLE_DEVICES=0 python train.py train --base Qwen/Qwen3-0.6B-Base --max-samples 200 --epochs 1 --out out-smoke`.
- **Colab:** the same commands on a GPU runtime (Runtime, then Change runtime type). Mount Google Drive first
  (`from google.colab import drive; drive.mount('/content/drive')`) and point `--out` into it, for example
  `--out /content/drive/MyDrive/tiny-reasoner`, so your checkpoints survive a disconnect.
- **Two GPUs:** `train.py` uses one. If your session shows two, start it with `CUDA_VISIBLE_DEVICES=0`, because
  otherwise the Trainer may split each batch across both and change your effective batch size. Use the other GPU
  for evaluation.
- **Sessions end.** A free machine is wiped when the session closes. On Kaggle, files in `/kaggle/working` are
  kept only from a saved run of the notebook (Save Version, then Save & Run All). `train.py` and `grpo.py` write
  a checkpoint every `--save-steps` steps. If a run is cut off, start the same command again with `--resume`.
  That only works if the checkpoints survived, for example because `--out` is on Google Drive.

When a run finishes, copy `out/adapter` off the machine straight away: download it, or push it to the Hugging
Face Hub (make a free account, then create a token with write access at huggingface.co/settings/tokens):

```python
from huggingface_hub import HfApi, login

login()  # paste the token when asked
api = HfApi()
api.create_repo("YOUR-NAME/tiny-reasoner-adapter", private=True, exist_ok=True)
api.upload_folder(folder_path="out/adapter", repo_id="YOUR-NAME/tiny-reasoner-adapter")
```

## Things worth knowing

- **Long traces get dropped, not truncated.** Samples longer than `--max-len` (capped at the base model's
  context limit) are skipped so the final answer is never cut off. `inspect` shows what share fits at 2048,
  4096 and 8192 tokens. Training therefore favors the shorter problems.
- **Loss is on the response only.** The prompt tokens are masked, so the model learns to produce reasoning and
  answers, not to repeat questions.
- **No system prompt.** DeepSeek recommends putting all instructions in the user turn. The dataset's long
  system prompt is replaced by the `<think>` format.
- **The end token.** `train.py` teaches the model to end with `<|im_end|>`, but Qwen3 base models declare
  `<|endoftext|>` as their end token. `merge` fixes this in the saved model (tokenizer and generation settings),
  which TRL and anything else that calls `generate()` rely on. `chat.py` and `harness.py` stop at both.
- **Sampling.** Use temperature 0.5 to 0.7 (0.6 default) and force the response to start with `<think>\n`.
  `chat.py` and `harness.py` do both. If an answer stops in the middle of its thinking, raise `--max-new-tokens`
  (chat) or `--max-tokens` (harness). An answer that hits the cap counts as wrong in a score.
- **Tokenizer differs from the official model.** DeepSeek changed the tokenizer and chat template for the
  distills. This project keeps Qwen's ChatML format, so your model needs `--template chatml` (the default).
- **A T4 uses fp16.** GPUs without bf16 support train in fp16. If the loss turns into `nan`, move to a GPU with
  bf16 (L4, A10, A100, RTX 30/40 series).
- **Test sets stay out of training.** `harness.py gen` removes any training problem that also appears in
  GSM8K-test, MATH-500 or AIME 2024. Bespoke-Stratos is not checked this way.

## Knobs (train.py)

| Flag | Default | Notes |
|---|---|---|
| `--base` | `Qwen/Qwen3-1.7B-Base` | any Hugging Face causal LM with the same layer names |
| `--data` | Bespoke-Stratos | a JSONL file with `question`, `thought`, `solution` (also on `inspect`) |
| `--max-samples` | 2000 | `0` uses every usable row |
| `--max-len` | 4096 | lower it to save memory, raise it to keep long traces |
| `--epochs` | 2 | |
| `--lr` | 1e-4 | typical LoRA range is 1e-4 to 2e-4 |
| `--batch-size` / `--grad-accum` | 1 / 16 | effective batch size 16 |
| `--lora-r` | 64 | alpha is set to 2×r |
| `--save-steps` | 50 | write a checkpoint every N optimizer steps (what `--resume` continues from); `0` = once per epoch |
| `--resume` | off | continue from the newest checkpoint in `--out`; use the same options as the first run |

`grpo.py` has its own `--help`; the ones to know are `--num-generations` (8), `--prompts-per-step` (2),
`--max-completion-length` (1024), `--max-steps` and `--lr` (2e-5, a starting guess).

## What has and has not been tested

Checked offline, with stand-ins for the libraries (torch, transformers, datasets, peft, trl): row conversion,
the prompt format and that `train.py`, `harness.py` and `rewards.py` build the same prompt, loss masking,
dropping over-long samples, reading your own JSONL, `convert`, the end-token fix, the step count, the fp16/bf16
choice, the checkpoint lookup behind `--resume`, how `train` and `merge` pass their options on, the answer
filters in `harness.py gen`, the score summary, and how `grpo.py` computes its batch sizes and passes its
settings to TRL.

Not run end to end: training, resuming, merging, scoring with vLLM or Math-Verify, GRPO, the GGUF export and
Ollama. They need a GPU, a network and the real libraries, which were not available where this was written, so
the stand-ins cannot show that the real libraries accept those arguments. Math-Verify in particular was never
run: `rewards.py` has a plainer fallback, but the fallback is not meant for MATH answers. The GRPO and Ollama
settings follow the current TRL and Ollama documentation, the learning rates and step counts are starting
guesses. Start with `inspect` and the 200-sample smoke test, and expect to fix small version differences.
