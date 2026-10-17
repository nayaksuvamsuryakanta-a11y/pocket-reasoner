# Project Code Summary

## 1. File list

Line counts below are physical text lines as present before this summary was created. The `tools/` row is a directory, so it has no file line count.

| Name | Lines | Purpose |
|---|---:|---|
| [README.md](../README.md#L1) | 245 | Explains the eight-week small reasoning-model plan, usage commands, and export path. |
| [.gitignore](../.gitignore#L1) | 9 | Excludes environments, generated outputs/data, model weights, and `.env` files. |
| [requirements.txt](../requirements.txt#L1) | 16 | Lists required Python packages and comments on optional GPU packages. |
| [train.py](../train.py#L1) | 433 | Converts reasoning data, runs LoRA supervised fine-tuning, and merges its adapter. |
| [harness.py](../harness.py#L1) | 579 | Evaluates models, displays results/answers, and can generate checked teacher data. |
| [rewards.py](../rewards.py#L1) | 138 | Builds GRPO prompts and scores answer format and correctness. |
| [grpo.py](../grpo.py#L1) | 194 | Runs a GRPO reinforcement-learning stage using the model from supervised training. |
| [chat.py](../chat.py#L1) | 58 | Generates one answer from a selected model and prompt. |
| [tools/read_samples.py](read_samples.py#L1) | 59 | Downloads/reads sample data through the training helpers, prints samples, and saves them. |
| [tools/samples.txt](samples.txt#L1) | 2228 | Contains three rendered dataset examples, including their questions, thinking, solutions, and token counts. |
| [tools/notes_week1_monday.md](notes_week1_monday.md#L1) | 35 | Holds the worksheet notes for the three rendered samples. |
| [tools/codebase_summary.md](codebase_summary.md#L1) | 216 | This line-cited project overview. |
| `tools/` | directory | Contains the sample-generation script, generated sample text, and worksheet ([read_samples.py](read_samples.py#L31), [samples.txt](samples.txt#L1), [notes_week1_monday.md](notes_week1_monday.md#L1)). |

## 2. Python files

### train.py

This is the supervised fine-tuning (SFT: teaching a model from example answers) entry point. It can inspect or convert the Bespoke-Stratos data, train LoRA adapters, and merge an adapter back into its base model ([train.py](../train.py#L1), [train.py](../train.py#L383)).

**Commands and flags:** The required subcommand is `inspect`, `convert`, `train`, or `merge` ([train.py](../train.py#L385)). `inspect`: `--base=Qwen/Qwen3-1.7B-Base` (L388; constant at L37), `--n=500` (L389), `--data=None` (L390), `--seed=42` (L391). `convert`: `--n=2000` (L395), `--out=data/bespoke.jsonl` (L396), `--seed=42` (L397). `train`: `--base=Qwen/Qwen3-1.7B-Base` (L401-L402), `--out=out` (L403), `--data=None` (L404-L405), `--max-samples=2000` (L406), `--max-len=4096` (L407-L408), `--epochs=2` (L409), `--lr=1e-4` (L410), `--batch-size=1` (L411), `--grad-accum=16` (L412), `--lora-r=64` (L413), `--load-in-4bit=off` (L414), `--save-steps=50` (L415-L416), `--resume=off` (L417-L418), `--seed=42` (L419). `merge`: `--adapter=out/adapter` (L423), `--output=out/merged` (L424), `--base=None` (L425; it reads the base name from the adapter when omitted).

**Main functions:**

| Function | What it does |
|---|---|
| [`convert_row`](../train.py#L54) | Extracts the question, thinking, and solution from one Bespoke-Stratos conversation. |
| [`build_prompt`](../train.py#L67) | Formats a question as a ChatML user turn and assistant prefix. |
| [`build_response`](../train.py#L72) | Places the thinking in `<think>` tags, then appends the solution and end token. |
| [`encode`](../train.py#L77) | Tokenizes the prompt and response separately. |
| [`precision`](../train.py#L83) | Chooses bf16, fp16, or fp32 from the available CUDA hardware. |
| [`load_causal_lm`](../train.py#L96) | Loads a Transformers causal language model while supporting old/new precision argument names. |
| [`plan_steps`](../train.py#L111) | Estimates optimizer and warm-up steps. |
| [`last_checkpoint`](../train.py#L125) | Finds the newest `checkpoint-N` folder for resuming. |
| [`read_jsonl_rows`](../train.py#L133) | Validates custom JSONL rows with `question`, `thought`, and `solution` fields. |
| [`triples`](../train.py#L157) | Yields examples from a custom file or the Hugging Face dataset. |
| [`load_examples`](../train.py#L171) | Encodes examples, drops invalid/too-long rows, and masks prompt labels. |
| [`cmd_inspect`](../train.py#L199) | Reports format validity and token-length statistics. |
| [`cmd_train`](../train.py#L226) | Builds the model, LoRA adapter, trainer, and checkpoint output. |
| [`cmd_convert`](../train.py#L327) | Writes Bespoke-Stratos examples as JSONL. |
| [`set_stop_tokens`](../train.py#L344) | Adds ChatML `<|im_end|>` to the saved model's stop-token IDs when available. |
| [`cmd_merge`](../train.py#L365) | Merges a LoRA adapter into its base weights and saves the merged model. |
| [`main`](../train.py#L383) | Defines the four subcommands and dispatches to their handlers. |

**Imports:** It imports no other project module. Its top-level imports are Python standard-library modules (L28-L35); model/data libraries are imported inside commands, including `datasets` at L165, and `torch`, `peft`, and `transformers` at L227-L229.

### harness.py

This is the evaluation and data-generation tool: it samples answers, checks them against known answers, summarizes scores, and filters teacher answers into possible training rows ([harness.py](../harness.py#L1), [harness.py](../harness.py#L318), [harness.py](../harness.py#L475)).

**Commands and flags:** Commands are `eval`, `table`, `peek`, and `gen` ([harness.py](../harness.py#L534), [harness.py](../harness.py#L547), [harness.py](../harness.py#L551), [harness.py](../harness.py#L559)). Shared generation defaults: `--samples=4` (L517), `--max-tokens=4096` for `eval` and `6000` for `gen` (L518, with values passed at L544 and L571), `--temperature=0.6` (L520), `--top-p=0.95` (L521), `--seed=0` (L522), `--engine=vllm` (L523-L524), `--batch-size=2` (L525), `--max-model-len=None` (L526), and `--gpu-memory-utilization=0.9` (L527). `eval`: required `--model` (L535), `--name=None` (L536), `--template=chatml` (L537), `--bench=gsm8k,math500` (L540), `--n=100` problems per benchmark (L541-L542), `--out=results` (L543). `table`: positional `files` defaults to an empty list (L548). `peek`: required positional `file` (L552), `--n=5` (L553), `--wrong=off` (L554), `--bench=None` (L555), `--seed=0` (L556). `gen`: required `--teacher` (L560), `--template=tokenizer` (L562), `--problems=gsm8k,math` (L563), `--n=400` per source (L564), `--keep=2` correct answers per problem (L565), `--max-trace-tokens=3500` (L566-L567), `--no-decontaminate=off` (L568-L569), and `--out=data/teacher.jsonl` (L570).

**Main functions:**

| Function | What it does |
|---|---|
| [`gsm8k_gold`](../harness.py#L42) | Extracts the GSM8K answer after `####`. |
| [`last_boxed`](../harness.py#L47) | Reads the last `\boxed{...}` answer, including nested braces. |
| [`pick_subset`](../harness.py#L64) | Chooses a repeatable subset using a seed. |
| [`load_eval_problems`](../harness.py#L72) | Loads GSM8K, MATH-500, or AIME 2024 test problems. |
| [`load_train_problems`](../harness.py#L92) | Loads GSM8K or MATH training problems and known answers. |
| [`build_prompt`](../harness.py#L114) | Adds the boxed-answer instruction and the chosen chat template. |
| [`stop_ids_for`](../harness.py#L127) | Selects stop-token IDs for generation. |
| [`Gen`](../harness.py#L143) | Stores generated text, whether it finished, and its token count. |
| [`gpu_dtype_name`](../harness.py#L149) | Chooses the vLLM precision name from GPU capability. |
| [`generate_vllm`](../harness.py#L158) | Generates answers with optional vLLM. |
| [`read_generated_ids`](../harness.py#L175) | Removes tokens after a stop token and reports whether one was found. |
| [`generate_hf`](../harness.py#L183) | Generates answers with standard Transformers. |
| [`generate`](../harness.py#L217) | Selects vLLM or Transformers generation. |
| [`require_math_verify`](../harness.py#L227) | Stops with an install hint when Math-Verify is unavailable. |
| [`final_part`](../harness.py#L234) | Keeps the text after the last `</think>` for answer checking. |
| [`parse_gold`](../harness.py#L239) | Parses a known answer with Math-Verify. |
| [`check_answer`](../harness.py#L248) | Checks a finished answer and rejects cut-off generations. |
| [`make_checker`](../harness.py#L265) | Caches parsed known answers while checking many outputs. |
| [`percentile`](../harness.py#L277) | Selects a percentile from sorted values. |
| [`summarize`](../harness.py#L283) | Computes pass@1, uncertainty, token use, and completion statistics. |
| [`describe`](../harness.py#L307) | Formats one benchmark's score summary for printing. |
| [`cmd_eval`](../harness.py#L318) | Runs model evaluation and writes a JSON summary and JSONL answers. |
| [`cmd_table`](../harness.py#L370) | Prints saved evaluation summaries side by side. |
| [`cmd_peek`](../harness.py#L394) | Prints selected raw answers, optionally only wrong answers. |
| [`norm_question`](../harness.py#L414) | Normalizes a question for overlap checks. |
| [`looks_repetitive`](../harness.py#L418) | Flags very compressible long text as likely repetition. |
| [`filter_traces`](../harness.py#L424) | Keeps finished, correct, non-repetitive, short, unique, non-overlapping teacher answers. |
| [`cmd_gen`](../harness.py#L475) | Generates teacher answers and writes filtered training rows plus a card JSON file. |
| [`add_generation_options`](../harness.py#L516) | Adds flags shared by evaluation and teacher generation. |
| [`main`](../harness.py#L530) | Defines the four commands and their arguments. |

**Imports:** It does not import another project module. Top-level imports are standard-library modules (L22-L31); model/data dependencies are loaded in functions, such as `datasets` at L73 and `transformers` at L320/L477. vLLM is imported only by the vLLM engine ([harness.py](../harness.py#L158)); Math-Verify is checked by the evaluation/generation commands ([harness.py](../harness.py#L227), [harness.py](../harness.py#L318)).

### rewards.py

This module builds the same kind of thinking prompt as training and provides format and correctness reward functions for GRPO (group-relative policy optimization: comparing several answers to the same problem) ([rewards.py](../rewards.py#L1), [rewards.py](../rewards.py#L32), [rewards.py](../rewards.py#L99)). Running `python rewards.py` shows a small demonstration; it defines no command-line flags ([rewards.py](../rewards.py#L126)).

**Main functions:** [`make_prompt`](../rewards.py#L32) formats a question; [`gsm8k_gold`](../rewards.py#L37) extracts a GSM8K answer; [`completion_text`](../rewards.py#L42) handles string or chat-message completions; [`last_boxed`](../rewards.py#L51) extracts a boxed answer; [`_number`](../rewards.py#L68) tries to parse a simple numeric string; [`same_answer`](../rewards.py#L78) compares a prediction with the known answer; [`format_reward`](../rewards.py#L99) scores closing `</think>` and boxing; [`correctness_reward`](../rewards.py#L116) scores a correct answer after `</think>`.

**Imports:** No other project file is imported. `re` is from the standard library (L21); Math-Verify is optional here, with a simpler fallback when it is missing (L23-L27, L78-L97).

### grpo.py

This script applies GRPO to the SFT model, samples several completions per problem, and uses the two reward functions from `rewards.py` ([grpo.py](../grpo.py#L1), [grpo.py](../grpo.py#L173)). It saves a new LoRA adapter rather than replacing the original model ([grpo.py](../grpo.py#L186-L190)).

**Flags:** `--model=out/merged` (L88), `--out=out-grpo` (L89), `--problems=gsm8k` (L90), `--n=2000` per source (L91), `--max-steps=100` (L92), `--num-generations=8` (L93), `--prompts-per-step=2` (L94), `--per-device-batch=2` (L95), `--max-completion-length=1024` (L96-L97), `--max-prompt-tokens=512` (L98), `--temperature=0.8` (L99), `--lr=2e-5` (L100), `--lora-r=32` (L101), `--save-steps=25` (L102), `--resume=off` (L103), `--vllm=off` (L104), `--vllm-memory=0.3` (L105), and `--seed=42` (L106).

**Main functions:** [`plan_batches`](../grpo.py#L38) checks that each generation group divides into the requested micro-batch; [`largest_divisor_up_to`](../grpo.py#L54) suggests a valid batch size; [`reward_self_check`](../grpo.py#L59) removes problems whose own answer the reward checker cannot parse; [`build_rows`](../grpo.py#L69) makes token-limited prompt/answer rows; [`main`](../grpo.py#L86) loads dependencies, builds the trainer, trains, and saves the adapter.

**Imports:** It imports `rewards.py` for answer rewards and prompt building (L29, L31), `harness.py` for training-source names and problem loading (L30), and `train.py` for LoRA targets, checkpoint lookup, model loading, precision selection, and stop-token setup (L32).

### chat.py

This script formats one prompt using ChatML or a model's own tokenizer template, generates an answer, and prints it ([chat.py](../chat.py#L20), [chat.py](../chat.py#L39-L58)). It does not import another project module ([chat.py](../chat.py#L12-L15)).

**Flags:** optional positional `prompt` defaults to a divisors-of-36 question (L22); `--model=out/merged` (L23), `--template=chatml` (L24), `--max-new-tokens=2048` (L25-L26), `--temperature=0.6` (L27), and `--top-p=0.95` (L28).

**Main function:** [`main`](../chat.py#L20) loads the model/tokenizer, formats the prompt, generates tokens, and prints the decoded answer.

**Imports:** `torch` and Transformers' `AutoModelForCausalLM`/`AutoTokenizer` are imported at module load (L14-L15); these packages are listed in requirements.txt (L1, L3).

### tools/read_samples.py

This top-level script takes the first usable rows from the configured public dataset, formats each sample, counts marker words, then prints and saves the result ([tools/read_samples.py](read_samples.py#L31-L59)). It defines no named functions; its steps run when the script is executed.

**Flags:** `--n=3` samples (L26), `--seed=42` (L27), and `--out=tools/samples.txt` (L28).

**Imports:** It imports dataset/model tokenizer packages (L12-L13) and imports `DEFAULT_BASE`, `DATASET`, `triples`, `build_prompt`, `build_response`, and `encode` from `train.py` (L22).

## 3. How the files connect

1. **Prepare data.** The default training source is `bespokelabs/Bespoke-Stratos-17k` ([train.py](../train.py#L37-L38), [train.py](../train.py#L157-L169)). Check it with `python train.py inspect` ([README.md](../README.md#L62-L63)). Optional conversions write `data/bespoke.jsonl` with `python train.py convert --n 2000 --out data/bespoke.jsonl` ([train.py](../train.py#L394-L398), [train.py](../train.py#L327-L342)). Optional teacher-data generation is `python harness.py gen --teacher MODEL --out data/teacher.jsonl`; it writes that JSONL and a matching `.card.json` ([README.md](../README.md#L96-L103), [harness.py](../harness.py#L498-L507)).
2. **Train (SFT).** Run `python train.py train` or pass a custom JSONL using `--data`; the trainer writes resumable checkpoints under `out/checkpoints/` and the final adapter under `out/adapter/` ([train.py](../train.py#L295-L324)).
3. **Merge.** Run `python train.py merge`; it combines the adapter with its base model and writes `out/merged/` ([README.md](../README.md#L71-L72), [train.py](../train.py#L365-L380)).
4. **Evaluate.** Run `python harness.py eval --model out/merged --name sft`; it writes `results/sft.json` and `results/sft.samples.jsonl` by default ([README.md](../README.md#L84-L90), [harness.py](../harness.py#L354-L366)).
5. **Reinforcement learning.** Run `python grpo.py --model out/merged --problems gsm8k --n 2000 --max-steps 100 --out out-grpo-easy`; this writes `out-grpo-easy/checkpoints/` and `out-grpo-easy/adapter/`. Merge with `python train.py merge --adapter out-grpo-easy/adapter --output out-grpo-easy/merged`, then evaluate that merged model ([README.md](../README.md#L106-L111), [grpo.py](../grpo.py#L147-L168), [grpo.py](../grpo.py#L186-L190)).
6. **Chat/export.** Try it locally with `python chat.py --model out/merged "your question"`; chat prints its answer and does not save a result file ([README.md](../README.md#L74-L75), [chat.py](../chat.py#L57-L58)). For Ollama, the README converts the merged model to GGUF with external llama.cpp tools and then creates/runs an Ollama model ([README.md](../README.md#L119-L150)).

## 4. Data format

For the default dataset, `convert_row` takes the first `user`/`human` message as the question and the first `assistant`/`gpt`/`model` message as the answer; it extracts text between the thought and solution markers, and rejects rows missing those fields/markers ([train.py](../train.py#L43-L64)). A custom JSONL line must have non-empty string fields `question`, `thought`, and `solution`; invalid rows are rejected ([train.py](../train.py#L133-L155)).

The prompt text and response text are built separately, then tokenized without automatic special tokens ([train.py](../train.py#L67-L80)):

```python
return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
return f"<think>\n{thought.strip()}\n</think>\n\n{solution.strip()}<|im_end|>"
prompt_ids = tok(build_prompt(question), add_special_tokens=False)["input_ids"]
resp_ids = tok(build_response(thought, solution), add_special_tokens=False)["input_ids"]
```

The training example concatenates those token IDs. Prompt tokens get label `-100` (ignored by the loss); response tokens keep their token IDs and receive loss. Padding labels also use `-100`, and samples longer than the length limit are dropped rather than cut off ([train.py](../train.py#L171-L196), [train.py](../train.py#L275-L285)). The model therefore learns to produce the thinking, solution, and `<|im_end|>` response, but not to reproduce the question.

```python
"input_ids": prompt_ids + resp_ids,
"labels": [-100] * len(prompt_ids) + resp_ids,
```

## 5. Settings that matter

| Setting | Default | Meaning and source |
|---|---:|---|
| SFT max length | 4096 tokens | `--max-len`; capped again at the base model's context limit ([train.py](../train.py#L407-L408), [train.py](../train.py#L236-L241)). |
| SFT learning rate | 0.0001 | `--lr` ([train.py](../train.py#L410)). |
| SFT LoRA size | r=64 | `--lora-r`; alpha is twice r ([train.py](../train.py#L413), [train.py](../train.py#L260-L262)). |
| SFT epochs | 2 | `--epochs` ([train.py](../train.py#L409)). |
| SFT device batch | 1 | `--batch-size`; gradient accumulation is 16, giving 16 examples per optimizer update when all batches are full ([train.py](../train.py#L411-L412)). |
| SFT sample cap | 2000 usable examples | `--max-samples`; 0 means no cap ([train.py](../train.py#L406)). |
| Eval generation | 4 answers/problem, 4096 max tokens, 100 problems/benchmark | Defaults in `harness.py` ([harness.py](../harness.py#L517-L519), [harness.py](../harness.py#L541-L544)). |
| Teacher generation | 400 problems/source, 2 retained answers/problem, 6000 max generation tokens, 3500 max retained trace tokens | Defaults in `harness.py` ([harness.py](../harness.py#L564-L571)). |
| GRPO | 100 steps, 8 generations/problem, 2 problems/step, batch 2, completion cap 1024, LR 0.00002, LoRA r=32 | Defaults in `grpo.py` ([grpo.py](../grpo.py#L91-L101)). GRPO uses steps rather than an epoch setting ([grpo.py](../grpo.py#L147-L158)). |

## 6. GPU and local requirements

| File/operation | Packages from requirements.txt or imports | Hardware notes |
|---|---|---|
| `train.py` | `torch`, `transformers`, `datasets`, `peft`, and `accelerate` ([requirements.txt](../requirements.txt#L1-L6), [train.py](../train.py#L165-L167), [train.py](../train.py#L227-L229)). `bitsandbytes` is optional for `--load-in-4bit` ([requirements.txt](../requirements.txt#L10-L11)). | A GPU is recommended; code selects CUDA precision when available and warns that CPU training is very slow ([train.py](../train.py#L83-L94), [train.py](../train.py#L242-L245)). |
| `harness.py eval/gen` | `transformers`, `datasets`, and Math-Verify ([requirements.txt](../requirements.txt#L3-L7), [harness.py](../harness.py#L318-L324), [harness.py](../harness.py#L475-L486)). vLLM is optional but is the default engine; `--engine hf` uses Transformers instead ([harness.py](../harness.py#L158-L173), [harness.py](../harness.py#L523-L527), [requirements.txt](../requirements.txt#L12-L16)). | GPU recommended. vLLM needs a supported GPU; Transformers mode can run without vLLM but is described as slow ([requirements.txt](../requirements.txt#L12-L15), [harness.py](../harness.py#L523-L525)). |
| `harness.py table/peek` | Python standard library; it reads saved JSON/JSONL rather than loading a model ([harness.py](../harness.py#L370-L412)). | No GPU needed for these commands. |
| `rewards.py` | Standard library plus optional Math-Verify; the fallback is simpler ([rewards.py](../rewards.py#L21-L27), [rewards.py](../rewards.py#L78-L97)). | No GPU needed to calculate rewards. |
| `grpo.py` | `torch`, `datasets`, `peft`, `transformers`, `trl`, and Math-Verify ([requirements.txt](../requirements.txt#L1-L9), [grpo.py](../grpo.py#L109-L117)). | A GPU is strongly recommended; the code warns that GRPO is extremely slow without CUDA ([grpo.py](../grpo.py#L139-L145)). vLLM use is optional ([grpo.py](../grpo.py#L169-L171)). |
| `chat.py` | `torch` and `transformers` ([requirements.txt](../requirements.txt#L1-L3), [chat.py](../chat.py#L14-L15)). | Can load CPU fp32, but GPU is faster ([chat.py](../chat.py#L31-L35)). |
| `tools/read_samples.py` | `datasets`, `transformers`, and the helpers in `train.py` ([tools/read_samples.py](read_samples.py#L12-L13), [tools/read_samples.py](read_samples.py#L22)). | It tokenizes examples and does not train; downloading the dataset/tokenizer requires network access ([tools/read_samples.py](read_samples.py#L31-L40)). |

The requirements file does not pin a Python version, so the supported Python version is not clear from the code ([requirements.txt](../requirements.txt#L1-L16)).

## 7. Things that look risky or unfinished

- **End-to-end status:** The README says real training, resume, merge, vLLM/Math-Verify scoring, GRPO, GGUF export, and Ollama were not run end to end ([README.md](../README.md#L231-L245)).
- **Optional default engine:** `harness.py` defaults to `vllm`, but vLLM is commented out of requirements; use `--engine hf` if it is not installed ([harness.py](../harness.py#L523-L525), [requirements.txt](../requirements.txt#L12-L16)).
- **GPU precision differs:** training picks bf16 only for newer CUDA devices and fp16 for older ones, while `chat.py` chooses bf16 for any CUDA device. Chat may therefore choose a precision unsupported or slow on an older GPU ([train.py](../train.py#L83-L94), [harness.py](../harness.py#L149-L156), [chat.py](../chat.py#L31-L35)).
- **Merge precision is fixed:** `train.py merge` always loads the base as bfloat16 instead of using the hardware-aware `precision()` helper; verify this on CPU or older GPUs ([train.py](../train.py#L83-L94), [train.py](../train.py#L365-L379)).
- **Math fallback is limited:** `rewards.py` falls back to comparing boxed text/numbers when Math-Verify is absent, and the README says this fallback is not meant for MATH answers ([rewards.py](../rewards.py#L78-L97), [README.md](../README.md#L240-L244)). GRPO itself exits if Math-Verify is missing ([grpo.py](../grpo.py#L109-L110)).
- **Windows setup friction:** README setup and data-mixing examples use Unix `source` and `cat` commands; a Windows PowerShell user needs equivalent commands ([README.md](../README.md#L41-L45), [README.md](../README.md#L100-L103)).
- **Export is external:** llama.cpp and Ollama are not Python requirements; the README instructs the user to clone/install llama.cpp and invoke Ollama separately ([README.md](../README.md#L119-L150), [requirements.txt](../requirements.txt#L1-L16)).
- **TODO/FIXME comments:** None were found in the six Python source files inspected ([train.py](../train.py#L1), [harness.py](../harness.py#L1), [rewards.py](../rewards.py#L1), [grpo.py](../grpo.py#L1), [chat.py](../chat.py#L1), [tools/read_samples.py](read_samples.py#L1)).

## 8. Current state of the folder

The visible project files are listed in Section 1; the inventory excludes `.venv/`, `__pycache__/`, and `.git/` contents as requested. A `.git` directory exists, so this appears to be a Git working tree, but its contents were not inspected. `.gitignore` exists and contains ([.gitignore](../.gitignore#L1-L9)):

```text
.venv/
__pycache__/
out/
out-*/
data/
*.gguf
*.safetensors
*.bin
.env
```