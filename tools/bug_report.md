# Bug Audit Report

## 1. Overview

- `train.py` prepares reasoning examples, runs LoRA supervised fine-tuning, and merges adapters ([train.py:159](../train.py#L159), [train.py:228](../train.py#L228), [train.py:369](../train.py#L369)).
- `harness.py` evaluates generated answers and can make filtered teacher-data rows ([harness.py:318](../harness.py#L318), [harness.py:475](../harness.py#L475)).
- `rewards.py` computes format/correctness rewards; `grpo.py` trains an adapter using those rewards ([rewards.py:99](../rewards.py#L99), [grpo.py:175](../grpo.py#L175)).
- `chat.py` runs one prompt; `tools/read_samples.py` formats examples from the public training set ([chat.py:20](../chat.py#L20), [read_samples.py:31](read_samples.py#L31)).
- `tools/codebase_summary.md` is **out of date**: its inventory omits the Tuesday worksheet and group-advantage script, and it still says chat uses bf16 on every CUDA GPU ([codebase_summary.md:16](codebase_summary.md#L16), [codebase_summary.md:195](codebase_summary.md#L195)); current `chat.py` checks compute capability ([chat.py:32](../chat.py#L32)).
- A static pass over the CLI declarations found no parsed option that is plainly unused ([train.py:385](../train.py#L385), [harness.py:530](../harness.py#L530), [grpo.py:86](../grpo.py#L86), [chat.py:20](../chat.py#L20)).

## 2. Findings

| ID | File:line | Severity | Evidence quote | How verified |
|---|---|---|---|---|
| F-01 | [train.py:218](../train.py#L218) | confirmed bug | `statistics.quantiles(lengths, n=10)[-1] if len(lengths) > 1 else lengths[0]` | The one-row check failed before the fix with `StatisticsError: must have at least two data points`; it passes now and reports p90 as the sole length. |
| F-02 | [train.py:117](../train.py#L117), [grpo.py:45](../grpo.py#L45) | confirmed bug | `if batch_size < 1 or grad_accum < 1:`; `if per_device_batch < 1:` | Before guards, zero values caused `ZeroDivisionError`/`ZeroDivisionError` at the division and modulo. After guards, the checks require clear `SystemExit` errors and pass. |
| R-01 | [harness.py:280](../harness.py#L280) | risk | `int(q * len(sorted_values))` | With sorted values 1 through 10 and q=0.9, the current index selects 10. The code does not name its percentile convention; this differs from nearest-rank p90, which selects 9 for this input. |
| R-02 | [harness.py:288](../harness.py#L288) | risk | `if n > 1 else 0.0` | A one-problem summary reports `ci95=0.0`, even though a confidence interval cannot be estimated from one problem. Confirmed with one in-memory sample. |
| R-03 | [harness.py:455](../harness.py#L455), [harness.py:464](../harness.py#L464) | risk | `taken, capped = {}, []` followed later by `key = (row["question"], row["thought"])` | A tiny three-answer example with the first two thoughts identical and `keep=2` returns one unique row; the third unique answer is discarded by the cap before deduplication. It does satisfy the stated maximum. |
| R-04 | [README.md:234](../README.md#L234), [train.py:69](../train.py#L69), [rewards.py:34](../rewards.py#L34) | risk | README says the three scripts “build the same prompt”; training's prompt ends at the assistant prefix, while rewards adds the boxed-answer suffix and `<think>` opener. | Compared the literal format strings. `harness.py` also adds the suffix and `<think>` opener ([harness.py:116](../harness.py#L116), [harness.py:123](../harness.py#L123)). This is a documentation mismatch; whether the format difference is intentional is not clear from the code. |
| R-05 | [README.md:44](../README.md#L44) | risk | `source .venv/bin/activate` | This setup command is for a Bash-style shell and has no PowerShell alternative in the setup section. The README later provides Linux/Colab/Kaggle commands ([README.md:155](../README.md#L155)); native PowerShell setup is not documented. |
| R-06 | [.gitignore:9](../.gitignore#L9) | risk | `.env` | The exact `.env` name is ignored, as are `.gguf`, `.safetensors`, and `.bin` model files ([.gitignore:6](../.gitignore#L6), [.gitignore:7](../.gitignore#L7), [.gitignore:8](../.gitignore#L8)); `.env.local` and other `.env.*` names are not covered by these patterns. A scan for common token-shaped strings found no matches in current Python, Markdown, or text files. Exact index membership was not queried. |
| R-07 | [train.py:375](../train.py#L375) | risk | `base = load_causal_lm(base_name, dtype=torch.bfloat16)` | Merge hard-codes bf16 instead of using the hardware-aware precision helper ([train.py:83](../train.py#L83)). Whether this fails or is slow on the target CPU/GPU depends on PyTorch and hardware; not changed without that verification. |
| S-01 | [codebase_summary.md:16](codebase_summary.md#L16), [codebase_summary.md:195](codebase_summary.md#L195) | style | The file list reports `chat.py` as 58 lines and says it selects bf16 on any CUDA GPU. | The current tree has additional Tuesday/audit files, and `chat.py` now branches on GPU capability. The summary should be refreshed. |

## 3. Fixed

- **F-01, inspect with one valid row:** commit `bf08f31f7f49d104dba71f50683b75e84b3968cb`. The check supplies one hand-made row and checks token-statistics output ([audit_checks.py:32](audit_checks.py#L32)). Before the fix, it failed with `StatisticsError`; after the fix, it passed.
- **F-02, nonpositive batch divisors:** commit `74e660d36da1629e8e14c86c1bf3f6ff3f21ddb3`. The checks exercise zero `batch_size`, `grad_accum`, and `per_device_batch` ([audit_checks.py:61](audit_checks.py#L61), [audit_checks.py:68](audit_checks.py#L68)); before the guards all three errored with `ZeroDivisionError`, and after they all pass by raising `SystemExit`.
- The audit test harness was changed to use only the standard library and source-extracted functions in commit `dc3b9c52156a54b72a31cf3c85cf232e4f1a6322` ([audit_checks.py:19](audit_checks.py#L19)). Current result: `Ran 3 tests ... OK`. No training, model load, or evaluation was run.

## 4. Not fixed

- **R-01:** No change proposed yet; select and document a percentile convention first. If nearest-rank is intended, use index `ceil(q * n) - 1` with a lower bound of zero.
- **R-02:** No change proposed yet; decide whether a one-problem interval should be reported as unavailable (`null`/`n/a`) instead of zero. Changing the result shape may affect consumers.
- **R-03:** Proposed change: discard duplicate `(question, thought)` rows while traversing candidates, then enforce the per-problem `keep` limit, preserving first-seen order. Not changed because the current “first N, then dedupe” behavior may be intentional.
- **R-04:** Proposed documentation change: state that the scripts share a related ChatML format but evaluation/reward prompts add an instruction suffix and pre-open `<think>`. No prompt template was changed.
- **R-05:** Proposed documentation change: add the equivalent PowerShell virtual-environment activation command and identify Bash-only commands.
- **R-06:** Proposed ignore rules: cover `.env.*` while exempting a non-secret `.env.example`; decide whether additional model extensions such as `.pt`/`.pth` should be ignored. No ignore rules were changed.
- **R-07:** Potential code change: select merge dtype using the existing hardware-aware helper. Not applied because merge dtype behavior on this machine and target hardware could not be tested.
- **S-01:** Proposed documentation change: refresh the file inventory and remove the stale chat precision statement. No existing documentation file was modified.

## 5. Needs my decision

- Which percentile convention should `harness.py` use for p90: nearest-rank or its current index rule?
- Should a single-problem confidence interval be `n/a` rather than zero?
- Should deduplication happen before the per-problem keep cap, so later unique answers can fill the quota?
- Is the prompt difference documented in R-04 intentional, or should the README describe the templates more precisely?
- Should `.env.*` be ignored, and should `.env.example` be explicitly exempted?

## 6. Could not verify

- GPU training, checkpoint resume, adapter merge, model evaluation, GRPO training, and chat inference were not run. The README also says the end-to-end training/evaluation/export paths have not been tested ([README.md:240](../README.md#L240)).
- PyTorch was not imported for training or evaluation. Actual merge precision behavior needs the target CPU/GPU and PyTorch build.
- Transformers, PEFT, TRL, vLLM, Math-Verify, and datasets runtime behavior/version compatibility was not checked; `.venv` was skipped. In particular, the installed TRL reward-callback column behavior and vLLM finish-reason behavior remain unverified.
- Remote dataset loading/schema, Windows console handling of arbitrary model output, and actual PowerShell setup were not exercised.
- Credential scanning covered common token patterns in current `.py`, `.md`, and `.txt` files. It is not proof against all secret formats; `git ls-files` was not run because it was outside the permitted git-command list.