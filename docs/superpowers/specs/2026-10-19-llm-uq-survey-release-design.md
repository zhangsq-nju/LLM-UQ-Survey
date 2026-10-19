# LLM-UQ-Survey Minimal Release Design

Date: 2026-10-19 (Asia/Shanghai)

## Goal and decision

Create a runnable, minimal code release of the existing UQ survey implementation at `/home/malenia/GithubProject/LLM-UQ-Survey`, then publish it as the private GitHub repository `Mark-Liangs/LLM-UQ-Survey`. The user selected the writing-plans stage after receiving the recommended design in chat. This document records that design; this planning stage does not create or push the remote repository.

The selected approach preserves all 21 methods and 243 registered experiment combinations while extracting only implementations proven equivalent by AST comparison. The alternative was copying all 243 scripts without consolidation; it changes fewer paths but retains unnecessary duplication.

The sole prerequisite-import exception is adding an unaliased top-level `import ast` to the shared LM-Polygraph and author-aligned backends: their existing HumanEval `assemble_code()` helpers call `ast` without importing it. Parity normalization ignores exactly that standalone import for those two groups and requires it in both released modules; all other AST comparisons and BSDetector byte comparisons remain unchanged. Original source files remain untouched.

## Global constraints

- Source directory: `/home/malenia/下载/UQ_survey/UQ_senior`.
- Destination directory: `/home/malenia/GithubProject/LLM-UQ-Survey`.
- GitHub repository: `Mark-Liangs/LLM-UQ-Survey`; visibility: `private`; default branch: `main`.
- Preserve all 21 methods and 243 registered method/family/dataset/setting combinations.
- Keep exactly 18 method implementation files: 3 shared backends and 15 original BSDetector scripts.
- Do not modify the source directory or the neighboring `AgentUQ` repository.
- Exclude datasets, generated responses, experiment results, model weights, credentials, and caches from the release.
- Execute local Python commands and dependency checks with `conda run --no-capture-output -n AgentUQ-newvllm`; stop for user guidance if neither authorized AgentUQ Conda environment exists.
- Use Python 3.11 for documented setup; retain `transformers>=4.51` and the two existing pinned Git dependency revisions.
- Preserve algorithms, prompt text, evaluation defaults, output score direction, and provenance references.
- Write the published README in English and use repository-relative figure links.
- Do not run full model experiments, call paid model APIs, or download datasets/model weights for release validation.
- Do not invent a software license; retain any existing upstream provenance comments.

## Source evidence

The source has no Git history, README, or license file. Its registry resolves to 243 existing Python scripts without missing or unregistered files:

| Registry implementation | Routes | Normalization that proves equivalence | Destination |
| --- | ---: | --- | --- |
| `lm_polygraph` | 156 | Ignore the final main-guard call with fixed route arguments and exactly the standalone unaliased top-level prerequisite `import ast`. | `UQ-methods/lm_polygraph.py` |
| `paper_author_aligned` | 66 | Ignore top-level `METHOD`, `FAMILY`, `DATASET`, and `SETTING` assignments and exactly the standalone unaliased top-level prerequisite `import ast`. | `UQ-methods/paper_author_aligned.py` |
| `qa_topology_reason_ged` | 6 | Ignore only top-level `DATASET` and `FAMILY` assignments. | `UQ-methods/qa_topology_reason_ged.py` |
| implicit `local` / BSDetector | 15 | No shared-body equivalence; preserve bytes. | Existing `UQ-methods/bsdetector__*.py` paths |

The source `data/` is approximately 6.7 GB and is excluded from the release along with experiment results.

## File layout and responsibilities

```text
README.md                         Paper, figures, installation, working usage, outputs, citation.
requirements.txt                  Existing runtime dependencies and upstream revision pins.
.gitignore                        Data, results, model, environment and cache exclusions.
run_uq.py                         Public CLI; validates the registry route and dispatches it.
registry.json                     The original 243 route keys and metadata, with shared script paths.
uq_runtime/{__init__,runner,worker}.py
                                  Existing native routing, AST override execution and output normalization.
UQ-methods/lm_polygraph.py          The shared LM-Polygraph implementation.
UQ-methods/paper_author_aligned.py  Shared ConU, SelfCheckGPT, MARS and CoT-UQ implementation.
UQ-methods/qa_topology_reason_ged.py Shared topology implementation.
UQ-methods/bsdetector__*.py         The 15 distinct original implementations.
data_generation/*.py               User-provided questions to generated responses and correctness labels.
metric_evaluation/*.py             Scores and correctness labels to seven evaluation metrics.
metric_evaluation/requirements.txt Existing evaluator dependencies.
assets/uncertainty-method-ranking-{reasoning,answer-only}.{pdf,png}
                                  Original PDF figures and deterministic PNG previews.
tests/test_release.py              Focused regression checks for shared routing and extraction.
docs/superpowers/specs/            This design record.
docs/superpowers/plans/            The executable release plan.
```

No dataset placeholders, generated fixture files, packaging framework, CI workflow, dashboard, or model wrapper layer is required.

## Runtime changes

Copy the three shared implementations from existing canonical scripts. Apart from the prerequisite import below, the LM-Polygraph backend changes only its final `main(fixed=...)` call to `main()`. The public CLI appends canonical registry method/family/dataset/setting arguments when launching that shared backend, preserving the old fixed-route behavior and supported route restrictions.

Also add the prerequisite `import ast` to the LM-Polygraph and author-aligned shared backends under the narrow exception above. No helper algorithm changes are permitted.

Keep the author and topology implementations' default constant assignments. Extend `uq_runtime.runner.prepare_backend(entry, args, raw_output, extra)` to include the selected registry entry's `METHOD`, `FAMILY`, `DATASET`, and `SETTING` in its override dictionary. The existing `execute_backend(path, argv, overrides)` replaces these top-level assignments before evaluating functions, class defaults and parsers. Do not pass route flags to author/topology parsers that do not accept them.

Change only registry `script` fields for the three shared groups. Preserve all route keys and other metadata. Keep all 15 BSDetector files byte-for-byte. Collect original paper/author repository comment references from each consolidated group into its shared file so that comments specific to the canonical script do not erase other attribution.

## Data flow and README contract

The English README follows the concise layout of [SGA](https://github.com/zhangsq-nju/SGA): project name, paper title, authors and paper link, short description, figures, getting started, models and user-provided input format, experiments, important parameters, outputs, and BibTeX citation.

Paper: [The Origins of Stochasticity: Comprehensive Investigations on Uncertainty Quantification for Large Language Models](https://arxiv.org/abs/2606.22792), Xiang-Jun Ou, Shuang Liang, Xin-Yu Hu, Rong-Hao Huang, Jing Wang, and Shao-Qun Zhang (2026). The scope is 21 UQ methods, Qwen3/Llama 3.2/DeepSeek-V3 families, TriviaQA/GSM8K/HumanEval, and answer-only/reasoning-augmented generation; the registry is authoritative for supported combinations, including DeepSeek answer-only restrictions.

Distinguish these two actual flows:

1. LM-Polygraph: original question-and-gold JSONL → `run_uq.py` in `generate` mode → `score` with `score_type=uncertainty` → evaluator using `--method lm_polygraph_<method>`.
2. Native methods: original question-and-gold JSONL → `python -m data_generation.generate` → stage1 greedy/sampled or reasoning responses → `run_uq.py` → evaluator using the original method name.

Users supply datasets and checkpoints themselves. The README must not promise automatic dataset downloads or claim that generation produces the same historical paper results. State that MARS requires an external author-trained importance checkpoint, SelfCheckGPT requires `en_core_web_sm`, and HumanEval correctness evaluation requires Docker plus its image and associated labels.

The figures are copied from `/home/malenia/下载/uncertainty-method-ranking-reasoning.pdf` and `/home/malenia/下载/uncertainty-method-ranking-answer-only.pdf`. Use deterministic PDF rasterization for previews; retain the original PDFs as linked vector figures.

## Verification and error behavior

Use meaningful regression checks for the behavior-changing consolidation: registry coverage, AST parity of each consolidated source, byte parity of all BSDetector files, canonical LM subprocess arguments, native override metadata for every route, and a synthetic `execute_backend` probe confirming overrides are applied before default argument evaluation.

The focused suite has nine tests. Its additional HumanEval helper regression imports both released backends without loading models, assembles a complete function that requires prompt imports, and executes it to verify the result. A separate normalization regression confirms only the exact prerequisite import is exempted: aliases, mixed imports, from-imports, nested imports, and topology imports remain significant.

Use an ephemeral multi-route CLI dry-run and a four-row CPU evaluation smoke test with both correct and incorrect labels. Dry-run confirms route selection and argument construction; it does not execute the real backend or prove all model-resource prerequisites. Run local checks in the existing authorized Conda environment. No test inputs or generated outputs are committed.

Preserve existing validation, missing-score errors, overwrite protection and unsupported-route errors. Do not refactor unrelated source behavior. If equivalence fails, retain the distinct implementation until the discrepancy is resolved rather than changing its algorithm.

Before publishing, review the tracked file list, relative README links, Git diff, and commit. After publishing, verify the owner, repository name, private visibility, default branch and remote/local commit equality. An existing target repo must be inspected and reused safely; never force-push or change an existing public repository's visibility automatically.

## Completion criteria

The destination contains the minimal runnable code and both figures, the focused checks pass, and the remote is `Mark-Liangs/LLM-UQ-Survey`, private, with the reviewed local `main` commit pushed. The final report includes the repository link, local path, validation performed and the practical limit that full GPU/API experiments were not run.
