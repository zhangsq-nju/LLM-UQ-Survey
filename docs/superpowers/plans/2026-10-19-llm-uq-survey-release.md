# LLM-UQ-Survey Minimal Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the existing UQ survey implementation as the private `Mark-Liangs/LLM-UQ-Survey` repository with minimal necessary code, a paper README and both supplied main figures, without datasets or generated results.

**Architecture:** Preserve the existing CLI, registry and generation/evaluation modules. Consolidate only the three AST-equivalent implementation groups into shared scripts, preserving all 243 routes, and keep the 15 distinct BSDetector scripts unchanged. Reuse the existing AST execution mechanism to inject route metadata before backend definitions and parser defaults are evaluated. The sole prerequisite-import exception adds a standalone unaliased top-level `import ast` to the LM and author-aligned shared backends; their existing HumanEval code-assembly helpers otherwise fail with NameError. All other algorithms, source AST comparisons and BSDetector bytes remain unchanged.

**Tech Stack:** Python 3.11, argparse, standard-library unittest/AST/subprocess, existing PyTorch/Transformers/LM-Polygraph dependencies, Git, GitHub CLI and Poppler PDF tools.

**Spec:** `docs/superpowers/specs/2026-10-19-llm-uq-survey-release-design.md` (read this file before execution).

## Global Constraints

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

---

## File Structure and Ownership

All paths below are relative to `/home/malenia/GithubProject/LLM-UQ-Survey`. The destination currently contains only the design and this plan; it is not yet a Git repository or a GitHub repository. Detect its actual state before execution, since another session may have progressed it.

| Files | Responsibility | Task |
| --- | --- | --- |
| `run_uq.py`, `registry.json` | Public CLI and authoritative supported routes. | 1 |
| `uq_runtime/__init__.py`, `runner.py`, `worker.py` | Existing native argument routing, AST overrides and subprocess execution. | 1 |
| `UQ-methods/lm_polygraph.py` | One shared backend for 156 routes. | 1 |
| `UQ-methods/paper_author_aligned.py` | One shared backend for 66 routes. | 1 |
| `UQ-methods/qa_topology_reason_ged.py` | One shared backend for 6 routes. | 1 |
| `UQ-methods/bsdetector__*.py` | Fifteen distinct unchanged implementations. | 1 |
| `data_generation/*.py` | Questions to stage1 responses and correctness sidecars. | 1 |
| `metric_evaluation/*.py`, `metric_evaluation/requirements.txt` | Scores and correctness to seven metrics. | 1 |
| `requirements.txt`, `.gitignore` | Existing runtime dependencies and release exclusions. | 1 |
| `tests/test_release.py` | Focused regression checks for the behavior-changing extraction. | 1 |
| `README.md`, `assets/uncertainty-method-ranking-{reasoning,answer-only}.{pdf,png}` | Research documentation and supplied figures. | 2 |
| Existing design and plan documents | Execution context and checked task tracking. | 1–3 |

Do not add packaging, configuration frameworks, CI, notebooks, dataset placeholders or a separate migration tool to the published repository. The one-time extraction code below is executed from the plan, not retained as a runtime module.

## Task 1: Release the Existing Pipeline with Equivalent Shared Backends

**Files:**
- Create from the source whitelist: `run_uq.py`, `registry.json`, `requirements.txt`, `uq_runtime/*.py`, `data_generation/*.py`, `metric_evaluation/*.py`, `metric_evaluation/requirements.txt`.
- Create: `.gitignore`, `UQ-methods/lm_polygraph.py`, `UQ-methods/paper_author_aligned.py`, `UQ-methods/qa_topology_reason_ged.py`, `tests/test_release.py`.
- Preserve: fifteen `UQ-methods/bsdetector__*.py` files.
- Modify after copying: `run_uq.py:35–39`, `uq_runtime/runner.py:112–114`, and `registry.json` script paths only. Line references refer to the source snapshot inspected on 2026-10-19.

**Interfaces:**
- Consumes: the source `registry.json`, mapping route strings to dictionaries containing `method`, `family`, `dataset`, `setting`, `script`, and optional `implementation`.
- Preserves: `run_uq.main()`, `data_generation.generate.main()`, `metric_evaluation.evaluate.main()`, `uq_runtime.runner.prepare_backend(entry, args, raw_output, extra) -> (Path, list[str], dict)` and `execute_backend(path, argv, overrides)`.
- Produces: the same 243 registry keys with 18 distinct implementation paths; LM-Polygraph `main(argv=None, fixed=None)` remains usable; native overrides contain uppercase route constants.

- [x] **Step 1: Check the destination and copy only necessary code.**

Run read-only checks first:

```bash
conda env list
gh auth status
ls -la /home/malenia/GithubProject/LLM-UQ-Survey
git -C /home/malenia/GithubProject/LLM-UQ-Survey rev-parse --show-toplevel
```

Expected initially: `AgentUQ-newvllm` exists, GitHub user is `Mark-Liangs`, and the destination is not a Git repository. If already initialized, inspect `git status --short`, `git remote -v` and recent commits; preserve any work and do not overwrite progressed files. If the destination resolves inside another repository, stop before initializing or staging it. An independent new repository does not require a linked worktree; apply worktree guidance at execution time if the actual destination is an existing repository requiring isolation.

Run this one-time copy only on the initial docs-only destination. It deliberately copies source method scripts before the failing tests, then Step 4 removes the redundant copies.

```bash
cd /home/malenia/GithubProject/LLM-UQ-Survey
conda run --no-capture-output -n AgentUQ-newvllm python -B - <<'PY'
from pathlib import Path
import shutil

source = Path('/home/malenia/下载/UQ_survey/UQ_senior')
target = Path.cwd()
files = [source / name for name in ('run_uq.py', 'registry.json', 'requirements.txt')]
for folder in ('uq_runtime', 'data_generation', 'metric_evaluation', 'UQ-methods'):
    files.extend(sorted((source / folder).glob('*.py')))
files.append(source / 'metric_evaluation/requirements.txt')
assert len(list((source / 'UQ-methods').glob('*.py'))) == 243
for item in files:
    destination = target / item.relative_to(source)
    if destination.exists():
        raise SystemExit(f'Refusing to overwrite existing implementation: {destination}')
for item in files:
    destination = target / item.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(item, destination)
print(f'Copied {len(files)} code/dependency files; no data directories copied')
PY
git init -b main
```

Write `.gitignore` with this content:

```gitignore
/data/
/outputs/
/models/
/checkpoints/
.venv/
__pycache__/
*.py[cod]
.DS_Store
__MACOSX/
.env
.env.*
*.jsonl
*.jsonl.gz
*.csv
*.tsv
*.parquet
*.arrow
*.npy
*.npz
*.pt
*.pth
*.safetensors
```

- [x] **Step 2: Write the focused regression tests before changing runtime behavior.**

Create `tests/test_release.py` with the following complete test module. The source-dependent parity check is enabled explicitly by an environment variable; ordinary users can run the other checks without possessing the original directory. Synthetic rows are written only into temporary directories.

```python
import argparse
import ast
from collections import Counter
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_uq
from uq_runtime.runner import execute_backend, prepare_backend

REGISTRY = json.loads((ROOT / 'registry.json').read_text())
SHARED = {
    'lm_polygraph': 'UQ-methods/lm_polygraph.py',
    'paper_author_aligned': 'UQ-methods/paper_author_aligned.py',
    'qa_topology_reason_ged': 'UQ-methods/qa_topology_reason_ged.py',
}
CONSTANTS = {'METHOD', 'FAMILY', 'DATASET', 'SETTING'}


def normalized_ast(path, implementation):
    tree = ast.parse(path.read_text(), filename=str(path))
    body = []
    for node in tree.body:
        if (implementation in ('lm_polygraph', 'paper_author_aligned')
                and isinstance(node, ast.Import) and len(node.names) == 1
                and node.names[0].name == 'ast' and node.names[0].asname is None):
            continue
        if implementation == 'lm_polygraph' and isinstance(node, ast.If):
            expected = ast.parse("__name__ == '__main__'", mode='eval').body
            if ast.dump(node.test) == ast.dump(expected):
                continue
        ignored = {'METHOD', 'FAMILY', 'DATASET', 'SETTING'}
        if implementation == 'qa_topology_reason_ged':
            ignored = {'DATASET', 'FAMILY'}
        if implementation != 'lm_polygraph' and isinstance(node, ast.Assign):
            if all(isinstance(t, ast.Name) and t.id in ignored for t in node.targets):
                continue
        body.append(node)
    tree.body = body
    return ast.dump(tree, include_attributes=False)


class ReleaseTests(unittest.TestCase):
    def test_assemble_code_preserves_prompt_imports_and_runs_complete_function(self):
        prompt = ('import math\nfrom typing import List\n\n'
                  'def roots(values: List[float]) -> List[float]:\n'
                  '    """Return the square roots."""\n')
        answer = ('def roots(values: List[float]) -> List[float]:\n'
                  '    return [math.sqrt(value) for value in values]\n')
        for implementation in ('lm_polygraph', 'paper_author_aligned'):
            with self.subTest(implementation=implementation):
                path = ROOT / SHARED[implementation]
                backend = runpy.run_path(str(path))
                code = backend['assemble_code'](prompt, answer, 'roots')
                namespace = {}
                exec(compile(code, '<assembled HumanEval fixture>', 'exec'), namespace)
                self.assertEqual(namespace['roots']([4., 9.]), [2., 3.])
                tree = ast.parse(path.read_text())
                self.assertTrue(any(isinstance(node, ast.Import) and len(node.names) == 1
                                    and node.names[0].name == 'ast'
                                    and node.names[0].asname is None for node in tree.body))

    def test_parity_normalization_only_exempts_exact_ast_prerequisite_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            before, after = (Path(temporary) / name for name in ('before.py', 'after.py'))
            before.write_text('value = 1\n')
            for implementation in ('lm_polygraph', 'paper_author_aligned'):
                with self.subTest(implementation=implementation):
                    after.write_text('import ast\nvalue = 1\n')
                    self.assertEqual(normalized_ast(before, implementation),
                                     normalized_ast(after, implementation))
                    for statement in ('import ast as alias', 'import ast, sys',
                                      'from ast import parse', 'def nested():\n    import ast'):
                        after.write_text(statement + '\nvalue = 1\n')
                        self.assertNotEqual(normalized_ast(before, implementation),
                                            normalized_ast(after, implementation))
            after.write_text('import ast\nvalue = 1\n')
            self.assertNotEqual(normalized_ast(before, 'qa_topology_reason_ged'),
                                normalized_ast(after, 'qa_topology_reason_ged'))

    def test_registry_preserves_scope_and_uses_18_implementations(self):
        self.assertEqual(len(REGISTRY), 243)
        self.assertEqual(len({entry['method'] for entry in REGISTRY.values()}), 21)
        self.assertEqual(Counter(entry.get('implementation', 'local') for entry in REGISTRY.values()),
                         Counter(lm_polygraph=156, paper_author_aligned=66,
                                 qa_topology_reason_ged=6, local=15))
        paths = {entry['script'] for entry in REGISTRY.values()}
        self.assertEqual(len(paths), 18)
        self.assertEqual({str(p.relative_to(ROOT)) for p in (ROOT / 'UQ-methods').glob('*.py')}, paths)
        for entry in REGISTRY.values():
            self.assertTrue((ROOT / entry['script']).is_file())
            implementation = entry.get('implementation', 'local')
            if implementation in SHARED:
                self.assertEqual(entry['script'], SHARED[implementation])

    @unittest.skipUnless(os.environ.get('UQ_SOURCE_ROOT'), 'Original source is not supplied')
    def test_every_route_preserves_original_implementation(self):
        source = Path(os.environ['UQ_SOURCE_ROOT'])
        original = json.loads((source / 'registry.json').read_text())
        self.assertEqual(set(REGISTRY), set(original))
        for key, before in original.items():
            with self.subTest(route=key):
                after = REGISTRY[key]
                self.assertEqual({k: v for k, v in before.items() if k != 'script'},
                                 {k: v for k, v in after.items() if k != 'script'})
                implementation = before.get('implementation', 'local')
                old_path, new_path = source / before['script'], ROOT / after['script']
                if implementation in SHARED:
                    self.assertEqual(normalized_ast(old_path, implementation),
                                     normalized_ast(new_path, implementation))
                else:
                    self.assertEqual(old_path.read_bytes(), new_path.read_bytes())

    def test_native_routes_supply_metadata_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'stage1.jsonl'
            source.write_text('{}\n')
            for key, entry in REGISTRY.items():
                if entry.get('implementation') == 'lm_polygraph':
                    continue
                with self.subTest(route=key):
                    args = argparse.Namespace(method=entry['method'], family=entry['family'],
                        model_path='fixture/model', input_file=str(source), correct_file=str(source),
                        problem_file=None, nli_model='fixture/nli', device='cpu')
                    _, _, overrides = prepare_backend(entry, args, Path(temporary) / 'raw.jsonl', [])
                    self.assertEqual({name: overrides[name] for name in CONSTANTS},
                                     {name: entry[name.lower()] for name in CONSTANTS})

    def test_overrides_apply_before_function_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            backend = Path(temporary) / 'probe.py'
            backend.write_text(
                "METHOD='conu'\nFAMILY='qwen'\nDATASET='gsm8k'\nSETTING='answer_only'\n"
                "def probe(method=METHOD, family=FAMILY, dataset=DATASET, setting=SETTING):\n"
                "    return [method, family, dataset, setting]\n"
                "import json\nprint(json.dumps(probe()))\n")
            expected = dict(METHOD='cot_uq_sp', FAMILY='llama',
                            DATASET='humaneval', SETTING='reasoning_augmented')
            stdout = io.StringIO()
            with patch.object(sys, 'argv', list(sys.argv)), contextlib.redirect_stdout(stdout):
                execute_backend(backend, [str(backend)], expected)
            self.assertEqual(json.loads(stdout.getvalue()), list(expected.values()))

    def test_lm_routes_forward_canonical_parameters(self):
        for key, entry in REGISTRY.items():
            if entry.get('implementation') != 'lm_polygraph':
                continue
            with self.subTest(route=key):
                argv = ['run_uq.py', '--method', entry['method'], '--family', entry['family'],
                        '--dataset', entry['dataset'], '--setting', entry['setting']]
                with patch.object(sys, 'argv', argv), patch.object(run_uq.subprocess, 'run') as launch:
                    launch.return_value = subprocess.CompletedProcess([], 0)
                    with self.assertRaises(SystemExit) as raised:
                        run_uq.main()
                self.assertEqual(raised.exception.code, 0)
                command = launch.call_args.args[0]
                self.assertEqual(command[2], str(ROOT / SHARED['lm_polygraph']))
                self.assertEqual(command[-8:], ['--method', entry['method'], '--family', entry['family'],
                                               '--dataset', entry['dataset'], '--setting', entry['setting']])

    def test_all_cli_routes_dry_run_without_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, labels, output = (directory / name for name in ('input.jsonl', 'labels.jsonl', 'uq.jsonl'))
            source.write_text(json.dumps({'id': 'smoke', 'question': 'What is one plus one?',
                'gt': {'value': '2'}, 'greedy': {'text': '2'}, 'samples': [], 'responses': []}) + '\n')
            labels.write_text('{"id":"smoke","correct":1}\n')
            for key, entry in REGISTRY.items():
                with self.subTest(route=key):
                    command = [sys.executable, '-B', 'run_uq.py', '--method', entry['method'],
                        '--family', entry['family'], '--model-path', 'fixture/' + entry['family'],
                        '--dataset', entry['dataset'], '--setting', entry['setting'],
                        '--input-file', str(source), '--output-file', str(output), '--device', 'cpu', '--dry-run']
                    if entry['dataset'] == 'humaneval' and entry.get('implementation') != 'lm_polygraph':
                        command += ['--correct-file', str(labels)]
                    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    report = json.loads(completed.stdout)
                    if entry.get('implementation') == 'lm_polygraph':
                        self.assertEqual(report['method'], entry['method'])
                        self.assertEqual(report['implementation'], 'lm_polygraph')
                    else:
                        self.assertEqual(report['route'], key)
                    self.assertFalse(output.exists())

    def test_evaluator_handles_known_uncertainty_direction(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, output = directory / 'scores.jsonl', directory / 'metrics.csv'
            rows = [{'id': str(i), 'correct': correct, 'score': score}
                    for i, (correct, score) in enumerate([(1, .1), (0, .9), (1, .2), (0, .8)])]
            source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            command = [sys.executable, '-B', '-m', 'metric_evaluation.evaluate',
                '--input-file', str(source), '--output-file', str(output),
                '--score-key', 'score', '--score-type', 'uncertainty', '--bootstrap-runs', '0']
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with output.open(encoding='utf-8-sig', newline='') as stream:
                metrics = {row['metric']: row for row in csv.DictReader(stream)}
            self.assertEqual(set(metrics), {'AUROC', 'AUPRC', 'AUARC', 'AURC', 'ECE', 'MCE', 'Brier'})
            self.assertAlmostEqual(float(metrics['AUROC']['mean']), 1.)
            self.assertAlmostEqual(float(metrics['AUPRC']['mean']), 1.)
            report = json.loads(output.with_suffix('.evaluation.json').read_text())
            self.assertEqual(report['counts']['valid'], 4)
            self.assertEqual(report['score_type'], 'uncertainty')


if __name__ == '__main__':
    unittest.main()
```

- [x] **Step 3: Run the targeted tests and confirm the relevant failures.**

```bash
conda run --no-capture-output -n AgentUQ-newvllm python -B -m unittest discover -s tests -p test_release.py -v -k registry_preserves
conda run --no-capture-output -n AgentUQ-newvllm python -B -m unittest discover -s tests -p test_release.py -v -k native_routes_supply
```

Expected: the first test fails because there are 243 distinct scripts rather than 18; the second fails because route constants are absent from `prepare_backend` overrides. Fix test setup failures before implementing; a missing import is not the intended red signal.

- [x] **Step 4: Mechanically extract the three shared bodies and rewrite only their registry paths.**

Run the following one-time code from the destination root. It independently checks the equivalence premise before deleting any redundant destination copy. It never writes to the source.

```bash
conda run --no-capture-output -n AgentUQ-newvllm python -B - <<'PY'
import ast
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path('tests').resolve()))
from test_release import normalized_ast

source = Path('/home/malenia/下载/UQ_survey/UQ_senior')
target = Path.cwd()
registry = json.loads((source / 'registry.json').read_text())
groups = {
    'lm_polygraph': ('UQ-methods/max_seq_prob__qwen__gsm8k__answer_only.py', 'UQ-methods/lm_polygraph.py', 156),
    'paper_author_aligned': ('UQ-methods/selfcheckgpt__qwen__gsm8k__answer_only.py', 'UQ-methods/paper_author_aligned.py', 66),
    'qa_topology_reason_ged': ('UQ-methods/topo_uq__qwen__gsm8k__reasoning_augmented.py', 'UQ-methods/qa_topology_reason_ged.py', 6),
}
for implementation, (canonical, destination, count) in groups.items():
    members = [entry for entry in registry.values() if entry.get('implementation') == implementation]
    assert len(members) == count, (implementation, len(members))
    signature = normalized_ast(source / canonical, implementation)
    for entry in members:
        assert normalized_ast(source / entry['script'], implementation) == signature, entry['script']
    text = (source / canonical).read_text()
    if implementation == 'lm_polygraph':
        guard = ast.parse(text).body[-1]
        assert isinstance(guard, ast.If)
        lines = text.splitlines(keepends=True)
        lines[guard.lineno - 1:guard.end_lineno] = ["if __name__ == '__main__':\n", '    main()\n']
        text = ''.join(lines)
    if implementation in ('lm_polygraph', 'paper_author_aligned'):
        text = text.replace('import argparse\n', 'import argparse\nimport ast\n', 1)
    references = set()
    for entry in members:
        for line in (source / entry['script']).read_text().splitlines()[:8]:
            if line.startswith(('# Original paper/author repository:', '# Implementation repository:')):
                references.add(line)
    additional = sorted(references - set(text.splitlines()))
    lines = text.splitlines(keepends=True)
    lines[1:1] = [line + '\n' for line in additional]
    (target / destination).write_text(''.join(lines))
    for entry in members:
        entry['script'] = destination
retained = {entry['script'] for entry in registry.values()}
for script in (target / 'UQ-methods').glob('*.py'):
    if str(script.relative_to(target)) not in retained:
        script.unlink()
(target / 'registry.json').write_text(json.dumps(registry, indent=2, ensure_ascii=False) + '\n')
assert len(retained) == 18
print('Retained all 243 routes using 18 implementations')
PY
```

In `uq_runtime/runner.py`, immediately after constructing the existing `overrides` dictionary in `prepare_backend`, add:

```python
    overrides.update({
        'METHOD': entry['method'],
        'FAMILY': entry['family'],
        'DATASET': entry['dataset'],
        'SETTING': entry['setting'],
    })
```

Keep `execute_backend` unchanged: it already replaces top-level assignments before executing the module. Do not pass these route constants as new flags to author/topology backend parsers.

In the `lm_polygraph` branch of `run_uq.main()`, retain the subprocess call and add canonical route flags after the existing `argv` construction:

```python
            argv = [arg for arg in sys.argv[1:] if arg != '--']
            argv.extend(['--method', method, '--family', family,
                         '--dataset', args.dataset, '--setting', args.setting])
            result = subprocess.run([sys.executable, '-B', str(script)] + argv)
```

Prerequisite-import exception: add only an unaliased top-level `import ast` to the two shared backends above, leaving the source untouched. `normalized_ast` ignores exactly that standalone import for those groups, while the executable helper regression requires it in both released modules. Aliases, mixed imports, from-imports, nested imports and the topology group retain strict comparison.

The registry has already validated those four values. Appending them preserves route-fixed behavior without hardcoding a canonical method/family into the shared backend. Keep all other runtime code unchanged.

- [x] **Step 5: Run the complete focused suite, including source parity and CPU smoke checks.**

```bash
UQ_SOURCE_ROOT=/home/malenia/下载/UQ_survey/UQ_senior conda run --no-capture-output -n AgentUQ-newvllm python -B -m unittest discover -s tests -p test_release.py -v
```

Expected: all nine tests pass, including comparisons for 243 original routes under the narrow prerequisite-import exception, metadata for 87 native routes, canonical forwarding for 156 LM routes, 243 dry-runs, four-row evaluation, executable code assembly in both affected shared backends, and a regression constraining the normalization exception. No models load and no output appears inside the repository. Use a yielding command session if execution exceeds the tool's initial timeout and communicate progress within 60 seconds.

Observed during planning: the authorized environment has NumPy, SciPy, scikit-learn, PyTorch, Transformers, math-verify and LM-Polygraph import specs. `selfcheckgpt` is absent in that environment; the checks above do not need it because the native dry-run does not import real model backends. Do not claim a SelfCheckGPT model execution passed or install extra packages into the shared environment merely to run this release check.

- [x] **Step 6: Review and commit the independently working code release.**

```bash
git add .gitignore run_uq.py registry.json requirements.txt uq_runtime UQ-methods data_generation metric_evaluation tests docs/superpowers/specs/2026-10-19-llm-uq-survey-release-design.md docs/superpowers/plans/2026-10-19-llm-uq-survey-release.md
git diff --cached --check
git diff --cached --stat
git diff --cached -- run_uq.py uq_runtime/runner.py
git commit -m "feat: prepare minimal LLM UQ survey code release"
```

Review the source-parity and runtime changes before the commit. Mark completed task checkboxes in the plan; stage those plan updates with this commit or the next task commit. Do not configure a fabricated Git author if the machine's Git identity is unavailable.

## Task 2: Publish an Accurate Paper README and the Two Supplied Figures

**Files:**
- Create: `README.md`.
- Create: `assets/uncertainty-method-ranking-reasoning.pdf`, `assets/uncertainty-method-ranking-reasoning.png`.
- Create: `assets/uncertainty-method-ranking-answer-only.pdf`, `assets/uncertainty-method-ranking-answer-only.png`.

**Interfaces:**
- Consumes: Task 1's unchanged public CLI and data/evaluation schema; both source PDF paths; paper metadata at `https://arxiv.org/abs/2606.22792`; reference structure at `https://github.com/zhangsq-nju/SGA`.
- Produces: repository-relative previews and PDF links, complete English usage examples, accurate output and resource requirements. No runtime API changes.

- [x] **Step 1: Copy original PDFs and generate deterministic previews.**

```bash
mkdir -p assets
cp /home/malenia/下载/uncertainty-method-ranking-reasoning.pdf assets/uncertainty-method-ranking-reasoning.pdf
cp /home/malenia/下载/uncertainty-method-ranking-answer-only.pdf assets/uncertainty-method-ranking-answer-only.pdf
pdftoppm -png -singlefile -scale-to 2400 assets/uncertainty-method-ranking-reasoning.pdf assets/uncertainty-method-ranking-reasoning
pdftoppm -png -singlefile -scale-to 2400 assets/uncertainty-method-ranking-answer-only.pdf assets/uncertainty-method-ranking-answer-only
cmp /home/malenia/下载/uncertainty-method-ranking-reasoning.pdf assets/uncertainty-method-ranking-reasoning.pdf
cmp /home/malenia/下载/uncertainty-method-ranking-answer-only.pdf assets/uncertainty-method-ranking-answer-only.pdf
```

Use the image viewer to inspect both PNGs: text must be legible, all labels must fit, and the images must respectively show reasoning-augmented and answer-only rankings. Do not use image generation or redraw the supplied scientific figures.

- [x] **Step 2: Write the English README using the content in the next section.**

The shell commands inside this README are instructions for repository users, not commands to execute during documentation validation. The release agent continues using `AgentUQ-newvllm` for its own Python checks.

<!-- README-CONTENT-START -->

````markdown
# LLM-UQ-Survey

Code for **The Origins of Stochasticity: Comprehensive Investigations on Uncertainty Quantification for Large Language Models**.

Xiang-Jun Ou, Shuang Liang, Xin-Yu Hu, Rong-Hao Huang, Jing Wang, and Shao-Qun Zhang · [Paper](https://arxiv.org/abs/2606.22792)

This repository implements the survey's evaluation pipeline for 21 uncertainty quantification methods across Qwen3, Llama 3.2 and DeepSeek-V3 model families, TriviaQA, GSM8K and HumanEval, and answer-only and reasoning-augmented generation. Supported combinations are listed in `registry.json`.

![Method rankings for reasoning-augmented generation](assets/uncertainty-method-ranking-reasoning.png)
[Reasoning-augmented figure (PDF)](assets/uncertainty-method-ranking-reasoning.pdf)

![Method rankings for answer-only generation](assets/uncertainty-method-ranking-answer-only.png)
[Answer-only figure (PDF)](assets/uncertainty-method-ranking-answer-only.pdf)

## Getting Started

```bash
git clone git@github.com:Mark-Liangs/LLM-UQ-Survey.git
cd LLM-UQ-Survey
conda create -n llm-uq-survey python=3.11
conda activate llm-uq-survey
pip install -r requirements.txt
```

Install a PyTorch build suitable for your CUDA installation. A GPU is recommended for local generation and model-based scoring. Run commands from the repository root. The upstream LM-Polygraph and SelfCheckGPT revisions are pinned in `requirements.txt`.

### Models and Data

Provide a local checkpoint or Hugging Face model ID through `--model-path`. For DeepSeek, use the API model ID and set `DEEPSEEK_API_KEY` in your environment. DeepSeek reasoning-augmented routes are outside the supported experiment scope.

Datasets, checkpoints, generated responses and experiment results are not included. Supply question-and-gold JSONL yourself. GSM8K and TriviaQA accept records such as:

```json
{"id":"example-1","question":"What is one plus one?","answer":"2"}
```

TriviaQA records may include answer aliases. HumanEval requires a function prompt, `test` and `entry_point`; correctness evaluation requires Docker with the `python:3.11-slim` image available locally.

## Running Experiments

List the 243 supported method/family/dataset/setting combinations:

```bash
python run_uq.py --list
python run_uq.py --help
```

The implementation has two input flows. LM-Polygraph methods generate directly from original questions; native methods score responses created by `data_generation.generate`.

### 1. LM-Polygraph Methods

This example reads original GSM8K question-and-gold records, generates answers and calculates MaxSeqProb:

```bash
python run_uq.py \
  --method max_seq_prob --family qwen \
  --model-path Qwen/Qwen3-4B --dataset gsm8k \
  --setting answer_only \
  --input-file data/gsm8k.jsonl \
  --output-file outputs/max_seq_prob.jsonl --device cuda

python -m metric_evaluation.evaluate \
  --method lm_polygraph_max_seq_prob \
  --input-file outputs/max_seq_prob.jsonl \
  --output-file outputs/max_seq_prob.csv
```

LM-Polygraph outputs store `score` as uncertainty. Use the `lm_polygraph_` method prefix during evaluation, or supply `--score-key score --score-type uncertainty` explicitly. These methods also expose `--mode stats` for compatible precomputed upstream statistics.

### 2. Native Methods

Generate stage1 responses before scoring native methods. For SelfCheckGPT, install its sentence segmentation model first:

```bash
python -m spacy download en_core_web_sm

python -m data_generation.generate \
  --family qwen --model-path Qwen/Qwen3-4B \
  --dataset gsm8k --setting answer_only \
  --input-file data/gsm8k.jsonl \
  --output-file outputs/stage1.jsonl

python run_uq.py \
  --method selfcheckgpt --family qwen \
  --model-path Qwen/Qwen3-4B --dataset gsm8k \
  --setting answer_only \
  --input-file outputs/stage1.jsonl \
  --output-file outputs/selfcheckgpt.jsonl --device cuda

python -m metric_evaluation.evaluate \
  --method selfcheckgpt \
  --input-file outputs/selfcheckgpt.jsonl \
  --output-file outputs/selfcheckgpt.csv
```

For reasoning-augmented generation, use `--setting reasoning_augmented` consistently during generation and scoring, and use separate output paths. Consult `--list` before selecting the method/family/dataset combination.

MARS requires an author-trained importance checkpoint passed after `--` as `--importance-model-path`. Topo-UQ requires an embedding checkpoint through `--nli-model` or backend `--embedding-model-path`. Native HumanEval scoring requires the associated correctness labels through `--correct-file`.

Adding `--dry-run` to a generation or scoring command checks its input and route without loading a model. Native scoring dry-run checks routing and parameter construction; model-resource prerequisites are checked by the real backend.

## Important Parameters

| Parameter | Meaning |
| --- | --- |
| `--method` | UQ method; use `--list` for supported routes. |
| `--family` | `qwen`, `llama` or `deepseek`. |
| `--model-path` | Local checkpoint, Hugging Face ID or API model ID. |
| `--dataset` | `gsm8k`, `triviaqa` or `humaneval`. |
| `--setting` | `answer_only` (default) or `reasoning_augmented`. |
| `--input-file`, `--output-file` | Input and output paths; the input schema depends on the flow. |
| `--correct-file` | External correctness labels, including native HumanEval labels. |
| `--nli-model` | NLI or embedding checkpoint for compatible native backends. |
| `--device` | `cpu` or `cuda` for scoring. |
| `--overwrite` | Explicitly replace an existing output. |
| `--` | Pass backend-specific options after the public CLI options. |

Generation defaults to ten stochastic samples in addition to a greedy answer. Its sampling settings can be changed with `--num-samples`, `--temperature`, `--top-p` and `--max-new-tokens`. See `python -m data_generation.generate --help` for the full generation options.

## Outputs

The native example writes:

```text
outputs/
  stage1.jsonl                    # Generated responses and token statistics
  stage1.meta.json                # Generation configuration and status
  stage1.summary.json             # Generation/evaluation summary
  stage1.correct.jsonl            # Available correctness labels
  selfcheckgpt.jsonl              # Scored responses and uq_run metadata
  selfcheckgpt.csv                # Seven evaluation metrics
  selfcheckgpt.evaluation.json    # Input coverage, score direction and evaluation settings
```

Use the canonical `uq_run` fields when grouping native outputs; the legacy BSDetector `meta.dataset` field incorrectly labels the DeepSeek TriviaQA route as GSM8K.

Higher is better for AUROC, AUPRC and AUARC; lower is better for AURC, ECE, MCE and Brier. Evaluation defaults to 200 bootstrap samples. CSV `mean` and `std` use raw metric units; the display column multiplies all metrics except Brier by 100. Use matching calibration and coverage settings when comparing methods.

## Citation

If you use this code, please cite the [paper](https://arxiv.org/abs/2606.22792).

```bibtex
@article{ou2026origins,
  author  = {Xiang-Jun Ou and Shuang Liang and Xin-Yu Hu and
             Rong-Hao Huang and Jing Wang and Shao-Qun Zhang},
  title   = {The Origins of Stochasticity: Comprehensive Investigations on
             Uncertainty Quantification for Large Language Models},
  journal = {arXiv preprint arXiv:2606.22792},
  year    = {2026}
}
```
````

<!-- README-CONTENT-END -->

- [x] **Step 3: Verify the documented commands and local figure links.**

Run only help commands and the link check; do not execute the README's installation, model downloads or full experiments during this step.

```bash
conda run --no-capture-output -n AgentUQ-newvllm python -B run_uq.py --help
conda run --no-capture-output -n AgentUQ-newvllm python -B -m data_generation.generate --help
conda run --no-capture-output -n AgentUQ-newvllm python -B -m metric_evaluation.evaluate --help
conda run --no-capture-output -n AgentUQ-newvllm python -B - <<'PY'
from pathlib import Path
import re

readme = Path('README.md').read_text()
for target in re.findall(r'\]\(([^)]+)\)', readme):
    if not target.startswith(('https://', 'http://', '#')):
        assert Path(target).is_file(), target
for setting in ('reasoning', 'answer-only'):
    for extension in ('pdf', 'png'):
        path = Path('assets') / f'uncertainty-method-ranking-{setting}.{extension}'
        assert path.is_file() and path.stat().st_size > 1000, path
print('README relative links and four figure assets exist')
PY
```

Check each README command against the parser names and actual output schema. Task 1 already exercises every supported scoring route, so do not repeat the complete route suite solely for these documentation/asset changes.

- [x] **Step 4: Commit the paper documentation and figures.**

```bash
git add README.md assets docs/superpowers/plans/2026-10-19-llm-uq-survey-release.md
git diff --cached --check
git diff --cached --stat
git commit -m "docs: add paper README and method ranking figures"
```

## Task 3: Verify the Release Inventory and Publish the Private Repository

**Files:**
- Review the tracked files from Tasks 1 and 2; update completed checkboxes in this plan.
- No new runtime code or configuration is required.

**Interfaces:**
- Consumes: Task 1's passing verification result and Task 2's checked README/assets; a reviewed local `main` commit.
- Produces: `origin` pointing to `Mark-Liangs/LLM-UQ-Survey`, a pushed private remote, and evidence that remote `main` equals local `HEAD`.

- [x] **Step 1: Check the tracked inventory, syntax and final diff.**

```bash
git status --short
git diff --check
git log -2 --oneline
git ls-files
conda run --no-capture-output -n AgentUQ-newvllm python -B - <<'PY'
from pathlib import Path
import subprocess

tracked = [Path(name) for name in subprocess.check_output(['git', 'ls-files', '-z'], text=True).split('\0') if name]
excluded_roots = {'data', 'outputs', 'models', 'checkpoints', '__pycache__', '__MACOSX', '.venv'}
excluded_suffixes = {'.jsonl', '.csv', '.tsv', '.parquet', '.arrow', '.npy', '.npz', '.pt', '.pth', '.safetensors', '.pyc'}
for path in tracked:
    assert not set(path.parts).intersection(excluded_roots), path
    assert path.suffix not in excluded_suffixes, path
    assert path.name not in {'.env', '.DS_Store'} and not path.name.startswith('.env.'), path
    assert path.stat().st_size < 2_000_000, path
    if path.suffix == '.py':
        compile(path.read_text(), str(path), 'exec')
implementations = [path for path in tracked if path.parent == Path('UQ-methods') and path.suffix == '.py']
assert len(implementations) == 18
assert len([path for path in tracked if path.parts[0] == 'assets']) == 4
assert Path('README.md') in tracked and Path('registry.json') in tracked
print(f'Checked {len(tracked)} tracked files; syntax valid and no datasets/results/weights included')
PY
```

Expected: all assertions pass, only expected files appear, and no unstaged runtime edits remain. Inspect any unexpected files rather than adding them. Inspect code/configuration changes for real credentials before pushing; do not print credential values. If new runtime edits were needed after Task 1, rerun only the affected focused tests, then the complete suite if route/extraction concerns remain unresolved.

- [x] **Step 2: Verify the authenticated owner and target remote state.**

```bash
gh api user --jq .login
gh repo view Mark-Liangs/LLM-UQ-Survey --json nameWithOwner,isPrivate,url,defaultBranchRef
git remote -v
git branch --show-current
```

Expected in the initial release: authenticated login is `Mark-Liangs`, the target does not yet exist, there is no `origin`, and the local branch is `main`. Distinguish a definite repository-not-found response from an authentication/network failure before creating anything. If a target already exists, inspect its visibility, commits and ownership; reuse it only if safe. Do not delete a repository, overwrite unrelated history, force-push, or automatically alter an existing public repository's visibility.

- [x] **Step 3: Publish the reviewed commit as a private repository.**

The original user request explicitly authorizes creating and publishing this private repository; no additional permission is needed once the selected execution workflow reaches this step. With the initial state confirmed above, run:

```bash
gh repo create Mark-Liangs/LLM-UQ-Survey \
  --private --source=. --remote=origin \
  --description "Code for The Origins of Stochasticity: Comprehensive Investigations on Uncertainty Quantification for Large Language Models" \
  --push
```

If the already-existing target is the same private repository and contains only this release's history, use its verified `origin` and normal `git push -u origin main` instead. If the result of a create/push is uncertain, query repository metadata and remote refs before retrying. Do not create a public repository even temporarily.

- [x] **Step 4: Verify remote privacy and exact commit equality, then report the outcome.**

```bash
gh repo view Mark-Liangs/LLM-UQ-Survey --json nameWithOwner,isPrivate,url,defaultBranchRef
release_remote_ok="$(gh repo view Mark-Liangs/LLM-UQ-Survey --json nameWithOwner,isPrivate,defaultBranchRef --jq '.nameWithOwner == "Mark-Liangs/LLM-UQ-Survey" and .isPrivate == true and .defaultBranchRef.name == "main"')"
test "$release_remote_ok" = true
test "$(git rev-parse HEAD)" = "$(git ls-remote origin refs/heads/main | cut -f1)"
git status --short
```

Expected: private visibility, owner/name and default branch match the request, the hashes are equal, and the working tree is clean. Mark this task complete, commit the final plan-tracking update if one remains, push normally, and verify hash equality again for that final commit.

Final user-facing report, in Chinese: link the repository and local path; state that it is private, that data was excluded, and that shared extraction retained all method coverage. Summarize the passing route/parity/CPU checks. State that full GPU/API experiments were not run; do not claim paper-result reproduction or a complete dependency installation.

## Plan Author Self-Review

- Spec coverage: Task 1 covers the whitelist, exact method/route coverage, shared extraction, unchanged algorithms/provenance, source isolation and meaningful verification. Task 2 covers both original figures, English README, actual input flows, resource requirements, output semantics and citation. Task 3 covers file inventory, private publication, owner/branch and commit verification.
- Placeholder scan: every code-changing step includes concrete code or an exact existing-source copy operation; the full README and regression module are supplied. No unresolved implementation decisions are deferred.
- Interface consistency: shared filenames match registry/test maps; route dictionary names map to the same uppercase AST override names; LM-only flags stay in its subprocess path; native parser signatures are preserved; tests use the evaluator's actual `mean`, `counts.valid` and `score_type` fields.
- Validation limits: dry-run routing is distinct from backend execution; the synthetic override probe covers evaluation order; the CPU smoke check covers score direction; no full-model or historical-result claim is made.
