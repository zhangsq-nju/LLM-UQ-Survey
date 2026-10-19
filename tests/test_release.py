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
    if implementation in ('lm_polygraph', 'paper_author_aligned'):
        # HumanEval tests may use a reference function prompt beside the full upstream input.
        reference_call = ast.parse("assemble_code(example['gt'].get('prompt', example['question']), answer, example['gt']['entry_point'])", mode='eval').body
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(reference_call):
                node.args[0] = ast.parse("example['question']", mode='eval').body
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
