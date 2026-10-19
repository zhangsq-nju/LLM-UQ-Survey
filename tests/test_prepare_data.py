import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_generation.schema import build_messages, normalize_example, read_examples


class PrepareDataTests(unittest.TestCase):
    def test_prepares_upstream_three_shot_inputs_without_changing_them(self):
        original_input = 'Upstream 3-shot prompt.\nExamples and instructions stay unchanged.\x85\n'
        reference = {'task_id': 'HumanEval/3', 'prompt': 'def increment(x):\n    """Add one."""\n',
                     'entry_point': 'increment', 'test': 'def check(candidate):\n    assert candidate(2) == 3\n'}

        def load(repository, name=None, split=None, revision=None):
            self.assertEqual(split, 'test')
            if repository == 'openai/openai_humaneval':
                return [reference]
            self.assertEqual(name, '3-shot')
            answer = '18'
            qid = 'source-id'
            if repository.endswith('/trivia_stop_strings'):
                answer = {'value': 'New York City', 'aliases': ['NYC', 'New York City']}
            elif repository.endswith('/humaneval_stop_strings'):
                qid, answer = 'HumanEval/3', '    return x + 1\n'
            row = {'id': qid, 'input': original_input, 'output': answer}
            return [row, dict(row)] if repository.endswith('/trivia_stop_strings') else [row]

        lm = runpy.run_path(str(ROOT / 'UQ-methods/lm_polygraph.py'))
        with tempfile.TemporaryDirectory() as temporary, patch('datasets.load_dataset', side_effect=load), \
                patch.object(sys, 'argv', ['prepare', '--output-dir', temporary]), contextlib.redirect_stdout(io.StringIO()):
            runpy.run_module('data_generation.prepare', run_name='__main__')
            for dataset, count in (('gsm8k', 1), ('triviaqa', 2), ('humaneval', 1)):
                rows = list(read_examples(Path(temporary) / f'{dataset}.jsonl'))
                self.assertEqual(len(rows), count)
                self.assertEqual(len({row['id'] for row in rows}), count)
                for row in rows:
                    self.assertEqual(row['question'], original_input)
                    self.assertEqual(normalize_example(row, dataset, 0), row)
                    self.assertEqual(lm['normalize_example'](row, dataset, 0), row)
                    for setting in ('answer_only', 'reasoning_augmented'):
                        for builder in (build_messages, lm['build_messages']):
                            messages = builder(row, dataset, setting)
                            self.assertEqual([message['role'] for message in messages], ['system', 'user'])
                            self.assertEqual(messages[-1]['content'], original_input)
                if dataset == 'triviaqa':
                    self.assertEqual(set(rows[0]['gt']['aliases']), {'New York City', 'NYC'})
                if dataset == 'humaneval':
                    self.assertEqual(rows[0]['gt']['prompt'], reference['prompt'])
                    self.assertEqual(rows[0]['gt']['test'], reference['test'])

    def test_humaneval_uses_reference_function_for_tests(self):
        from data_generation.evaluation import Evaluator
        example = {'question': 'Unchanged upstream 3-shot input.',
                   'gt': {'prompt': 'def increment(x):\n    """Add one."""\n',
                          'entry_point': 'increment',
                          'test': 'def check(candidate):\n    assert candidate(2) == 3\n'}}

        def docker_run(command, **kwargs):
            if command[1] == 'run':
                try:
                    exec(compile(kwargs['input'], '<HumanEval fixture>', 'exec'), {})
                except (SyntaxError, AssertionError) as exc:
                    return subprocess.CompletedProcess(command, 1, '', str(exc))
            return subprocess.CompletedProcess(command, 0, '', '')

        classes = [Evaluator] + [runpy.run_path(str(ROOT / 'UQ-methods' / name))['Evaluator']
                                for name in ('lm_polygraph.py', 'paper_author_aligned.py')]
        for evaluator_class in classes:
            evaluator = evaluator_class('humaneval', human_mode='skip')
            evaluator.human_mode = 'docker'
            with patch('subprocess.run', side_effect=docker_run):
                self.assertEqual(evaluator(example, '    return x + 1\n')['correct'], 1)


if __name__ == '__main__':
    unittest.main()
