#!/usr/bin/env python3

from __future__ import annotations
import ast
import re
import shutil
import string
import subprocess
import textwrap


def normalize_trivia(text):
    text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def assemble_code(prompt, answer, entry_point):
    try:
        tree = ast.parse(answer)
        if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry_point for n in tree.body):
            
            imports = '\n'.join(ast.unparse(n) for n in ast.parse(prompt).body if isinstance(n,(ast.Import,ast.ImportFrom)))
            return imports + '\n' + answer
    except SyntaxError:
        pass
    return prompt.rstrip() + '\n' + textwrap.indent(textwrap.dedent(answer).strip(), '    ') + '\n'


class Evaluator:
    def __init__(self, dataset, human_mode='docker', image='python:3.11-slim', timeout=10):
        self.dataset, self.human_mode, self.image, self.timeout = dataset, human_mode, image, timeout
        if dataset == 'gsm8k':
            try:
                from math_verify import parse, verify
            except ImportError as exc:
                raise RuntimeError('GSM8K correctness requires pip install math-verify') from exc
            self.parse, self.verify = parse, verify
        if dataset == 'humaneval' and human_mode == 'docker':
            if not shutil.which('docker'):
                raise RuntimeError('HumanEval testing requires Docker; use --human-eval skip only to generate unevaluated data')
            check = subprocess.run(['docker','image','inspect',image],capture_output=True,timeout=20)
            if check.returncode:
                raise RuntimeError(f'Docker image unavailable: {image}; pull it before generation')

    def __call__(self, example, answer):
        if self.dataset == 'triviaqa':
            pred = normalize_trivia(answer)
            correct = bool(pred) and any(pred == normalize_trivia(x) for x in example['gt']['aliases'])
            return {'correct': int(correct), 'status': 'normalized_alias_match'}
        if self.dataset == 'gsm8k':
            try:
                pred = self.parse(answer)
                gold = self.parse(example['gt']['value'])
                correct = bool(pred) and bool(gold) and bool(self.verify(gold,pred))
                return {'correct': int(correct), 'status': 'math_verify'}
            except Exception as exc:
                return {'correct': None, 'status': 'evaluation_error', 'evaluation_error': str(exc)}
        if self.human_mode == 'skip':
            return {'correct': None, 'status': 'not_evaluated'}
        code = assemble_code(example['gt'].get('prompt', example['question']), answer, example['gt']['entry_point'])
        harness = code + '\n' + example['gt']['test'] + '\ncheck(' + example['gt']['entry_point'] + ')\n'
        
        name = 'uq-eval-' + __import__('uuid').uuid4().hex
        command = ['docker','run','--rm','-i','--name',name,'--network','none','--read-only',
                   '--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','64',
                   '--memory','256m','--cpus','1','--user','65534:65534',
                   '--tmpfs','/tmp:rw,noexec,nosuid,size=16m',self.image,'python','-I','-B','-']
        try:
            result = subprocess.run(command,input=harness,text=True,capture_output=True,timeout=self.timeout)
            if result.returncode in {125,126,127}:
                raise RuntimeError('Docker infrastructure failure: '+result.stderr[-1000:])
            return {'correct': int(result.returncode==0), 'status': 'unit_test_pass' if result.returncode==0 else 'unit_test_fail',
                    'test_stderr': result.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            return {'correct':0,'status':'unit_test_timeout'}
        finally:
            subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
