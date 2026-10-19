#!/usr/bin/env python3
# Original paper/author repository: https://github.com/cognizant-ai-labs/semantic-density-paper
# Original paper/author repository: https://github.com/jlko/semantic_uncertainty
# Original paper/author repository: https://github.com/zlin7/UQ-NLG
# Implementation repository: https://github.com/IINemo/lm-polygraph

from __future__ import annotations
import argparse
import ast
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import gzip
import re
import shutil
import subprocess
import string
import textwrap


REVISION = '1b24f22d0cd426d3f89e7c98dc4b95115b6f3586'


METHODS = {
    'max_seq_prob': ('max_probability', 'MaximumSequenceProbability'),
    'max_tok_prob': ('max_probability', 'MaximumTokenProbability'),
    'mean_tok_ent': ('token_entropy', 'MeanTokenEntropy'),
    'ppl': ('perplexity', 'Perplexity'),
    'num_set': ('num_sem_sets', 'NumSemSets'),
    'eigv': ('eig_val_laplacian', 'EigValLaplacian'),
    'deg': ('deg_mat', 'DegMat'),
    'ecc': ('eccentricity', 'Eccentricity'),
    'semantic_entropy': ('semantic_entropy', 'SemanticEntropy'),
    'semantic_density': ('semantic_density', 'SemanticDensity'),
    'p_true': ('p_true', 'PTrue'),
    'verbalized_probability': ('verbalized_2s', 'Verbalized2S'),
    'token_sar': ('token_sar', 'TokenSAR'),
    'pmi': ('pointwise_mutual_information', 'PointwiseMutualInformation'),
    'mean_pmi': ('pointwise_mutual_information', 'MeanPointwiseMutualInformation'),
}

def normalize_method(value):
    import re
    key = re.sub('[^a-z0-9]', '', value.lower())
    aliases = {re.sub('[^a-z0-9]', '', k): k for k in METHODS}
    aliases.update({'perplexity':'ppl', 'maximumsequenceprobability':'max_seq_prob',
                    'maximumtokenprobability':'max_tok_prob', 'meantokenentropy':'mean_tok_ent'})
    if key not in aliases:
        raise argparse.ArgumentTypeError(f'No verified LM-Polygraph counterpart: {value}. Use --list; MARS and CoT-UQ are not aliases for TokenSAR.')
    return aliases[key]


def make_estimator(method, kwargs):
    module, name = METHODS[method]
    cls = getattr(importlib.import_module('lm_polygraph.estimators.' + module), name)
    if method == 'verbalized_probability':
        kwargs = {'confidence_prompt': 'Give your confidence that your previous answer is correct. Output exactly Confidence: followed by a number between 0 and 1.',
                  'confidence_regex': r'Confidence:\s*(0(?:\.\d+)?|1(?:\.0+)?)\b', **kwargs}
    return cls(**kwargs)


def numeric(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [numeric(x) for x in value]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Upstream returned a non-finite score; output not committed')
    return result


def evaluate_stats(estimator, stats):
    
    import numpy as np
    missing = set(estimator.stats_dependencies) - stats.keys()
    if missing:
        raise ValueError(f'Missing upstream statistics: {sorted(missing)}')
    converted = {}
    for key, value in stats.items():
        
        
        if 'matrix' in key or key == 'token_similarity':
            value = np.asarray(value)
        converted[key] = value
    
    
    if type(estimator).__name__ == 'PointwiseMutualInformation':
        if len(stats['greedy_log_likelihoods']) != 1:
            raise ValueError('stats must contain exactly one question')
        return numeric(estimator(converted))
    result = estimator(converted)
    if len(result) != 1:
        raise ValueError('stats must describe exactly one question (batch dimension = 1)')
    return numeric(result[0])


def load_model(args):
    from lm_polygraph.utils.generation_parameters import GenerationParameters
    params = GenerationParameters(max_new_tokens=args.max_new_tokens)
    if args.family == 'deepseek':
        from lm_polygraph.utils.model import BlackboxModel
        from openai import OpenAI
        key = os.environ.get('DEEPSEEK_API_KEY')
        if not key:
            raise ValueError('DEEPSEEK_API_KEY is required')
        model = BlackboxModel(openai_api_key=key, model_path=args.model_path,
                              generation_parameters=params, supports_logprobs=args.api_logprobs)
        model.openai_api = OpenAI(api_key=key, base_url=args.api_base_url)
        return model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_polygraph.utils.model import WhiteboxModel
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype='auto').to(args.device)
    base.eval()
    class RenderedPromptModel(WhiteboxModel):
        
        
        def tokenize(self, texts):
            rendered = []
            for text in texts:
                if isinstance(text, list):
                    options = {'enable_thinking':False} if args.family == 'qwen' else {}
                    text = self.tokenizer.apply_chat_template(text, tokenize=False,
                                add_generation_prompt=True, **options)
                rendered.append(text)
            return self.tokenizer(rendered, padding=True, return_tensors='pt', add_special_tokens=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return RenderedPromptModel(base, tokenizer, model_path=args.model_path,
                               generation_parameters=params, instruct=False)


def provenance():
    dist = importlib.metadata.distribution('lm-polygraph')
    direct = dist.read_text('direct_url.json')
    return {'version':dist.version, 'installation':json.loads(direct) if direct else None,
            'audited_revision':REVISION, 'repository':'https://github.com/IINemo/lm-polygraph'}


def main(argv=None, fixed=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', type=normalize_method)
    parser.add_argument('--mode', choices=['generate','stats'], default='generate')
    parser.add_argument('--family', choices=['qwen','llama','deepseek'])
    parser.add_argument('--model-path')
    parser.add_argument('--dataset', choices=['gsm8k','triviaqa','humaneval'])
    parser.add_argument('--setting', choices=['answer_only','reasoning_augmented'], default='answer_only')
    parser.add_argument('--input-file')
    parser.add_argument('--output-file')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--estimator-kwargs', type=json.loads, default={})
    parser.add_argument('--human-eval', choices=['docker','skip'], default='docker')
    parser.add_argument('--api-base-url', default='https://api.deepseek.com')
    parser.add_argument('--api-logprobs', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.set_defaults(**(fixed or {}))
    args = parser.parse_args(argv)
    for key, value in (fixed or {}).items():
        if getattr(args,key) != value:
            parser.error(f'This file fixes --{key.replace("_","-")} to {value}')
    if args.list:
        print(json.dumps(METHODS, indent=2)); return
    for name in ['method','dataset','input_file','output_file']:
        if not getattr(args,name):parser.error(f'--{name.replace("_","-")} required')
    if not isinstance(args.estimator_kwargs,dict):parser.error('--estimator-kwargs must be a JSON object')
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):parser.error('Token and question limits must be positive')
    if args.mode == 'generate' and not args.model_path:parser.error('--model-path required for generation')
    if args.mode == 'generate' and not args.family:
        args.family = next((f for f in ['qwen','llama','deepseek'] if f in args.model_path.lower()),None)
        if args.family is None:parser.error('Specify --family for an unrecognized model path')
    if args.mode == 'generate' and args.method == 'pmi':
        parser.error('Upstream high-level API truncates flattened PMI to one token; use --mode stats or method mean_pmi')
    if args.mode == 'stats' and args.method == 'verbalized_probability':
        parser.error('Verbalized2S requires a live model; use --mode generate')
    if args.family == 'deepseek' and args.mode == 'generate':
        if args.setting != 'answer_only':parser.error('DeepSeek reasoning_augmented is outside the verified experiment scope')
        if args.method not in {'num_set','eigv','deg','ecc'}:parser.error('Only verified blackbox graph routes are enabled for DeepSeek generation; use stats for other estimators')
    inp, out = Path(args.input_file).resolve(), Path(args.output_file).resolve()
    if not inp.is_file():parser.error('Input does not exist')
    if inp == out:parser.error('Output cannot replace input')
    if out.exists() and not args.overwrite:parser.error('Output exists; use --overwrite')
    if args.dry_run:
        print(json.dumps({'implementation':'lm_polygraph', 'method':args.method, 'class':METHODS[args.method],
                          'mode':args.mode, 'setting':args.setting, 'estimator_kwargs':args.estimator_kwargs,
                          'input':str(inp), 'output':str(out)},ensure_ascii=False)); return
    try:
        estimator = make_estimator(args.method,args.estimator_kwargs)
        source = provenance()
    except ImportError as exc:
        raise RuntimeError('Install dependencies from requirements.txt') from exc
    if args.mode == 'generate':
        from transformers import set_seed
        from lm_polygraph.utils import estimate_uncertainty
        set_seed(args.seed)
        evaluator = Evaluator(args.dataset,human_mode=args.human_eval)
        model = load_model(args)
    out.parent.mkdir(parents=True,exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+out.name+'.',suffix='.tmp',dir=out.parent)
    seen = set()
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            for index,row in enumerate(read_examples(inp)):
                if args.limit is not None and index >= args.limit:break
                if args.mode == 'stats':
                    qid = str(row.get('id',index))
                    score = evaluate_stats(estimator,row['stats'])
                    answer, correct = row.get('answer'), row.get('correct')
                    evaluation = {'status':'provided_upstream_statistics_label'}
                    question = row.get('question')
                else:
                    example = normalize_example(row,args.dataset,index)
                    qid, question = example['id'], example['question']
                    messages = build_messages(example,args.dataset,args.setting)
                    if args.family == 'deepseek':
                        prompt = '\n\n'.join(m['content'] for m in messages)
                    else:
                        kwargs = {'enable_thinking':False} if args.family == 'qwen' else {}
                        prompt = model.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**kwargs)
                    result = estimate_uncertainty(model,estimator,input_text=prompt)
                    score = numeric(result.uncertainty)
                    answer = result.generation_text
                    start, end, found = answer_span(answer,args.setting)
                    evaluation = (evaluator(example,clean_answer(answer[start:end],args.dataset))
                                  if found else {'correct':None,'status':'missing_final_answer'})
                    correct = evaluation['correct']
                if qid in seen:raise ValueError(f'Duplicate question ID: {qid}')
                seen.add(qid)
                if correct is not None and (isinstance(correct,str) or correct not in (0,1)):
                    raise ValueError(f'{qid}: correct must be 0, 1 or null')
                record = {'id':qid,'question':question,'answer':answer,'correct':correct,
                          'evaluation':evaluation, 'method':'lm_polygraph_' + args.method,'base_method':args.method,'implementation':'lm_polygraph',
                          'setting':args.setting,'dataset':args.dataset,'model_path':args.model_path,
                          'score':score,'score_type':'uncertainty','level':estimator.level,
                          'provenance':source,'estimator':str(estimator),
                          'run':{'mode':args.mode,'seed':args.seed,'max_new_tokens':args.max_new_tokens,
                                 'estimator_kwargs':args.estimator_kwargs, 'device':args.device,
                                 'sampling_configuration':'upstream registered calculator defaults (audited samples_n=10)'}}
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        if not seen:raise ValueError('No questions processed')
        os.replace(temporary,out)
    finally:
        Path(temporary).unlink(missing_ok=True)





def read_examples(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        if path.suffix == '.json':
            rows = json.load(stream)
            if isinstance(rows, dict):
                rows = rows.get('data', rows.get('examples'))
            if not isinstance(rows, list):
                raise ValueError('JSON input must be an array or contain data/examples array')
            yield from rows
        else:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f'Input line {number} is not an object')
                    yield row

def normalize_example(row, dataset, index):
    qid = next((row[k] for k in ['id', 'question_id', 'task_id', 'qid'] if row.get(k) is not None), str(index))
    question = next((row[k] for k in ['question', 'prompt', 'problem', 'input', 'query'] if isinstance(row.get(k), str)), None)
    if not question:
        raise ValueError(f'{qid}: missing question/prompt')
    gold = next((row[k] for k in ['gt', 'ground_truth', 'correct answer', 'correct_answer', 'answer', 'answers', 'target'] if row.get(k) is not None), None)
    if dataset == 'humaneval':
        gt = dict(gold) if isinstance(gold, dict) else {}
        for key in ['test', 'entry_point', 'canonical_solution']:
            if key in row:
                gt[key] = row[key]
        if not gt.get('test') or not gt.get('entry_point'):
            raise ValueError(f'{qid}: HumanEval requires test and entry_point')
    else:
        if isinstance(gold, dict):
            value = gold.get('value', gold.get('normalized_value', gold.get('text')))
            aliases = gold.get('aliases', gold.get('normalized_aliases', [])) or []
        elif isinstance(gold, list):
            value = gold[0] if gold else None
            aliases = gold
        else:
            value, aliases = (gold, row.get('aliases', []) or [])
        if value is None or str(value).strip() == '':
            raise ValueError(f'{qid}: missing gold answer')
        value = str(value)
        if dataset == 'gsm8k' and '####' in value:
            value = value.rsplit('####', 1)[1].strip()
        gt = {'value': value, 'aliases': list(dict.fromkeys([value] + [str(x) for x in aliases]))}
    return {'id': str(qid), 'question': question, 'gt': gt}

def build_messages(example, dataset, setting):
    task = {'gsm8k': 'Solve the math word problem. The final answer must be a number.', 'triviaqa': 'Answer the trivia question. The final answer must be a short noun phrase.', 'humaneval': 'Solve the Python programming task. Preserve the required function name and signature. Return a complete Python function, without tests.'}[dataset]
    if setting == 'reasoning_augmented':
        instruction = task + '\nExplain your reasoning in numbered steps, then write exactly "Final Answer:" followed by the final answer. For Python, put the final code after that marker.'
    else:
        instruction = task + '\nReturn only the final answer. No reasoning or explanations. For Python return code only, without Markdown fences.'
    return [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': example['question']}]

def answer_span(text, setting):
    if setting == 'answer_only':
        return (0, len(text), True)
    matches = list(re.finditer('(?im)^\\s*Final\\s+Answer\\s*:\\s*', text))
    if matches:
        return (matches[-1].end(), len(text), True)
    return (len(text), len(text), False)

def clean_answer(text, dataset):
    text = text.strip()
    if dataset == 'humaneval':
        match = re.search('```(?:python)?\\s*\\n(.*?)```', text, re.S | re.I)
        if match:
            return match.group(1).rstrip()
    return text

def normalize_trivia(text):
    text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub('\\b(a|an|the)\\b', ' ', text).split())

def assemble_code(prompt, answer, entry_point):
    try:
        tree = ast.parse(answer)
        if any((isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry_point for n in tree.body)):
            imports = '\n'.join((ast.unparse(n) for n in ast.parse(prompt).body if isinstance(n, (ast.Import, ast.ImportFrom))))
            return imports + '\n' + answer
    except SyntaxError:
        pass
    return prompt.rstrip() + '\n' + textwrap.indent(textwrap.dedent(answer).strip(), '    ') + '\n'

class Evaluator:

    def __init__(self, dataset, human_mode='docker', image='python:3.11-slim', timeout=10):
        self.dataset, self.human_mode, self.image, self.timeout = (dataset, human_mode, image, timeout)
        if dataset == 'gsm8k':
            try:
                from math_verify import parse, verify
            except ImportError as exc:
                raise RuntimeError('GSM8K correctness requires pip install math-verify') from exc
            self.parse, self.verify = (parse, verify)
        if dataset == 'humaneval' and human_mode == 'docker':
            if not shutil.which('docker'):
                raise RuntimeError('HumanEval testing requires Docker; use --human-eval skip only to generate unevaluated data')
            check = subprocess.run(['docker', 'image', 'inspect', image], capture_output=True, timeout=20)
            if check.returncode:
                raise RuntimeError(f'Docker image unavailable: {image}; pull it before generation')

    def __call__(self, example, answer):
        if self.dataset == 'triviaqa':
            pred = normalize_trivia(answer)
            correct = bool(pred) and any((pred == normalize_trivia(x) for x in example['gt']['aliases']))
            return {'correct': int(correct), 'status': 'normalized_alias_match'}
        if self.dataset == 'gsm8k':
            try:
                pred = self.parse(answer)
                gold = self.parse(example['gt']['value'])
                correct = bool(pred) and bool(gold) and bool(self.verify(gold, pred))
                return {'correct': int(correct), 'status': 'math_verify'}
            except Exception as exc:
                return {'correct': None, 'status': 'evaluation_error', 'evaluation_error': str(exc)}
        if self.human_mode == 'skip':
            return {'correct': None, 'status': 'not_evaluated'}
        code = assemble_code(example['gt'].get('prompt', example['question']), answer, example['gt']['entry_point'])
        harness = code + '\n' + example['gt']['test'] + '\ncheck(' + example['gt']['entry_point'] + ')\n'
        name = 'uq-eval-' + __import__('uuid').uuid4().hex
        command = ['docker', 'run', '--rm', '-i', '--name', name, '--network', 'none', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '64', '--memory', '256m', '--cpus', '1', '--user', '65534:65534', '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m', self.image, 'python', '-I', '-B', '-']
        try:
            result = subprocess.run(command, input=harness, text=True, capture_output=True, timeout=self.timeout)
            if result.returncode in {125, 126, 127}:
                raise RuntimeError('Docker infrastructure failure: ' + result.stderr[-1000:])
            return {'correct': int(result.returncode == 0), 'status': 'unit_test_pass' if result.returncode == 0 else 'unit_test_fail', 'test_stderr': result.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            return {'correct': 0, 'status': 'unit_test_timeout'}
        finally:
            subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)

if __name__ == '__main__':
    main()
