#!/usr/bin/env python3

from __future__ import annotations
import gzip
import json
import math
import re
from pathlib import Path


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
            value, aliases = gold, row.get('aliases', []) or []
        if value is None or str(value).strip() == '':
            raise ValueError(f'{qid}: missing gold answer')
        value = str(value)
        if dataset == 'gsm8k' and '####' in value:
            value = value.rsplit('####', 1)[1].strip()
        gt = {'value': value, 'aliases': list(dict.fromkeys([value] + [str(x) for x in aliases]))}
    return {'id': str(qid), 'question': question, 'gt': gt}


def build_messages(example, dataset, setting):
    task = {'gsm8k': 'Solve the math word problem. The final answer must be a number.',
            'triviaqa': 'Answer the trivia question. The final answer must be a short noun phrase.',
            'humaneval': 'Solve the Python programming task. Preserve the required function name and signature. Return a complete Python function, without tests.'}[dataset]
    if setting == 'reasoning_augmented':
        instruction = task + '\nExplain your reasoning with each step starting exactly as "Step 1:", "Step 2:", and so on, then write exactly "Final Answer:" followed by the final answer. For Python, put the final code after that marker.'
    else:
        instruction = task + '\nReturn only the final answer. No reasoning or explanations. For Python return code only, without Markdown fences.'
    
    return [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': example['question']}]


def answer_span(text, setting):
    if setting == 'answer_only':
        return 0, len(text), True
    matches = list(re.finditer(r'(?im)^\s*Final\s+Answer\s*:\s*', text))
    if matches:
        return matches[-1].end(), len(text), True
    return len(text), len(text), False


def clean_answer(text, dataset):
    text = text.strip()
    if dataset == 'humaneval':
        match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.S | re.I)
        if match:
            return match.group(1).rstrip()
    return text


def prob_info(steps):
    return {'token logprob': [s['chosen_logprob'] for s in steps],
            'max logprob': [s.get('max_logprob') for s in steps],
            'neg entropy': [-s['entropy'] if s.get('entropy') is not None else None for s in steps],
            'token id': [s.get('token_id') for s in steps],
            'token text': [s['text_piece'] for s in steps]}


def reasoning_response(generation, dataset, setting='reasoning_augmented'):
    text = generation['raw_text']
    start, end, found = answer_span(text, setting)
    
    answer_steps = [s for s in generation['steps'] if s['char_end'] > start]
    reason_steps = [s for s in generation['steps'] if s['char_end'] <= start]
    final = clean_answer(text[start:end], dataset) if found else ''
    reason = text[:start] if found else text
    units = []
    for match in re.finditer(r'[^\n]+(?:\n|$)', reason):
        selected = [s for s in reason_steps if match.start() < s['char_end'] <= match.end()]
        if selected:
            units.append({'text': match.group(), 'prob info': prob_info(selected)})
    return {'llm reason': reason, 'llm answer': final, 'step-wise info': units,
            'answer info': {'text': text[start:end] if found else '', 'prob info': prob_info(answer_steps)},
            'final_answer_marker_found': found, 'correct': generation['correct'],
            'raw_text': text, 'generation': generation}


def finalize_generation(generation, example, dataset, setting, evaluator):
    text = generation['raw_text']
    start, end, found = answer_span(text, setting)
    answer = clean_answer(text[start:end], dataset) if found else ''
    generation.update(text=answer, final_answer_marker_found=found)
    verdict = evaluator(example, answer) if found else {'correct': 0, 'status': 'missing_final_answer_marker'}
    generation.update(verdict)
    logprobs = [s['chosen_logprob'] for s in generation['steps']]
    valid = bool(logprobs) and all(v is not None and math.isfinite(v) for v in logprobs)
    generation['seq_logprob'] = sum(logprobs) if valid else None
    generation['avg_logprob'] = sum(logprobs)/len(logprobs) if valid else None
    generation['sequence_probability'] = math.exp(sum(logprobs)) if valid else None
    generation['probabilities_available'] = valid
    return generation
