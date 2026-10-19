"""Load the upstream 3-shot/test benchmarks as local experiment inputs."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from data_generation.schema import normalize_example


SOURCES = {
    'gsm8k': ('gsm8k_stop_strings', '45aace450ef42f149e4165a2d78275ef9c8ab011'),
    'triviaqa': ('trivia_stop_strings', 'de1471850c50b58af147db2abfb2c260164aed88'),
    'humaneval': ('humaneval_stop_strings', '6e96dd831805e3cbf7197e27f3c5171d36d89507'),
}


def prepare(dataset, output_dir, overwrite=False):
    output = Path(output_dir) / f'{dataset}.jsonl'
    if output.exists() and not overwrite:
        raise ValueError(f'{output} exists; use --overwrite to replace it')
    repository, revision = SOURCES[dataset]
    rows = load_dataset(f'graphUQ-ls-hxy/{repository}', name='3-shot', split='test', revision=revision)
    reference = {}
    if dataset == 'humaneval':
        reference = {row['task_id']: row for row in load_dataset(
            'openai/openai_humaneval', split='test', revision='7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544')}
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with output.open('w' if overwrite else 'x', encoding='utf-8') as stream:
        for index, row in enumerate(rows):
            record = {'id': row['id'], 'question': row['input'], 'answer': row['output']}
            if dataset == 'humaneval':
                task = reference[row['id']]
                record['gt'] = {key: task[key] for key in ('prompt', 'test', 'entry_point')}
                record['gt']['canonical_solution'] = row['output']
            example = normalize_example(record, dataset, index)
            if example['id'] in seen:
                example['id'] += f'__row_{index}'
            seen.add(example['id'])
            stream.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f'{dataset}: {len(rows)} examples -> {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['all', *SOURCES], default='all')
    parser.add_argument('--output-dir', default='data')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    try:
        for dataset in SOURCES if args.dataset == 'all' else [args.dataset]:
            prepare(dataset, args.output_dir, args.overwrite)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
