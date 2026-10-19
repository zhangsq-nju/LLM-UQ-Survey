#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import sys
from uq_runtime.runner import normalize_method, infer_family

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--method')
    parser.add_argument('--family')
    parser.add_argument('--model-path')
    parser.add_argument('--dataset')
    parser.add_argument('--setting', default='answer_only')
    parser.add_argument('--list', action='store_true')
    args, _ = parser.parse_known_args()
    registry = json.loads((ROOT/'registry.json').read_text())
    if args.list:
        print('\n'.join(sorted(registry)))
        return
    if args.method:
        try:
            method = normalize_method(args.method)
            family = args.family or infer_family(args.model_path)
            key = f'{method}__{family}__{args.dataset}__{args.setting}'
        except (ValueError, argparse.ArgumentTypeError) as exc:
            raise SystemExit(str(exc))
        if key not in registry:
            raise SystemExit('Unsupported method/family/dataset/setting; use --list')
        entry = registry[key]
        if entry.get('implementation') == 'lm_polygraph':
            script = ROOT/entry['script']
            argv = [arg for arg in sys.argv[1:] if arg != '--']
            argv.extend(['--method', method, '--family', family,
                         '--dataset', args.dataset, '--setting', args.setting])
            result = subprocess.run([sys.executable, '-B', str(script)] + argv)
            raise SystemExit(result.returncode)
    from uq_runtime.runner import main as run_native
    run_native()


if __name__ == '__main__':
    main()
