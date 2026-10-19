#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {
    "max_seq_prob": ["maximum_sequence_probability"],
    "mean_tok_ent": ["mean_token_entropy"],
    "max_tok_prob": ["maximum_token_probability"],
    "ppl": ["perplexity"],
    "num_set": ["number_of_semantic_sets", "metrics.number_of_semantic_sets"],
    "eigv": ["sum_of_eigenvalues_of_graph_laplacian", "metrics.sum_of_eigenvalues_of_graph_laplacian"],
    "deg": ["degree_mean", "metrics.the_degree_matrix.degree_mean", "the_degree_matrix.degree_mean", "metrics.degree_matrix.degree_mean", "degree_matrix.degree_mean"],
    "ecc": ["eccentricity_mean", "metrics.eccentricity.eccentricity_mean", "eccentricity.eccentricity_mean"],
    "semantic_entropy": ["semantic_entropy", "semantic_entropy_uncertainty"],
    "semantic_density": ["semantic_density_confidence", "semantic_density"],
    "bsdetector": ["bsdetector_uncertainty", "uncertainty"],
    "selfcheckgpt": ["selfcheck_uncertainty", "selfcheck_nli_uncertainty", "uncertainty"],
    "conu": ["conu_uncertainty", "uncertainty"],
    "mars": ["mars_uncertainty", "uncertainty"],
    "p_true": ["p_true_uncertainty", "ptrue_uncertainty", "uncertainty"],
    "verbalized_probability": ["verbalized_probability_uncertainty", "uncertainty"],
    "cot_uq_me": ["cotuq_probas_mean_uncertainty", "uncertainties.Probas-mean"],
    "cot_uq_mi": ["cotuq_probas_min_uncertainty", "uncertainties.Probas-min"],
    "cot_uq_sar": ["cotuq_token_sar_uncertainty", "uncertainties.Token-SAR"],
    "cot_uq_sp": ["cotuq_self_probing_uncertainty", "uncertainties.Self-Probing"],
    "topo_uq": ["topouq_uncertainty", "uncertainty", "topo_uncertainty"],
}


def normalize_method(value):
    key = re.sub(r"[^a-z0-9]", "", value.lower())
    aliases = {re.sub(r"[^a-z0-9]", "", m): m for m in FIELDS}
    aliases.update({"numset": "num_set", "numsets": "num_set", "selfcheck": "selfcheckgpt",
                    "perplexity": "ppl", "maximumsequenceprobability": "max_seq_prob",
                    "maximumtokenprobability": "max_tok_prob", "meantokenentropy": "mean_tok_ent",
                    "probasmean": "cot_uq_me", "probasmin": "cot_uq_mi"})
    if key not in aliases:
        raise argparse.ArgumentTypeError(f"Unknown method: {value}; use --list")
    return aliases[key]


def infer_family(model):
    value = (model or "").lower()
    for family in ["qwen", "llama", "deepseek"]:
        if family in value:
            return family
    raise ValueError("模型路径无法推断 family，请传 --family qwen/llama/deepseek")


def read_options(path):
    result = {}
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.FunctionDef) and node.name == "_uq_standalone")]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            options = [a.value for a in node.args if isinstance(a, ast.Constant)
                       and isinstance(a.value, str) and a.value.startswith("--")]
            for option in options:
                result[option] = node
    return result


def prepare_backend(entry, args, raw_output, extra):
    path = ROOT / entry["script"]
    options = read_options(path)
    argv = [str(path)]
    def add(names, value):
        if value is None:
            return
        option = next((n for n in names if n in options), None)
        if option:
            argv.extend([option, str(value)])
    add(["--input-file", "--input_file"], str(Path(args.input_file).resolve()))
    add(["--output-file", "--output_file"], str(raw_output))
    
    if args.method not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
        add(["--model-path", "--model_name_or_path", "--judge-model-path"], args.model_path)
    add(["--api-model"], args.model_path)
    add(["--model-family", "--judge-family"], args.family)
    add(["--correct-file", "--correct_file"], args.correct_file)
    add(["--problem-file"], args.problem_file)
    add(["--nli-model", "--nli-model-path"], args.nli_model)
    add(["--device"], args.device)
    
    protected = {"--input-file", "--input_file", "--output-file", "--output_file",
                 "--model-path", "--model_name_or_path", "--judge-model-path",
                 "--model-family", "--judge-family", "--api-model"}
    if any(x.split("=")[0] in protected for x in extra):
        raise ValueError("-- 后不能覆盖入口的 input/output/model/family 参数")
    argv.extend(extra)
    provided = {x.split("=")[0] for x in argv if x.startswith("--")}
    for opt, node in options.items():
        required = any(k.arg == "required" and isinstance(k.value, ast.Constant) and k.value.value
                       for k in node.keywords)
        aliases = [a.value for a in node.args if isinstance(a, ast.Constant)]
        if required and not provided.intersection(aliases):
            raise ValueError(f"该方法还需要参数 {opt}；可通过统一参数或 -- {opt} VALUE 传入")
    overrides = {"INPUT_FILE": str(Path(args.input_file).resolve()), "OUTPUT_FILE": str(raw_output),
                 "MODEL_NAME": args.model_path, "JUDGE_MODEL_NAME": args.model_path,
                 "API_MODEL": args.model_path, "MODEL_FAMILY": args.family}
    overrides.update({
        'METHOD': entry['method'],
        'FAMILY': entry['family'],
        'DATASET': entry['dataset'],
        'SETTING': entry['setting'],
    })
    if args.nli_model:
        overrides.update({"NLI_MODEL_NAME": args.nli_model, "NLI_MODEL": args.nli_model})
    if args.device:
        overrides["DEVICE"] = args.device
    return path, argv, {k: v for k, v in overrides.items() if v is not None}


def execute_backend(path, argv, overrides):
    
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [x.id for x in targets if isinstance(x, ast.Name)]
            for name in names:
                if name in overrides:
                    node.value = ast.Constant(overrides[name])
    ast.fix_missing_locations(tree)
    module = types.ModuleType("__main__")
    module.__file__ = str(path)
    module.__package__ = None
    namespace = module.__dict__
    namespace["_UQ_EXECUTING"] = True
    sys.argv = argv
    previous = sys.modules.get("__main__")
    sys.modules["__main__"] = module
    try:
        exec(compile(tree, str(path), "exec"), namespace)
    finally:
        sys.modules["__main__"] = previous


def lookup(row, path):
    value = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def extract_score(row, method):
    for field in FIELDS[method]:
        value = lookup(row, field)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return field, value
    
    fallback = {"conu": ["conu_confidence"], "selfcheckgpt": ["selfcheck_confidence"],
                "p_true": ["p_true", "ptrue", "p_true_confidence"],
                "verbalized_probability": ["verbalized_probability", "confidence"]}
    for field in fallback.get(method, []):
        value = lookup(row, field)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return field, value
    raise ValueError(f"输出缺少 {method} 的有效分数；id={row.get('id')}。未把缺失值当作 0。")


def main(fixed=None):
    parser = argparse.ArgumentParser(description="运行清洗后的 UQ 方法；输入为已有模型回答的 stage1 JSONL。")
    parser.add_argument("--method", type=normalize_method)
    parser.add_argument("--family", choices=["qwen", "llama", "deepseek"])
    parser.add_argument("--model-path", help="本地/HuggingFace 模型路径，DeepSeek 为 API model name")
    parser.add_argument("--dataset", choices=["gsm8k", "triviaqa", "humaneval"])
    parser.add_argument("--setting", choices=["answer_only", "reasoning_augmented"], default="answer_only")
    parser.add_argument("--input-file", help="stage1 JSONL 路径（含回答/采样/所需概率）")
    parser.add_argument("--output-file")
    parser.add_argument("--correct-file", help="HumanEval 单元测试所得的逐样本正确性 JSONL")
    parser.add_argument("--problem-file", help="可选的原始题目文件")
    parser.add_argument("--nli-model", help="NLI 模型名称或本地路径")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只核验参数和路由，不加载模型")
    parser.add_argument("--overwrite", action="store_true")
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    args = parser.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    for name, value in (fixed or {}).items():
        setattr(args, name, value)
    registry = json.loads((ROOT / "registry.json").read_text())
    if args.list:
        for key in sorted(registry):
            print(key)
        return
    try:
        for name in ["method", "dataset", "input_file", "output_file"]:
            if not getattr(args, name):
                raise ValueError(f"缺少 --{name.replace('_', '-')}")
        args.family = args.family or infer_family(args.model_path)
        key = f"{args.method}__{args.family}__{args.dataset}__{args.setting}"
        if key not in registry:
            raise ValueError(f"尚无已核实实现：{key}；可用 --list 查看支持的组合")
        entry = dict(registry[key])
        entry.setdefault('implementation', 'local')
        model_free = {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl", "num_set", "eigv",
                      "deg", "ecc", "semantic_entropy", "semantic_density", "conu", "selfcheckgpt", "topo_uq", "mars", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp"}
        if args.method not in model_free and not args.model_path:
            raise ValueError("该方法需要 --model-path 指定打分模型或 API model name")
        input_path = Path(args.input_file).resolve()
        output = Path(args.output_file).resolve()
        if not input_path.is_file():
            raise ValueError(f"输入不存在：{input_path}")
        for value in [args.correct_file, args.problem_file]:
            if value and not Path(value).is_file():
                raise ValueError(f"辅助输入不存在：{value}")
        if output in [input_path] + [Path(x).resolve() for x in [args.correct_file, args.problem_file] if x]:
            raise ValueError("输出路径不能覆盖输入或正确性/题目文件")
        if output.exists() and not args.overwrite:
            raise ValueError("输出文件已存在；如需替换请传 --overwrite")
        for name in ["correct_file", "problem_file", "model_path", "nli_model"]:
            value = getattr(args, name)
            if value and Path(value).exists():
                setattr(args, name, str(Path(value).resolve()))
        
        with input_path.open() as stream:
            prefix = stream.readline(4 * 1024 * 1024)
        if not prefix.strip():
            raise ValueError("输入文件为空")
        if not prefix.endswith("\n") and len(prefix) >= 4 * 1024 * 1024:
            first = None  
        else:
            first = json.loads(prefix)
            if not isinstance(first, dict):
                raise ValueError("stage1 JSONL 每行必须是对象")
        if first is not None:
            expected = "responses" in first if args.setting == "reasoning_augmented" else any(k in first for k in ["greedy", "greedy_answer", "samples", "sampled_answers"])
            if not expected:
                raise ValueError("Input must contain stage1 model responses for the selected setting.")
        path, backend_argv, overrides = prepare_backend(entry, args, output.with_suffix(".raw.jsonl"), extra)
        if args.dry_run:
            print(json.dumps({"route": key, "script": entry["script"],
                              "argv": backend_argv, "model_loading": False}, ensure_ascii=False, indent=2))
            return
        if args.family == "deepseek" and args.method == "bsdetector" and not os.environ.get("DEEPSEEK_API_KEY"):
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="uq-", dir=output.parent) as tmp:
            raw = Path(tmp) / "raw.jsonl"
            path, backend_argv, overrides = prepare_backend(entry, args, raw, extra)
            task = Path(tmp) / "task.json"
            task.write_text(json.dumps({"path": str(path), "argv": backend_argv, "overrides": overrides}))
            os.chmod(task, 0o600)
            environment = os.environ.copy()
            environment.update(UQ_FAMILY=args.family, UQ_MODEL_PATH=args.model_path or "",
                               UQ_INPUT_FILE=str(input_path), UQ_OUTPUT_FILE=str(raw))
            subprocess.run([sys.executable, "-m", "uq_runtime.worker", str(task)], cwd=ROOT,
                           env=environment, check=True)
            cleaned = Path(tmp) / "output.jsonl"
            count = 0
            with raw.open() as src, cleaned.open("w") as dst:
                for line in src:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("error"):
                        raise ValueError(f"后端样本失败：{row.get('id')}：{row['error']}")
                    field, score = extract_score(row, args.method)
                    row["uq_run"] = {"method": args.method, "family": args.family, "dataset": args.dataset,
                                     "setting": args.setting, "model_path": args.model_path,
                                     "score_field": field, "score": score, "script": entry["script"],
                                     "implementation": entry["implementation"]}
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
            if not count:
                raise ValueError("后端没有输出有效结果")
            os.replace(cleaned, output)
        print(f"已写入 {count} 条结果：{output}")
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"错误：{exc}\n")
