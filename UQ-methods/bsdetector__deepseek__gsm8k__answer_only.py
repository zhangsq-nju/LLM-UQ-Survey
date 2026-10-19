#!/usr/bin/env python3
# Author-related repository: https://github.com/cleanlab/tlm

if __name__ == "__main__" and not globals().get("_UQ_EXECUTING", False):
    
    import ast as _uq_ast
    from pathlib import Path as _uq_Path
    _uq_tree = _uq_ast.parse(_uq_Path(__file__).read_text())
    _uq_helper = next(n for n in _uq_tree.body if isinstance(n, _uq_ast.FunctionDef) and n.name == "_uq_standalone")
    exec(compile(_uq_ast.Module(body=[_uq_helper], type_ignores=[]), __file__, "exec"), globals())
    _uq_standalone()
    raise SystemExit(0)



import json
import os
import re
import time
import threading
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from math_verify import parse, verify






INPUT_FILE = __import__("os").environ.get("UQ_INPUT_FILE", "")
OUTPUT_FILE = __import__("os").environ.get("UQ_OUTPUT_FILE", "")


DEEPSEEK_API_KEY = __import__("os").environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = __import__("os").environ.get("UQ_MODEL_PATH", "")


NLI_MODEL_NAME = "microsoft/deberta-large-mnli"


ALPHA = 0.8
BETA = 0.5


NUM_AUX_SAMPLES = 5


NUM_REFLECTION_ROUNDS = 2


MAX_WORKERS_PER_ITEM = 4   
REQUEST_GAP_SECONDS = 0.05


MAX_RETRIES = 8
RETRY_BASE_SLEEP = 2.0


MAX_NEW_TOKENS_AUX = 128
MAX_NEW_TOKENS_REFLECT = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"





if not DEEPSEEK_API_KEY:
    raise ValueError("请先设置环境变量 DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=BASE_URL,
    timeout=120.0,
    max_retries=0,  
)

print("Loading NLI model...", flush=True)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)


nli_model = AutoModelForSequenceClassification.from_pretrained(
    NLI_MODEL_NAME,
    torch_dtype=torch.float32,
)
nli_model.to(DEVICE)
nli_model.eval()

id2label = {int(k): v.lower() for k, v in nli_model.config.id2label.items()}
CONTRA_IDX = None
for idx, label in id2label.items():
    if "contradiction" in label:
        CONTRA_IDX = idx
        break
if CONTRA_IDX is None:
    raise ValueError(f"Cannot find contradiction label in {id2label}")

write_lock = threading.Lock()
print_lock = threading.Lock()





def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs, flush=True)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()


def load_done_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "id" in obj:
                    done.add(obj["id"])
            except Exception:
                continue
    return done


def normalize_ground_truth(gt_field: Any) -> Dict[str, Any]:
    
    if isinstance(gt_field, str):
        value = gt_field.strip()
        return {"value": value, "aliases": [value]}

    if isinstance(gt_field, dict):
        value = str(gt_field.get("value", "")).strip()
        aliases = [str(x).strip() for x in gt_field.get("aliases", [])]
        if value and value not in aliases:
            aliases = [value] + aliases
        return {"value": value, "aliases": aliases}

    value = str(gt_field).strip()
    return {"value": value, "aliases": [value]}


def normalize_number_string(text: str) -> str:
    text = text.strip().replace(",", "")
    try:
        val = float(text)
        if val.is_integer():
            return str(int(val))
        return str(val)
    except Exception:
        return text


def extract_numeric_answer(text: str) -> str:
    
    text = text.strip()

    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return normalize_number_string(m.group(1))

    text_no_comma = text.replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text_no_comma)
    if nums:
        return normalize_number_string(nums[0])

    return text.strip()


def get_question_id(obj: Dict[str, Any]) -> str:
    if "id" in obj:
        return str(obj["id"])
    if "question_id" in obj:
        return str(obj["question_id"])
    return ""


def get_gt(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "gt" in obj:
        return normalize_ground_truth(obj["gt"])
    if "ground_truth" in obj:
        return normalize_ground_truth(obj["ground_truth"])
    return {"value": "", "aliases": [""]}


def get_greedy_answer(obj: Dict[str, Any]) -> str:
    if "greedy_answer" in obj:
        return str(obj["greedy_answer"]).strip()

    if "greedy" in obj and isinstance(obj["greedy"], dict):
        return str(obj["greedy"].get("text", "")).strip()

    return ""





def mathv_correct(ref_text: str, target_text: str) -> bool:
    try:
        ref = parse(ref_text)
        target = parse(target_text)
        return bool(verify(ref, target))
    except Exception:
        return False


def is_correct(answer: str, gt: Dict[str, Any]) -> int:
    
    ref = gt.get("value", "")
    if ref and mathv_correct(ref, answer):
        return 1

    
    for alias in gt.get("aliases", []):
        if mathv_correct(alias, answer):
            return 1

    return 0





def request_with_retry(kwargs: Dict[str, Any]):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if REQUEST_GAP_SECONDS > 0:
                time.sleep(REQUEST_GAP_SECONDS)
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_err = e
            sleep_time = RETRY_BASE_SLEEP * attempt
            safe_print(f"[retry {attempt}/{MAX_RETRIES}] {type(e).__name__}: {e}")
            time.sleep(sleep_time)
    raise last_err


def call_deepseek(
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    resp = request_with_retry(
        dict(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False,
        )
    )
    return (resp.choices[0].message.content or "").strip()





def build_aux_sample_messages(question: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Please strictly use the following template to provide answer:\n"
                "explanation: [insert step-by-step analysis]\n"
                "answer: [provide your answer]"
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}",
        },
    ]


def generate_one_aux_sample(question: str, sample_index: int) -> Dict[str, Any]:
    messages = build_aux_sample_messages(question)
    raw_text = call_deepseek(
        messages=messages,
        temperature=1.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS_AUX,
    )
    answer = extract_numeric_answer(raw_text)
    return {
        "sample_index": sample_index,
        "messages": messages,
        "raw_text": raw_text,
        "answer": answer,
    }





CHOICE2SCORE = {
    "A": 1.0,
    "B": 0.0,
    "C": 0.5,
}


def parse_reflection_choice(text: str) -> Optional[str]:
    m = re.search(r"answer\s*:\s*([ABC])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([ABC])\b", text.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def build_reflection_messages(question: str, proposed_answer: str, round_index: int) -> List[Dict[str, str]]:
    if round_index == 0:
        user_content = (
            f"Question: {question}\n"
            f"Proposed Answer: {proposed_answer}\n"
            "Is the proposed answer: (A) Correct (B) Incorrect (C) I am not sure.\n"
            "The output should strictly use the following template:\n"
            "explanation: [insert analysis]\n"
            "answer: [choose one letter from among choices A through C]"
        )
    else:
        user_content = (
            f"Question: {question}\n"
            f"Proposed Answer: {proposed_answer}\n"
            "Are you really sure the proposed answer is correct? Choose again: "
            "(A) Correct (B) Incorrect (C) I am not sure.\n"
            "The output should strictly use the following template:\n"
            "explanation: [insert analysis]\n"
            "answer: [choose one letter from among choices A through C]"
        )

    return [
        {
            "role": "system",
            "content": (
                "You are evaluating whether a proposed numeric answer to a math word problem is correct. "
                "Follow the requested output format exactly."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def run_one_reflection(question: str, proposed_answer: str, round_index: int) -> Dict[str, Any]:
    messages = build_reflection_messages(question, proposed_answer, round_index)
    raw_text = call_deepseek(
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS_REFLECT,
    )
    choice = parse_reflection_choice(raw_text)
    score = CHOICE2SCORE.get(choice, 0.5)
    return {
        "round_index": round_index,
        "messages": messages,
        "raw_text": raw_text,
        "choice": choice,
        "score": score,
    }





def contradiction_prob(premise: str, hypothesis: str) -> float:
    inputs = nli_tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(nli_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = nli_model(**inputs).logits[0].float()
        probs = F.softmax(logits, dim=-1)

    return float(probs[CONTRA_IDX].item())


def bidirectional_similarity(a: str, b: str) -> float:
    p1 = contradiction_prob(a, b)
    p2 = contradiction_prob(b, a)
    return 0.5 * ((1.0 - p1) + (1.0 - p2))





def exact_match_indicator(a: str, b: str) -> int:
    return int(normalize_number_string(extract_numeric_answer(a)) == normalize_number_string(extract_numeric_answer(b)))


def compute_observed_consistency(greedy_answer: str, aux_answers: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
    if not aux_answers:
        return 0.0, []

    details = []
    vals = []

    greedy_norm = extract_numeric_answer(greedy_answer)

    for y_i in aux_answers:
        y_i_norm = extract_numeric_answer(y_i)
        s_i = bidirectional_similarity(y_i_norm, greedy_norm)
        r_i = exact_match_indicator(greedy_norm, y_i_norm)
        o_i = ALPHA * s_i + (1.0 - ALPHA) * r_i

        details.append(
            {
                "sample_answer": y_i_norm,
                "similarity_s": s_i,
                "indicator_r": r_i,
                "observed_item_o": o_i,
            }
        )
        vals.append(o_i)

    O = float(sum(vals) / len(vals))
    return O, details


def compute_bsdetector_for_item(obj: Dict[str, Any]) -> Dict[str, Any]:
    qid = get_question_id(obj)
    question = str(obj["question"])
    gt = get_gt(obj)

    
    greedy_answer_raw = get_greedy_answer(obj)
    correct = is_correct(greedy_answer_raw, gt)

    aux_results: List[Optional[Dict[str, Any]]] = [None] * NUM_AUX_SAMPLES
    reflection_results: List[Optional[Dict[str, Any]]] = [None] * NUM_REFLECTION_ROUNDS

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PER_ITEM) as pool:
        future_to_info = {}

        for i in range(NUM_AUX_SAMPLES):
            fut = pool.submit(generate_one_aux_sample, question, i)
            future_to_info[fut] = ("aux", i)

        for i in range(NUM_REFLECTION_ROUNDS):
            fut = pool.submit(run_one_reflection, question, greedy_answer_raw, i)
            future_to_info[fut] = ("ref", i)

        for fut in as_completed(future_to_info):
            kind, idx = future_to_info[fut]
            result = fut.result()
            if kind == "aux":
                aux_results[idx] = result
            else:
                reflection_results[idx] = result

    aux_results = [x for x in aux_results if x is not None]
    reflection_results = [x for x in reflection_results if x is not None]

    aux_answers = [x["answer"] for x in aux_results]

    O, observed_details = compute_observed_consistency(greedy_answer_raw, aux_answers)

    reflection_scores = [x["score"] for x in reflection_results]
    S = float(sum(reflection_scores) / len(reflection_scores)) if reflection_scores else 0.5

    confidence = float(BETA * O + (1.0 - BETA) * S)
    uncertainty = float(1.0 - confidence)

    return {
        "id": qid,
        "question": question,
        "gt": gt,
        "greedy_answer": greedy_answer_raw,
        "correct": correct,
        "observed_consistency": O,
        "self_reflection_certainty": S,
        "bsdetector_confidence": confidence,
        "bsdetector_uncertainty": uncertainty,
        "auxiliary_samples": aux_results,
        "observed_details": observed_details,
        "self_reflection_details": reflection_results,
        "meta": {
            "method": "BSDETECTOR",
            "dataset": "gsm8k",
            "llm_model": MODEL_NAME,
            "nli_model": NLI_MODEL_NAME,
            "alpha": ALPHA,
            "beta": BETA,
            "num_aux_samples": NUM_AUX_SAMPLES,
            "num_reflection_rounds": NUM_REFLECTION_ROUNDS,
            "correctness_checker": "math_verify",
        },
    }





def main():
    done = load_done_ids(OUTPUT_FILE)
    safe_print(f"existing records: {len(done)}")

    with open(INPUT_FILE, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin, start=1):
            obj = json.loads(line)
            qid = get_question_id(obj)

            if qid in done:
                continue

            try:
                result = compute_bsdetector_for_item(obj)
                append_jsonl(OUTPUT_FILE, result)
                safe_print(
                    f"done: {qid} | line={line_idx} | "
                    f"correct={result['correct']} | "
                    f"conf={result['bsdetector_confidence']:.4f} | "
                    f"unc={result['bsdetector_uncertainty']:.4f}"
                )
            except Exception as e:
                err = {
                    "id": qid,
                    "question": obj.get("question", ""),
                    "error": str(e),
                }
                append_jsonl(OUTPUT_FILE, err)
                safe_print(f"error: {qid} | line={line_idx} | {e}")


if __name__ == "__main__":
    main()

def _uq_standalone():
    
    import argparse as _ap
    import ast as _ast
    import json as _json
    import os as _os
    from pathlib import Path as _Path
    import sys as _sys
    import tempfile as _tempfile
    import types as _types

    _parser = _ap.ArgumentParser(description="bsdetector / deepseek / gsm8k / answer_only (standalone)")
    _parser.add_argument("--input-file", required=True)
    _parser.add_argument("--output-file", required=True)
    _parser.add_argument("--model-path")
    _parser.add_argument("--correct-file")
    _parser.add_argument("--problem-file")
    _parser.add_argument("--nli-model")
    _parser.add_argument("--device", choices=["cpu", "cuda"])
    _parser.add_argument("--overwrite", action="store_true")
    _parser.add_argument("--dry-run", action="store_true")
    _argsv = _sys.argv[1:]
    _cut = _argsv.index("--") if "--" in _argsv else len(_argsv)
    _args = _parser.parse_args(_argsv[:_cut])
    _extra = _argsv[_cut + 1:]
    _source = _Path(__file__).resolve()
    _input = _Path(_args.input_file).resolve()
    _output = _Path(_args.output_file).resolve()
    if not _input.is_file():
        _parser.error("input file does not exist")
    if _input == _output:
        _parser.error("output must differ from input")
    if _output.exists() and not _args.overwrite:
        _parser.error("output exists; use --overwrite")
    for _name in ["correct_file", "problem_file", "model_path", "nli_model"]:
        _value = getattr(_args, _name)
        if _value and _Path(_value).exists():
            setattr(_args, _name, str(_Path(_value).resolve()))
    if _args.correct_file and _output == _Path(_args.correct_file).resolve():
        _parser.error("output must differ from correctness input")
    if _args.problem_file and _output == _Path(_args.problem_file).resolve():
        _parser.error("output must differ from problem input")
    _tree = _ast.parse(_source.read_text(), filename=str(_source))
    _tree.body = [n for n in _tree.body
                  if not (isinstance(n, _ast.FunctionDef) and n.name == "_uq_standalone")
                  and not (isinstance(n, _ast.If) and "_UQ_EXECUTING" in _ast.unparse(n.test))]
    _options = {}
    for _node in _ast.walk(_tree):
        if isinstance(_node, _ast.Call) and isinstance(_node.func, _ast.Attribute) and _node.func.attr == "add_argument":
            for _opt in _node.args:
                if isinstance(_opt, _ast.Constant) and isinstance(_opt.value, str) and _opt.value.startswith("--"):
                    _options[_opt.value] = _node
    _argv = [str(_source)]
    def _add(_names, _value):
        if _value is not None:
            _opt = next((x for x in _names if x in _options), None)
            if _opt:
                _argv.extend([_opt, str(_value)])
    _add(["--input-file", "--input_file"], str(_input))
    if "bsdetector" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
        _add(["--model-path", "--model_name_or_path", "--judge-model-path"], _args.model_path)
    _add(["--api-model"], _args.model_path)
    _add(["--model-family", "--judge-family"], "deepseek")
    _add(["--correct-file", "--correct_file"], _args.correct_file)
    _add(["--problem-file"], _args.problem_file)
    _add(["--nli-model", "--nli-model-path"], _args.nli_model)
    _add(["--device"], _args.device)
    _protected = {"--input-file", "--input_file", "--output-file", "--output_file", "--model-path",
                  "--model_name_or_path", "--judge-model-path", "--model-family", "--judge-family", "--api-model"}
    if any(x.split("=")[0] in _protected for x in _extra):
        _parser.error("use the main CLI options for input/output/model/family")
    _argv.extend(_extra)
    
    _provided = {x.split("=")[0] for x in _argv if x.startswith("--")}
    _provided.update({"--output-file", "--output_file"})
    for _opt, _node in _options.items():
        _required = any(k.arg == "required" and isinstance(k.value, _ast.Constant) and k.value.value for k in _node.keywords)
        _aliases = [x.value for x in _node.args if isinstance(x, _ast.Constant)]
        if _required and not _provided.intersection(_aliases):
            _parser.error("missing backend argument: " + _opt)
    if "bsdetector" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "bsdetector", "family": "deepseek",
                           "dataset": "gsm8k", "setting": "answer_only", "model_loading": False}, indent=2))
        return
    _output.parent.mkdir(parents=True, exist_ok=True)
    with _tempfile.TemporaryDirectory(prefix="uq-", dir=_output.parent) as _tmp:
        _raw = _Path(_tmp) / "result.jsonl"
        _add(["--output-file", "--output_file"], str(_raw))
        _overrides = {"INPUT_FILE": str(_input), "OUTPUT_FILE": str(_raw),
                      "MODEL_NAME": _args.model_path, "JUDGE_MODEL_NAME": _args.model_path,
                      "API_MODEL": _args.model_path, "MODEL_deepseek": "deepseek"}
        if _args.nli_model:
            _overrides.update(NLI_MODEL_NAME=_args.nli_model, NLI_MODEL=_args.nli_model)
        if _args.device:
            _overrides["DEVICE"] = _args.device
        for _node in _tree.body:
            if isinstance(_node, (_ast.Assign, _ast.AnnAssign)):
                _targets = _node.targets if isinstance(_node, _ast.Assign) else [_node.target]
                for _target in _targets:
                    if isinstance(_target, _ast.Name) and _overrides.get(_target.id) is not None:
                        _node.value = _ast.Constant(_overrides[_target.id])
        _ast.fix_missing_locations(_tree)
        _os.environ.update(UQ_deepseek="deepseek", UQ_MODEL_PATH=_args.model_path or "",
                           UQ_INPUT_FILE=str(_input), UQ_OUTPUT_FILE=str(_raw))
        _module = _types.ModuleType("__main__")
        _module.__file__ = str(_source)
        _sys.modules["__main__"] = _module
        _sys.argv = _argv
        exec(compile(_tree, str(_source), "exec"), _module.__dict__)
        _count = 0
        with _raw.open() as _stream:
            for _line in _stream:
                if _line.strip():
                    _row = _json.loads(_line)
                    if _row.get("error"):
                        raise RuntimeError(str(_row["error"]))
                    _count += 1
        if not _count:
            raise RuntimeError("backend produced no records")
        _os.replace(_raw, _output)
        print(f"Wrote {_count} records to {_output}")

