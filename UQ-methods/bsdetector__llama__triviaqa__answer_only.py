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



import os
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

import inspect
import json
import re
import string
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)






INPUT_FILE = __import__("os").environ.get("UQ_INPUT_FILE", "")
OUTPUT_FILE = __import__("os").environ.get("UQ_OUTPUT_FILE", "")


MODEL_NAME = __import__("os").environ.get("UQ_MODEL_PATH", "")
NLI_MODEL_NAME = "microsoft/deberta-large-mnli"

ALPHA = 0.8
BETA = 0.5
NUM_AUX_SAMPLES = 5
NUM_REFLECTION_ROUNDS = 2

MAX_NEW_TOKENS_AUX = 24
MAX_NEW_TOKENS_REFLECT = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DTYPE = torch.float32





print(f"Loading judge model from: {MODEL_NAME}", flush=True)
judge_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

judge_tokenizer.padding_side = "left"
if judge_tokenizer.pad_token_id is None:
    judge_tokenizer.pad_token_id = judge_tokenizer.eos_token_id
if judge_tokenizer.pad_token is None and judge_tokenizer.eos_token is not None:
    judge_tokenizer.pad_token = judge_tokenizer.eos_token

judge_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)

if not torch.cuda.is_available():
    judge_model.to(DEVICE)

judge_model.eval()
judge_model.generation_config.use_cache = True
judge_model.generation_config.pad_token_id = judge_tokenizer.pad_token_id
if judge_tokenizer.eos_token_id is not None:
    judge_model.generation_config.eos_token_id = judge_tokenizer.eos_token_id

print(f"Loading NLI model from: {NLI_MODEL_NAME}", flush=True)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)

nli_model = AutoModelForSequenceClassification.from_pretrained(
    NLI_MODEL_NAME,
    torch_dtype=torch.float32,
    use_safetensors=False,
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
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_done_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "id" in obj:
                    done.add(str(obj["id"]))
            except Exception:
                continue
    return done


def get_question_id(obj: Dict[str, Any]) -> str:
    if "id" in obj:
        return str(obj["id"])
    if "question_id" in obj:
        return str(obj["question_id"])
    return ""


def get_gt(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "gt" in obj:
        gt = obj["gt"]
    elif "ground_truth" in obj:
        gt = obj["ground_truth"]
    elif "answer" in obj:
        gt = obj["answer"]
    else:
        gt = {"value": "", "aliases": [""]}

    if isinstance(gt, str):
        value = gt.strip()
        return {"value": value, "aliases": [value]}

    if isinstance(gt, dict):
        value = str(gt.get("value", "")).strip()
        aliases = [str(x).strip() for x in gt.get("aliases", []) if str(x).strip()]
        if value and value not in aliases:
            aliases = [value] + aliases
        return {"value": value, "aliases": aliases}

    value = str(gt).strip()
    return {"value": value, "aliases": [value]}


def get_greedy_answer(obj: Dict[str, Any]) -> str:
    if "greedy_answer" in obj:
        return str(obj["greedy_answer"]).strip()
    if "greedy" in obj and isinstance(obj["greedy"], dict):
        return str(obj["greedy"].get("text", "")).strip()
    if "prediction" in obj:
        return str(obj["prediction"]).strip()
    return ""


def strip_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ['"', "'", "“", "”"]:
        return text[1:-1].strip()
    return text


def extract_trivia_answer(text: str) -> str:
    text = str(text).strip()

    patterns = [
        r"(?i)answer\s*:\s*(.+)$",
        r"(?i)final answer\s*:\s*(.+)$",
        r"(?i)the answer is\s+(.+)$",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.MULTILINE)
        if m:
            cand = m.group(1).strip()
            cand = cand.split("\n")[0].strip()
            return strip_quotes(cand).rstrip(". ")

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines:
        return strip_quotes(lines[-1]).rstrip(". ")

    return strip_quotes(text).rstrip(". ")


def normalize_answer(text: str) -> str:
    text = str(text).lower().strip()
    text = strip_quotes(text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def alias_match(pred: str, gt: Dict[str, Any]) -> int:
    pred_norm = normalize_answer(extract_trivia_answer(pred))
    if not pred_norm:
        return 0

    candidates = []
    if gt.get("value"):
        candidates.append(gt["value"])
    candidates.extend(gt.get("aliases", []))

    seen = set()
    for cand in candidates:
        cand_norm = normalize_answer(cand)
        if not cand_norm or cand_norm in seen:
            continue
        seen.add(cand_norm)
        if pred_norm == cand_norm:
            return 1
        if pred_norm in cand_norm or cand_norm in pred_norm:
            return 1
    return 0





def build_fallback_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user").strip().lower()
        content = m.get("content", "")
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
        else:
            parts.append(f"<|user|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def apply_chat_template_compat(messages: List[Dict[str, str]]) -> str:
    if hasattr(judge_tokenizer, "apply_chat_template") and judge_tokenizer.chat_template:
        kwargs = {
            "conversation": messages,
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            sig = inspect.signature(judge_tokenizer.apply_chat_template)
            if "enable_thinking" in sig.parameters:
                kwargs["enable_thinking"] = False
            return judge_tokenizer.apply_chat_template(**kwargs)
        except Exception:
            pass
    return build_fallback_prompt(messages)





def call_llama_batch(
    messages_list: List[List[Dict[str, str]]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    num_return_sequences: int = 1,
) -> List[str]:
    prompts = [apply_chat_template_compat(messages) for messages in messages_list]

    inputs = judge_tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(judge_model.device)

    input_lengths = inputs["attention_mask"].sum(dim=1)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=judge_tokenizer.pad_token_id,
        eos_token_id=judge_tokenizer.eos_token_id,
        use_cache=True,
        num_return_sequences=num_return_sequences,
    )

    if do_sample:
        gen_kwargs.update(
            dict(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        )
    else:
        gen_kwargs.update(dict(do_sample=False))

    with torch.inference_mode():
        outputs = judge_model.generate(**gen_kwargs)

    texts: List[str] = []
    for out_idx in range(outputs.shape[0]):
        src_idx = out_idx // num_return_sequences
        prompt_len = int(input_lengths[src_idx].item())
        gen_ids = outputs[out_idx, prompt_len:]
        raw_text = judge_tokenizer.decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        texts.append(raw_text)

    return texts





def build_aux_sample_messages(question: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You answer factual trivia questions. "
                "Respond briefly with the final answer only as a short phrase."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\nAnswer:",
        },
    ]


def generate_aux_samples_batch(question: str, num_samples: int) -> List[Dict[str, Any]]:
    messages = build_aux_sample_messages(question)
    raw_texts = call_llama_batch(
        messages_list=[messages],
        max_new_tokens=MAX_NEW_TOKENS_AUX,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        num_return_sequences=num_samples,
    )

    results = []
    for i, raw_text in enumerate(raw_texts):
        results.append(
            {
                "sample_index": i,
                "messages": messages,
                "raw_text": raw_text,
                "answer": extract_trivia_answer(raw_text),
            }
        )
    return results





CHOICE2SCORE = {
    "A": 1.0,
    "B": 0.0,
    "C": 0.5,
}


def parse_reflection_choice(text: str) -> Optional[str]:
    m = re.search(r"answer\s*:\s*([ABC])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([ABC])\b", str(text).strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def build_reflection_messages(question: str, proposed_answer: str, round_index: int) -> List[Dict[str, str]]:
    if round_index == 0:
        user_content = (
            f"Question: {question}\n"
            f"Proposed Answer: {proposed_answer}\n"
            "Is the proposed answer: (A) Correct (B) Incorrect (C) I am not sure.\n"
            "Use the following format exactly:\n"
            "explanation: [brief analysis]\n"
            "answer: [A or B or C]"
        )
    else:
        user_content = (
            f"Question: {question}\n"
            f"Proposed Answer: {proposed_answer}\n"
            "Are you really sure the proposed answer is correct? Choose again: "
            "(A) Correct (B) Incorrect (C) I am not sure.\n"
            "Use the following format exactly:\n"
            "explanation: [brief analysis]\n"
            "answer: [A or B or C]"
        )

    return [
        {
            "role": "system",
            "content": (
                "You are evaluating whether a short answer to a trivia question is correct. "
                "Follow the requested output format exactly."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def run_reflections_batch(question: str, proposed_answer: str) -> List[Dict[str, Any]]:
    messages_list = [
        build_reflection_messages(question, proposed_answer, i)
        for i in range(NUM_REFLECTION_ROUNDS)
    ]

    raw_texts = call_llama_batch(
        messages_list=messages_list,
        max_new_tokens=MAX_NEW_TOKENS_REFLECT,
        do_sample=False,
        num_return_sequences=1,
    )

    results = []
    for i, raw_text in enumerate(raw_texts):
        choice = parse_reflection_choice(raw_text)
        score = CHOICE2SCORE.get(choice, 0.5)
        results.append(
            {
                "round_index": i,
                "messages": messages_list[i],
                "raw_text": raw_text,
                "choice": choice,
                "score": score,
            }
        )
    return results





def contradiction_prob_batch(pairs: List[Tuple[str, str]]) -> List[float]:
    if not pairs:
        return []

    premises = [p for p, _ in pairs]
    hypotheses = [h for _, h in pairs]

    inputs = nli_tokenizer(
        premises,
        hypotheses,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(nli_model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = nli_model(**inputs).logits.float()
        probs = F.softmax(logits, dim=-1)

    return probs[:, CONTRA_IDX].tolist()





def compute_observed_consistency(greedy_answer: str, aux_answers: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
    if not aux_answers:
        return 0.0, []

    greedy_norm = extract_trivia_answer(greedy_answer)
    aux_norms = [extract_trivia_answer(x) for x in aux_answers]

    forward_pairs = [(y, greedy_norm) for y in aux_norms]
    backward_pairs = [(greedy_norm, y) for y in aux_norms]

    p1s = contradiction_prob_batch(forward_pairs)
    p2s = contradiction_prob_batch(backward_pairs)

    details = []
    vals = []

    greedy_em = normalize_answer(greedy_norm)
    for y_i_norm, p1, p2 in zip(aux_norms, p1s, p2s):
        s_i = 0.5 * ((1.0 - p1) + (1.0 - p2))
        r_i = int(normalize_answer(y_i_norm) == greedy_em)
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
    correct = alias_match(greedy_answer_raw, gt)

    aux_results = generate_aux_samples_batch(question, NUM_AUX_SAMPLES)
    reflection_results = run_reflections_batch(question, greedy_answer_raw)

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
            "dataset": "triviaqa",
            "llm_model": MODEL_NAME,
            "nli_model": NLI_MODEL_NAME,
            "alpha": ALPHA,
            "beta": BETA,
            "num_aux_samples": NUM_AUX_SAMPLES,
            "num_reflection_rounds": NUM_REFLECTION_ROUNDS,
            "correctness_checker": "normalized_alias_match",
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

    _parser = _ap.ArgumentParser(description="bsdetector / llama / triviaqa / answer_only (standalone)")
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
    _add(["--model-family", "--judge-family"], "llama")
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
        print(_json.dumps({"script": str(_source), "method": "bsdetector", "family": "llama",
                           "dataset": "triviaqa", "setting": "answer_only", "model_loading": False}, indent=2))
        return
    _output.parent.mkdir(parents=True, exist_ok=True)
    with _tempfile.TemporaryDirectory(prefix="uq-", dir=_output.parent) as _tmp:
        _raw = _Path(_tmp) / "result.jsonl"
        _add(["--output-file", "--output_file"], str(_raw))
        _overrides = {"INPUT_FILE": str(_input), "OUTPUT_FILE": str(_raw),
                      "MODEL_NAME": _args.model_path, "JUDGE_MODEL_NAME": _args.model_path,
                      "API_MODEL": _args.model_path, "MODEL_llama": "llama"}
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
        _os.environ.update(UQ_llama="llama", UQ_MODEL_PATH=_args.model_path or "",
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

