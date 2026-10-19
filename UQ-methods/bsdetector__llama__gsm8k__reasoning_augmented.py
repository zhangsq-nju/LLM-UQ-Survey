#!/usr/bin/env python3
# Author-related repository: https://github.com/cleanlab/tlm

from __future__ import annotations

if __name__ == "__main__" and not globals().get("_UQ_EXECUTING", False):
    
    import ast as _uq_ast
    from pathlib import Path as _uq_Path
    _uq_tree = _uq_ast.parse(_uq_Path(__file__).read_text())
    _uq_helper = next(n for n in _uq_tree.body if isinstance(n, _uq_ast.FunctionDef) and n.name == "_uq_standalone")
    exec(compile(_uq_ast.Module(body=[_uq_helper], type_ignores=[]), __file__, "exec"), globals())
    _uq_standalone()
    raise SystemExit(0)


import argparse
import json
import math
import os
import sys
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from math_verify import parse, verify


_EPS = 1e-12
print_lock = threading.Lock()
write_lock = threading.Lock()


def safe_print(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    with print_lock:
        print(*args, **kwargs)


def load_records(path: str) -> List[Dict[str, Any]]:
    
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    
    jsonl_ok = True
    tmp: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                tmp.append(obj)
            else:
                jsonl_ok = False
                break
        except Exception:
            jsonl_ok = False
            break
    if jsonl_ok and tmp:
        return tmp

    
    obj = json.loads(text)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]
    raise ValueError(f"Unsupported input JSON structure in {path}")


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with write_lock:
        dirpath = os.path.dirname(os.path.abspath(path))
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    dirpath = os.path.dirname(os.path.abspath(path))
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


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





def mathv_correct(ref_text: str, target_text: str) -> bool:
    try:
        ref = parse(str(ref_text))
        target = parse(str(target_text))
        return bool(verify(ref, target))
    except Exception:
        return False


def mathv_equivalent(a: str, b: str) -> bool:
    return mathv_correct(a, b)





def get_record_id(record: Dict[str, Any], fallback: int) -> str:
    for key in ["id", "question_id", "qid"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return f"line_{fallback}"


def get_question(record: Dict[str, Any]) -> str:
    for key in ["question", "problem", "prompt", "input"]:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def get_correct_answer(record: Dict[str, Any]) -> str:
    for key in ["correct answer", "correct_answer", "answer", "gt", "ground_truth"]:
        if key not in record:
            continue
        val = record.get(key)
        if isinstance(val, dict):
            if val.get("value") is not None:
                return str(val["value"])
            aliases = val.get("aliases")
            if isinstance(aliases, list) and aliases:
                return str(aliases[0])
        if val is not None:
            return str(val)
    return ""


def get_responses(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    responses = record.get("responses", [])
    if isinstance(responses, list):
        return [x for x in responses if isinstance(x, dict)]
    return []


def get_response_answer(resp: Dict[str, Any]) -> str:
    for key in ["llm answer", "llm_answer", "answer", "final_answer", "text", "raw_text", "response"]:
        val = resp.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    
    for key in ["llm reason", "llm_reason", "reason", "reasoning_augmented"]:
        val = resp.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def get_response_reason(resp: Dict[str, Any]) -> str:
    for key in ["llm reason", "llm_reason", "reason", "reasoning_augmented", "rationale"]:
        val = resp.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def get_answer_mean_token_prob(resp: Dict[str, Any]) -> Optional[float]:
    
    answer_info = resp.get("answer info") or resp.get("answer_info") or {}
    if not isinstance(answer_info, dict):
        return None
    prob_info = answer_info.get("prob info") or answer_info.get("prob_info") or {}
    if not isinstance(prob_info, dict):
        return None
    token_logprobs = prob_info.get("token logprob") or prob_info.get("token_logprob")
    if not isinstance(token_logprobs, list) or not token_logprobs:
        return None
    vals = []
    for x in token_logprobs:
        try:
            vals.append(float(x))
        except Exception:
            pass
    if not vals:
        return None
    mean_lp = sum(vals) / len(vals)
    return float(math.exp(max(-100.0, min(0.0, mean_lp))))





def cluster_answers_by_math(answers: List[str]) -> Tuple[List[int], List[str]]:
    reps: List[str] = []
    cluster_ids: List[int] = []
    for ans in answers:
        assigned = False
        for cid, rep in enumerate(reps):
            if mathv_equivalent(ans, rep):
                cluster_ids.append(cid)
                assigned = True
                break
        if not assigned:
            reps.append(ans)
            cluster_ids.append(len(reps) - 1)
    return cluster_ids, reps


def choose_target_response(
    responses: List[Dict[str, Any]],
    target_mode: str,
) -> Tuple[int, Dict[str, Any]]:
    if not responses:
        raise ValueError("record has no responses")

    if target_mode == "first":
        return 0, responses[0]

    answers = [get_response_answer(r) for r in responses]
    cluster_ids, _reps = cluster_answers_by_math(answers)
    counts: Dict[int, int] = {}
    for cid in cluster_ids:
        counts[cid] = counts.get(cid, 0) + 1
    majority_cid = max(counts.keys(), key=lambda c: counts[c])
    for i, cid in enumerate(cluster_ids):
        if cid == majority_cid:
            return i, responses[i]
    return 0, responses[0]


def compute_observed_consistency(
    target_answer: str,
    responses: List[Dict[str, Any]],
    target_index: int,
    max_aux_responses: Optional[int],
) -> Tuple[float, List[Dict[str, Any]]]:
    aux_items = []
    for idx, resp in enumerate(responses):
        if idx == target_index:
            continue
        ans = get_response_answer(resp)
        if ans:
            aux_items.append((idx, ans))

    if max_aux_responses is not None:
        aux_items = aux_items[:max_aux_responses]

    if not aux_items:
        return 0.5, []

    vals: List[float] = []
    details: List[Dict[str, Any]] = []
    for idx, ans in aux_items:
        same = mathv_equivalent(target_answer, ans)
        val = 1.0 if same else 0.0
        vals.append(val)
        details.append(
            {
                "response_index": idx,
                "sample_answer": ans,
                "equivalent_to_target": bool(same),
                "observed_item_o": val,
            }
        )
    return float(sum(vals) / len(vals)), details





class LocalJudge:
    def __init__(
        self,
        model_path: str,
        model_family: str = "auto",
        dtype: str = "auto",
        device: str = "cuda",
        use_p_a_only: bool = False,
    ) -> None:
        self.model_path = model_path
        self.model_family = model_family.lower()
        self.use_p_a_only = use_p_a_only

        torch_dtype = self._resolve_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() and device == "cuda" else None,
            trust_remote_code=True,
        )
        if not (torch.cuda.is_available() and device == "cuda"):
            self.model.to(torch.device("cpu"))
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.a_id = self._get_single_token_id("A")
        self.b_id = self._get_single_token_id("B")
        self.c_id = self._get_single_token_id("C")
        if self.a_id is None or self.b_id is None or self.c_id is None:
            raise ValueError(
                "Cannot map A/B/C to single tokens. Try a different judge model/tokenizer."
            )

    @staticmethod
    def _resolve_dtype(dtype: str):
        dtype = dtype.lower()
        if dtype == "auto":
            if torch.cuda.is_available():
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _get_single_token_id(self, token_str: str) -> Optional[int]:
        candidates = [token_str, " " + token_str, "\n" + token_str]
        for cand in candidates:
            ids = self.tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
        return None

    def build_messages(
        self,
        question: str,
        reasoning: str,
        answer: str,
        round_index: int,
    ) -> List[Dict[str, str]]:
        system = (
            "You are a strict judge for math reasoning answers. "
            "You must decide whether the proposed final answer is correct."
        )
        if question.strip():
            problem_part = f"Problem:\n{question.strip()}\n\n"
        else:
            problem_part = (
                "Problem: [not provided in the input file]\n"
                "Use the reasoning trace to judge whether the final answer is internally supported.\n\n"
            )

        if round_index == 0:
            instruction = (
                "Is the proposed final answer correct?\n"
                "A. Correct\n"
                "B. Incorrect\n"
                "C. Not sure\n\n"
                "Answer with exactly one letter: A, B, or C."
            )
        else:
            instruction = (
                "Check again carefully. Is the proposed final answer correct?\n"
                "A. Correct\n"
                "B. Incorrect\n"
                "C. Not sure\n\n"
                "Answer with exactly one letter: A, B, or C."
            )

        user = (
            problem_part
            + f"Reasoning trace:\n{reasoning.strip()}\n\n"
            + f"Proposed final answer:\n{answer.strip()}\n\n"
            + instruction
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        family = self.model_family
        try:
            if family == "qwen":
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            if family == "llama":
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            if family == "auto":
                
                try:
                    return self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    return self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
        except Exception:
            pass

        
        chunks = []
        for m in messages:
            chunks.append(f"{m['role'].upper()}: {m['content']}")
        chunks.append("ASSISTANT:")
        return "\n\n".join(chunks)

    @torch.no_grad()
    def score_once(
        self,
        question: str,
        reasoning: str,
        answer: str,
        round_index: int,
    ) -> Dict[str, Any]:
        messages = self.build_messages(question, reasoning, answer, round_index)
        prompt = self.format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs).logits[0, -1, :].float()
        abc_logits = torch.tensor(
            [logits[self.a_id].item(), logits[self.b_id].item(), logits[self.c_id].item()],
            dtype=torch.float32,
            device=logits.device,
        )
        probs = F.softmax(abc_logits, dim=-1)
        p_a = float(probs[0].item())
        p_b = float(probs[1].item())
        p_c = float(probs[2].item())
        score = p_a if self.use_p_a_only else p_a + 0.5 * p_c
        return {
            "round_index": round_index,
            "p_a_correct": p_a,
            "p_b_incorrect": p_b,
            "p_c_not_sure": p_c,
            "score": float(score),
            "prompt_len": int(inputs["input_ids"].shape[1]),
        }

    def score_reflections(
        self,
        question: str,
        reasoning: str,
        answer: str,
        num_rounds: int,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        details = [self.score_once(question, reasoning, answer, i) for i in range(num_rounds)]
        scores = [float(x["score"]) for x in details]
        certainty = float(sum(scores) / len(scores)) if scores else 0.5
        return certainty, details





def compute_bsdetector_for_record(
    record: Dict[str, Any],
    record_index: int,
    judge: LocalJudge,
    target_mode: str,
    beta: float,
    num_reflection_rounds: int,
    max_aux_responses: Optional[int],
    keep_responses: bool,
) -> Dict[str, Any]:
    rid = get_record_id(record, record_index)
    question = get_question(record)
    correct_answer = get_correct_answer(record)
    responses = get_responses(record)
    if not responses:
        raise ValueError("no responses found")

    target_index, target_resp = choose_target_response(responses, target_mode)
    target_answer = get_response_answer(target_resp)
    target_reason = get_response_reason(target_resp)

    correct = 1 if mathv_correct(correct_answer, target_answer) else 0

    
    response_correctness: List[Dict[str, Any]] = []
    num_correct_responses = 0
    for i, resp in enumerate(responses):
        ans = get_response_answer(resp)
        c = 1 if mathv_correct(correct_answer, ans) else 0
        num_correct_responses += c
        response_correctness.append({"response_index": i, "answer": ans, "correct": c})

    observed_consistency, observed_details = compute_observed_consistency(
        target_answer=target_answer,
        responses=responses,
        target_index=target_index,
        max_aux_responses=max_aux_responses,
    )

    self_reflection_certainty, reflection_details = judge.score_reflections(
        question=question,
        reasoning=target_reason,
        answer=target_answer,
        num_rounds=num_reflection_rounds,
    )

    beta = max(0.0, min(1.0, float(beta)))
    confidence = float(beta * observed_consistency + (1.0 - beta) * self_reflection_certainty)
    confidence = max(0.0, min(1.0, confidence))
    uncertainty = float(1.0 - confidence)

    result: Dict[str, Any] = {
        "id": rid,
        "correct_answer": correct_answer,
        "target_mode": target_mode,
        "target_response_index": target_index,
        "target_answer": target_answer,
        "correct": int(correct),
        "observed_consistency": observed_consistency,
        "self_reflection_certainty": self_reflection_certainty,
        "bsdetector_confidence": confidence,
        "bsdetector_uncertainty": uncertainty,
        "uncertainty": uncertainty,
        "num_responses": len(responses),
        "num_aux_responses_used": len(observed_details),
        "num_correct_responses": int(num_correct_responses),
        "response_accuracy_among_samples": float(num_correct_responses / len(responses)) if responses else None,
        "observed_details": observed_details,
        "self_reflection_details": reflection_details,
        "response_correctness": response_correctness,
        "meta": {
            "method": "BSDETECTOR-reasoning-local-judge",
            "correctness_checker": "math_verify",
            "correctness_source": "record['correct answer']",
            "auxiliary_samples_source": "record['responses']",
            "self_reflection_source": "local_judge_model_next_token_ABC",
            "beta": beta,
            "num_reflection_rounds": num_reflection_rounds,
            "question_available": bool(question.strip()),
        },
    }
    if question:
        result["question"] = question
    if keep_responses:
        result["responses"] = responses
    return result





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute BSDetector uncertainty for reasoning-task multi-sample records with local judge model."
    )
    parser.add_argument("--input-file", required=True, help="Input reasoning samples JSONL/JSON file")
    parser.add_argument("--output-file", required=True, help="Output JSONL path")
    parser.add_argument("--model-path", required=True, help="Local judge model path, e.g. Qwen/Llama")
    parser.add_argument("--model-family", default="auto", choices=["auto", "qwen", "llama", "plain"])
    parser.add_argument("--hf-endpoint", default=None, help="Optional HF_ENDPOINT, e.g. https://hf-mirror.com")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--target-mode", default="first", choices=["first", "majority"])
    parser.add_argument("--beta", type=float, default=0.5, help="Final confidence = beta*O + (1-beta)*S")
    parser.add_argument("--num-reflection-rounds", type=int, default=2)
    parser.add_argument("--max-aux-responses", type=int, default=None)
    parser.add_argument("--use-p-a-only", action="store_true", help="Self-reflection score = P(A) instead of P(A)+0.5P(C)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-responses", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Correctness source: record['correct answer'] + math_verify")
    safe_print("Auxiliary samples source: existing responses")
    safe_print("Self-reflection source: local judge model")
    safe_print(f"Judge model: {args.model_path}")
    safe_print(f"Judge family: {args.model_family}")
    safe_print(f"Target mode: {args.target_mode}")
    safe_print(f"Beta: {args.beta}")

    out_dir = os.path.dirname(os.path.abspath(args.output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)

    done = load_done_ids(args.output_file)
    if done:
        safe_print(f"existing records: {len(done)}")

    safe_print("Loading local judge model...")
    judge = LocalJudge(
        model_path=args.model_path,
        model_family=args.model_family,
        dtype=args.dtype,
        device=args.device,
        use_p_a_only=args.use_p_a_only,
    )
    safe_print("Judge model loaded.")

    records = load_records(args.input_file)
    safe_print(f"loaded records: {len(records)}")

    processed = 0
    for idx, record in enumerate(records, start=1):
        if args.limit is not None and processed >= args.limit:
            break

        rid = get_record_id(record, idx)
        if str(rid) in done:
            continue

        try:
            result = compute_bsdetector_for_record(
                record=record,
                record_index=idx,
                judge=judge,
                target_mode=args.target_mode,
                beta=args.beta,
                num_reflection_rounds=args.num_reflection_rounds,
                max_aux_responses=args.max_aux_responses,
                keep_responses=args.keep_responses,
            )
            append_jsonl(args.output_file, result)
            processed += 1
            if args.verbose:
                safe_print(
                    f"done: {rid} | line={idx} | correct={result['correct']} | "
                    f"O={result['observed_consistency']:.4f} | "
                    f"S={result['self_reflection_certainty']:.4f} | "
                    f"conf={result['bsdetector_confidence']:.4f} | "
                    f"unc={result['bsdetector_uncertainty']:.4f}"
                )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            err = {
                "id": rid,
                "error": str(e),
            }
            append_jsonl(args.output_file, err)
            safe_print(f"error: {rid} | line={idx} | {e}")

    safe_print(f"Finished. processed={processed}")
    safe_print(f"Saved to: {args.output_file}")


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

    _parser = _ap.ArgumentParser(description="bsdetector / llama / gsm8k / reasoning_augmented (standalone)")
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
                           "dataset": "gsm8k", "setting": "reasoning_augmented", "model_loading": False}, indent=2))
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

