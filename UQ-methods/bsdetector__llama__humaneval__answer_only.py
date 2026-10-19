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
import ast
import json
import math
import os
import re
import sys
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer





write_lock = threading.Lock()
print_lock = threading.Lock()


def safe_print(*args: Any, **kwargs: Any) -> None:
    with print_lock:
        print(*args, **kwargs, flush=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as e:
                print(f"[warn] skip bad json line {line_no}: {e}", file=sys.stderr)
    return records


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


def load_done_ids(path: str) -> Set[str]:
    done: Set[str] = set()
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





def resolve_record_id(record: Dict[str, Any]) -> Optional[str]:
    for key in ("id", "question_id", "task_id"):
        if record.get(key) is not None:
            return str(record[key])
    return None


def build_correct_map(records: Sequence[Dict[str, Any]], question_id_key: str, correct_key: str) -> Dict[str, int]:
    
    out: Dict[str, int] = {}
    for rec in records:
        qid = rec.get(question_id_key)
        if qid is None and question_id_key != "id":
            qid = rec.get("id")
        if qid is None:
            qid = rec.get("task_id")
        if qid is None:
            continue
        try:
            correct = int(rec.get(correct_key, 0))
        except Exception:
            correct = 0
        out[str(qid)] = 1 if correct else 0
    return out


def get_nested_text(obj: Dict[str, Any], path: Sequence[str]) -> Optional[str]:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if cur is None:
        return None
    return str(cur)


def get_problem_text(obj: Dict[str, Any]) -> str:
    
    candidates: List[Optional[str]] = [
        get_nested_text(obj, ["question"]),
        get_nested_text(obj, ["prompt"]),
        get_nested_text(obj, ["problem"]),
        get_nested_text(obj, ["gt", "prompt"]),
        get_nested_text(obj, ["gt", "question"]),
    ]
    for x in candidates:
        if x and x.strip():
            return x.strip()
    return ""


def get_entry_point(obj: Dict[str, Any]) -> str:
    candidates: List[Optional[str]] = [
        get_nested_text(obj, ["entry_point"]),
        get_nested_text(obj, ["gt", "entry_point"]),
        get_nested_text(obj, ["metadata", "entry_point"]),
    ]
    for x in candidates:
        if x and x.strip():
            return x.strip()
    return ""


def get_test_text(obj: Dict[str, Any]) -> str:
    candidates: List[Optional[str]] = [
        get_nested_text(obj, ["test"]),
        get_nested_text(obj, ["gt", "test"]),
        get_nested_text(obj, ["metadata", "test"]),
    ]
    for x in candidates:
        if x and x.strip():
            return x.strip()
    return ""


def get_greedy_prediction(obj: Dict[str, Any]) -> str:
    candidates: List[Optional[str]] = [
        get_nested_text(obj, ["greedy", "text"]),
        get_nested_text(obj, ["greedy_answer"]),
        get_nested_text(obj, ["prediction"]),
        get_nested_text(obj, ["completion"]),
        get_nested_text(obj, ["text"]),
        get_nested_text(obj, ["raw_text"]),
    ]
    for x in candidates:
        if x is not None:
            return x.strip()
    return ""





def strip_code_fences(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""

    
    fenced = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        
        return max((x.strip() for x in fenced), key=len)

    
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("code", "answer", "completion"):
                if key in obj and isinstance(obj[key], str):
                    return obj[key].strip()
    except Exception:
        pass

    
    m = re.search(r"(?is)(?:^|\n)\s*(?:answer|code|solution)\s*:\s*(.*)$", text)
    if m:
        text = m.group(1).strip()

    return text.strip()


def remove_after_tests_or_main(code: str) -> str:
    
    cut_patterns = [
        r"\n\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:",
        r"\n\s*#\s*test",
        r"\n\s*#\s*example",
        r"\n\s*assert\s+",
        r"\n\s*print\s*\(",
    ]
    cut_idx = len(code)
    for pat in cut_patterns:
        m = re.search(pat, code, flags=re.IGNORECASE)
        if m:
            cut_idx = min(cut_idx, m.start())
    return code[:cut_idx].rstrip()


def extract_code_answer(text: str) -> str:
    code = strip_code_fences(text)
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = remove_after_tests_or_main(code)
    return code.strip("\n")


def strip_python_comments_and_blank_lines(code: str) -> str:
    lines: List[str] = []
    for line in code.splitlines():
        raw = line.rstrip()
        stripped = raw.strip()
        if not stripped:
            continue
        
        if stripped.startswith("#"):
            continue
        lines.append(raw)
    return "\n".join(lines).strip()


def try_ast_dump(code: str) -> Optional[str]:
    candidates = [code]
    
    candidates.append("def __candidate__():\n" + "\n".join("    " + ln for ln in code.splitlines()))
    for cand in candidates:
        try:
            tree = ast.parse(cand)
            return ast.dump(tree, annotate_fields=False, include_attributes=False)
        except Exception:
            continue
    return None


def normalize_code_for_match(code: str) -> str:
    code = extract_code_answer(code)
    code = strip_python_comments_and_blank_lines(code)
    ast_norm = try_ast_dump(code)
    if ast_norm is not None:
        return ast_norm
    
    return re.sub(r"\s+", " ", code).strip()


def exact_code_indicator(a: str, b: str) -> int:
    return int(normalize_code_for_match(a) == normalize_code_for_match(b))


def code_for_nli(code: str, max_chars: int = 3000) -> str:
    code = extract_code_answer(code)
    code = strip_python_comments_and_blank_lines(code)
    if len(code) > max_chars:
        code = code[:max_chars] + "\n# [truncated]"
    return "Python solution:\n" + code





def resolve_torch_dtype(dtype_name: str) -> Optional[torch.dtype]:
    name = dtype_name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


class LocalCausalLM:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "auto",
        device_map: str = "auto",
        disable_thinking: bool = True,
    ) -> None:
        self.model_path = model_path
        self.disable_thinking = disable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        torch_dtype = resolve_torch_dtype(dtype)
        use_device_map = device_map if torch.cuda.is_available() and device != "cpu" else None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=use_device_map,
            trust_remote_code=True,
        )
        if use_device_map is None:
            target_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
            self.model.to(target_device)
        self.model.eval()
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.use_cache = True

    @property
    def device(self) -> torch.device:
        return self.model.device

    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        
        if self.disable_thinking:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                pass
            except Exception:
                pass
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            chunks: List[str] = []
            for m in messages:
                chunks.append(f"{m.get('role', 'user').upper()}: {m.get('content', '')}")
            chunks.append("ASSISTANT:")
            return "\n\n".join(chunks)

    def generate_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        num_return_sequences: int = 1,
    ) -> List[str]:
        prompts = [self.format_chat_prompt(messages) for messages in messages_list]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        prompt_len = inputs["input_ids"].shape[1]
        gen_kwargs: Dict[str, Any] = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
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
            outputs = self.model.generate(**gen_kwargs)

        texts: List[str] = []
        for out in outputs:
            gen_ids = out[prompt_len:]
            raw_text = self.tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            texts.append(raw_text)
        return texts





class NLIModel:
    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 512) -> None:
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        target_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model.to(target_device)
        self.model.eval()

        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.contra_idx: Optional[int] = None
        for idx, label in id2label.items():
            if "contradiction" in label:
                self.contra_idx = idx
                break
        if self.contra_idx is None:
            raise ValueError(f"Cannot find contradiction label in {id2label}")

    @property
    def device(self) -> torch.device:
        return self.model.device

    def contradiction_prob_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        inputs = self.tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits.float()
            probs = F.softmax(logits, dim=-1)
        return probs[:, int(self.contra_idx)].tolist()





def build_aux_sample_messages(problem_text: str, entry_point: str = "") -> List[Dict[str, str]]:
    entry_hint = f"\nThe target entry point is `{entry_point}`." if entry_point else ""
    return [
        {
            "role": "system",
            "content": (
                "You are solving a HumanEval Python programming task. "
                "Return only the Python solution code. Do not include Markdown fences, explanations, tests, or comments."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Complete the following programming problem.{entry_hint}\n\n"
                f"Problem:\n{problem_text}\n\n"
                "Output only the final Python code."
            ),
        },
    ]


def generate_aux_samples_batch(
    llm: LocalCausalLM,
    problem_text: str,
    entry_point: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    messages = build_aux_sample_messages(problem_text, entry_point)
    raw_texts = llm.generate_batch(
        messages_list=[messages],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_samples,
    )
    results: List[Dict[str, Any]] = []
    for i, raw_text in enumerate(raw_texts):
        code = extract_code_answer(raw_text)
        results.append(
            {
                "sample_index": i,
                "messages": messages,
                "raw_text": raw_text,
                "code": code,
            }
        )
    return results


CHOICE2SCORE = {"A": 1.0, "B": 0.0, "C": 0.5}


def parse_reflection_choice(text: str) -> Optional[str]:
    text = str(text)
    m = re.search(r"answer\s*:\s*([ABC])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([ABC])\b", text.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def build_reflection_messages(problem_text: str, proposed_code: str, entry_point: str, round_index: int) -> List[Dict[str, str]]:
    entry_hint = f"\nEntry point: {entry_point}" if entry_point else ""
    if round_index == 0:
        question = "Is the proposed Python code correct for this programming task?"
    else:
        question = "Think again carefully. Is the proposed Python code really correct for this programming task?"

    user_content = (
        f"HumanEval problem:{entry_hint}\n{problem_text}\n\n"
        f"Proposed code:\n```python\n{extract_code_answer(proposed_code)}\n```\n\n"
        f"{question}\n"
        "(A) Correct\n(B) Incorrect\n(C) I am not sure\n\n"
        "Use exactly this format:\n"
        "explanation: <brief analysis>\n"
        "answer: <A/B/C>"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are evaluating whether a proposed Python solution solves a HumanEval task. "
                "Do not execute code. Judge from the problem statement and the code."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def run_reflections_batch(
    llm: LocalCausalLM,
    problem_text: str,
    proposed_code: str,
    entry_point: str,
    num_rounds: int,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    messages_list = [
        build_reflection_messages(problem_text, proposed_code, entry_point, i)
        for i in range(num_rounds)
    ]
    raw_texts = llm.generate_batch(
        messages_list=messages_list,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_return_sequences=1,
    )
    results: List[Dict[str, Any]] = []
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





def compute_observed_consistency(
    nli: NLIModel,
    greedy_code: str,
    aux_codes: List[str],
    alpha: float,
) -> Tuple[float, List[Dict[str, Any]]]:
    if not aux_codes:
        return 0.0, []

    greedy_nli = code_for_nli(greedy_code)
    aux_nli = [code_for_nli(x) for x in aux_codes]

    forward_pairs = [(y, greedy_nli) for y in aux_nli]
    backward_pairs = [(greedy_nli, y) for y in aux_nli]

    p1s = nli.contradiction_prob_batch(forward_pairs)
    p2s = nli.contradiction_prob_batch(backward_pairs)

    details: List[Dict[str, Any]] = []
    vals: List[float] = []
    for aux_code, p1, p2 in zip(aux_codes, p1s, p2s):
        similarity_s = 0.5 * ((1.0 - p1) + (1.0 - p2))
        indicator_r = exact_code_indicator(aux_code, greedy_code)
        observed_item_o = alpha * similarity_s + (1.0 - alpha) * indicator_r
        details.append(
            {
                "sample_code": extract_code_answer(aux_code),
                "contradiction_forward": float(p1),
                "contradiction_backward": float(p2),
                "similarity_s": float(similarity_s),
                "indicator_r": int(indicator_r),
                "observed_item_o": float(observed_item_o),
            }
        )
        vals.append(float(observed_item_o))

    observed_consistency = float(sum(vals) / len(vals)) if vals else 0.0
    return observed_consistency, details


def compute_bsdetector_for_item(
    obj: Dict[str, Any],
    correct_map: Dict[str, int],
    llm: LocalCausalLM,
    nli: NLIModel,
    args: argparse.Namespace,
    line_idx: int,
) -> Tuple[Dict[str, Any], bool]:
    rid = resolve_record_id(obj) or f"line_{line_idx}"
    missing_correct = rid not in correct_map
    correct = correct_map.get(rid, 0)

    problem_text = get_problem_text(obj)
    entry_point = get_entry_point(obj)
    test_text = get_test_text(obj)
    greedy_raw = get_greedy_prediction(obj)
    greedy_code = extract_code_answer(greedy_raw)

    aux_results = generate_aux_samples_batch(
        llm=llm,
        problem_text=problem_text,
        entry_point=entry_point,
        num_samples=args.num_aux_samples,
        max_new_tokens=args.max_new_tokens_aux,
        temperature=args.aux_temperature,
        top_p=args.aux_top_p,
        top_k=args.aux_top_k,
    )
    reflection_results = run_reflections_batch(
        llm=llm,
        problem_text=problem_text,
        proposed_code=greedy_code,
        entry_point=entry_point,
        num_rounds=args.num_reflection_rounds,
        max_new_tokens=args.max_new_tokens_reflect,
    )

    aux_codes = [str(x.get("code", "")) for x in aux_results]
    observed_consistency, observed_details = compute_observed_consistency(
        nli=nli,
        greedy_code=greedy_code,
        aux_codes=aux_codes,
        alpha=args.alpha,
    )

    reflection_scores = [float(x.get("score", 0.5)) for x in reflection_results]
    self_reflection_certainty = (
        float(sum(reflection_scores) / len(reflection_scores)) if reflection_scores else 0.5
    )

    confidence = float(args.beta * observed_consistency + (1.0 - args.beta) * self_reflection_certainty)
    uncertainty = float(1.0 - confidence)

    out: Dict[str, Any] = {
        "id": rid,
        "correct": int(correct),
        "observed_consistency": observed_consistency,
        "self_reflection_certainty": self_reflection_certainty,
        "bsdetector_confidence": confidence,
        "bsdetector_uncertainty": uncertainty,
        
        "uncertainty": uncertainty,
        "greedy_code": greedy_code,
        "auxiliary_samples": aux_results,
        "observed_details": observed_details,
        "self_reflection_details": reflection_results,
        "meta": {
            "method": "BSDETECTOR",
            "dataset": "humaneval",
            "llm_model": args.model_path,
            "nli_model": args.nli_model,
            "alpha": args.alpha,
            "beta": args.beta,
            "num_aux_samples": args.num_aux_samples,
            "num_reflection_rounds": args.num_reflection_rounds,
            "correctness_checker": "external_correct_file_only",
            "correct_file": args.correct_file,
        },
    }

    if args.keep_question:
        out["question"] = problem_text
        if entry_point:
            out["entry_point"] = entry_point
        if args.keep_test and test_text:
            out["test"] = test_text
    if args.keep_prediction:
        out["prediction"] = greedy_raw

    return out, missing_correct





def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute BSDetector uncertainty for HumanEval result files.")
    parser.add_argument("--input-file", required=True, help="HumanEval stage1 model output JSONL")
    parser.add_argument("--correct-file", required=True, help="Correctness JSONL; correctness is loaded only from this file")
    parser.add_argument("--output-file", required=True, help="Output JSONL path")
    parser.add_argument("--model-path", required=True, help="Local model path used for auxiliary samples and self-reflection")
    parser.add_argument("--nli-model", default="microsoft/deberta-large-mnli", help="NLI model for contradiction score")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device preference")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map, default=auto when CUDA is available")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], help="Causal LM dtype")
    parser.add_argument("--question-id-key", default="question_id", help="Key name for question id in correct file")
    parser.add_argument("--correct-key", default="correct", help="Key name for correctness label in correct file")

    parser.add_argument("--alpha", type=float, default=0.8, help="Weight of NLI similarity in observed consistency")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight of observed consistency in final confidence")
    parser.add_argument("--num-aux-samples", type=int, default=5, help="Number of sampled auxiliary solutions")
    parser.add_argument("--num-reflection-rounds", type=int, default=2, help="Number of self-reflection rounds")
    parser.add_argument("--max-new-tokens-aux", type=int, default=256, help="Max new tokens for auxiliary code generation")
    parser.add_argument("--max-new-tokens-reflect", type=int, default=128, help="Max new tokens for reflection generation")
    parser.add_argument("--nli-max-length", type=int, default=512, help="Max token length for NLI input")
    parser.add_argument("--aux-temperature", type=float, default=1.0, help="Sampling temperature for auxiliary solutions")
    parser.add_argument("--aux-top-p", type=float, default=1.0, help="top_p for auxiliary solutions")
    parser.add_argument("--aux-top-k", type=int, default=50, help="top_k for auxiliary solutions")

    parser.add_argument("--keep-question", action="store_true", help="Keep HumanEval prompt/question in output")
    parser.add_argument("--keep-prediction", action="store_true", help="Keep raw greedy prediction in output")
    parser.add_argument("--keep-test", action="store_true", help="Keep HumanEval test text if --keep-question is also set")
    parser.add_argument("--verbose", action="store_true", help="Print per-record progress")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file instead of resuming")
    parser.add_argument("--max-items", type=int, default=None, help="Optional debug limit")
    parser.add_argument("--disable-thinking", action="store_true", default=True, help="Try enable_thinking=False for Qwen3 chat template")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    data = load_jsonl(args.input_file)
    if args.max_items is not None:
        data = data[: args.max_items]

    correct_records = load_jsonl(args.correct_file)
    correct_map = build_correct_map(correct_records, args.question_id_key, args.correct_key)

    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)

    done = load_done_ids(args.output_file)
    safe_print(f"[info] loaded input records: {len(data)}")
    safe_print(f"[info] loaded correct labels: {len(correct_map)}")
    safe_print(f"[info] existing output records: {len(done)}")

    safe_print(f"[info] loading local LM: {args.model_path}")
    llm = LocalCausalLM(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        disable_thinking=args.disable_thinking,
    )

    safe_print(f"[info] loading NLI model: {args.nli_model}")
    nli = NLIModel(args.nli_model, device=args.device, max_length=args.nli_max_length)

    missing_correct = 0
    processed = 0

    for line_idx, obj in enumerate(data, start=1):
        rid = resolve_record_id(obj) or f"line_{line_idx}"
        if rid in done:
            continue

        try:
            result, is_missing = compute_bsdetector_for_item(
                obj=obj,
                correct_map=correct_map,
                llm=llm,
                nli=nli,
                args=args,
                line_idx=line_idx,
            )
            if is_missing:
                missing_correct += 1
            append_jsonl(args.output_file, result)
            processed += 1

            if args.verbose:
                safe_print(
                    f"done: {rid} | line={line_idx} | correct={result['correct']} | "
                    f"O={result['observed_consistency']:.4f} | "
                    f"S={result['self_reflection_certainty']:.4f} | "
                    f"conf={result['bsdetector_confidence']:.4f} | "
                    f"unc={result['bsdetector_uncertainty']:.4f}"
                )
        except KeyboardInterrupt:
            safe_print("[info] KeyboardInterrupt detected. Exiting gracefully.")
            raise
        except Exception as e:
            err = {
                "id": rid,
                "correct": int(correct_map.get(rid, 0)),
                "error": repr(e),
                "meta": {
                    "method": "BSDETECTOR",
                    "dataset": "humaneval",
                    "correctness_checker": "external_correct_file_only",
                    "correct_file": args.correct_file,
                },
            }
            if args.keep_question:
                err["question"] = get_problem_text(obj)
            if args.keep_prediction:
                err["prediction"] = get_greedy_prediction(obj)
            append_jsonl(args.output_file, err)
            processed += 1
            safe_print(f"error: {rid} | line={line_idx} | {repr(e)}")

    safe_print(f"[info] wrote/appended {processed} records to {args.output_file}")
    if missing_correct > 0:
        safe_print(
            f"[warn] {missing_correct} records were missing in --correct-file and were set to correct=0"
        )


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

    _parser = _ap.ArgumentParser(description="bsdetector / llama / humaneval / answer_only (standalone)")
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
                           "dataset": "humaneval", "setting": "answer_only", "model_loading": False}, indent=2))
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

