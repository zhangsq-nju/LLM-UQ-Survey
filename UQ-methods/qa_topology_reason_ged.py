#!/usr/bin/env python3
# Original paper/author repository: https://github.com/LongchaoDa/LLM-Topology

import argparse
import json
import os
import random
import tempfile
import itertools
import string
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

try:
    from math_verify import parse, verify
except Exception:
    parse = None
    verify = None


ID_KEYS = ["id", "question_id", "qid", "task_id", "problem_id"]
QUESTION_KEYS = ["question", "input", "prompt", "problem"]
DATASET = 'gsm8k'
FAMILY = 'qwen'
GOLD_KEYS = ["correct answer", "answer", "gold", "gold_answer", "target"]


def read_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def get_first(obj: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return default


def normalize_space(text: Any) -> str:
    return "" if text is None else re.sub(r"\s+", " ", str(text)).strip()


def get_gold(item: Dict[str, Any]) -> str:
    value = get_first(item, ["gt", "ground_truth"] + GOLD_KEYS, "")
    if isinstance(value, dict):
        value = value.get("value", value.get("normalized_value", value.get("text", "")))
    if isinstance(value, list):
        value = value[0] if value else ""
    text = "" if value is None else str(value)
    if DATASET == "gsm8k" and "####" in text:
        text = text.rsplit("####", 1)[1]
    return normalize_space(text)


def get_question(item: Dict[str, Any]) -> str:
    return str(get_first(item, QUESTION_KEYS, "")).strip()


def get_item_id(item: Dict[str, Any], fallback: int) -> str:
    return str(get_first(item, ID_KEYS, fallback))


def get_prediction_text(response: Dict[str, Any]) -> str:
    for key in ["llm answer", "answer", "prediction", "output", "response"]:
        if response.get(key) is not None:
            return str(response[key])
    info = response.get("answer info")
    if isinstance(info, dict) and info.get("text"):
        return str(info["text"])
    return str(response.get("llm reason", ""))


def mathv_correct(ref_text: str, target_text: str) -> bool:
    if parse is None or verify is None:
        raise RuntimeError("math_verify is required: pip install math-verify")
    try:
        gold, prediction = parse(str(ref_text)), parse(str(target_text))
        return bool(gold) and bool(prediction) and bool(verify(gold, prediction))
    except Exception as exc:
        raise RuntimeError("math-verify evaluation failed") from exc


def choose_response_indices(n: int, response_index: int, max_responses: int) -> List[int]:
    if n <= 0:
        return []
    if response_index >= 0:
        return [response_index] if response_index < n else []
    indices = list(range(n))
    return indices[:max_responses] if max_responses > 0 else indices


def _extract_json_value(text: str) -> Any:
    text = re.sub(r"^```(?:json)?\s*", "", str(text or "").strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char in "[{":
            try:
                value, _ = decoder.raw_decode(text[start:])
                return value
            except json.JSONDecodeError:
                pass
    raise ValueError("No valid JSON object or array in model output")


def parse_knowledge_points(text: str, max_points: int) -> List[str]:
    points: List[str] = []
    try:
        value = _extract_json_value(text)
        if isinstance(value, dict):
            value = value.get("sub_questions") or value.get("knowledge_points") or value.get("questions")
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    item = item.get("question") or item.get("text") or item.get("content")
                item = normalize_space(item)
                if item:
                    points.append(item)
    except ValueError:
        pass
    if not points:
        for line in str(text or "").splitlines():
            line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if line:
                points.append(normalize_space(line))
    unique, seen = [], set()
    for point in points:
        key = point.casefold()
        if key not in seen:
            unique.append(point)
            seen.add(key)
    return unique[:max_points]


@dataclass
class QAPair:
    edge_id: str
    question: str
    node_id: str
    answer: str

    def to_dict(self) -> Dict[str, str]:
        return {"edge_id": self.edge_id, "question": self.question,
                "node_id": self.node_id, "answer": self.answer}


@dataclass
class GraphLink:
    source: str
    target: str
    edge: str

    def to_dict(self) -> Dict[str, str]:
        return {"source": self.source, "target": self.target, "edge": self.edge}


@dataclass
class ReasoningGraph:
    question: str
    qa_pairs: List[QAPair]
    links: List[GraphLink]
    final_answer: str
    result_edge: str
    raw_knowledge_output: str = ""
    raw_topology_output: str = ""
    _node_map: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _edge_map: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._node_map = {"NodeRaw": self.question, "NodeResult": self.final_answer}
        self._edge_map = {"ResultEdge": self.result_edge or self.final_answer}
        for pair in self.qa_pairs:
            self._node_map[pair.node_id] = pair.answer
            self._edge_map[pair.edge_id] = pair.question

    @property
    def node_texts(self) -> List[str]:
        used = {x.source for x in self.links} | {x.target for x in self.links}
        ordered = ["NodeRaw"] + [x.node_id for x in self.qa_pairs] + ["NodeResult"]
        return [self._node_map[x] for x in ordered if x in used and self._node_map.get(x)]

    @property
    def edge_texts(self) -> List[str]:
        
        return [self._edge_map.get(link.edge, link.edge) for link in self.links]

    def to_dict(self, keep_raw: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "question": self.question,
            "qa_pairs": [x.to_dict() for x in self.qa_pairs],
            "links": [x.to_dict() for x in self.links],
            "final_answer": self.final_answer,
            "result_edge": self.result_edge,
        }
        if keep_raw:
            data["raw_knowledge_output"] = self.raw_knowledge_output
            data["raw_topology_output"] = self.raw_topology_output
        return data


def _canonical_node_id(value: Any) -> str:
    text = normalize_space(value)
    folded = text.casefold().replace("_", "")
    if folded in {"noderaw", "raw", "question"}:
        return "NodeRaw"
    if folded in {"noderesult", "resultnode", "result", "answer"}:
        return "NodeResult"
    match = re.search(r"(?:node)?\s*(\d+)", text, flags=re.I)
    return f"Node{int(match.group(1))}" if match else text


def _canonical_edge_id(value: Any) -> str:
    text = normalize_space(value)
    folded = text.casefold().replace("_", "")
    if folded in {"resultedge", "finaledge", "result", "final"}:
        return "ResultEdge"
    match = re.search(r"(?:edge)?\s*(\d+)", text, flags=re.I)
    return f"Edge{int(match.group(1))}" if match else text


def _parse_legacy_structure(text: str) -> List[Dict[str, str]]:
    pattern = re.compile(r"\[\s*([^,\]]+)\s*,\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]")
    return [{"source": a, "target": b, "edge": e} for a, b, e in pattern.findall(text)]


def graph_from_dict(data: Dict[str, Any], fallback_question: str = "") -> ReasoningGraph:
    if not isinstance(data, dict):
        raise ValueError("Topology must be an object")
    question = normalize_space(data.get("question") or fallback_question)
    qa_pairs = []
    for index, pair in enumerate(data.get("qa_pairs") or data.get("pairs") or []):
        if not isinstance(pair, dict):
            continue
        edge_id = _canonical_edge_id(pair.get("edge_id") or pair.get("edge") or f"Edge{index}")
        node_id = _canonical_node_id(pair.get("node_id") or pair.get("node") or f"Node{index}")
        subq = normalize_space(pair.get("question") or pair.get("sub_question") or pair.get("edge_text"))
        answer = normalize_space(get_first(pair, ["answer", "response", "node_text"]))
        if subq and answer:
            qa_pairs.append(QAPair(edge_id, subq, node_id, answer))
    final_answer = normalize_space(get_first(data, ["final_answer", "answer", "result"]))
    result_edge = normalize_space(data.get("result_edge") or data.get("conclusion") or final_answer)
    allowed_nodes = {"NodeRaw", "NodeResult"} | {x.node_id for x in qa_pairs}
    allowed_edges = {"ResultEdge"} | {x.edge_id for x in qa_pairs}
    links_raw = data.get("links") or data.get("steps") or data.get("structure") or []
    if isinstance(links_raw, str):
        links_raw = _parse_legacy_structure(links_raw)
    links = []
    for link in links_raw:
        if isinstance(link, (list, tuple)) and len(link) == 3:
            source, target, edge = link
        elif isinstance(link, dict):
            source = link.get("source") or link.get("from") or link.get("start")
            target = link.get("target") or link.get("to") or link.get("end")
            edge = link.get("edge") or link.get("edge_id") or link.get("relation")
        else:
            continue
        candidate = GraphLink(_canonical_node_id(source), _canonical_node_id(target),
                              _canonical_edge_id(edge))
        if (candidate.source in allowed_nodes and candidate.target in allowed_nodes
                and candidate.edge in allowed_edges and candidate not in links):
            links.append(candidate)
    if not question or not qa_pairs or not final_answer or not links:
        raise ValueError("Topology misses question, Q/A pairs, final answer, or links")
    if not any(x.source == "NodeRaw" for x in links):
        raise ValueError("Topology has no edge leaving NodeRaw")
    if not any(x.target == "NodeResult" for x in links):
        raise ValueError("Topology has no edge entering NodeResult")
    if len({x.node_id for x in qa_pairs}) != len(qa_pairs) or len({x.edge_id for x in qa_pairs}) != len(qa_pairs):
        raise ValueError("Duplicate topology node/edge IDs")
    if any(x.node_id in {"NodeRaw", "NodeResult"} or x.edge_id == "ResultEdge" for x in qa_pairs):
        raise ValueError("Q/A pairs must not overwrite reserved topology IDs")
    reachable, pending = {"NodeRaw"}, ["NodeRaw"]
    while pending:
        current = pending.pop()
        for link in links:
            if link.source == current and link.target not in reachable:
                reachable.add(link.target)
                pending.append(link.target)
    if "NodeResult" not in reachable:
        raise ValueError("Topology has no path from NodeRaw to NodeResult")
    return ReasoningGraph(question, qa_pairs, links, final_answer, result_edge or final_answer,
                          normalize_space(data.get("raw_knowledge_output")),
                          normalize_space(data.get("raw_topology_output")))


class LocalGenerator:
    def __init__(self, model_path: str, device: str, chat_template_mode: str,
                 temperature: float, top_p: float, max_new_tokens: int, seed: int, family: str = FAMILY) -> None:
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.chat_template_mode = chat_template_mode
        self.family = family
        self.temperature, self.top_p = temperature, top_p
        self.max_new_tokens, self.seed = max_new_tokens, seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        kwargs: Dict[str, Any] = {
            "torch_dtype": torch.float32 if self.device == "cpu" else "auto"
        }
        if self.device.startswith("cuda"):
            kwargs["device_map"] = "auto" if self.device == "cuda" else {"": self.device}
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if self.device == "cpu":
            self.model.to("cpu")
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _prompt(self, system: str, user: str) -> str:
        use_chat = self.chat_template_mode == "on" or (
            self.chat_template_mode == "auto" and getattr(self.tokenizer, "chat_template", None))
        if use_chat:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            options = {"enable_thinking": False} if self.family == "qwen" else {}
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **options)
        return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"

    @torch.inference_mode()
    def generate(self, system: str, user: str, sample_seed: int,
                 max_new_tokens: Optional[int] = None) -> str:
        use_chat = self.chat_template_mode == "on" or (self.chat_template_mode == "auto" and bool(getattr(self.tokenizer, "chat_template", None)))
        inputs = self.tokenizer(self._prompt(system, user), return_tensors="pt", add_special_tokens=not use_chat)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        torch.manual_seed(self.seed + sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed + sample_seed)
        do_sample = self.temperature > 0
        kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs.update(temperature=self.temperature, top_p=self.top_p)
        output = self.model.generate(**inputs, **kwargs)
        tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()


class TopologyElicitor:
    def __init__(self, generator: LocalGenerator, max_points: int, retries: int) -> None:
        self.generator, self.max_points, self.retries = generator, max_points, retries

    def _knowledge(self, question: str, sample_seed: int) -> Tuple[List[str], str]:
        system = ("Identify sufficient and concise knowledge needed to solve a question. "
                  "Return only valid JSON; do not directly solve the original question.")
        user = (f"Question: {question}\nList at most {self.max_points} necessary knowledge points "
                'as sub-questions. Return exactly: {"sub_questions":["...","..."]}')
        last = ""
        for retry in range(self.retries + 1):
            last = self.generator.generate(system, user, sample_seed * 1000 + retry)
            points = parse_knowledge_points(last, self.max_points)
            if points:
                return points, last
            user += "\nPrevious output was invalid. Return one valid JSON object only."
        raise ValueError(f"Failed to elicit knowledge points: {last[:300]}")

    def _answer(self, original: str, subq: str, seed: int) -> str:
        system = ("Answer one knowledge sub-question precisely and concisely. "
                  "Return only the answer text, without a label.")
        return normalize_space(self.generator.generate(
            system, f"Original question: {original}\nSub-question: {subq}", seed, 192))

    def _topology(self, question: str, pairs: Sequence[QAPair], seed: int
                  ) -> Tuple[List[GraphLink], str, str, str]:
        pair_text = "\n".join(f"{x.edge_id}: {x.question}\n{x.node_id}: {x.answer}" for x in pairs)
        system = ("Connect supplied Edge-Node question-answer pairs into a directed reasoning "
                  "topology that solves the question. Return only valid JSON.")
        user = f"""Original question: {question}
Available pairs:
{pair_text}

Rules:
1. NodeRaw is the original question; NodeResult is the reasoning end.
2. Each link is [source node, target node, edge]. Use supplied IDs only.
3. The topology may branch or merge; do not force a chain.
4. It must contain a path from NodeRaw to NodeResult.
5. The final link uses ResultEdge.

Return exactly:
{{"links":[{{"source":"NodeRaw","target":"Node0","edge":"Edge0"}},
{{"source":"Node0","target":"NodeResult","edge":"ResultEdge"}}],
"final_answer":"...","result_edge":"concluding reasoning operation"}}"""
        last = ""
        for retry in range(self.retries + 1):
            last = self.generator.generate(system, user, seed * 1000 + 500 + retry)
            try:
                data = _extract_json_value(last)
                graph = graph_from_dict({
                    "question": question,
                    "qa_pairs": [x.to_dict() for x in pairs],
                    "links": data.get("links") or data.get("steps") or data.get("structure"),
                    "final_answer": get_first(data, ["final_answer", "answer", "result"]),
                    "result_edge": data.get("result_edge") or data.get("conclusion"),
                }, question)
                return graph.links, graph.final_answer, graph.result_edge, last
            except (AttributeError, TypeError, ValueError):
                user += "\nPrevious output was invalid. Use valid IDs and output JSON only."
        raise ValueError(f"Failed to construct topology: {last[:300]}")

    def elicit(self, question: str, sample_index: int) -> ReasoningGraph:
        points, raw_knowledge = self._knowledge(question, sample_index)
        pairs = []
        for index, subq in enumerate(points):
            answer = self._answer(question, subq, sample_index * 100 + index + 1)
            if not answer:
                raise ValueError(f"Empty answer for sub-question {index}")
            pairs.append(QAPair(f"Edge{index}", subq, f"Node{index}", answer))
        links, final, result_edge, raw_topology = self._topology(question, pairs, sample_index)
        return ReasoningGraph(question, pairs, links, final, result_edge,
                              raw_knowledge, raw_topology)


class SentenceEmbedder:
    

    def __init__(self, model_path: str, device: str, batch_size: int,
                 max_length: int, pooling: str = "cls") -> None:
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.batch_size, self.max_length, self.pooling = batch_size, max_length, pooling
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
        self.model.to(self.device).float().eval()
        self.cache: Dict[str, np.ndarray] = {}

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        keys = [normalize_space(x) for x in texts]
        missing = list(dict.fromkeys(x for x in keys if x not in self.cache))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            hidden = self.model(**inputs).last_hidden_state.float()
            if self.pooling == "cls":
                
                pooled = hidden[:, 0, :]
            else:
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            for key, vector in zip(batch, pooled.cpu().numpy().astype(np.float32)):
                self.cache[key] = vector
        return (np.stack([self.cache[x] for x in keys]) if keys
                else np.empty((0, 0), dtype=np.float32))


def cosine_similarity_matrix(left: Sequence[str], right: Sequence[str],
                             embedder: SentenceEmbedder) -> np.ndarray:
    if not left or not right:
        return np.empty((len(left), len(right)), dtype=np.float32)
    return np.clip(embedder.encode(left) @ embedder.encode(right).T, -1.0, 1.0)


def _deletion_cost(index: int, own_sim: np.ndarray, cross_sim: np.ndarray) -> float:
    
    cross = float(np.max(cross_sim[index])) if cross_sim.shape[1] else 0.0
    unique = (1.0 if own_sim.shape[0] <= 1
              else 1.0 - float(np.mean(np.delete(own_sim[index], index))))
    return 0.5 * (cross + unique)


def _match_elements(left: Sequence[str], right: Sequence[str],
                    embedder: SentenceEmbedder) -> Dict[str, Any]:
    if linear_sum_assignment is None:
        raise RuntimeError("scipy is required for Hungarian graph matching")
    left, right = list(left), list(right)
    if not left and not right:
        return {"cost": 0.0, "substitution": 0.0, "edit": 0.0, "matches": []}
    cross = cosine_similarity_matrix(left, right, embedder)
    sub_matrix = 1.0 - cross
    if left and right:
        rows, cols = linear_sum_assignment(sub_matrix)
        matches = list(zip(rows.tolist(), cols.tolist()))
    else:
        matches = []
    matched_left, matched_right = {x for x, _ in matches}, {y for _, y in matches}
    substitution = float(sum(sub_matrix[x, y] for x, y in matches))
    left_self = cosine_similarity_matrix(left, left, embedder)
    right_self = cosine_similarity_matrix(right, right, embedder)
    edit = sum(_deletion_cost(i, left_self, cross) for i in range(len(left))
               if i not in matched_left)
    edit += sum(_deletion_cost(j, right_self, cross.T) for j in range(len(right))
                if j not in matched_right)
    return {"cost": substitution + edit, "substitution": substitution,
            "edit": float(edit), "matches": matches}


def reason_ged(graph_a: ReasoningGraph, graph_b: ReasoningGraph,
               embedder: SentenceEmbedder, node_weight: float = 1.0,
               edge_weight: float = 1.0, normalize: bool = False
               ) -> Tuple[float, Dict[str, Any]]:
    nodes = _match_elements(graph_a.node_texts, graph_b.node_texts, embedder)
    edges = _match_elements(graph_a.edge_texts, graph_b.edge_texts, embedder)
    raw = node_weight * nodes["cost"] + edge_weight * edges["cost"]
    denominator = 1.0
    if normalize:
        denominator = max(1.0,
            node_weight * max(len(graph_a.node_texts), len(graph_b.node_texts)) +
            edge_weight * max(len(graph_a.edge_texts), len(graph_b.edge_texts)))
    details = {
        "node_substitution_cost": nodes["substitution"],
        "node_insertion_deletion_cost": nodes["edit"],
        "edge_substitution_cost": edges["substitution"],
        "edge_insertion_deletion_cost": edges["edit"],
        "raw_reason_ged": float(raw),
        "normalization_denominator": float(denominator),
    }
    return float(raw / denominator), details


def pairwise_topouq(graphs: Sequence[ReasoningGraph], embedder: SentenceEmbedder,
                    node_weight: float, edge_weight: float,
                    normalize_ged: bool) -> Dict[str, Any]:
    count = len(graphs)
    if count < 2:
        raise ValueError("Topo-UQ requires at least two topologies")
    matrix = np.zeros((count, count), dtype=np.float64)
    values, pair_details = [], []
    for i in range(count):
        for j in range(i + 1, count):
            distance, details = reason_ged(graphs[i], graphs[j], embedder,
                                           node_weight, edge_weight, normalize_ged)
            matrix[i, j] = matrix[j, i] = distance
            if not np.isfinite(distance):
                raise ValueError("Non-finite Reason-GED distance")
            values.append(distance)
            pair_details.append({"left": i, "right": j, "distance": distance, **details})
    array = np.asarray(values, dtype=np.float64)
    
    variance = float(np.var(matrix)) if count else 0.0
    return {
        "uncertainty": variance,
        "topouq_variance": variance,
        "reason_ged_mean_pairwise": float(np.mean(array)) if array.size else 0.0,
        "reason_ged_max_pairwise": float(np.max(array)) if array.size else 0.0,
        "reason_ged_min_pairwise": float(np.min(array)) if array.size else 0.0,
        "distance_matrix": matrix.tolist(),
        "pair_details": pair_details,
    }


def extract_precomputed_graphs(item: Dict[str, Any], question: str,
                               max_count: int) -> List[ReasoningGraph]:
    candidates: List[Any] = []
    item_graphs = item.get("reasoning_topologies") or item.get("topologies")
    if isinstance(item_graphs, list):
        candidates.extend(item_graphs)
    if isinstance(item.get("responses"), list):
        for response in item["responses"]:
            if isinstance(response, dict):
                graph = response.get("reasoning_topology") or response.get("topology")
                if graph is not None:
                    candidates.append(graph)
    graphs = []
    for candidate in candidates:
        try:
            if isinstance(candidate, str):
                candidate = _extract_json_value(candidate)
            graphs.append(graph_from_dict(candidate, question))
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid precomputed topology") from exc
        if len(graphs) >= max_count:
            break
    return graphs



def load_correct_map(path):
    labels = {}
    for _, row in read_jsonl(path):
        qid = get_first(row, ID_KEYS)
        index = row.get("response_index", row.get("target_response_index", 0))
        value = row.get("correct")
        if qid is None or value is None or isinstance(value,str) or value not in (0,1) or not isinstance(index,int) or index < 0:
            raise ValueError("Correctness rows require id, nonnegative response_index and binary correct")
        key = (str(qid),index)
        if key in labels:
            raise ValueError(f"Duplicate correctness ID/index: {key}")
        labels[key] = int(value)
    return labels


def response_correct(item, item_id, index, prediction, args):
    if args.correct_file:
        key = (str(item_id),index)
        if key not in args.correct_map:
            raise ValueError(f"Missing correctness label for {key}")
        return args.correct_map[key]
    if DATASET == "humaneval":
        raise ValueError("HumanEval requires unit-test labels via --correct-file")
    if DATASET == "gsm8k":
        gold = get_gold(item)
        if not gold:
            raise ValueError("GSM8K requires a gold answer")
        return int(mathv_correct(gold,prediction))
    def normalize(text):
        text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
        return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())
    value = get_first(item,["gt","ground_truth"]+GOLD_KEYS)
    if isinstance(value,dict):
        aliases = list(value.get("aliases",value.get("normalized_aliases",[])) or [])
        value = value.get("value",value.get("normalized_value",value.get("text")))
        if value is not None:
            aliases.append(value)
    else:
        aliases = value if isinstance(value,list) else [value]
        aliases = aliases + list(item.get("aliases",[]) or [])
    aliases = [normalize(x) for x in aliases if x is not None]
    if not aliases:
        raise ValueError("TriviaQA requires gold answer aliases")
    predicted = normalize(prediction)
    return int(bool(predicted) and predicted in aliases)


def build_output_row(item: Dict[str, Any], item_id: str, gold: str,
                     response_index: int, response: Dict[str, Any],
                     selected: Sequence[int], graphs: Sequence[ReasoningGraph],
                     uq: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    prediction = get_prediction_text(response)
    row: Dict[str, Any] = {
        "id": item_id, "question_id": item_id, "response_index": response_index,
        "correct": response_correct(item, item_id, response_index, prediction, args),
        "dataset": DATASET, "family": FAMILY, "setting": "reasoning_augmented",
        "implementation": "qa_topology_reason_ged",
        "topouq_uncertainty": uq["uncertainty"],
        "uncertainty": uq["uncertainty"],
        "topouq_variance": uq["topouq_variance"],
        "reason_ged_mean_pairwise": uq["reason_ged_mean_pairwise"],
        "reason_ged_max_pairwise": uq["reason_ged_max_pairwise"],
        "reason_ged_min_pairwise": uq["reason_ged_min_pairwise"],
        "method": "Topo-UQ (QA topology + Reason-GED)",
        "topology_sampling": {"elicitation_uses_only_original_question": True,
                              "precomputed_topologies": uq.get("precomputed_topologies",0), "seed": args.seed,
                              "model_path": args.model_path, "temperature": args.temperature,
                              "top_p": args.top_p, "max_knowledge_points": args.max_knowledge_points},
        "distance_config": {"embedding_model": args.embedding_model_path or args.nli_model_path,
                            "pooling": args.embedding_pooling, "normalize_ged": args.normalize_ged,
                            "node_weight": args.node_weight, "edge_weight": args.edge_weight,
                            "variance_scope": "full_symmetric_matrix_including_diagonal", "ddof": 0},
        "selected_response_indices": list(selected),
        "num_topologies": len(graphs),
    }
    if args.keep_question:
        row["question"] = get_question(item)
    if args.keep_gold:
        row["gold"] = gold
    if args.keep_prediction:
        row["prediction"] = prediction
    if args.keep_candidates:
        row["candidate_predictions"] = [get_prediction_text(item["responses"][i]) for i in selected]
    if args.keep_topologies:
        row["reasoning_topologies"] = [x.to_dict(args.keep_raw_elicitation) for x in graphs]
    if args.keep_distance_matrix:
        row["distance_matrix"] = uq["distance_matrix"]
    if args.keep_pair_details:
        row["reason_ged_pair_details"] = uq["pair_details"]
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-style Topo-UQ / Reason-GED")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--correct-file", required=DATASET == "humaneval")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-path", default=None,
                        help="Causal LM for three-stage topology elicitation")
    parser.add_argument("--embedding-model-path", default=None,
                        help="BERT/sentence encoder for node/edge semantics")
    parser.add_argument("--nli-model-path", default=None,
                        help="Deprecated alias for --embedding-model-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-family", default=FAMILY, choices=["qwen", "llama", "auto"])
    parser.add_argument("--chat-template-mode", default="auto", choices=["off", "on", "auto"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-knowledge-points", type=int, default=10)
    parser.add_argument("--elicitation-retries", type=int, default=2)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-pooling", default="cls", choices=["cls", "mean"],
                        help="CLS matches the authors' released BERT embedding pipeline")
    parser.add_argument("--response-index", type=int, default=0,
                        help="Response emitted for correctness; -1 emits all selected")
    parser.add_argument("--max-responses", type=int, default=10)
    parser.add_argument("--num-topologies", type=int, default=None,
                        help="Graph samples; default=min(response count,max-responses)")
    parser.add_argument("--node-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--normalize-ged", action="store_true",
                        help="Size-normalize GED (off matches the paper's summed GED)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep-question", action="store_true")
    parser.add_argument("--keep-prediction", action="store_true")
    parser.add_argument("--keep-gold", action="store_true")
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--keep-topologies", action="store_true")
    parser.add_argument("--keep-raw-elicitation", action="store_true")
    parser.add_argument("--keep-distance-matrix", action="store_true")
    parser.add_argument("--keep-pair-details", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    parser.add_argument("--include-answer-unit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-units", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--length-penalty-weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--nli-max-length", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--nli-batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--keep-chains", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    obsolete = [key for key in ["include_answer_unit","max_units","length_penalty_weight",
                                "nli_max_length","nli_batch_size","keep_chains"]
                if getattr(args,key) is not None and getattr(args,key) is not False]
    if obsolete:
        parser.error("Chain-only options are not applicable: " + ", ".join(obsolete) +
                     "; use embedding-* / keep-topologies options or the _approx.py version")
    return args


def main() -> None:
    args = parse_args()
    if args.model_family not in {"auto", FAMILY}:
        raise ValueError(f"This file is for family {FAMILY}")
    if os.path.realpath(args.input_file) == os.path.realpath(args.output_file):
        raise ValueError("Output must not replace input")
    if not os.path.isfile(args.input_file):
        raise ValueError("Input file does not exist")
    if os.path.exists(args.output_file) and not args.overwrite:
        raise ValueError("Output exists; use --overwrite")
    if args.response_index < -1 or args.max_responses < 1 or args.max_knowledge_points < 1:
        raise ValueError("Invalid response/knowledge limits")
    if args.num_topologies is not None and args.num_topologies < 2:
        raise ValueError("At least two independent topologies required")
    if (args.limit is not None and args.limit < 1) or args.elicitation_retries < 0:
        raise ValueError("Invalid limit/retry count")
    if args.embedding_batch_size < 1 or args.embedding_max_length < 1 or args.max_new_tokens < 1:
        raise ValueError("Embedding/generation limits must be positive")
    if not (np.isfinite(args.temperature) and args.temperature >= 0 and 0 < args.top_p <= 1):
        raise ValueError("Invalid sampling parameters")
    if not all(np.isfinite(x) and x >= 0 for x in [args.node_weight,args.edge_weight]) or args.node_weight + args.edge_weight <= 0:
        raise ValueError("Distance weights must be finite, nonnegative and not both zero")
    embedding_path = args.embedding_model_path or args.nli_model_path
    if not embedding_path:
        raise ValueError("Provide --embedding-model-path (or --nli-model-path alias)")
    if args.correct_file and DATASET != "humaneval":
        raise ValueError("--correct-file is for HumanEval; GSM8K uses math-verify and TriviaQA uses aliases")
    args.correct_map = load_correct_map(args.correct_file) if args.correct_file else {}
    if args.dry_run:
        print(json.dumps({"method":"topo_uq", "family":FAMILY, "dataset":DATASET,
                          "implementation":"qa_topology_reason_ged", "embedding_model":embedding_path}))
        return
    if DATASET == "gsm8k" and (parse is None or verify is None):
        raise RuntimeError("math_verify is required: pip install math-verify")
    if linear_sum_assignment is None:
        raise RuntimeError("scipy is required: pip install scipy")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    embedding_path = args.embedding_model_path or args.nli_model_path
    if not embedding_path:
        raise ValueError("Provide --embedding-model-path (or --nli-model-path alias)")
    embedder = SentenceEmbedder(embedding_path, args.device, args.embedding_batch_size,
                                args.embedding_max_length, args.embedding_pooling)
    elicitor: Optional[TopologyElicitor] = None
    outputs: List[Dict[str, Any]] = []
    iterator: Iterable[Tuple[int, Dict[str, Any]]] = read_jsonl(args.input_file)
    if args.limit is not None:
        iterator = itertools.islice(iterator, args.limit)

    seen_ids = set()
    for item_count, (line_no, item) in enumerate(tqdm(iterator, desc="Topo-UQ"), 1):
        item_id, question, gold = get_item_id(item, line_no), get_question(item), get_gold(item)
        if item_id in seen_ids:
            raise ValueError(f"Duplicate question ID: {item_id}")
        seen_ids.add(item_id)
        responses = item.get("responses", [])
        if not question:
            raise ValueError(f"Line {line_no} (id={item_id}) has no question")
        if not isinstance(responses, list) or not responses:
            raise ValueError(f"Line {line_no} (id={item_id}) has no responses")
        if not all(isinstance(response, dict) for response in responses):
            raise ValueError(f"id={item_id} responses must be objects")
        if args.response_index >= len(responses):
            raise ValueError(f"id={item_id} response index out of range")
        selected = choose_response_indices(len(responses), -1, args.max_responses)
        target_count = args.num_topologies if args.num_topologies is not None else len(selected)
        if target_count < 2:
            raise ValueError("Topo-UQ requires at least two independent topologies")
        graphs = extract_precomputed_graphs(item, question, target_count)
        if len(graphs) < target_count:
            if not args.model_path:
                raise ValueError(f"id={item_id} has {len(graphs)}/{target_count} topologies; "
                                 "provide --model-path to elicit the rest")
            if elicitor is None:
                generator = LocalGenerator(args.model_path, args.device,
                    args.chat_template_mode, args.temperature, args.top_p,
                    args.max_new_tokens, args.seed, FAMILY)
                elicitor = TopologyElicitor(generator, args.max_knowledge_points,
                                            args.elicitation_retries)
            while len(graphs) < target_count:
                graphs.append(elicitor.elicit(question, item_count * 10000 + len(graphs)))

        precomputed_count = len(extract_precomputed_graphs(item, question, target_count))
        uq = pairwise_topouq(graphs, embedder, args.node_weight,
                             args.edge_weight, args.normalize_ged)
        uq["precomputed_topologies"] = precomputed_count
        output_indices = selected if args.response_index < 0 else (
            [args.response_index] if args.response_index < len(responses) else [])
        if args.verbose:
            print(f"[item {item_count}] id={item_id}, graphs={len(graphs)}, "
                  f"Topo-UQ={uq['topouq_variance']:.6f}, "
                  f"mean Reason-GED={uq['reason_ged_mean_pairwise']:.6f}", flush=True)
        for index in output_indices:
            outputs.append(build_output_row(item, item_id, gold, index, responses[index],
                                            selected, graphs, uq, args))

    if not outputs:
        raise ValueError("No output rows; input may be empty")
    directory = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".topouq-", suffix=".jsonl", dir=directory)
    os.close(descriptor)
    try:
        write_jsonl(temporary, outputs)
        os.replace(temporary, args.output_file)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Saved {len(outputs)} rows to {args.output_file}")


if __name__ == "__main__":
    main()
