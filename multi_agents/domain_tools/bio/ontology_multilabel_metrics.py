from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Set


TermSetByItem = Mapping[str, Set[str]]
ScoresByItem = Mapping[str, Mapping[str, float]]
ParentsByTerm = Mapping[str, Set[str]]


@dataclass(frozen=True)
class FmaxResult:
    fmax: float
    threshold: float
    precision: float
    recall: float


def ancestors(term: str, parents_by_term: ParentsByTerm) -> Set[str]:
    """Return all ancestors of a term using a parent adjacency map."""
    seen: Set[str] = set()
    stack = list(parents_by_term.get(term, set()))
    while stack:
        parent = stack.pop()
        if parent in seen:
            continue
        seen.add(parent)
        stack.extend(parents_by_term.get(parent, set()))
    return seen


def close_terms(terms: Iterable[str], parents_by_term: ParentsByTerm) -> Set[str]:
    """Apply ontology ancestor closure to a set of labels."""
    closed: Set[str] = set()
    for term in terms:
        closed.add(term)
        closed.update(ancestors(term, parents_by_term))
    return closed


def close_truth(
    truth: TermSetByItem,
    parents_by_term: Optional[ParentsByTerm] = None,
) -> Dict[str, Set[str]]:
    if not parents_by_term:
        return {item: set(terms) for item, terms in truth.items()}
    return {item: close_terms(terms, parents_by_term) for item, terms in truth.items()}


def close_scores(
    scores: ScoresByItem,
    parents_by_term: Optional[ParentsByTerm] = None,
) -> Dict[str, Dict[str, float]]:
    """Propagate prediction scores to ancestor labels using max child score."""
    if not parents_by_term:
        return {
            item: {term: float(score) for term, score in term_scores.items()}
            for item, term_scores in scores.items()
        }

    closed: Dict[str, Dict[str, float]] = {}
    for item, term_scores in scores.items():
        propagated: Dict[str, float] = {}
        for term, raw_score in term_scores.items():
            score = float(raw_score)
            propagated[term] = max(propagated.get(term, 0.0), score)
            for parent in ancestors(term, parents_by_term):
                propagated[parent] = max(propagated.get(parent, 0.0), score)
        closed[item] = propagated
    return closed


def filter_terms(
    values: Mapping[str, Mapping[str, float] | Set[str]],
    allowed_terms: Optional[Set[str]],
):
    if allowed_terms is None:
        return values
    filtered = {}
    for item, terms in values.items():
        if isinstance(terms, set):
            filtered[item] = {term for term in terms if term in allowed_terms}
        else:
            filtered[item] = {
                term: score for term, score in terms.items() if term in allowed_terms
            }
    return filtered


def item_centric_fmax(
    truth: TermSetByItem,
    scores: ScoresByItem,
    parents_by_term: Optional[ParentsByTerm] = None,
    allowed_terms: Optional[Set[str]] = None,
    threshold_count: int = 101,
) -> FmaxResult:
    """Compute an item-centric Fmax validation proxy for sparse multi-label tasks."""
    if threshold_count < 2:
        raise ValueError("threshold_count must be at least 2")

    closed_truth = close_truth(truth, parents_by_term)
    closed_scores = close_scores(scores, parents_by_term)
    closed_truth = filter_terms(closed_truth, allowed_terms)
    closed_scores = filter_terms(closed_scores, allowed_terms)

    items = sorted(set(closed_truth) | set(closed_scores))
    thresholds = [i / (threshold_count - 1) for i in range(threshold_count)]
    best = FmaxResult(fmax=0.0, threshold=0.0, precision=0.0, recall=0.0)

    for threshold in thresholds:
        precisions = []
        recalls = []
        for item in items:
            true_terms = set(closed_truth.get(item, set()))
            pred_terms = {
                term
                for term, score in closed_scores.get(item, {}).items()
                if score >= threshold
            }
            if not true_terms and not pred_terms:
                continue
            precisions.append(len(true_terms & pred_terms) / len(pred_terms) if pred_terms else 0.0)
            recalls.append(len(true_terms & pred_terms) / len(true_terms) if true_terms else 0.0)

        precision = sum(precisions) / len(precisions) if precisions else 0.0
        recall = sum(recalls) / len(recalls) if recalls else 0.0
        fscore = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        if fscore > best.fmax:
            best = FmaxResult(fscore, threshold, precision, recall)

    return best


def mean_namespace_fmax(
    truth: TermSetByItem,
    scores: ScoresByItem,
    namespace_terms: Mapping[str, Set[str]],
    parents_by_term: Optional[ParentsByTerm] = None,
    threshold_count: int = 101,
) -> Dict[str, FmaxResult | float]:
    """Compute one Fmax per namespace plus their arithmetic mean."""
    results: Dict[str, FmaxResult | float] = {}
    values = []
    for namespace, allowed_terms in namespace_terms.items():
        result = item_centric_fmax(
            truth=truth,
            scores=scores,
            parents_by_term=parents_by_term,
            allowed_terms=allowed_terms,
            threshold_count=threshold_count,
        )
        results[namespace] = result
        values.append(result.fmax)
    results["mean_fmax"] = sum(values) / len(values) if values else 0.0
    return results


def frequency_baseline(
    truth: TermSetByItem,
    target_items: Iterable[str],
    top_k: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """Assign global label-frequency scores to every target item."""
    counts = defaultdict(int)
    for terms in truth.values():
        for term in terms:
            counts[term] += 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_k]
    if not ranked:
        return {item: {} for item in target_items}
    max_count = ranked[0][1]
    term_scores = {term: count / max_count for term, count in ranked}
    return {item: dict(term_scores) for item in target_items}


protein_centric_fmax = item_centric_fmax
