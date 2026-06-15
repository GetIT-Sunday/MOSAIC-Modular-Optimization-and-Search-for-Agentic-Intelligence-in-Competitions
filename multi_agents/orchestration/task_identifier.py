from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .profile import CompetitionProfile, load_profiles, profile_file_match_score


@dataclass(frozen=True)
class CompetitionSignal:
    competition_name: str
    files: List[str]
    overview_text: str = ""


@dataclass(frozen=True)
class ProfileDecision:
    profile_name: str
    confidence: float
    reasons: List[str]


BIO_KEYWORDS = {
    "protein",
    "sequence",
    "fasta",
    "ontology",
    "gene ontology",
    "go term",
    "amino acid",
    "multilabel",
    "multi-label",
}

TABULAR_KEYWORDS = {
    "tabular",
    "csv",
    "classification",
    "regression",
    "train.csv",
    "test.csv",
}


def keyword_score(text: str, keywords: Iterable[str]) -> float:
    text_lower = text.lower()
    hits = sum(1 for keyword in keywords if keyword in text_lower)
    return hits / max(len(set(keywords)), 1)


def identify_profile(
    signal: CompetitionSignal,
    profiles: Optional[List[CompetitionProfile]] = None,
) -> ProfileDecision:
    profiles = profiles or load_profiles()
    best_name = "tabular_classic"
    best_score = -1.0
    best_reasons: List[str] = []

    for profile in profiles:
        file_score = profile_file_match_score(profile, signal.files)
        text_score = 0.0
        reasons = [f"file_match={file_score:.2f}"]

        if profile.name == "bio_sequence_multilabel":
            text_score = keyword_score(signal.overview_text, BIO_KEYWORDS)
            fasta_files = [file for file in signal.files if file.endswith((".fasta", ".fa"))]
            if fasta_files:
                text_score += 0.25
                reasons.append("fasta_files_present")
            if not fasta_files and text_score == 0:
                file_score = 0.0
                reasons.append("bio_evidence_absent")
        elif profile.name == "tabular_classic":
            text_score = keyword_score(signal.overview_text, TABULAR_KEYWORDS)
            csv_files = [file for file in signal.files if Path(file).suffix == ".csv"]
            if len(csv_files) >= 3:
                text_score += 0.25
                reasons.append("multiple_csv_files_present")

        text_score = min(1.0, text_score)
        score = min(1.0, 0.65 * file_score + 0.35 * text_score)
        reasons.append(f"text_match={text_score:.2f}")
        if score > best_score:
            best_score = score
            best_name = profile.name
            best_reasons = reasons

    return ProfileDecision(best_name, round(best_score, 3), best_reasons)
