"""Evaluation metrics for handwriting OCR: Character Error Rate (CER) and Word Error Rate (WER)."""

from typing import List, Sequence, Tuple


def levenshtein_distance(seq1: Sequence, seq2: Sequence) -> int:
    """Compute standard minimum edit distance (Levenshtein) between two sequences."""
    n1, n2 = len(seq1), len(seq2)
    dp = list(range(n2 + 1))

    for i in range(1, n1 + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n2 + 1):
            temp = dp[j]
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp

    return dp[n2]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) = Levenshtein(chars) / max(1, len(ref))."""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    dist = levenshtein_distance(ref_chars, hyp_chars)
    return float(dist) / float(max(1, len(ref_chars)))


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) = Levenshtein(words) / max(1, len(ref_words))."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = levenshtein_distance(ref_words, hyp_words)
    return float(dist) / float(len(ref_words))


def compute_corpus_metrics(
    references: List[str],
    hypotheses: List[str],
) -> Tuple[float, float]:
    """Compute macro-averaged corpus CER and WER."""
    if not references:
        return 0.0, 0.0

    total_cer = sum(compute_cer(r, h) for r, h in zip(references, hypotheses))
    total_wer = sum(compute_wer(r, h) for r, h in zip(references, hypotheses))
    n = len(references)
    return round(total_cer / n, 4), round(total_wer / n, 4)
