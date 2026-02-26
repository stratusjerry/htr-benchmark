import jiwer


def normalize_text(text: str) -> str:
    """Lowercase, strip, and collapse whitespace."""
    return " ".join(text.lower().split())


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    if not reference:
        return 1.0 if hypothesis else 0.0
    return jiwer.cer(reference, hypothesis)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    if not reference:
        return 1.0 if hypothesis else 0.0
    return jiwer.wer(reference, hypothesis)


def evaluate(reference: str, hypothesis: str) -> dict:
    """Evaluate a transcription against ground truth.

    Returns dict with: cer, wer, ref_char_count, hyp_char_count,
    ref_word_count, hyp_word_count.
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    return {
        "cer": round(compute_cer(ref_norm, hyp_norm), 4),
        "wer": round(compute_wer(ref_norm, hyp_norm), 4),
        "ref_char_count": len(ref_norm),
        "hyp_char_count": len(hyp_norm),
        "ref_word_count": len(ref_norm.split()),
        "hyp_word_count": len(hyp_norm.split()),
    }
