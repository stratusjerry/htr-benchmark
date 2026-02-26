from htr_benchmark.evaluate import compute_cer, compute_wer, evaluate, normalize_text


def test_normalize_text():
    assert normalize_text("  Hello   World  ") == "hello world"
    assert normalize_text("Line1\nLine2\tLine3") == "line1 line2 line3"
    assert normalize_text("") == ""


def test_compute_cer_identical():
    assert compute_cer("hello", "hello") == 0.0


def test_compute_cer_different():
    cer = compute_cer("hello", "helo")
    assert 0 < cer < 1


def test_compute_cer_empty_reference():
    assert compute_cer("", "something") == 1.0
    assert compute_cer("", "") == 0.0


def test_compute_wer_identical():
    assert compute_wer("hello world", "hello world") == 0.0


def test_compute_wer_different():
    wer = compute_wer("the cat sat", "the dog sat")
    assert 0 < wer < 1


def test_compute_wer_empty_reference():
    assert compute_wer("", "something") == 1.0
    assert compute_wer("", "") == 0.0


def test_evaluate_perfect_match():
    result = evaluate("Hello World", "hello world")
    assert result["cer"] == 0.0
    assert result["wer"] == 0.0


def test_evaluate_keys():
    result = evaluate("hello", "world")
    expected_keys = {"cer", "wer", "ref_char_count", "hyp_char_count", "ref_word_count", "hyp_word_count"}
    assert set(result.keys()) == expected_keys
