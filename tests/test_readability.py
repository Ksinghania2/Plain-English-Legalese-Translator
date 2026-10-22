from legalese.readability import flesch_kincaid_grade, flesch_reading_ease, count_syllables
from legalese.tokenizer import split_sentences, count_words


def test_syllables():
    assert count_syllables("hello") >= 1
    assert count_syllables("a") == 1


def test_grade_level():
    g = flesch_kincaid_grade("The cat sat on the mat. The dog ran fast.")
    assert g >= 0


def test_reading_ease_range():
    e = flesch_reading_ease("The cat sat on the mat.")
    assert 0 <= e <= 100


def test_split_sentences():
    spans = split_sentences("First sentence. Second sentence!")
    assert len(spans) == 2


def test_count_words():
    assert count_words("one two three") == 3
