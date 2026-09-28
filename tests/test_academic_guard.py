"""Mechanical academic guards: language hint vocabulary and markup skeleton checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service" / "scripts"))

from academic_guard import academic_error, language_hint, structure_error


def test_research_vocabulary_counts_as_english():
    assert language_hint("Calibration, validation, variance, residuals.") == "en"
    assert language_hint("Gridded raster dataset aggregation.") == "en"


@pytest.mark.parametrize(
    "text",
    [
        "The out of sample fit",
        "the out-of-sample fit",
        "an order of magnitude",
        "equal by construction",
        "the ground truth",
    ],
)
def test_multi_word_terms_are_matched_as_phrases(text):
    # One function word plus the phrase: enough only when the phrase counts.
    assert language_hint(text + " and the rest.") == "en"


def test_et_al_alone_does_not_outweigh_portuguese():
    text = "O estudo de Silva et al. não é conclusivo, mas a análise permanece em aberto."
    assert language_hint(text) == "pt"


def test_academic_error_catches_a_translated_rewrite():
    original = "The estimate remains within the threshold of the baseline error."
    rewritten = "A estimativa permanece dentro do limiar do erro de referência em uma análise."
    assert academic_error(original, rewritten) == "language changed"


def test_typst_structure_compares_strong_and_emphasis_markers():
    assert structure_error("a *b* _c_", "x *y* _z_", "typst") is None
    assert structure_error("a *b* c", "a b c", "typst") == "Typst markup changed"
    assert structure_error("\\emph{x}", "x") == "LaTeX structure changed"
    assert academic_error("A *strong* claim.", "A strong claim.", "typst") == "Typst markup changed"
