"""Typst span protection for Layer B: masking, restoration, prompts, rewrite paths."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service" / "scripts"))

import rewrite_text
from rewrite_text import apply_strategy, build_prompt, rewrite
from typst_mask import (
    TypstMask,
    call_arguments,
    looks_like_typst,
    markup_spans,
    mask_typst,
    placeholder_error,
    restore_typst,
)

THESIS = """#import "template.typ": *
#set document(title: "Load disaggregation")
= Data and study area <sec:data>
// a note to self
We model the load $rho(i) in [0, 100]$ as in @horsch2018[p. 3], see @sec:gibs.
#figure(table(columns: 2, [a], [b]), caption: [The share of $x$ per region, 2012--2024.]) <tab:x>
#note[This remains a *strong* assumption, see https://example.org/x.]
$ D_(c,s) (i, h) = sum_k w_k $ <eq:product-form>
The estimate is −1 × 10² at 5 °C, escaped \\$5 and a~b.
```python
print("$not math$")
```
"""

_LLM = {"backend": "ollama", "model": "m", "base_url": "http://127.0.0.1:11434", "api_key": None}


def _rewrite_kwargs(**overrides):
    kwargs = dict(
        tactic="academic",
        lang="French",
        original_lang="English",
        timeout=10,
        layer_a_after=False,
        temperature=0.9,
        candidates=1,
        syntax="typst",
        **_LLM,
    )
    kwargs.update(overrides)
    return kwargs


# --- masking ---------------------------------------------------------------


def test_roundtrip_restores_source_exactly():
    masked, mask = mask_typst(THESIS, mode="on")
    out, stats = restore_typst(masked, mask)
    assert out == THESIS
    assert stats["missing"] == stats["duplicated"] == 0
    assert placeholder_error(masked, mask) is None


def test_code_math_references_and_raw_leave_the_masked_text():
    masked, _mask = mask_typst(THESIS, mode="on")
    for span in (
        "#import",
        "#set document",
        "$rho(i) in [0, 100]$",
        "@horsch2018[p. 3]",
        "@sec:gibs",
        "<sec:data>",
        "<eq:product-form>",
        "table(columns",
        "https://example.org/x",
        "// a note to self",
        'print("$not math$")',
        "2012--2024",
    ):
        assert span not in masked, span


def test_prose_in_content_blocks_and_captions_stays_open():
    masked, _mask = mask_typst(THESIS, mode="on")
    assert "Data and study area" in masked
    assert "This remains a *strong* assumption" in masked
    assert "The share of" in masked and "per region, 2012" in masked
    assert "We model the load" in masked


def test_typographic_signs_escapes_and_nbsp_are_protected():
    masked, _mask = mask_typst("Down −1 × 3, costs \\$5, a~b and c \\\nnext", mode="on")
    for sign in ("−", "×", "\\$", "~"):
        assert sign not in masked
    assert "Down" in masked and "costs" in masked


def test_auto_mode_skips_plain_prose_and_off_never_masks():
    plain = "A plain sentence with no markup at all."
    assert not looks_like_typst(plain)
    assert mask_typst(plain) == (plain, TypstMask())
    assert mask_typst(THESIS, mode="off") == (THESIS, TypstMask())


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown protect mode"):
        mask_typst(THESIS, mode="sometimes")


def test_unbalanced_source_is_refused():
    with pytest.raises(ValueError, match="Unbalanced Typst brackets"):
        mask_typst("#note[never closed", mode="on")


def test_prefix_avoids_collision_with_existing_tokens():
    text = "Literal [[WMX0000]] and $x$."
    masked, mask = mask_typst(text, mode="on")
    assert mask.prefix != "WMX"
    assert restore_typst(masked, mask)[0] == text


def test_dropped_and_duplicated_placeholders_are_reported():
    masked, mask = mask_typst("See $x$ and @ref-a.", mode="on")
    _out, stats = restore_typst(masked.replace(mask.token(1), ""), mask)
    assert stats["missing"] == 1
    _out, stats = restore_typst(masked + mask.token(0), mask)
    assert stats["duplicated"] == 1
    assert "Typst placeholders" in placeholder_error(masked + mask.token(0), mask)


# --- scanner helpers used by the metadata cleaner ---------------------------


def test_markup_spans_skip_raw_urls_and_strings():
    text = (
        "Text // real comment\n"
        "```\n// inside raw\n```\n"
        "See https://a.b/c.\n"
        '#let s = "not // a comment"\n'
        "#note[inner // nested comment\n]\n"
        "/* block */\n"
    )
    comments = [text[a:b] for kind, a, b in markup_spans(text) if kind == "comment"]
    assert comments == ["// real comment", "// nested comment", "/* block */"]


def test_call_arguments_split_top_level_only():
    text = '#set document(title: "a, b", author: ("X", "Y"), keywords: [c, d], date: none)'
    open_pos = text.index("(")
    arguments, end = call_arguments(text, open_pos)
    assert end == len(text)
    assert [name for name, _s, _e in arguments] == ["title", "author", "keywords", "date"]
    assert text[arguments[1][1] : arguments[1][2]] == 'author: ("X", "Y")'


# --- prompts and rewrite paths ------------------------------------------------


def test_prompt_names_typst_spans_and_guard():
    _masked, mask = mask_typst("See $x$.", mode="on")
    prompt = build_prompt("academic", "See [[WMX0000]].", mask=mask)
    assert "Typst code, references or raw spans" in prompt
    guard = build_prompt("academic", "See $x$.", latex_guard=True, syntax="typst")
    assert "Typst code (anything introduced by #)" in guard
    assert guard.endswith("See $x$.")


def test_rewrite_restores_all_protected_typst_spans(monkeypatch):
    masked, mask = mask_typst(THESIS)
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: masked)
    out, info = rewrite(THESIS, **_rewrite_kwargs())
    assert out == THESIS
    assert info["typst_protected"] == mask.count
    assert info["typst"]["missing"] == 0
    assert "latex" not in info


def test_rewrite_rejects_lost_typst_span(monkeypatch):
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: "Prose only.")
    with pytest.raises(RuntimeError, match="protected math/Typst"):
        rewrite("We cite @smith2020 for $x$.", **_rewrite_kwargs())


def test_print_prompt_uses_typst_guard():
    out, info = rewrite(THESIS, **_rewrite_kwargs(backend="print-prompt", model=None))
    assert info["typst_protection"] == "prompt-guard"
    assert "Typst code (anything introduced by #)" in out


def test_apply_strategy_masks_typst_and_checks_markup(monkeypatch):
    prompts: list[str] = []

    def fake(base_url, model, prompt, *args, **kwargs):
        prompts.append(prompt)
        body = prompt.split("\n\n---\n", 1)[1]
        return body.replace("*strong*", "strong")  # drops the Typst strong markup

    monkeypatch.setattr(rewrite_text, "call_ollama", fake)
    source = "This remains a *strong* assumption for $x$ in @sec:a."
    out, stats = apply_strategy(source, [("academic", 0.3)], syntax="typst", **_LLM)
    assert out == source
    assert stats["ok"] is False and "Typst markup changed" in stats["errors"][0]
    assert stats["typst_protected"] == 2
    assert all("$x$" not in p and "@sec:a" not in p for p in prompts)


def test_cli_picks_typst_for_a_typ_path(tmp_path, capsys, monkeypatch):
    source = tmp_path / "t.typ"
    source.write_text("We model $x$ as in @smith2020.\n", encoding="utf-8")
    argv = ["rewrite_text.py", "--backend", "print-prompt", "--tactic", "academic", str(source)]
    monkeypatch.setattr(sys, "argv", argv)
    assert rewrite_text.main() == 0
    assert "Typst code (anything introduced by #)" in capsys.readouterr().out
