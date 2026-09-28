#!/usr/bin/env python3
"""Protect mathematics, Typst code, references and raw spans from Layer B.

The Layer B rewrite hands prose to a generative model, which is free to reword
anything it is given. In academic sources that freedom is a hazard rather than a
feature: an index, a sign, a label, a citation key or a function call carries no
token-sampling watermark worth attacking, and a model that "improves" one
corrupts the document silently.

This module lifts those spans out of the text before the rewrite and puts them
back afterwards. Each protected span becomes an opaque placeholder
(``[[WMX0001]]``) that survives a paraphrase, a masked-LM infill (the token
carries digits, so ``_mlm_infill`` never selects it) and the deterministic
humanizer pass (which would otherwise turn a Typst en dash ``--`` into a comma).

What is protected, in the forms the thesis uses:

* math, inline and display: ``$rho(i) in [0, 100]$``,
  ``$ D_(c,s) (i, h) = ... $ <eq:product-form>``;
* raw text: fenced blocks (```` ```python ... ``` ````) and inline code
  (```O4630```, ```FC_TRA_ROAD_E```);
* comments (``// ...`` and ``/* ... */``) and bare URLs;
* references and labels: ``@sec:gibs``, ``@tab:gibs-layers``,
  ``@eq:mass-balance``, ``@app:consistency``, ``@fig:four-moves``,
  ``@horsch2018[p. 3]``, ``<sec:decoding>``;
* code introduced by ``#``: keywords (``#import``, ``#include``, ``#let``,
  ``#set``, ``#show``) to the end of the statement, symbols
  (``#sym.circle.filled``, ``#sym.arrow.r``, ``#sym.checkmark``, ``#sym.dot``,
  ``#sym.times``) and the head and arguments of every call (``#chapter(short:
  "...")``, ``#es("nrg_bal_c")``, ``#entsoe-load``, ``#ref(<sec:clouds>,
  supplement: none)``, ``#figure(...)``, ``#v(1em)``);
* markup that a rewrite tends to drop or "fix": heading markers (``==``),
  escapes (``\\$``, ``\\#``), forced line breaks (`` \\``), the non-breaking
  space ``~``, the en/em dash shorthands ``--`` / ``---`` (``2012--2024``,
  ``Eurostat--ENTSO-E``) and the typographic signs the thesis writes directly
  in prose (the minus sign U+2212, the multiplication sign U+00D7, ``±``, ``°``,
  ``²``, ``†``).

Prose inside code is deliberately left open, the way the LaTeX mask leaves
``abstract`` or ``theorem`` open: the trailing content block of a call
(``#chapter(...)[Data and study area]``, ``#note[...]``, ``#emph[...]``,
``#link(url)[text]``, ``#align(center)[...]``) and the ``caption:`` of a figure
are handed back to the rewrite, and the masker runs again inside them so their
own math and references stay protected. Their brackets are protected on their
own, so a model cannot unbalance the call. Layout calls listed in
``PROTECTED_CALLS`` (``table``, ``grid``, ...) are protected whole, content
included: rewording a cell breaks the alignment.

Restoration is reported, never assumed: a model can drop or duplicate a
placeholder, and ``restore_typst`` counts both instead of pretending the round
trip was lossless.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Calls protected whole, trailing content blocks included: their content is
#: layout or data rather than prose.
PROTECTED_CALLS: frozenset[str] = frozenset(
    {
        "table",
        "table.header",
        "table.footer",
        "table.cell",
        "table.hline",
        "table.vline",
        "grid",
        "grid.cell",
        "image",
        "raw",
        "math.equation",
        "place",
        "stack",
        "bibliography",
        "outline",
    }
)

#: Named arguments whose content block is prose, reopened to the rewrite even
#: inside a protected call (``#figure(table(...), caption: [...])``).
PROSE_ARGUMENTS: frozenset[str] = frozenset({"caption"})

#: Keywords whose statement is protected up to the end of the line (or of the
#: bracket it opens, for multi-line ``#let`` definitions).
STATEMENT_KEYWORDS: frozenset[str] = frozenset(
    {"import", "include", "let", "set", "show", "if", "else", "for", "while", "return"}
)

#: Accepted values for the protection mode (CLI ``--protect-typst``, `/clean`
#: option ``protect_typst``, env ``WATERMARKS_PROTECT_TYPST``).
PROTECT_TYPST_MODES = frozenset({"auto", "on", "off"})

_DEFAULT_PREFIX = "WMX"

_IDENT = r"[A-Za-z_][A-Za-z0-9_-]*"
# Label and reference names: ``sec:gibs``, ``tab:inventory-jrc``. A trailing
# ``.`` or ``:`` ends the sentence, not the name (Typst reads it the same way).
_NAME = r"[A-Za-z_](?:[\w-]|[:.](?=[\w-]))*"

# A cheap "is there anything here worth protecting" probe for mode="auto".
_TYPST_HINT_RE = re.compile(
    r"(?<!\\)#[A-Za-z_{\[(\"]"
    r"|(?<!\\)\$"
    r"|(?<![\\\w])@" + _NAME + r"|(?<![\\\w])<" + _NAME + r">"
    r"|^[ \t]*=+[ \t]"
    r"|`",
    re.M,
)

_PATTERN = re.compile(
    # Raw first: a raw block can contain anything, including $, # and //.
    r"(?P<fence>(?P<tick>`{3,})[\s\S]*?(?P=tick))"
    r"|(?P<code>(?<!\\)`[^`]+`)"
    r"|(?P<blockcomment>/\*[\s\S]*?\*/)"
    # A bare URL is a link in Typst markup; its ``//`` is not a comment.
    r"|(?P<url>https?://[^\s<>\[\]()\"]+)"
    r"|(?P<comment>(?<![:\\])//[^\n]*)"
    # Math has no prose to offer and may span lines ($ ... $ display blocks).
    r"|(?P<math>(?<!\\)\$(?:\\[\s\S]|[^$\\])*?(?<!\\)\$)"
    r"|(?P<hash>(?<!\\)#(?=[A-Za-z_{\[(\"]))"
    # A reference keeps its supplement: ``@horsch2018[p. 3]``.
    r"|(?P<ref>(?<![\\\w])@" + _NAME + r"(?:\[[^\]\n]*\])?)"
    r"|(?P<label>(?<![\\\w])<" + _NAME + r">)"
    r"|(?P<heading>^[ \t]*=+(?=[ \t]))"
    r"|(?P<linebreak>(?<=[ \t])\\(?=[ \t]*$))"
    r"|(?P<escape>\\(?:u\{[0-9A-Fa-f]+\}|[^\sA-Za-z0-9]))"
    r"|(?P<dash>(?<!-)---?(?!-))"
    r"|(?P<nbsp>(?<=\S)~(?=\S))"
    r"|(?P<sign>[\u2212\u00d7±°²†])",
    re.M,
)


@dataclass(frozen=True)
class TypstMask:
    """The spans lifted out of a text, and the token naming used for them."""

    spans: tuple[str, ...] = ()
    prefix: str = _DEFAULT_PREFIX

    @property
    def count(self) -> int:
        """Number of protected spans."""
        return len(self.spans)

    def token(self, index: int) -> str:
        """The placeholder standing for span *index*."""
        return f"[[{self.prefix}{index:04d}]]"

    @property
    def token_re(self) -> re.Pattern[str]:
        """Pattern matching this mask's placeholders."""
        return re.compile(rf"\[\[{re.escape(self.prefix)}(\d{{4,}})\]\]")


# --- a small Typst scanner -------------------------------------------------
#
# Regexes cannot pair brackets, so code after ``#`` is walked by hand. The
# scanner returns *pieces*: ("mask", start, end) for code, ("open", start, end)
# for prose content handed back to the rewrite.

_CLOSING = {"(": ")", "[": "]", "{": "}"}


def _skip_string(text: str, pos: int) -> int:
    """Position just after the string literal opening at *pos*."""
    pos += 1
    while pos < len(text):
        if text[pos] == "\\":
            pos += 2
            continue
        if text[pos] == '"':
            return pos + 1
        pos += 1
    raise ValueError("Unterminated Typst string; review the source first")


def _balanced_end(text: str, pos: int) -> int:
    """Position just after the bracket opening at *pos*, strings and raw aware."""
    stack = [_CLOSING[text[pos]]]
    pos += 1
    while pos < len(text) and stack:
        char = text[pos]
        if char == "\\":
            pos += 2
            continue
        # Strings only exist in code; inside content ``"`` is a plain quote.
        if char == '"' and stack[-1] != "]":
            pos = _skip_string(text, pos)
            continue
        if char == "`" and stack[-1] == "]":
            fence = re.match(r"`{3,}|`", text[pos:])
            close = text.find(fence.group(0), pos + len(fence.group(0)))
            pos = len(text) if close == -1 else close + len(fence.group(0))
            continue
        # Math inside content may hold unpaired brackets: ``$[ell, u)$``.
        if char == "$" and stack[-1] == "]":
            math = _PATTERN.match(text, pos)
            if math and math.lastgroup == "math":
                pos = math.end()
                continue
        if text.startswith("//", pos) and text[pos - 1] != ":":
            newline = text.find("\n", pos)
            pos = len(text) if newline == -1 else newline
            continue
        if char in _CLOSING:
            stack.append(_CLOSING[char])
        elif char == stack[-1]:
            stack.pop()
        elif char in ")]}" and stack[-1] != "]":
            raise ValueError("Unbalanced Typst brackets; review the source first")
        pos += 1
    if stack:
        raise ValueError("Unbalanced Typst brackets; review the source first")
    return pos


def _argument_holes(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Inner ranges of ``caption: [...]`` arguments directly inside ``(...)``."""
    holes: list[tuple[int, int]] = []
    pos = start + 1
    while pos < end - 1:
        char = text[pos]
        if char == '"':
            pos = _skip_string(text, pos)
            continue
        if char in _CLOSING:
            match = re.compile(r"(" + _IDENT + r")\s*:\s*$").search(text, start + 1, pos)
            close = _balanced_end(text, pos)
            if char == "[" and match and match.group(1) in PROSE_ARGUMENTS:
                holes.append((pos + 1, close - 1))
            pos = close
            continue
        pos += 1
    return holes


def _statement_end(text: str, pos: int) -> int:
    """End of a keyword statement: the end of its line, brackets balanced."""
    while pos < len(text) and text[pos] != "\n":
        if text[pos] == '"':
            pos = _skip_string(text, pos)
        elif text[pos] in _CLOSING:
            pos = _balanced_end(text, pos)
        else:
            pos += 1
    return pos


def _hash_pieces(text: str, start: int, extra_calls: frozenset[str]) -> list[tuple[str, int, int]]:
    """Split the code expression at ``text[start] == '#'`` into mask/open pieces."""
    pos = start + 1
    if text[pos] == '"':
        return [("mask", start, _skip_string(text, pos))]
    if text[pos] in "({":
        return [("mask", start, _balanced_end(text, pos))]
    if text[pos] == "[":
        close = _balanced_end(text, pos)
        return [("mask", start, pos + 1), ("open", pos + 1, close - 1), ("mask", close - 1, close)]

    name = re.compile(_IDENT).match(text, pos)
    pos = name.end()
    if name.group(0) in STATEMENT_KEYWORDS:
        return [("mask", start, _statement_end(text, pos))]

    # Field access: ``#sym.circle.filled``, ``#table.header``. A dot followed
    # by anything but a name ends the sentence.
    field = re.compile(r"(?:\." + _IDENT + r")*").match(text, pos)
    callee = text[name.start() : field.end()]
    pos = field.end()
    whole = callee in PROTECTED_CALLS or callee in extra_calls

    pieces: list[tuple[str, int, int]] = []
    head = start
    while pos < len(text) and text[pos] in "([":
        close = _balanced_end(text, pos)
        if text[pos] == "(" and not whole:
            for hole_start, hole_end in _argument_holes(text, pos, close):
                pieces.append(("mask", head, hole_start))
                pieces.append(("open", hole_start, hole_end))
                head = hole_end
        elif text[pos] == "[" and not whole:
            pieces.append(("mask", head, pos + 1))
            pieces.append(("open", pos + 1, close - 1))
            head = close - 1
        pos = close
        # Method calls chain on: ``#x.at(0)``.
        more = re.compile(r"\." + _IDENT + r"(?=\()").match(text, pos)
        if more:
            pos = more.end()
    pieces.append(("mask", head, pos))
    return [piece for piece in pieces if piece[0] == "open" or piece[1] < piece[2]]


# --- public API ------------------------------------------------------------


def looks_like_typst(text: str) -> bool:
    """True when *text* carries math, Typst code, references or raw text."""
    return bool(_TYPST_HINT_RE.search(text))


def _pick_prefix(text: str) -> str:
    """A placeholder prefix that does not already occur in *text*."""
    prefix = _DEFAULT_PREFIX
    while re.search(rf"\[\[{re.escape(prefix)}\d", text):
        prefix += "X"
    return prefix


def mask_typst(
    text: str,
    *,
    mode: str = "auto",
    extra_calls: tuple[str, ...] = (),
) -> tuple[str, TypstMask]:
    """Replace protected spans in *text* with placeholders.

    ``mode`` is ``"auto"`` (protect when the text looks like Typst),
    ``"on"`` (always) or ``"off"`` (never). ``extra_calls`` adds function names
    to protect whole, like ``PROTECTED_CALLS``. Returns the masked text and the
    mask needed to restore it; an unprotected text comes back unchanged with an
    empty mask, so callers run the same code path either way.
    """
    if mode not in PROTECT_TYPST_MODES:
        raise ValueError(f"unknown protect mode {mode!r}; expected auto|on|off")
    if mode == "off" or not text:
        return text, TypstMask()
    if mode == "auto" and not looks_like_typst(text):
        return text, TypstMask()

    prefix = _pick_prefix(text)
    naming = TypstMask(prefix=prefix)
    calls = frozenset(extra_calls)
    spans: list[str] = []
    parts: list[str] = []

    def keep(start: int, end: int) -> None:
        if start < end:
            parts.append(text[start:end])

    def protect(start: int, end: int) -> None:
        # Adjacent protected pieces share one placeholder.
        if parts and spans and parts[-1] == naming.token(len(spans) - 1):
            spans[-1] += text[start:end]
            return
        spans.append(text[start:end])
        parts.append(naming.token(len(spans) - 1))

    def walk(start: int, end: int) -> None:
        cursor = start
        while cursor < end:
            match = _PATTERN.search(text, cursor, end)
            if not match:
                break
            keep(cursor, match.start())
            if match.lastgroup == "hash":
                pieces = _hash_pieces(text, match.start(), calls)
            else:
                pieces = [("mask", match.start(), match.end())]
            for kind, piece_start, piece_end in pieces:
                if kind == "mask":
                    protect(piece_start, piece_end)
                else:
                    walk(piece_start, piece_end)
            cursor = pieces[-1][2]
            # A label or reference right after protected code joins its span:
            # ``$ ... $ <eq:mass-balance>`` stays one placeholder.
            if match.lastgroup in {"math", "hash"}:
                tail = re.compile(r"[ \t]*<" + _NAME + r">").match(text, cursor, end)
                if tail:
                    protect(cursor, tail.end())
                    cursor = tail.end()
        keep(cursor, end)

    walk(0, len(text))
    return "".join(parts), TypstMask(spans=tuple(spans), prefix=prefix)


def restore_typst(text: str, mask: TypstMask) -> tuple[str, dict[str, int]]:
    """Put the protected spans back, reporting what the round trip actually did.

    A rewrite model may drop a placeholder or emit it twice. Both are recorded
    (``missing`` / ``duplicated``) rather than silently absorbed, because a
    dropped placeholder means the output lost an equation or a reference.
    """
    stats = {"protected": mask.count, "restored": 0, "missing": 0, "duplicated": 0}
    if not mask.count:
        return text, stats

    seen: dict[int, int] = {}

    def _put_back(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if not 0 <= index < mask.count:
            # A token the model invented; leave it visible instead of guessing.
            return match.group(0)
        seen[index] = seen.get(index, 0) + 1
        return mask.spans[index]

    out = mask.token_re.sub(_put_back, text)
    stats["restored"] = len(seen)
    stats["missing"] = mask.count - len(seen)
    stats["duplicated"] = sum(n - 1 for n in seen.values())
    return out, stats


def markup_spans(text: str) -> list[tuple[str, int, int]]:
    """``("comment" | "code", start, end)`` for comments and ``#`` code, in order.

    Walks the text the way ``mask_typst`` does, so raw spans, math and URLs are
    never mistaken for comments, and a string inside code never ends it early.
    Content blocks handed back to prose are walked too, so a comment inside
    ``#note[...]`` is found; code is reported whole, its comments included.
    """
    spans: list[tuple[str, int, int]] = []

    def walk(start: int, end: int) -> None:
        cursor = start
        while cursor < end:
            match = _PATTERN.search(text, cursor, end)
            if not match:
                return
            if match.lastgroup in {"comment", "blockcomment"}:
                spans.append(("comment", match.start(), match.end()))
            if match.lastgroup != "hash":
                cursor = match.end()
                continue
            pieces = _hash_pieces(text, match.start(), frozenset())
            spans.append(("code", match.start(), pieces[-1][2]))
            for kind, piece_start, piece_end in pieces:
                if kind == "open":
                    walk(piece_start, piece_end)
            cursor = pieces[-1][2]

    walk(0, len(text))
    return spans


def call_arguments(text: str, open_pos: int) -> tuple[list[tuple[str | None, int, int]], int]:
    """Arguments of the call whose ``(`` is at *open_pos*, and the end of the call.

    Each argument is ``(name, start, end)``; the second value is the position
    just after the closing ``)``.

    *name* is the argument name for ``name: value`` and None for a positional
    argument; ``start``/``end`` delimit the whole argument, surrounding blanks
    excluded. Strings, nested brackets and comments never split an argument.
    """
    end_of_call = _balanced_end(text, open_pos)
    close = end_of_call - 1
    items: list[tuple[int, int]] = []
    head = pos = open_pos + 1
    while pos < close:
        char = text[pos]
        if char == '"':
            pos = _skip_string(text, pos)
        elif char in _CLOSING:
            pos = _balanced_end(text, pos)
        elif text.startswith("//", pos):
            newline = text.find("\n", pos)
            pos = close if newline == -1 else min(newline, close)
        elif text.startswith("/*", pos):
            end = text.find("*/", pos + 2)
            pos = close if end == -1 else min(end + 2, close)
        elif char == ",":
            items.append((head, pos))
            head = pos = pos + 1
        else:
            pos += 1
    items.append((head, close))

    arguments: list[tuple[str | None, int, int]] = []
    for item_start, item_end in items:
        start, end = item_start, item_end
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        named = re.compile(r"(" + _IDENT + r")\s*:(?!:)").match(text, start, end)
        arguments.append((named.group(1) if named else None, start, end))
    return arguments, end_of_call


def placeholder_error(text: str, mask: TypstMask) -> str | None:
    """Reject missing, duplicate, invented or reordered protected spans."""
    if not mask.count:
        return None
    indices = [int(m.group(1)) for m in mask.token_re.finditer(text)]
    if indices != list(range(mask.count)):
        return "protected math/Typst placeholders were lost, duplicated, invented or reordered"
    return None
