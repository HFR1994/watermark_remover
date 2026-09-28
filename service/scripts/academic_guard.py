"""Conservative mechanical guards; these do not prove scientific equivalence."""

import re
from collections import Counter

_PORTUGUESE_WORDS = frozenset(
    {
        "o",
        "os",
        "uma",
        "um",
        "de",
        "da",
        "das",
        "dos",
        "em",
        "não",
        "para",
        "com",
        "por",
        "é",
        "são",
        "mas",
        "neste",
        "esta",
        "essa",
        "que",
        "ao",
        "no",
        "na",
        "do",
        "permanece",
        "estudo",
        "análise",
        "relação",
        "hipóteses",
        "parâmetro",
        "indeterminado",
    }
)

_ENGLISH_WORDS = frozenset(
    {
        # function words
        "the",
        "this",
        "that",
        "these",
        "those",
        "which",
        "with",
        "within",
        "from",
        "their",
        "is",
        "are",
        "was",
        "were",
        "an",
        "but",
        "and",
        # general research vocabulary
        "remains",
        "study",
        "analysis",
        "relationship",
        "assumption",
        "assumptions",
        "parameter",
        "parameters",
        "indeterminate",
        "methodology",
        "method",
        "methods",
        "approach",
        "approaches",
        "framework",
        "procedure",
        "protocol",
        "criterion",
        "threshold",
        "scope",
        "limitation",
        "limitations",
        "literature",
        "evidence",
        "finding",
        "outcome",
        "derivation",
        "derive",
        "derived",
        "definition",
        "define",
        "defined",
        "denote",
        "notation",
        "implies",
        "yields",
        "equivalent",
        "plausible",
        "empirically",
        "diagnosis",
        "diagnostic",
        "subsequent",
        # statistics and validation
        "estimate",
        "estimates",
        "variance",
        "deviation",
        "coefficient",
        "correlation",
        "distribution",
        "sample",
        "pooled",
        "mean",
        "median",
        "percentile",
        "ratio",
        "ratios",
        "proportion",
        "proportional",
        "residual",
        "residuals",
        "outlier",
        "fitted",
        "bias",
        "error",
        "errors",
        "accuracy",
        "precision",
        "metric",
        "metrics",
        "statistic",
        "statistical",
        "sensitivity",
        "calibration",
        "calibrated",
        "validation",
        "validated",
        "baseline",
        "benchmark",
        "prior",
        "priors",
        "predictor",
        "observed",
        "observation",
        "observations",
        "measured",
        "discrepancy",
        "agreement",
        "consistent",
        "consistency",
        "reproducible",
        "reproducibility",
        "margin",
        "constraint",
        "constraints",
        "constant",
        "independent",
        "heterogeneous",
        "homogeneous",
        "uniform",
        "uniformly",
        "linear",
        "additive",
        "decomposition",
        "exponent",
        "magnitude",
        "iterative",
        "convergence",
        "converges",
        "interpolation",
        "extrapolation",
        "normalised",
        "normalisation",
        "weighted",
        "weighting",
        "scaling",
        "rescaled",
        "rescaling",
        "saturate",
        "saturation",
        "noise",
        "signal",
        "sparse",
        # spatial and data terms
        "spatial",
        "temporal",
        "resolution",
        "coarse",
        "coarser",
        "finer",
        "aggregate",
        "aggregated",
        "aggregation",
        "disaggregation",
        "disaggregated",
        "allocation",
        "attribution",
        "gridded",
        "raster",
        "rasterised",
        "centroid",
        "proxy",
        "proxies",
        "dataset",
        "datasets",
        "harmonised",
        "harmonisation",
        "compute",
        "computed",
    }
)

# Multi-word terms, matched as phrases; a hyphenated spelling counts too
# ("out-of-sample", "order-of-magnitude").
_ENGLISH_PHRASES = re.compile(
    r"\b(?:"
    + "|".join(
        r"[\s-]+".join(map(re.escape, phrase.split()))
        for phrase in (
            "out of sample",
            "ground truth",
            "by construction",
            "order of magnitude",
            "et al",
        )
    )
    + r")\b"
)


def language_hint(text: str) -> str | None:
    """Detect only clear Portuguese/English evidence; return None when uncertain."""
    lowered = text.lower()
    words = Counter(re.findall(r"[^\W\d_]+", lowered))
    pt = sum(words[word] for word in _PORTUGUESE_WORDS)
    en = sum(words[word] for word in _ENGLISH_WORDS) + len(_ENGLISH_PHRASES.findall(lowered))
    if pt >= 3 and pt >= 2 * en + 1:
        return "pt"
    if en >= 3 and en >= 2 * pt + 1:
        return "en"
    return None


_CONTROLS = r"\\[A-Za-z@]+\*?|\\[^A-Za-z]|[{}]"
# Typst markup left in prose once code, math and references are masked: the
# strong and emphasis toggles.
_TYPST_CONTROLS = r"(?<!\\)[*_]"


def structure_error(original: str, candidate: str, syntax: str = "latex") -> str | None:
    """Reject a rewrite whose markup skeleton differs from the input's.

    Both texts are expected with their math spans already replaced by
    placeholders; what is compared is the markup left in the prose, in order:
    for LaTeX the commands and braces (``\\section``, ``\\emph``, ``{``, ``}``),
    for Typst the strong and emphasis toggles (``*``, ``_``).
    """
    if syntax == "typst":
        if re.findall(_TYPST_CONTROLS, original) != re.findall(_TYPST_CONTROLS, candidate):
            return "Typst markup changed"
        return None
    if re.findall(_CONTROLS, original) != re.findall(_CONTROLS, candidate):
        return "LaTeX structure changed"
    return None


def academic_error(original: str, candidate: str, syntax: str = "latex") -> str | None:
    """Compare text with mathematical spans already replaced by placeholders."""
    source_language, target_language = language_hint(original), language_hint(candidate)
    if source_language and target_language and source_language != target_language:
        return "language changed"
    structure = structure_error(original, candidate, syntax)
    if structure:
        return structure
    numbers = r"(?<!\w)\d+(?:[.,]\d+)*"
    if Counter(re.findall(numbers, original)) != Counter(re.findall(numbers, candidate)):
        return "numbers changed"
    return None
