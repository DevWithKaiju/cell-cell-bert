"""
prompts.py

Prompt builders for decoder-based (LLM) evaluation of cell-cell relation
extraction, shared by both zero/few-shot inference (decoder_eval.py) and LoRA
SFT (sft.py).

Two prompt styles:
    - "fewshot" : instruction + 6 worked examples
    - "qa4re"   : QA4RE-style option selection

Each combines with two entity-presentation formats for the *target* sentence:
    - "replacement" : entities shown as [CELL0] / [CELL1]
    - "boundary"    : entities wrapped in <E0>name</E0> / <E1>name</E1>

Worked examples always use [CELL0]/[CELL1] notation regardless of presentation
(they illustrate structure, not facts), so the only difference between the two
presentation arms is how the target sentence itself is shown.

build_prompt() returns the user-turn body (a string); callers apply the chat
template themselves.
"""

from __future__ import annotations

PROMPT_STYLES = ("fewshot", "qa4re")
PRESENTATIONS = ("replacement", "boundary")

_MARKING_NOTE = {
    "replacement": (
        "In the input sentence, the two target cell entities are written as the "
        "placeholders `[CELL0]` and `[CELL1]`."
    ),
    "boundary": (
        "In the input sentence, the two target cell entities are wrapped in tags: "
        "the first as `<E0>...</E0>` and the second as `<E1>...</E1>`."
    ),
}

_LABEL_DEFINITION = """\
Decide whether the sentence states a **direct biological relationship** between the two target cell entities (call them A and B).

Answer `1` when the sentence describes any of the following:
  1. Interaction (A-B): non-directional, bidirectional, or mutual relationships such as
     physical binding, contact, mutual recognition, communication, or cross-regulation.
     This includes cases where the mere presence of one cell affects the other with no
     clear direction (e.g. interact, associate, bind, recognize, cross-talk).
  2. Regulation / control (A->B): directional influence where one cell regulates, induces,
     inhibits, promotes, mediates, or converts the other, including any stated effect on
     the other cell's behavior, differentiation, or activity (e.g. induce, promote,
     inhibit, mediate, regulate, convert, affect).
  3. Differentiation (A->A'): one cell type develops, transforms, or differentiates into
     the other (explicitly in vitro studies are included).

Answer `0` when there is **no direct biological relationship**, for example:
  - Marker / expression description only (e.g. "A expresses marker B", "A is a marker for B").
  - Methodological or experimental context only (both cells merely isolated, analyzed, or
     measured together, with no interaction described).
  - Indirect relationships mediated by a third entity.
  - The two cells have no described connection at all."""

_FEWSHOT_EXAMPLES = [
    ("Thus, [CELL0] from adult human pancreas can differentiate to [CELL1].", 1),
    (
        "We analyzed the expression profile of [CELL0], which is a well-known marker for [CELL1].",
        0,
    ),
    (
        "[CELL0] could differentiate into [CELL1] in vitro and have been shown to reconstitute the impaired myocardium in vivo.",
        1,
    ),
    (
        "Living tissue imaging provides evidence of interactions between differentiating [CELL0] and [CELL1].",
        1,
    ),
    (
        "The patient cohort showed a significant decrease in the number of circulating [CELL0] after treatment, while [CELL1] remained unchanged.",
        0,
    ),
    ("[CELL0] targeting [CELL1] were monitored by specific markers.", 1),
]


def _fewshot_prompt(sentence: str, presentation: str) -> str:
    lines = [
        "You are a biomedical expert.",
        "",
        _LABEL_DEFINITION,
        "",
        _MARKING_NOTE[presentation],
        "",
        "Output only the single digit `0` or `1`, with no explanation.",
        "",
        "### Examples",
    ]
    for ex_sentence, ex_label in _FEWSHOT_EXAMPLES:
        lines.append(f"Sentence: {ex_sentence}")
        lines.append(f"Output: {ex_label}")
    lines += ["", "### Input", f"Sentence: {sentence}", "", "### Output"]
    return "\n".join(lines)


_QA4RE_OPTIONS = (
    "1: A and B have a direct biological relationship "
    "(interaction, regulation/control, or differentiation).\n"
    "0: A and B have no direct biological relationship."
)

_QA4RE_EXAMPLES = [
    ("Thus, [CELL0] from adult human pancreas can differentiate to [CELL1].", 1),
    (
        "We analyzed the expression profile of [CELL0], which is a well-known marker for [CELL1].",
        0,
    ),
    (
        "Both [CELL0] and [CELL1] were isolated from peripheral blood using density gradient centrifugation.",
        0,
    ),
    ("[CELL0] targeting [CELL1] were monitored by specific markers.", 1),
]


def _qa4re_prompt(sentence: str, presentation: str) -> str:
    lines = [
        "You are a biomedical expert.",
        "",
        "Determine which option can be inferred from the given sentence.",
        "",
        _LABEL_DEFINITION,
        "",
        _MARKING_NOTE[presentation],
        "",
        "Answer with only the option number (`1` or `0`), without any explanation.",
        "",
        "### Examples",
    ]
    for ex_sentence, ex_label in _QA4RE_EXAMPLES:
        lines.append(f"Sentence: {ex_sentence}")
        lines.append("Options:")
        lines.append(_QA4RE_OPTIONS)
        lines.append("Which option can be inferred from the given sentence?")
        lines.append(f"Option: {ex_label}")
    lines += [
        "",
        "### Input",
        f"Sentence: {sentence}",
        "Options:",
        _QA4RE_OPTIONS,
        "Which option can be inferred from the given sentence?",
        "Option:",
    ]
    return "\n".join(lines)


def build_prompt(prompt_style: str, presentation: str, sentence: str) -> str:
    """Returns the user-turn body for (prompt_style, presentation)."""
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(
            f"Unknown prompt_style: {prompt_style!r}. Expected one of {PROMPT_STYLES}."
        )
    if presentation not in PRESENTATIONS:
        raise ValueError(
            f"Unknown presentation: {presentation!r}. Expected one of {PRESENTATIONS}."
        )
    return (
        _fewshot_prompt(sentence, presentation)
        if prompt_style == "fewshot"
        else _qa4re_prompt(sentence, presentation)
    )
