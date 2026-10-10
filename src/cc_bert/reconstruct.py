"""
reconstruct.py

Turns the published offset-only HuggingFace dataset
(DevWithKaiju/cell-cell-communication) back into the sentence-level CSVs the
training/eval code consumes:
    data/semmed/<seed>/{train,test}_{boundary,replacement}.csv  (id, sentence, label)
    data/pubmed/<seed>/test_{boundary,replacement}.csv          (id, sentence, label)
    data/cpt/relations_resolved.csv                              (id, pmid, sentence_plain,
                                                                   sentence_boundary, sentence_replacement)

Two markup representations are built directly from the published character
offsets:
    - "boundary"    : <E0>name</E0> / <E1>name</E1>
    - "replacement" : [CELL0] / [CELL1]

See scripts/01_reconstruct_dataset.py for the CLI that drives this module.
"""

import os

import pandas as pd
from datasets import load_dataset

from .utils import normalize_boundary_tags, normalize_replacement_tokens

HF_REPO_ID = "DevWithKaiju/cell-cell-communication"
PRESENTATIONS = ("boundary", "replacement")


def load_hf_configs(repo_id=HF_REPO_ID):
    """Loads all five dataset configs as pandas DataFrames."""
    config_names = [
        "pubmed_relations",
        "pubmed_seed_splits",
        "semmed_relations",
        "semmed_seed_splits",
        "cpt_relations",
    ]
    return {
        name: load_dataset(repo_id, name)["train"].to_pandas() for name in config_names
    }


def _parse_multi_span(start_val, end_val):
    """Parses the "12|45"-style pipe-separated multi-occurrence span encoding
    used when an entity mention is replaced more than once in the same
    sentence. Returns a list of (start, end) tuples."""
    if pd.isna(start_val) or start_val == "":
        return []
    starts = [int(x) for x in str(start_val).split("|")]
    ends = [int(x) for x in str(end_val).split("|")]
    if len(starts) != len(ends):
        raise ValueError(f"span start/end count mismatch: {start_val!r} vs {end_val!r}")
    return list(zip(starts, ends))


def _apply_markup(sentence, entity_0_spans, entity_1_spans, presentation):
    """Inserts boundary tags or replacement tokens at the given (sentence-
    relative) spans, processing right-to-left so earlier offsets stay valid."""
    spans = [(s, e, 0) for s, e in entity_0_spans] + [
        (s, e, 1) for s, e in entity_1_spans
    ]
    spans.sort(key=lambda x: x[0], reverse=True)
    out = sentence
    for s, e, which in spans:
        if presentation == "boundary":
            replacement = f"<E{which}>{out[s:e]}</E{which}>"
        elif presentation == "replacement":
            replacement = f"[CELL{which}]"
        else:
            raise ValueError(f"Unknown presentation: {presentation!r}")
        out = out[:s] + replacement + out[e:]
    return out


def _normalize(sentence, presentation):
    return (
        normalize_boundary_tags(sentence)
        if presentation == "boundary"
        else normalize_replacement_tokens(sentence)
    )


def reconstruct_pubmed_like_relations(df, abst_of, presentation):
    """Reconstructs sentence text for pubmed_relations/cpt_relations rows
    (entity offsets are relative to the sentence, sentence offsets are
    absolute into the PubMed abstract identified by `pmid`).

    Returns (df_out, report_rows). df_out has columns id, sentence
    (+ label, if present in `df`). report_rows lists (id, status) for every
    input row, so callers can see exactly what was dropped and why.
    """
    has_label = "label" in df.columns
    records, report = [], []
    for row in df.itertuples():
        abst = abst_of.get(row.pmid)
        if abst is None:
            report.append((row.id, "pmid_not_in_index"))
            continue
        sentence = abst[row.sentence_start : row.sentence_end]
        e0_spans = _parse_multi_span(row.entity_0_start, row.entity_0_end)
        e1_spans = _parse_multi_span(row.entity_1_start, row.entity_1_end)
        if not e0_spans or not e1_spans:
            report.append((row.id, "missing_entity_span"))
            continue
        marked = _normalize(
            _apply_markup(sentence, e0_spans, e1_spans, presentation), presentation
        )
        record = {"id": row.id, "sentence": marked}
        if has_label:
            record["label"] = row.label
        records.append(record)
        report.append((row.id, "ok"))
    return pd.DataFrame(records), report


def reconstruct_cpt_relations(df, abst_of):
    """Reconstructs cpt_relations rows in all three forms CPT training needs:
    plain text (no entity markup), boundary-tagged, and replacement-token.
    Row order is preserved (not reordered/deduplicated), matching the actual
    CPT pool -- see cpt.py for how the plain/marked mix is built.

    Returns (df_out, report_rows) with df_out columns:
    id, pmid, sentence_plain, sentence_boundary, sentence_replacement.
    """
    records, report = [], []
    for row in df.itertuples():
        abst = abst_of.get(row.pmid)
        if abst is None:
            report.append((row.id, "pmid_not_in_index"))
            continue
        sentence = abst[row.sentence_start : row.sentence_end]
        e0_spans = _parse_multi_span(row.entity_0_start, row.entity_0_end)
        e1_spans = _parse_multi_span(row.entity_1_start, row.entity_1_end)
        if not e0_spans or not e1_spans:
            report.append((row.id, "missing_entity_span"))
            continue
        records.append(
            {
                "id": row.id,
                "pmid": row.pmid,
                "sentence_plain": sentence,
                "sentence_boundary": _normalize(
                    _apply_markup(sentence, e0_spans, e1_spans, "boundary"), "boundary"
                ),
                "sentence_replacement": _normalize(
                    _apply_markup(sentence, e0_spans, e1_spans, "replacement"),
                    "replacement",
                ),
            }
        )
        report.append((row.id, "ok"))
    return pd.DataFrame(records), report


def reconstruct_semmed_relations(df, semmed_lookup, presentation):
    """Reconstructs sentence text for semmed_relations rows. Entity offsets
    are already sentence-relative in the published data, so this slices
    directly against the SemMedDB sentence text (no abstract lookup needed).

    `semmed_lookup`: {pred_id: (pmid, sentence, sent_start, sent_end,
    subject_start, subject_end, object_start, object_end)}, as returned by
    semmed_db.load_semmed_sentences().
    """
    records, report = [], []
    for row in df.itertuples():
        info = semmed_lookup.get(row.pred_id)
        if info is None:
            report.append((row.id, "pred_id_not_resolved"))
            continue
        _pmid, sentence, _sent_start, _sent_end, *_aux = info
        e0_spans = [(row.entity_0_start, row.entity_0_end)]
        e1_spans = [(row.entity_1_start, row.entity_1_end)]
        marked = _normalize(
            _apply_markup(sentence, e0_spans, e1_spans, presentation), presentation
        )
        records.append({"id": row.id, "sentence": marked, "label": row.label})
        report.append((row.id, "ok"))
    return pd.DataFrame(records), report


def verify_semmed_offsets(df, semmed_lookup, sample_size=None):
    """Sanity-checks that the local SemMedDB copy agrees with the offsets the
    dataset was built from: recomputes sentence-relative subject/object spans
    from PREDICATION_AUX and compares them to the published entity_0/1 offsets.

    A low match rate usually means the local SemMedDB copy is a different
    version than VER43_R (used to build the published dataset). Returns
    (n_checked, n_matched).
    """
    rows = df if sample_size is None else df.head(sample_size)
    n_checked = n_matched = 0
    for row in rows.itertuples():
        info = semmed_lookup.get(row.pred_id)
        if info is None:
            continue
        _pmid, _sentence, sent_start, _sent_end, subj_s, subj_e, obj_s, obj_e = info
        spans = sorted(
            [
                (subj_s - sent_start, subj_e - sent_start),
                (obj_s - sent_start, obj_e - sent_start),
            ]
        )
        n_checked += 1
        if spans == [
            (row.entity_0_start, row.entity_0_end),
            (row.entity_1_start, row.entity_1_end),
        ]:
            n_matched += 1
    return n_checked, n_matched


def write_seed_splits_for_presentation(
    df_resolved, df_seed_splits, data_root, source, presentation
):
    """Writes data/<source>/<seed>/{split}_{presentation}.csv for every seed."""
    merged = df_seed_splits.merge(df_resolved, on="id", how="inner")
    n_total = len(df_seed_splits)
    n_merged = len(merged)
    if n_merged < n_total:
        print(
            f"[reconstruct] {source}/{presentation}: {n_total - n_merged} / {n_total} "
            f"seed-split rows could not be resolved to sentence text (see build report)"
        )

    for (seed, split), group in merged.groupby(["seed", "split"]):
        out_dir = os.path.join(data_root, source, str(seed))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{split}_{presentation}.csv")
        group[["id", "sentence", "label"]].to_csv(out_path, index=False)
        print(f"  wrote {out_path} ({len(group)} rows)")
