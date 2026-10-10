"""
semmed_db.py

Resolves SemMedDB sentence text and verifies entity offsets for rows of the
published `semmed_relations` config, from a local copy of the official SemMedDB
distribution (obtained separately -- it requires its own license, see
https://lhncbc.nlm.nih.gov/ii/tools/SemRep_SemMedDB_SKR/SemMedDB_download.html).

semmed_relations publishes (entity_0_start, entity_0_end, entity_1_start,
entity_1_end) as offsets *relative to the SemMedDB sentence itself* (not the
containing abstract), keyed by `pred_id` (SemMedDB's PREDICATION_ID). To
reconstruct a row:
    1. PREDICATION_ID -> SENTENCE_ID            (PREDICATION table)
    2. SENTENCE_ID    -> SENTENCE text           (SENTENCE table)
    3. PREDICATION_ID -> subject/object spans    (PREDICATION_AUX table)
    4. subtract the sentence's own start offset from the aux spans to get
       sentence-relative positions, and slice `SENTENCE.SENTENCE` with them.

This module expects the three tables as separate files (CSV, optionally
gzipped). SemMedDB's official distribution ships them *without* a header row,
in the column order below (per NLM's documented schema for VER43). If your
download is a newer/older version with a different column layout, pass the
column lists explicitly to build_semmed_index().
"""

import csv
import gzip
import sqlite3
from collections import Counter

PREDICATION_COLUMNS = [
    "PREDICATION_ID",
    "SENTENCE_ID",
    "PMID",
    "PREDICATE",
    "SUBJECT_CUI",
    "SUBJECT_NAME",
    "SUBJECT_SEMTYPE",
    "SUBJECT_NOVELTY",
    "OBJECT_CUI",
    "OBJECT_NAME",
    "OBJECT_SEMTYPE",
    "OBJECT_NOVELTY",
]
SENTENCE_COLUMNS = [
    "SENTENCE_ID",
    "PMID",
    "TYPE",
    "NUMBER",
    "SENT_START_INDEX",
    "SENT_END_INDEX",
    "SECTION_HEADER",
    "SENTENCE",
]
PREDICATION_AUX_COLUMNS = [
    "PREDICATION_AUX_ID",
    "PREDICATION_ID",
    "SUBJECT_TEXT",
    "SUBJECT_DIST",
    "SUBJECT_MAXDIST",
    "SUBJECT_START_INDEX",
    "SUBJECT_END_INDEX",
    "SUBJECT_SCORE",
    "INDICATOR_TYPE",
    "PREDICATE_START_INDEX",
    "PREDICATE_END_INDEX",
    "OBJECT_TEXT",
    "OBJECT_DIST",
    "OBJECT_MAXDIST",
    "OBJECT_START_INDEX",
    "OBJECT_END_INDEX",
    "OBJECT_SCORE",
    "CURR_TIMESTAMP",
]


def _open_text(path):
    return (
        gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace")
        if path.endswith(".gz")
        else open(path, newline="", encoding="utf-8", errors="replace")
    )


def _rows(path, columns, has_header=False):
    """Yields dict rows from a (possibly gzipped) CSV, using `columns` as the
    field names unless the file already has a header row."""
    with _open_text(path) as f:
        reader = csv.reader(f)
        if has_header:
            header = next(reader)
            yield from (dict(zip(header, row)) for row in reader)
        else:
            yield from (dict(zip(columns, row)) for row in reader)


def build_semmed_index(
    predication_path,
    sentence_path,
    predication_aux_path,
    index_db_path,
    target_pred_ids,
    predication_columns=PREDICATION_COLUMNS,
    sentence_columns=SENTENCE_COLUMNS,
    predication_aux_columns=PREDICATION_AUX_COLUMNS,
    has_header=False,
):
    """Builds a compact local SQLite index covering only `target_pred_ids`
    (the pred_ids referenced by semmed_relations.csv), so the full SemMedDB
    distribution (tens of millions of rows) never has to be loaded at once.

    Writes table `semmed_index(pred_id, pmid, sentence, sent_start, sent_end,
    subject_start, subject_end, object_start, object_end)` plus a `skip_reasons`
    table recording why any target pred_id could not be resolved (e.g. not
    found, or an ambiguous 1-to-many PREDICATION_AUX match).
    """
    target_pred_ids = {int(p) for p in target_pred_ids}
    reason_counter = Counter()

    print(
        f"[semmed_db] pass 1/3: scanning {predication_path} for {len(target_pred_ids)} target pred_ids"
    )
    sentence_id_of = {}
    for row in _rows(predication_path, predication_columns, has_header):
        pred_id = int(row["PREDICATION_ID"])
        if pred_id in target_pred_ids:
            sentence_id_of[pred_id] = int(row["SENTENCE_ID"])
    for pred_id in target_pred_ids - sentence_id_of.keys():
        reason_counter["predication_not_found"] += 1
    print(f"  -> matched {len(sentence_id_of)} / {len(target_pred_ids)}")

    print(f"[semmed_db] pass 2/3: scanning {predication_aux_path}")
    aux_spans = {}
    aux_counts = Counter()
    for row in _rows(predication_aux_path, predication_aux_columns, has_header):
        pred_id = int(row["PREDICATION_ID"])
        if pred_id not in sentence_id_of:
            continue
        aux_counts[pred_id] += 1
        aux_spans[pred_id] = (
            int(row["SUBJECT_START_INDEX"]),
            int(row["SUBJECT_END_INDEX"]),
            int(row["OBJECT_START_INDEX"]),
            int(row["OBJECT_END_INDEX"]),
        )
    ambiguous = {p for p, n in aux_counts.items() if n != 1}
    for pred_id in ambiguous:
        reason_counter[f"predication_aux_count={aux_counts[pred_id]}"] += 1
        aux_spans.pop(pred_id, None)
    for pred_id in sentence_id_of.keys() - aux_counts.keys():
        reason_counter["predication_aux_not_found"] += 1
    print(
        f"  -> resolved spans for {len(aux_spans)} pred_ids ({len(ambiguous)} ambiguous)"
    )

    needed_sentence_ids = {sentence_id_of[p] for p in aux_spans}
    print(
        f"[semmed_db] pass 3/3: scanning {sentence_path} for {len(needed_sentence_ids)} sentence_ids"
    )
    sentence_row_of = {}
    for row in _rows(sentence_path, sentence_columns, has_header):
        sid = int(row["SENTENCE_ID"])
        if sid in needed_sentence_ids:
            sentence_row_of[sid] = (
                int(row["PMID"]),
                int(row["SENT_START_INDEX"]),
                int(row["SENT_END_INDEX"]),
                row["SENTENCE"],
            )
    print(f"  -> matched {len(sentence_row_of)} / {len(needed_sentence_ids)}")

    conn = sqlite3.connect(index_db_path)
    conn.execute("DROP TABLE IF EXISTS semmed_index")
    conn.execute(
        "CREATE TABLE semmed_index ("
        "pred_id INTEGER PRIMARY KEY, pmid INTEGER, sentence TEXT, "
        "sent_start INTEGER, sent_end INTEGER, "
        "subject_start INTEGER, subject_end INTEGER, object_start INTEGER, object_end INTEGER)"
    )
    conn.execute("DROP TABLE IF EXISTS skip_reasons")
    conn.execute("CREATE TABLE skip_reasons (pred_id INTEGER PRIMARY KEY, reason TEXT)")

    n_written = 0
    for pred_id, (subj_s, subj_e, obj_s, obj_e) in aux_spans.items():
        sentence_row = sentence_row_of.get(sentence_id_of[pred_id])
        if sentence_row is None:
            reason_counter["sentence_not_found"] += 1
            conn.execute(
                "INSERT INTO skip_reasons VALUES (?, ?)",
                (pred_id, "sentence_not_found"),
            )
            continue
        pmid, sent_start, sent_end, sentence = sentence_row
        conn.execute(
            "INSERT INTO semmed_index VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pred_id,
                pmid,
                sentence,
                sent_start,
                sent_end,
                subj_s,
                subj_e,
                obj_s,
                obj_e,
            ),
        )
        n_written += 1

    for pred_id in target_pred_ids - sentence_id_of.keys() - aux_spans.keys():
        conn.execute(
            "INSERT OR IGNORE INTO skip_reasons VALUES (?, ?)",
            (pred_id, "predication_or_aux_not_found"),
        )
    conn.commit()
    conn.close()

    print(
        f"[semmed_db] index written: {index_db_path} ({n_written} / {len(target_pred_ids)} resolved)"
    )
    print(f"[semmed_db] skip reasons: {dict(reason_counter)}")
    return n_written


def load_semmed_sentences(index_db_path, pred_ids):
    """Returns {pred_id: (pmid, sentence, sent_start, sent_end, subject_start,
    subject_end, object_start, object_end)} for the given pred_ids, reading
    from the index built by build_semmed_index()."""
    conn = sqlite3.connect(index_db_path)
    cur = conn.cursor()
    result = {}
    pred_ids = list(pred_ids)
    chunk_size = 500
    for i in range(0, len(pred_ids), chunk_size):
        chunk = pred_ids[i : i + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        cur.execute(
            f"SELECT pred_id, pmid, sentence, sent_start, sent_end, "
            f"subject_start, subject_end, object_start, object_end "
            f"FROM semmed_index WHERE pred_id IN ({placeholders})",
            chunk,
        )
        for row in cur.fetchall():
            result[row[0]] = row[1:]
    conn.close()
    return result
