"""
pubmed_ftp.py

Downloads the PubMed annual baseline (distributed over NLM's FTP site, mirrored
over HTTPS at the same path) and builds a local SQLite index of
pmid -> (title, abstract) text.

The published dataset's `sentence_start`/`sentence_end` offsets for PubMed-derived
rows (`pubmed_relations`, `cpt_relations`) are absolute character positions into
this abstract text, so reconstruction must use the *same* PubMed release the
offsets were computed against: the 2025 annual baseline
(https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/). Do not substitute a live
E-utilities fetch -- PubMed records can be revised after the baseline snapshot,
which would shift offsets.

Abstract text is reconstructed as the concatenation of each <AbstractText>
section, prefixed by its Label when present (e.g. "BACKGROUND: ..."), joined by
a single space -- the conventional flattening of a structured PubMed abstract.
This is an assumption about how the original offsets were computed (the private
raw-dump ingestion code is not available for direct comparison); use
reconstruct.py's --sample-check to manually spot-check reconstructed sentences
before relying on this at scale.
"""

import gzip
import hashlib
import os
import re
import sqlite3
import time

import requests
from lxml import etree

BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
FILENAME_RE = re.compile(r'href="(pubmed\d{2}n\d{4}\.xml\.gz)"')


def list_baseline_files(year="25", base_url=BASELINE_URL):
    """Returns the sorted list of baseline xml.gz filenames for the given
    2-digit year code (e.g. "25" for the 2025 annual baseline), scraped from
    NLM's directory listing."""
    resp = requests.get(base_url, timeout=60)
    resp.raise_for_status()
    names = sorted(set(FILENAME_RE.findall(resp.text)))
    names = [n for n in names if n.startswith(f"pubmed{year}n")]
    if not names:
        raise RuntimeError(
            f"No baseline files found for year code {year!r} at {base_url}"
        )
    return names


def _md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_md5(path, base_url, filename, session):
    """Downloads the NLM-provided .md5 sidecar and checks it against `path`.
    Returns True if verified, False if the sidecar is unavailable or mismatches.
    """
    resp = session.get(base_url + filename + ".md5", timeout=30)
    if resp.status_code != 200:
        return False
    m = re.search(r"([0-9a-fA-F]{32})", resp.text)
    if not m:
        return False
    return _md5_of(path) == m.group(1).lower()


def download_baseline(
    dest_dir, year="25", limit=None, base_url=BASELINE_URL, sleep_between=0.2
):
    """Downloads baseline xml.gz files into dest_dir, skipping files that are
    already present and pass md5 verification.

    Args:
        dest_dir: local directory to store the downloaded files.
        year: 2-digit baseline year code (e.g. "25").
        limit: if set, only download the first `limit` files (sorted by
            filename) -- useful for a quick smoke test before committing to the
            full ~40GB/1000+ file download.
        sleep_between: seconds to pause between downloads (politeness delay).

    Returns:
        List of local file paths that are present and verified after this call.
    """
    os.makedirs(dest_dir, exist_ok=True)
    filenames = list_baseline_files(year=year, base_url=base_url)
    if limit is not None:
        filenames = filenames[:limit]
        print(
            f"[pubmed_ftp] test mode: limiting to the first {limit} of the baseline's files"
        )

    session = requests.Session()
    local_paths = []
    for i, filename in enumerate(filenames, start=1):
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path) and _verify_md5(
            dest_path, base_url, filename, session
        ):
            print(f"[{i}/{len(filenames)}] already downloaded, verified: {filename}")
            local_paths.append(dest_path)
            continue

        print(f"[{i}/{len(filenames)}] downloading {filename} ...")
        with session.get(base_url + filename, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            tmp_path = dest_path + ".part"
            with open(tmp_path, "wb") as f:
                f.writelines(resp.iter_content(chunk_size=1024 * 1024))
            os.replace(tmp_path, dest_path)

        if not _verify_md5(dest_path, base_url, filename, session):
            raise RuntimeError(
                f"md5 verification failed for {filename} -- re-run to retry the download"
            )
        local_paths.append(dest_path)
        time.sleep(sleep_between)

    return local_paths


def _abstract_text(article_elem):
    """Flattens <Abstract>/<AbstractText> sections into one string, prefixing
    each section with its Label (if any), joined by a single space."""
    parts = []
    for abst_text in article_elem.findall(".//Abstract/AbstractText"):
        text = "".join(abst_text.itertext()).strip()
        if not text:
            continue
        label = abst_text.get("Label")
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts)


def iter_pubmed_records(xml_gz_path, target_pmids=None):
    """Streams (pmid, title, abstract) tuples out of one baseline xml.gz file.

    Args:
        xml_gz_path: path to a pubmedNNnNNNN.xml.gz file.
        target_pmids: optional set of pmids to keep; others are skipped without
            building their abstract text (keeps parsing fast when only a known
            subset of pmids is needed).
    """
    with gzip.open(xml_gz_path, "rb") as f:
        context = etree.iterparse(f, events=("end",), tag="PubmedArticle")
        for _, elem in context:
            pmid_elem = elem.find(".//MedlineCitation/PMID")
            if pmid_elem is None or pmid_elem.text is None:
                elem.clear()
                continue
            pmid = int(pmid_elem.text)
            if target_pmids is not None and pmid not in target_pmids:
                elem.clear()
                continue

            title_elem = elem.find(".//Article/ArticleTitle")
            title = (
                "".join(title_elem.itertext()).strip() if title_elem is not None else ""
            )
            abstract = _abstract_text(elem)

            yield pmid, title, abstract
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]


def build_abstract_index(xml_dir, index_db_path, target_pmids=None):
    """Parses every *.xml.gz in xml_dir and writes (pmid -> title, abstract)
    into a SQLite database at index_db_path (table `abstracts`).

    If a pmid appears in more than one file (rare, e.g. a correction), the
    occurrence from the file processed last (sorted filename order) wins.
    """
    conn = sqlite3.connect(index_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS abstracts (pmid INTEGER PRIMARY KEY, title TEXT, abstract TEXT)"
    )

    xml_files = sorted(f for f in os.listdir(xml_dir) if f.endswith(".xml.gz"))
    if not xml_files:
        raise RuntimeError(f"No .xml.gz files found in {xml_dir}")

    n_written = 0
    for i, fname in enumerate(xml_files, start=1):
        path = os.path.join(xml_dir, fname)
        print(f"[{i}/{len(xml_files)}] indexing {fname} ...")
        rows = list(iter_pubmed_records(path, target_pmids=target_pmids))
        conn.executemany(
            "INSERT OR REPLACE INTO abstracts (pmid, title, abstract) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        n_written += len(rows)
        print(f"  -> {len(rows)} matching records (cumulative: {n_written})")

    conn.close()
    print(
        f"[pubmed_ftp] abstract index written: {index_db_path} ({n_written} rows across {len(xml_files)} files)"
    )


def load_abstracts(index_db_path, pmids):
    """Returns {pmid: abstract_text} for the given pmids, reading from the
    index built by build_abstract_index(). Missing pmids are simply absent
    from the returned dict (callers should treat that as unresolved, not
    silently skip -- see reconstruct.py)."""
    conn = sqlite3.connect(index_db_path)
    cur = conn.cursor()
    abst_of = {}
    pmids = list(pmids)
    chunk_size = 500
    for i in range(0, len(pmids), chunk_size):
        chunk = pmids[i : i + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        cur.execute(
            f"SELECT pmid, abstract FROM abstracts WHERE pmid IN ({placeholders})",
            chunk,
        )
        abst_of.update(cur.fetchall())
    conn.close()
    return abst_of
