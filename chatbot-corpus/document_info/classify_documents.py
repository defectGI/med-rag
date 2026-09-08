"""
Scans the customer-facing documents under the BELGELER folder, classifies
them per document_node_schema.md and writes document_nodes.json.

owner_ids is resolved in two tiers (find_owner -> single `matched_node_id`):
  tier 1 - a code in the file path (DE\\d+ or DE..XX wildcard), validated
           against the product_code / variant_base / reference_code
           whitelist in product_nodes.json.
  tier 2 - if no code is found, the folder name is compared against the
           family/subfamily/subfamily_2 text.
If neither matches, the file lands in the "unresolved" list -- it is
not silently dropped.

When the matched node is a CATEGORY (family/subfamily/subfamily_2),
owner_ids is deterministically fan-out'd to the node ITSELF + ALL
descendant nodes in the tree (intermediate categories INCLUDED, leaf
products INCLUDED) via resolve_owner_ids + build_subtree_expander;
i.e. if a file belongs to a family, it is also considered to belong to
all subfamily/subfamily_2 nodes beneath it AND to all leaf products.
The raw matched node is still kept in `matched_node_id` (for
traceability; now also one element of owner_ids); when the fan-out
actually expands, `_expanded` is appended to `resolution_method` in
CODE (the previous separate, non-deterministic LLM-agent step was
retired).

The script handles both the first scan and re-scans. If the output
file already exists, it is NOT REBUILT — it is MERGED into: each record
is looked up first by rel_path, and if that misses, by content_hash, and
the matched record's `doc_id` + editorial/pipeline state (is_active,
parse, chunk) are KEPT as-is. This is not decoration; it is an identity
guarantee: if `doc_id` were regenerated, the parsed IR folders, the
`{doc_id}::c{n}` chunk IDs, and the entire viz would silently become
orphans. Carrying the parse/chunk state is part of the same guarantee:
staleness is already caught by the content_hash + parser_version /
chunker_version gates — resetting them would re-parse an unchanged
corpus from scratch.

scan_status is derived from the match: NEW (never seen) / UNCHANGED
(same path, same hash) / MODIFIED (same path, different hash) / MOVED
(same hash, different path). Records that existed in the previous scan
but are absent from this one are NOT DELETED — they stay sticky with
scan_status=DELETED (schema decision).

Paths are read from .env: PRODUCT_INFO_DIR's product_nodes.json lives
at a fixed path underneath (see .env.example). BELGELER_DIR is a separate
root — the BELGELER folder now lives outside product_info, at the
project's root. OUTPUT_PATH is also configurable.

Scanning/classification TUNING (which files are skipped, doc_type
keywords, code pattern, extensions) comes from config/default.toml
(see config.py) — not hard-coded in code; DOCUMENT_INFO_CONFIG env can
be used for a partial TOML override.
"""

import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from config import load_config

# Paths in .env are relative to the folder where this file lives
# (document_info/) -- the script keeps working if the repo is moved
# (as long as the same internal hierarchy is preserved). An absolute
# path is used as-is.
# Scanning/classification tuning (exclude lists, doc_type keywords, code
# pattern, extensions) now comes from config/default.toml (see config.py);
# it is not hard-coded in code. Paths remain in .env because they are
# inputs.
BASE_DIR = Path(__file__).resolve().parent


def _resolve(env_var: str) -> str:
    return str((BASE_DIR / os.environ[env_var]).resolve())


def build_subtree_expander(nodes):
    """node_id -> itself + ALL descendant nodes in the tree (deterministic).

    When a document resolves to a CATEGORY node
    (family/subfamily/subfamily_2), owner_ids is expanded to the node
    ITSELF + ALL its descendants -- intermediate categories (subfamily,
    subfamily_2) INCLUDED, leaf products INCLUDED. "A document belonging
    to a family also belongs to all its subfamilies and all its
    products" is the semantic decision (previously fan-out only reached
    leaf products, intermediate category nodes were excluded).
    If a product node is already a leaf, it returns its own single-
    element list.

    Expansion walks the tree deterministically via parent_id; the result
    is sorted by source_ref so the output is stable, independent of
    row order.
    """
    children = {}
    by_id = {}
    for n in nodes:
        by_id[n["node_id"]] = n
        children.setdefault(n["parent_id"], []).append(n["node_id"])

    def expand(node_id):
        node = by_id.get(node_id)
        if node is None:
            return []
        result = []
        stack = [node_id]
        seen = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if by_id.get(cur) is not None:
                result.append(cur)          # both category and product -- all owners
            stack.extend(children.get(cur, []))
        # sort deterministically by source_ref (independent of row order)
        result.sort(key=lambda nid: by_id[nid].get("source_ref") or nid)
        return result

    return expand


def load_whitelist_and_categories(product_nodes_path):
    with open(product_nodes_path, encoding="utf-8") as f:
        data = json.load(f)
    nodes = data["product_nodes"]

    code_to_node = {}
    for n in nodes:
        if n["product"]:
            for code in (n["product"]["product_code"], n["product"]["variant_base"]):
                if code:
                    code_to_node.setdefault(code, n["node_id"])
        if n["category"] and n["category"]["reference_code"]:
            code_to_node.setdefault(n["category"]["reference_code"], n["node_id"])

    # folder name -> node_id (by family/subfamily/subfamily_2 text, normalized)
    def norm(text):
        return re.sub(r"[^A-Z0-9]+", "", text.upper()) if text else None

    category_nodes_by_text = {}
    for n in nodes:
        if n["type"] in ("family", "subfamily", "subfamily_2"):
            # use the level THIS NODE added (type == field name) as the key text
            own_text = n[n["type"]]
            if own_text:
                category_nodes_by_text.setdefault(norm(own_text), n["node_id"])

    expand_to_subtree = build_subtree_expander(nodes)
    return code_to_node, category_nodes_by_text, norm, expand_to_subtree


def is_junk(file_name, scan):
    lower = file_name.lower()
    if lower in scan.junk_name_set():
        return True
    if file_name.startswith(scan.junk_prefix_tuple()):
        return True
    upper = file_name.upper()
    return any(k in upper for k in scan.exclude_file_keywords)


def classify_doc_type(file_name, cfg):
    upper = file_name.upper()
    for entry in cfg.doc_types:
        if any(k in upper for k in entry.keywords):
            return entry.type
    if file_name.lower().endswith(cfg.scan.stp_ext_tuple()):
        return "STP"
    ext = os.path.splitext(file_name)[1].lower()
    if ext in cfg.scan.image_ext_set():
        return "PRODUCT_IMAGE"
    return None


def find_owner(rel_path, code_to_node, category_nodes_by_text, norm, code_pattern):
    """Return the SINGLE raw matched node_id and resolution method from the file path.

    Fan-out (category -> leaf products) is NOT done here; the caller
    expands deterministically via `expand_to_subtree`. This way the raw
    match and the leaf owner_ids stay separate and traceability
    (`matched_node_id`) is preserved.
    """
    upper_path = rel_path.upper()

    # tier 1: code match (full whitelist validation)
    for token in code_pattern.findall(upper_path):
        if token in code_to_node:
            return code_to_node[token], "code_match"
        base = token.split("-")[0]
        if base in code_to_node:
            return code_to_node[base], "code_match"

    # tier 2: folder name -> family/subfamily/subfamily_2 text.
    # Walk from the most specific (deepest/closest) folder upward -- the
    # first match is the most accurate owner (don't fall back to a coarse
    # upper level like a family).
    folder_parts = rel_path.split(os.sep)[:-1]
    for part in reversed(folder_parts):
        key = norm(part)
        if key in category_nodes_by_text:
            return category_nodes_by_text[key], "folder_name_match"

    return None, None


def resolve_owner_ids(rel_path, code_to_node, category_nodes_by_text, norm,
                      code_pattern, expand_to_subtree):
    """find_owner + deterministic tree fan-out.

    Returns: (owner_ids, matched_node_id, resolution_method).
      - owner_ids: matched_node_id ITSELF + ALL descendant nodes in the
        tree (intermediate categories included, leaf products included).
        Single-element if it resolved directly to a product.
      - matched_node_id: the raw matched node (may be a category) --
        traceability; also one element of owner_ids.
      - resolution_method: match method; if the fan-out really expanded
        (matched != owner_ids, i.e. there are other nodes beneath) the
        `_expanded` suffix is appended consistently IN CODE (not
        delegated to the LLM).
    """
    matched_node_id, method = find_owner(
        rel_path, code_to_node, category_nodes_by_text, norm, code_pattern)
    if matched_node_id is None:
        return [], None, None

    owner_ids = expand_to_subtree(matched_node_id)
    # If it resolved to a category node and spread across the subtree (or
    # matched itself isn't a single element) that's an expansion ->
    # append _expanded to method.
    expanded = owner_ids != [matched_node_id]
    if expanded and owner_ids:
        method = f"{method}_expanded"
    return owner_ids, matched_node_id, method


def now_stamp():
    """Offset-local ISO-8601 -- repo-wide single timestamp standard."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_previous(output_path):
    """Previous scan's records: (rel_path -> record, content_hash -> record).

    If the file is missing/corrupt, return empty -- same path as the first
    scan. If two copies carry the same hash, only the first one enters
    the hash index: MOVED detection is correct for that one, the other
    becomes NEW (a fresh identity is more honest than fabricating one).
    """
    try:
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}, {}
    by_path, by_hash = {}, {}
    for rec in data.get("documents", []):
        by_path[rec["location"]["rel_path"]] = rec
        by_hash.setdefault(rec["scan"]["content_hash"], rec)
    return by_path, by_hash


def blank_pipeline_state():
    """The parse/chunk state of a never-processed document.

    There is deliberately NO `vector` block: no code was filling it, and
    an empty placeholder creates a "looks filled" risk. When the
    embedding stage arrives in this repo it will add its own block in
    the same pattern.
    """
    return {
        "parse": {
            "parser": None,
            "parser_version": None,
            "status": "PENDING",
            "parsed_from_hash": None,
            "parsed_json_path": None,
            "last_parsed": None,
            "error": None,
            "stages": None,
        },
        # Registry leg of the chunk stage. Filled by run_chunk_pipeline.py;
        # the "where in the corpus has chunking run?" question can be
        # answered without counting folders.
        "chunk": {
            "status": "PENDING",
            "chunked_at": None,
            "chunked_from_hash": None,
            "chunks_path": None,
            "chunker_version": None,
        },
        # Registry leg of the facts (pass 1) stage. Filled by
        # pipeline/run_facts_pipeline.py; "SKIPPED" is used for
        # out-of-scope doc_types / ScopeMismatch.
        "facts": {
            "status": "PENDING",
            "extracted_at": None,
            "extracted_from_hash": None,
            "extractor_version": None,
            "prompt_version": None,
        },
    }


def merge_record(record, previous):
    """Merge the freshly scanned `record` with the previous record.

    Preserved: `doc_id` (the identity -- the actual reason), `is_active`
    (editorial axis, the scan cannot know), and `parse`/`chunk`/`facts`
    (pipeline state; staleness is caught by their own hash/version
    gates). Location/hash/type/link fields come from the scan -- their
    truth is the file on disk.
    """
    record["identity"]["doc_id"] = previous["identity"]["doc_id"]
    record["is_active"] = previous.get("is_active", True)
    for key in ("parse", "chunk", "facts"):
        if key in previous:
            record[key] = previous[key]
    return record


def scan_status_for(previous, by_path, content_hash, rel_path):
    """scan_status (schema enum) derived from the match."""
    if previous is None:
        return "NEW"
    if by_path.get(rel_path) is previous:
        return "UNCHANGED" if previous["scan"]["content_hash"] == content_hash \
            else "MODIFIED"
    return "MOVED"  # hash matched but path differs


def main(*, dry_run: bool = False) -> dict:
    """`dry_run=True`: all scanning/classification/matching is computed in
    EXACTLY the same way (the same `scan_status` diff, the same
    `unresolved`/counters), but NOTHING is written to `output_path` --
    only the computed `output` dict is returned and a summary printed.
    This way the dry run shows the SAME sets as the real run's report
    (acceptance criterion) and leaves no trace on the file system
    (mtime/sha256 don't change)."""
    load_dotenv()
    # PRODUCT_INFO_DIR is OPTIONAL in this project: the Excel product catalog
    # was removed with the med-rag fork. If product_nodes.json is absent, the
    # whitelist is empty -- every document is still registered (parse does not
    # filter on owner), it just carries owner_ids=[] and counts as
    # "unresolved" in the summary.
    product_info_dir = os.environ.get("PRODUCT_INFO_DIR")
    product_nodes_path = (os.path.join(product_info_dir, "product_nodes.json")
                          if product_info_dir else None)
    belgeler_root = _resolve("BELGELER_DIR")
    output_path = _resolve("OUTPUT_PATH")

    cfg = load_config(os.environ.get("DOCUMENT_INFO_CONFIG") or None)
    scan = cfg.scan
    code_pattern = cfg.codes.compiled()
    exclude_dirs = scan.exclude_dir_set()

    if product_nodes_path and os.path.exists(product_nodes_path):
        code_to_node, category_nodes_by_text, norm, expand_to_subtree = \
            load_whitelist_and_categories(product_nodes_path)
    else:
        # Catalog-less mode: same shapes, everything resolves to nothing.
        code_to_node, category_nodes_by_text = {}, {}

        def norm(text):
            return re.sub(r"[^A-Z0-9]+", "", text.upper()) if text else None

        def expand_to_subtree(node_id):
            return [node_id] if node_id else []

    # If a previous scan exists, merge into it (identity preserved).
    prev_by_path, prev_by_hash = load_previous(output_path)
    gorulen_doc_ids = set()

    documents = []
    unresolved = []
    doc_type_counts = {}
    # All methods are counted, including _expanded variants (if the code
    # consistently writes _expanded, the counter should see it too).
    resolution_counts = {}
    status_counts = {"NEW": 0, "UNCHANGED": 0, "MODIFIED": 0, "MOVED": 0,
                     "DELETED": 0}

    for dirpath, dirnames, filenames in os.walk(belgeler_root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for file_name in filenames:
            if is_junk(file_name, scan):
                continue

            full_path = os.path.join(dirpath, file_name)
            rel_path = os.path.relpath(full_path, belgeler_root)

            with open(full_path, "rb") as f:
                content = f.read()
            content_hash = "sha256:" + hashlib.sha256(content).hexdigest()

            stat = os.stat(full_path)
            doc_type = classify_doc_type(file_name, cfg)
            owner_ids, matched_node_id, method = resolve_owner_ids(
                rel_path, code_to_node, category_nodes_by_text, norm,
                code_pattern, expand_to_subtree)

            doc_type_counts[doc_type or "UNKNOWN"] = doc_type_counts.get(doc_type or "UNKNOWN", 0) + 1
            method_key = method or "unresolved"
            resolution_counts[method_key] = resolution_counts.get(method_key, 0) + 1

            rel_key = rel_path.replace(os.sep, "/")
            # Path first, content hash if that misses: a moved file is
            # recognised by its hash.
            previous = prev_by_path.get(rel_key) or prev_by_hash.get(content_hash)
            status = scan_status_for(previous, prev_by_path, content_hash, rel_key)
            status_counts[status] += 1

            record = {
                "identity": {
                    "doc_id": str(uuid.uuid4()),
                    "file_name": file_name,
                    "extension": os.path.splitext(file_name)[1].lower(),
                },
                "location": {
                    "rel_path": rel_key,
                    "file_path": full_path,
                },
                "scan": {
                    "content_hash": content_hash,
                    "size_bytes": stat.st_size,
                    # Offset-local ISO (mtime converted from UTC).
                    "last_modified_time": datetime.fromtimestamp(
                        stat.st_mtime).astimezone().isoformat(timespec="seconds"),
                    "scan_status": status,
                },
                "doc_type": doc_type,
                "is_active": True,
                "links": {
                    "owner_ids": owner_ids,
                    "link_count": len(owner_ids),
                    # raw matched node (may be a category) -- kept for
                    # traceability when owner_ids is expanded across the tree.
                    "matched_node_id": matched_node_id,
                    "resolution_method": method,
                },
                **blank_pipeline_state(),
            }
            if previous is not None:
                merge_record(record, previous)
            gorulen_doc_ids.add(record["identity"]["doc_id"])
            documents.append(record)
            if not owner_ids:
                unresolved.append(rel_key)

    # Records present in the previous scan but no longer on disk: not
    # deleted, they stay sticky as DELETED (schema decision) -- the parsed
    # IR/chunk artifacts tied to their doc_id are still on disk, dropping
    # the record would orphan them.
    for rec in prev_by_path.values():
        if rec["identity"]["doc_id"] in gorulen_doc_ids:
            continue
        rec["scan"]["scan_status"] = "DELETED"
        status_counts["DELETED"] += 1
        documents.append(rec)

    output = {
        "generated_at": now_stamp(),
        "summary": {
            "scanned_files": len(documents) - status_counts["DELETED"],
            "resolved": len(documents) - status_counts["DELETED"] - len(unresolved),
            "unresolved": len(unresolved),
            "resolution_method_counts": resolution_counts,
            "doc_type_counts": doc_type_counts,
            "scan_status_counts": status_counts,
            "unresolved_files": unresolved,
            "note": "unresolved_files list is reserved for manual / LLM "
                    "review; not yet flagged in this file. scan_status is "
                    "derived by matching against the previous scan; DELETED "
                    "records are sticky.",
        },
        "documents": documents,
    }

    ozet = ", ".join(f"{k}={v}" for k, v in status_counts.items() if v)
    if dry_run:
        print(f"[dry-run] {len(documents)} kayit ({ozet or 'bos'}), "
              f"{len(unresolved)} tanesi eslesmedi -> YAZILMAYACAK ({output_path})")
        return output

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"{len(documents)} kayit ({ozet or 'bos'}), "
          f"{len(unresolved)} tanesi eslesmedi -> {output_path}")
    return output


if __name__ == "__main__":
    main()
