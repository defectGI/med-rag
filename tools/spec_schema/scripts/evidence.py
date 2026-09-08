"""Collect EVIDENCE from the corpus: label -> (value cell, taxonomy node).

`spec_keys.yaml` is built from this module's output. Nothing here is
guessed: every label, every value, every node comes from a real document.

Chain:
  all_chunks.json (node.text tables)
    -> doc_id
    -> document_nodes.json  links.owner_ids
    -> product_nodes.json   node_id -> family/subfamily/subfamily_2
    -> taxonomy node id

Usage (as a library):
    from evidence import collect
    ev = collect(chunks_path)     # {label: EvidenceSet}
"""

from __future__ import annotations

import collections
import dataclasses
import json
import pathlib
import re

# spec_schema/ lives directly under tools/, while chatbot-corpus/ sits
# next to tools/ at the repo root, so we walk up three levels (this file
# is .../tools/spec_schema/scripts/evidence.py).
CORPUS = pathlib.Path(__file__).resolve().parents[3] / "chatbot-corpus"
PRODUCT_NODES = CORPUS / "product_info" / "product_nodes.json"
DOCUMENT_NODES = CORPUS / "document_info" / "document_nodes.json"

EMPTY = {"-", "—", "–", "―", "", "n/a", "na", "tbd", "*", "...", "…"}

# Noise: pinout rows, document metadata, software UI screen text. The
# regex intentionally includes a mix of English and the corpus's own
# language, because the source datasheets themselves mix them.
NOISE = re.compile(
    r"^(pin|table|figure|görsel|burada|signal|gnd$|nc$|vcc|unused|reserved"
    r"|io[_ ]|mio\d|hpc\d|gth|gtr|ps[_ ]mio|[a-e]\d+$|\d|row$|som |bj770|pmod"
    r"|sfp$|test |boot mode select|hdmi[_ ]|usb\d|pci-e|x\d+y\d+|ch\d|k\d"
    r"|rev\.|size$|scale$|sheet$|revision|document name|product name"
    r"|address$|2224|sbl-|a3 \d|#+ ?table|step name|select |enabled$|file path"
    r"|file size|slave address|register address|number of bytes|arbitration"
    r"|polarity$|mode|data$|code$|description$|value$|parameter$|specification$"
    r"|function$|channel$|connector$|xj\d|xp\d|j\d pin|port\d|format$|length$"
    r"|cell \d|box id|tcp port|udp port|number of 256|source current"
    r"|channel number|fpga signal|e$|f$|g$|h$|a$|b$|c$|d$)", re.IGNORECASE)


def norm_label(s: str) -> str:
    s = s.lower().replace(" ", " ")
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"^#+\s*", "", s)
    s = re.sub(r"^\d+(\.\d+)*\s+", "", s)
    s = re.sub(r"[®™*:()\[\]]", " ", s)
    s = re.sub(r"[_/\\&]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .-–—")


def is_empty(cell: str) -> bool:
    return norm_label(cell) in EMPTY or cell.strip() in EMPTY


@dataclasses.dataclass
class Row:
    doc_id: str
    node: str          # taxonomy node id
    cells: list[str]   # cells AFTER the label; empties are PRESERVED
    raw: str           # raw tail of the line after the label
    source: str        # "table_row" | "prose"  -- how the label was found


def _taxonomy_index():
    """product_nodes.json node_id -> taxonomy (family, subfamily, subfamily_2)."""
    data = json.loads(PRODUCT_NODES.read_text(encoding="utf-8"))
    return {n["node_id"]: (n.get("family"), n.get("subfamily"),
                           n.get("subfamily_2"))
            for n in data["product_nodes"]}


def _doc_nodes():
    """doc_id -> list of owner node_ids."""
    data = json.loads(DOCUMENT_NODES.read_text(encoding="utf-8"))
    out = {}
    for d in data["documents"]:
        out[d["identity"]["doc_id"]] = (d.get("links", {}).get("owner_ids") or [])
    return out


def build_node_ids(taxonomy_yaml_nodes) -> dict[tuple, str]:
    """(family, subfamily, subfamily_2) -> spec_keys.yaml taxonomy id."""
    return {(t.get("family"), t.get("subfamily"), t.get("subfamily_2")): t["id"]
            for t in taxonomy_yaml_nodes}


def collect(chunks_path: pathlib.Path, node_ids: dict[tuple, str]):
    """label -> [Row]. Noisy labels are dropped."""
    tax = _taxonomy_index()
    owners = _doc_nodes()
    chunks = json.loads(pathlib.Path(chunks_path).read_text(encoding="utf-8"))

    out: dict[str, list[Row]] = collections.defaultdict(list)
    for doc_id, doc in chunks["documents"].items():
        # taxonomy nodes of the products that own this document
        nodes = set()
        for nid in owners.get(doc_id, []):
            path = tax.get(nid)
            if not path:
                continue
            # start from the most specific node, walk up to the first match
            for cut in (path, (path[0], path[1], None), (path[0], None, None)):
                if cut in node_ids:
                    nodes.add(node_ids[cut])
                    break
        if not nodes:
            continue

        for n in doc["nodes"]:
            for line in n["text"].split("\n"):
                line = line.strip()
                if not line.startswith("|"):
                    continue
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) < 2:
                    continue
                label, vals = cells[0], cells[1:]
                nl = norm_label(label)
                if not nl or len(nl) > 55 or NOISE.match(nl):
                    continue
                if all(is_empty(v) for v in vals):
                    continue
                for node in nodes:
                    out[nl].append(Row(doc_id, node, vals, " | ".join(vals),
                                       "table_row"))

            # --- second harvest: prose "LABEL: value" lines.
            # Some specs (notably MIL-STD-1553 coupler datasheets) appear as
            # a COLUMN HEADER in the table or in body prose, NEVER as a
            # row label:
            #   "TERMINATION RESISTOR VALUE: 78.7 OHMS ±1% 2W (R7, R8)"
            #   "- Fault Protection: 59 Ohms ±1% (R1-R4) in series with ..."
            # If these lines are skipped, those specs look evidence-less and
            # are dropped unfairly. `source` is tagged so they don't mix
            # with the table harvest.
            for line in n["text"].split("\n"):
                line = line.strip().lstrip("-•* ").strip()
                if not line or line.startswith("|") or ":" not in line:
                    continue
                if len(line) > 220:
                    continue
                lab, val = line.split(":", 1)
                val = val.strip()
                if not val or len(lab.split()) > 6 or len(lab) < 3:
                    continue
                # must be a real label, not an inline colon mid-sentence
                if lab != lab.strip() or any(c in lab for c in ".!?"):
                    continue
                nl = norm_label(lab)
                if not nl or len(nl) > 55 or NOISE.match(nl):
                    continue
                if is_empty(val):
                    continue
                for node in nodes:
                    out[nl].append(Row(doc_id, node, [val], val, "prose"))
    return out


def product_counts() -> dict[tuple, int]:
    """(family, subfamily, subfamily_2) -> number of products DIRECTLY attached to that node."""
    data = json.loads(PRODUCT_NODES.read_text(encoding="utf-8"))
    out: dict[tuple, int] = collections.Counter()
    for n in data["product_nodes"]:
        if n.get("type") != "product":
            continue
        out[(n.get("family"), n.get("subfamily"), n.get("subfamily_2"))] += 1
    return dict(out)


def value_cells(row: Row) -> list[str]:
    """The cells of a row that carry a VALUE (empty placeholders preserved)."""
    return row.cells
