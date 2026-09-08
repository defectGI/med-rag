"""`image_id` -> crop file on disk.

DELIBERATE ARCHITECTURAL EXCEPTION: `chatbot` normally never touches the
STORAGE of `parser`/`chunker`/`vectorize`; it imports only `retrieval` (pip)
as a dependency. Here -- with the user's explicit approval -- the
image-serving endpoint (`webapp.py::get_image`) reaches parser's crop blob
store (`parser/storage_paths.py::images_dir`, default `storage/images`) via an
INDEPENDENT env variable (`CHATBOT_IMAGE_STORAGE_DIR`, see .env.example).

NO CODE IS IMPORTED (only the filesystem is shared) -- instead of calling
parser's own `_store_blob`/`hash_hex`/`sha256_id` functions
(images/image_handler.py, parsers/base.py), the SAME file-naming rule
(`image_id` field is `sha256:<hex>`, file name is BARE hex + mime extension)
is MIRRORED here with plain path logic -- same principle as retrieval's top_n
embedder mirroring OpenAICompatEmbedder (see the answering_model.py module
docstring).
"""

from __future__ import annotations

import re
from pathlib import Path

# Bare sha256 hex: 64 chars, hexadecimal only -- doubles as a guard against
# path traversal (`..`, `/`, drive letters, etc.): ANYTHING that doesn't
# match is rejected.
_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def resolve_storage_root(raw: str) -> Path:
    """Converts the raw `CHATBOT_IMAGE_STORAGE_DIR` value to an ABSOLUTE path.

    Rationale: `resolve_image_path`'s glob resolves against the process CWD
    while `flask.send_file` resolves a RELATIVE path against `app.root_path`
    (the app's package directory) -- when the two diverge, the file is found
    by the glob yet serving it 500s (the endpoint doesn't even reach 404).
    Absolutizing in one place (including `~` expansion) closes that
    divergence; for a relative value the meaning stays CWD-based (the glob's
    old behavior) -- only the path handed to send_file is now absolute."""
    return Path(raw).expanduser().resolve()


def resolve_image_path(storage_root: Path, image_id: str | None) -> Path | None:
    """`image_id` (parser's `sha256:<hex>` format) -> the crop file under
    `storage_root`. Since the extension is unknown (it stays on the parser
    side, never carried into chatbot), `{hex}.*` is globbed -- multiple
    matches are unusual (content-addressed, sha256 dedup) but if present the
    first sorted result is returned for determinism.

    Malformed/empty `image_id` or a non-hex body -> `None` (path traversal +
    bad-request guard, in ONE place). Also `None` when the file is absent."""
    if not image_id:
        return None
    hex_part = image_id.split(":", 1)[1] if ":" in image_id else image_id
    if not _HEX_RE.match(hex_part):
        return None
    matches = sorted(storage_root.glob(f"{hex_part}.*"))
    return matches[0] if matches else None


__all__ = ["resolve_image_path", "resolve_storage_root"]
