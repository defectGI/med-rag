"""`image_store.resolve_image_path` -- image-serving architectural exception.
Tested against a real filesystem on disk (tmp_path); never touches network/GPU."""

from __future__ import annotations

from medrag.api.image_store import resolve_image_path


def test_resolves_prefixed_image_id_to_matching_file(tmp_path):
    hex_id = "a" * 64
    crop = tmp_path / f"{hex_id}.png"
    crop.write_bytes(b"data")
    assert resolve_image_path(tmp_path, f"sha256:{hex_id}") == crop


def test_resolves_bare_hex_without_sha256_prefix(tmp_path):
    hex_id = "b" * 64
    crop = tmp_path / f"{hex_id}.jpg"
    crop.write_bytes(b"data")
    assert resolve_image_path(tmp_path, hex_id) == crop


def test_returns_none_when_file_missing(tmp_path):
    assert resolve_image_path(tmp_path, "sha256:" + "c" * 64) is None


def test_returns_none_for_empty_or_none_image_id(tmp_path):
    assert resolve_image_path(tmp_path, "") is None
    assert resolve_image_path(tmp_path, None) is None


def test_returns_none_for_path_traversal_attempt(tmp_path):
    # Containing `..`/separators -- must be rejected by the 64-hex-digit
    # constraint and NEVER glob OUTSIDE storage_root.
    assert resolve_image_path(tmp_path, "../../etc/passwd") is None
    assert resolve_image_path(tmp_path, "sha256:../../../etc/passwd") is None


def test_returns_none_for_non_hex_characters():
    assert resolve_image_path(None, "sha256:not-hex-at-all!!") is None  # type: ignore[arg-type]


def test_deterministic_first_match_when_multiple_extensions(tmp_path):
    hex_id = "d" * 64
    (tmp_path / f"{hex_id}.jpg").write_bytes(b"jpg")
    (tmp_path / f"{hex_id}.png").write_bytes(b"png")
    result = resolve_image_path(tmp_path, hex_id)
    assert result is not None
    assert result.name == sorted([f"{hex_id}.jpg", f"{hex_id}.png"])[0]  # noqa: FURB192 -- deliberately mirrors the sorted(...)[0] behavior in image_store.resolve_image_path
