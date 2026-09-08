"""Uçtan uca duman testi (F2): çalışan med-rag stack'ine karşı kabul senaryosu.

Yalnızca stdlib (urllib) -- host'ta venv'siz de koşar:

    python tools/e2e_smoke.py --base-url http://localhost:8507 --sample belge.pdf

Adımlar: auth -> upload -> durum yoklaması (ready) -> içerik -> not oluştur ->
not hazır -> belge sil -> belge kayboldu -> not sil. `--chat` eklenirse ayrıca
`POST /api/chat` ile soru sorulur ve cevabın atıf (chunks) taşıdığı doğrulanır
(bu adım LLM+embedding gerektirir ve token harcar).

Çıkış kodu: 0 = tüm adımlar geçti; 1 = en az bir adım başarısız.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

TERMINAL_READY = {"ready"}
TERMINAL_BAD = {"error"}


class Client:
    """Çerez tutan minimal REST istemcisi (urllib üzerine)."""

    def __init__(self, base_url: str, password: str = "") -> None:
        self.base = base_url.rstrip("/")
        self.password = password
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar)
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body=None,
        data=None,
        headers: dict | None = None,
        timeout: int = 30,
    ):
        req = urllib.request.Request(self.base + path, method=method)
        req.add_header("Accept", "application/json")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        if json_body is not None:
            req.add_header("Content-Type", "application/json")
            req.data = json.dumps(json_body).encode()
        elif data is not None:
            req.data = data
        return self.opener.open(req, timeout=timeout)

    def login_if_needed(self) -> bool:
        with self.request("GET", "/api/auth/session") as resp:
            state = json.load(resp)
        if state.get("authenticated"):
            return True
        if not state.get("required"):
            return True  # auth kapalı
        if not self.password:
            raise SystemExit("sunucu şifre istiyor; --password verin")
        with self.request(
            "POST", "/api/auth/login", json_body={"password": self.password}
        ) as resp:
            return bool(json.load(resp).get("authenticated"))

    def upload(self, file_path: Path) -> str:
        boundary = f"----medrag-{uuid.uuid4().hex}"
        payload = file_path.read_bytes()
        body = (
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="files"; '
                f'filename="{file_path.name}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
            + payload
            + f"\r\n--{boundary}--\r\n".encode()
        )
        with self.request(
            "POST",
            "/api/library/documents",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        ) as resp:
            out = json.load(resp)
        kabul = out.get("kabul") or out.get("accepted") or []
        if not kabul:
            raise RuntimeError(f"upload kabul edilmedi: {out}")
        return kabul[0]["doc_id"]

    def documents(self) -> list[dict]:
        with self.request("GET", "/api/library/documents") as resp:
            return json.load(resp)["documents"]

    def wait_state(
        self, doc_id: str, want: set[str], timeout_s: int, label: str
    ) -> dict:
        deadline = time.monotonic() + timeout_s
        last = None
        while time.monotonic() < deadline:
            item = next((d for d in self.documents() if d["doc_id"] == doc_id), None)
            if item is None:
                if want == set():  # silme yoklaması: listeden düşmesini bekle
                    return {}
            else:
                state = (item.get("status") or {}).get("state")
                if state != last:
                    print(
                        f"    [{label}] durum: {state} "
                        f"({(item.get('status') or {}).get('detail') or '-'})"
                    )
                    last = state
                if state in want:
                    return item
            time.sleep(2.0)
        raise TimeoutError(
            f"{label}: {timeout_s} sn içinde {want} olamadı (son: {last})"
        )

    def chat(self, message: str) -> dict:
        with self.request(
            "POST", "/api/chat", json_body={"message": message}, timeout=300
        ) as resp:
            return json.load(resp)


PASS, FAIL = "✓", "✗"
sonuclar: list[tuple[bool, str]] = []


def adim(ok: bool, name: str, detail: str = "") -> None:
    sonuclar.append((ok, name))
    print(f"  {PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8507")
    ap.add_argument("--password", default="")
    ap.add_argument(
        "--sample", required=True, help="yüklenecek örnek dosya (pdf/docx/md ...)"
    )
    ap.add_argument(
        "--islenme-timeout",
        type=int,
        default=900,
        help="belge ready olana dek beklenecek sn (VLM yavaştır)",
    )
    ap.add_argument(
        "--chat", action="store_true", help="LLM'li tam tur: soru sor + atıf doğrula"
    )
    ap.add_argument(
        "--chat-question", default="Bu belgede geçen ana konu nedir? Kısaca özetle."
    )
    args = ap.parse_args()

    sample = Path(args.sample)
    if not sample.is_file():
        raise SystemExit(f"örnek dosya yok: {sample}")

    c = Client(args.base_url, args.password)
    print(f"hedef: {args.base_url}, örnek: {sample.name}")

    print("[1/8] auth")
    try:
        adim(c.login_if_needed(), "giriş/oturum")
    except urllib.error.URLError as exc:
        raise SystemExit(f"sunucuya ulaşılamadı: {exc}") from exc

    print("[2/8] upload")
    doc_id = c.upload(sample)
    adim(bool(doc_id), "dosya kabul edildi", doc_id)

    print(f"[3/8] işleme yoklaması (en fazla {args.islenme_timeout} sn)")
    try:
        item = c.wait_state(
            doc_id, TERMINAL_READY | TERMINAL_BAD, args.islenme_timeout, "belge"
        )
        state = (item.get("status") or {}).get("state")
        adim(
            state in TERMINAL_READY,
            "belge hazır",
            f"state={state}, detail={(item.get('status') or {}).get('detail')}",
        )
    except TimeoutError as exc:
        adim(False, "belge hazır", str(exc))
        state = None

    if state in TERMINAL_READY:
        print("[4/8] içerik")
        try:
            with c.request("GET", f"/api/library/documents/{doc_id}/content") as r:
                md = json.load(r).get("markdown") or ""
            adim(len(md) > 0, "ayrıştırılmış markdown geldi", f"{len(md)} karakter")
        except urllib.error.HTTPError as exc:
            adim(False, "ayrıştırılmış markdown geldi", f"HTTP {exc.code}")
    else:
        print("[4/8] içerik — atlandı (belge hazır değil)")

    print("[5/8] not")
    note_title = f"e2e-notu-{uuid.uuid4().hex[:8]}"
    with c.request(
        "POST",
        "/api/library/notes",
        json_body={
            "title": note_title,
            "content": f"Test notu. Referans: {note_title}",
        },
    ) as r:
        note = json.load(r)["note"]
    note_id = note["doc_id"]
    adim(bool(note_id), "not oluşturuldu", note_id)
    try:
        c.wait_state(note_id, TERMINAL_READY, 300, "not")
        adim(True, "not işlendi")
    except TimeoutError as exc:
        adim(False, "not işlendi", str(exc))

    if args.chat and state in TERMINAL_READY:
        print("[6/8] chat (LLM'li — yavaş)")
        try:
            out = c.chat(args.chat_question)
            reply = out.get("reply") or ""
            chunks = out.get("chunks") or []
            adim(bool(reply.strip()), "cevap geldi", f"{len(reply)} karakter")
            adim(len(chunks) > 0, "atıf (chunks) döndü", f"{len(chunks)} kanıt parçası")
        except Exception as exc:  # noqa: BLE001 -- kabul testi: hatayı raporla
            adim(False, "chat turu", repr(exc))
    else:
        print("[6/8] chat — atlandı (--chat verilmedi ya da belge hazır değil)")

    print("[7/8] belge silme")
    with c.request("DELETE", f"/api/library/documents/{doc_id}") as r:
        adim(json.load(r).get("ok") is True, "silme isteği kabul")
    try:
        c.wait_state(doc_id, set(), 120, "silme")
        kalan = [d["doc_id"] for d in c.documents() if d["doc_id"] == doc_id]
        adim(not kalan, "belge listeden düştü")
    except TimeoutError as exc:
        adim(False, "belge listeden düştü", str(exc))

    print("[8/8] not silme")
    with c.request("DELETE", f"/api/library/notes/{note_id}") as r:
        adim(json.load(r).get("ok") is True, "not silindi")

    basarisiz = [n for ok, n in sonuclar if not ok]
    print(f"\nSONUÇ: {len(sonuclar) - len(basarisiz)}/{len(sonuclar)} adım geçti")
    if basarisiz:
        print("başarısız adımlar:")
        for n in basarisiz:
            print(f"  - {n}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
