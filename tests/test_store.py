"""Test dell'archivio documenti (sirio/store.py) con fogli firma sintetici."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from sirio import pdf_io
from sirio.models import Document
from sirio.store import MAX_RESUME_ATTEMPTS, DocumentNotFound, DocumentStore, PartialImportError
from tests.synthetic import make_synthetic_sheet, sheet_to_pdf


@pytest.fixture(scope="module")
def sheet() -> np.ndarray:
    img, _truth = make_synthetic_sheet(seed=5, dpi=150)
    return img


@pytest.fixture(scope="module")
def png_bytes(sheet: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", sheet)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def store(tmp_path: Path) -> DocumentStore:
    return DocumentStore(tmp_path / "documenti")


def _variant(png: bytes, n: int) -> bytes:
    """Stessa immagine con un pixel diverso (sha256 diverso)."""
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    img[0, n % img.shape[1]] = (n % 256, 0, 0)
    return cv2.imencode(".png", img)[1].tobytes()


def test_add_image_writes_page_thumb_grid_and_doc(store: DocumentStore, png_bytes: bytes) -> None:
    rev0 = store.revision
    seen: list[Document] = []
    docs = store.add_file("cartella/Foglio Firma.png", png_bytes, on_document=seen.append)
    assert len(docs) == 1 and len(seen) == 1
    doc = docs[0]
    assert len(doc.id) == 12 and all(c in "0123456789abcdef" for c in doc.id)
    assert doc.source_file == "Foglio Firma.png"
    assert (doc.source_page, doc.page_count) == (1, 1)
    assert doc.status == "in_coda" and doc.progress == 0.0 and doc.status_message == "In coda"
    assert doc.source_sha256 and doc.created_at and doc.updated_at
    assert store.revision > rev0

    folder = store.root / doc.id
    for name in ("page.jpg", "thumb.jpg", "grid.json", "doc.json"):
        assert (folder / name).is_file(), name
    page = store.load_page(doc.id)
    assert max(page.shape[:2]) <= 3600
    thumb = cv2.imdecode(np.fromfile(str(store.thumb_path(doc.id)), np.uint8), cv2.IMREAD_COLOR)
    assert max(thumb.shape[:2]) <= 480
    grid = store.load_grid(doc.id)
    assert grid is not None
    assert (grid.width, grid.height) == (page.shape[1], page.shape[0])
    assert grid.detected and len(grid.col_x) == 11 and len(grid.row_y) == 32
    on_disk = Document.model_validate_json((folder / "doc.json").read_bytes())
    assert on_disk.id == doc.id and on_disk.status == "in_coda"
    assert not list(folder.glob("*.tmp")) and not list(folder.glob(".*.tmp"))


def test_dedup_same_file_and_page(store: DocumentStore, png_bytes: bytes) -> None:
    first = store.add_file("a.png", png_bytes)
    seen: list[Document] = []
    again = store.add_file("copia con altro nome.png", png_bytes, on_document=seen.append)
    assert [d.id for d in again] == [d.id for d in first]
    assert seen == []
    assert len(store.list()) == 1


def test_concurrent_import_of_same_file_is_deduplicated(store: DocumentStore, png_bytes: bytes) -> None:
    results: list[list[Document]] = []

    def worker() -> None:
        results.append(store.add_file("x.png", png_bytes))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.list()) == 1
    assert len({r[0].id for r in results}) == 1


def test_pdf_multipage_and_ordering(store: DocumentStore, sheet: np.ndarray, png_bytes: bytes) -> None:
    store.add_file("b.png", png_bytes)
    other = cv2.flip(sheet, 1)  # pagina diversa
    pdf = sheet_to_pdf([sheet, other], dpi=150)
    docs = store.add_file("fogli.pdf", pdf)
    assert [(d.source_page, d.page_count) for d in docs] == [(1, 2), (2, 2)]
    listed = store.list()
    assert [d.source_file for d in listed] == ["b.png", "fogli.pdf", "fogli.pdf"]
    assert [d.source_page for d in listed] == [1, 1, 2]
    # la pagina PDF e' renderizzata a 300 dpi (A4 ~ 2480 x 3508)
    page = store.load_page(docs[0].id)
    assert 3300 <= max(page.shape[:2]) <= 3600


def test_invalid_files_raise_italian_value_error(store: DocumentStore) -> None:
    with pytest.raises(ValueError, match="vuoto"):
        store.add_file("vuoto.pdf", b"")
    with pytest.raises(ValueError, match="PDF"):
        store.add_file("rotto.pdf", b"%PDF-1.4 questo non e' un pdf")
    with pytest.raises(ValueError):
        store.add_file("immagine.png", b"\x89PNG\r\n\x1a\nxxxx")
    assert store.list() == []
    assert [p for p in store.root.iterdir()] == []


def test_partial_import_keeps_first_pages(store: DocumentStore, sheet: np.ndarray,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = sheet_to_pdf([sheet, sheet, sheet], dpi=150)
    real_iter = pdf_io.iter_pages

    def broken(data: bytes, filename: str, dpi: int = 300):
        for i, img in real_iter(data, filename, dpi):
            if i == 1:
                raise ValueError("Impossibile leggere la pagina 2 del PDF «x.pdf»: il file è danneggiato.")
            yield i, img

    monkeypatch.setattr("sirio.store.pdf_io.iter_pages", broken)
    with pytest.raises(PartialImportError) as info:
        store.add_file("x.pdf", pdf)
    assert len(info.value.documents) == 1
    assert "pagina 2" in str(info.value)
    assert len(store.list()) == 1


class _Interruzione(BaseException):
    """Chiusura improvvisa del programma: nessuna gestione degli errori viene eseguita."""


def test_interrupted_multipage_import_resumes_at_next_start(tmp_path: Path, sheet: np.ndarray,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "documenti"
    pdf = sheet_to_pdf([sheet, cv2.flip(sheet, 1), sheet, cv2.flip(sheet, 0)], dpi=150)
    s1 = DocumentStore(root)
    real_import = DocumentStore._import_page

    def stop_after_page_2(self, image, name, page_no, n_pages, sha):
        if page_no == 3:
            raise _Interruzione()
        return real_import(self, image, name, page_no, n_pages, sha)

    monkeypatch.setattr(DocumentStore, "_import_page", stop_after_page_2)
    queued: list[Document] = []
    with pytest.raises(_Interruzione):
        s1.add_file("lotto.pdf", pdf, queued.append)
    monkeypatch.undo()
    assert [d.source_page for d in queued] == [1, 2]
    assert s1.pending_imports() == ["lotto.pdf"]

    # avvio successivo: le pagine 3 e 4 vengono importate e messe in coda, senza duplicare 1 e 2
    s2 = DocumentStore(root)
    assert [d.source_page for d in s2.list()] == [1, 2]
    queued.clear()
    added = s2.resume_pending_imports(queued.append)
    assert [d.source_page for d in added] == [3, 4]
    assert [d.source_page for d in queued] == [3, 4] and all(d.status == "in_coda" for d in queued)
    docs = s2.list()
    assert [(d.source_file, d.source_page, d.page_count) for d in docs] == [("lotto.pdf", n, 4) for n in (1, 2, 3, 4)]
    assert len({d.source_sha256 for d in docs}) == 1 and len({d.id for d in docs}) == 4
    assert s2.pending_imports() == [] and not list(root.glob(".import-*"))
    assert s2.resume_pending_imports(queued.append) == [] and len(queued) == 2
    # al riavvio successivo nulla da riprendere
    assert len(DocumentStore(root).list()) == 4


def test_pending_import_copy_removed_after_import_and_cleaned_at_start(
        store: DocumentStore, sheet: np.ndarray, png_bytes: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    root = store.root
    sha = hashlib.sha256(png_bytes).hexdigest()
    during: list[list[str]] = []
    real_import = DocumentStore._import_page

    def spy(self, *args):
        during.append(sorted(p.name for p in root.glob(".import-*")))
        return real_import(self, *args)

    monkeypatch.setattr(DocumentStore, "_import_page", spy)
    store.add_file("g.png", png_bytes)
    assert during == [[f".import-{sha}.bin", f".import-{sha}.json"]]   # copia presente durante l'importazione
    assert not list(root.glob(".import-*"))                            # ...ed eliminata alla fine
    monkeypatch.undo()

    # importazione non riuscita (errore segnalato all'utente): nessuna ripresa al prossimo avvio
    real_iter = pdf_io.iter_pages

    def broken(data: bytes, filename: str, dpi: int = 300):
        for i, img in real_iter(data, filename, dpi):
            if i == 1:
                raise ValueError("Impossibile leggere la pagina 2 del PDF «y.pdf»: il file è danneggiato.")
            yield i, img

    monkeypatch.setattr("sirio.store.pdf_io.iter_pages", broken)
    with pytest.raises(PartialImportError):
        store.add_file("y.pdf", sheet_to_pdf([sheet, sheet], dpi=150))
    assert not list(root.glob(".import-*"))
    monkeypatch.undo()

    # copia scritta a metà (senza .json) e file temporanei: eliminati all'avvio;
    # copia che non corrisponde allo sha256 del nome: scartata alla ripresa
    (root / f".import-{'a' * 64}.bin").write_bytes(b"x")
    (root / f"..import-{'b' * 64}.bin.k3x9q1.tmp").write_bytes(b"x")   # nome di _atomic_write_bytes
    (root / f".import-{'c' * 64}.bin").write_bytes(b"non corrisponde")
    (root / f".import-{'c' * 64}.json").write_text('{"file": "c.pdf"}', encoding="utf-8")
    s2 = DocumentStore(root)
    assert sorted(p.name for p in root.glob(".import-*")) == [f".import-{'c' * 64}.bin", f".import-{'c' * 64}.json"]
    assert not list(root.glob("*.tmp"))
    assert s2.pending_imports() == ["c.pdf"]
    assert s2.resume_pending_imports() == []
    assert not list(root.glob(".import-*")) and len(s2.list()) == 2


def test_pending_import_abandoned_after_repeated_failed_resumes(tmp_path: Path, sheet: np.ndarray,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    # un file che fa chiudere il programma durante la ripresa non deve bloccare ogni avvio
    root = tmp_path / "documenti"
    pdf = sheet_to_pdf([sheet, sheet, sheet], dpi=150)
    real_import = DocumentStore._import_page

    def crash_at_page_2(self, image, name, page_no, n_pages, sha):
        if page_no == 2:
            raise _Interruzione()
        return real_import(self, image, name, page_no, n_pages, sha)

    monkeypatch.setattr(DocumentStore, "_import_page", crash_at_page_2)
    with pytest.raises(_Interruzione):
        DocumentStore(root).add_file("pesante.pdf", pdf)
    for _ in range(MAX_RESUME_ATTEMPTS):
        s = DocumentStore(root)
        assert s.pending_imports() == ["pesante.pdf"]
        with pytest.raises(_Interruzione):
            s.resume_pending_imports()
    monkeypatch.undo()
    s = DocumentStore(root)
    assert s.resume_pending_imports() == []          # abbandonata: nessun nuovo tentativo
    assert s.pending_imports() == [] and not list(root.glob(".import-*"))
    assert [d.source_page for d in s.list()] == [1]  # le pagine gia' importate restano


def test_get_returns_copies_and_save_updates(store: DocumentStore, png_bytes: bytes) -> None:
    doc = store.add_file("c.png", png_bytes)[0]
    copy = store.get(doc.id)
    assert copy is not None
    copy.header.operatore = "ROSSI MARIO"
    assert store.get(doc.id).header.operatore is None  # non salvato
    rev = store.revision
    before = copy.updated_at
    store.save(copy)
    assert store.revision == rev + 1
    saved = store.get(doc.id)
    assert saved.header.operatore == "ROSSI MARIO"
    assert saved.updated_at >= before
    assert store.get("inesistente") is None
    assert store.get("../../etc") is None


def test_update_semantics(store: DocumentStore, png_bytes: bytes) -> None:
    doc = store.add_file("d.png", png_bytes)[0]
    rev = store.revision
    out = store.update(doc.id, lambda d: setattr(d, "status_message", "Ciao"))
    assert out is not None and out.status_message == "Ciao"
    assert store.revision == rev + 1
    # mutate che restituisce False: nessun salvataggio
    rev = store.revision
    store.update(doc.id, lambda d: False)
    assert store.revision == rev

    def boom(d: Document) -> None:
        d.status_message = "parziale"
        raise RuntimeError("errore")

    with pytest.raises(RuntimeError):
        store.update(doc.id, boom)
    assert store.get(doc.id).status_message == "Ciao"
    assert store.update("000000000000", lambda d: None) is None


def test_delete_and_clear(store: DocumentStore, png_bytes: bytes) -> None:
    a = store.add_file("e.png", png_bytes)[0]
    b = store.add_file("f.png", _variant(png_bytes, 1))[0]
    assert store.delete(a.id) is True
    assert not (store.root / a.id).exists()
    assert store.delete(a.id) is False
    with pytest.raises(DocumentNotFound):
        store.save(a)
    assert [d.id for d in store.list()] == [b.id]
    rev = store.revision
    store.clear()
    assert store.list() == [] and store.revision > rev
    assert not (store.root / b.id).exists()


def test_reload_from_disk_and_cleanup(tmp_path: Path, png_bytes: bytes) -> None:
    root = tmp_path / "documenti"
    s1 = DocumentStore(root)
    doc = s1.add_file("g.png", png_bytes)[0]
    s1.update(doc.id, lambda d: setattr(d, "status", "completato"))
    # importazione interrotta (senza doc.json), documento corrotto, cartella estranea
    (root / "aaaaaaaaaaaa").mkdir()
    (root / "aaaaaaaaaaaa" / "page.jpg").write_bytes(b"x")
    (root / "bbbbbbbbbbbb").mkdir()
    (root / "bbbbbbbbbbbb" / "doc.json").write_text("{non json", encoding="utf-8")
    (root / "altro").mkdir()
    s2 = DocumentStore(root)
    assert [d.id for d in s2.list()] == [doc.id]
    assert s2.get(doc.id).status == "completato"
    assert not (root / "aaaaaaaaaaaa").exists()
    assert (root / "bbbbbbbbbbbb").exists()  # i dati illeggibili non vengono distrutti
    assert (root / "altro").exists()
    # dedup anche dopo il riavvio
    assert s2.add_file("g.png", png_bytes)[0].id == doc.id


def test_paths_validate_ids(store: DocumentStore) -> None:
    with pytest.raises(DocumentNotFound):
        store.page_path("../segreto")
    with pytest.raises(DocumentNotFound):
        store.thumb_path("ABCDEF123456")
    assert store.load_grid("../x") is None
    with pytest.raises(FileNotFoundError):
        store.load_page("0123456789ab")


def test_counts_and_touch(store: DocumentStore, png_bytes: bytes) -> None:
    doc = store.add_file("h.png", png_bytes)[0]
    store.update(doc.id, lambda d: setattr(d, "status", "errore"))
    counts = store.counts()
    assert counts["errore"] == 1 and counts["in_coda"] == 0
    rev = store.revision
    assert store.touch() == rev + 1


def test_real_sample_pdf(tmp_path: Path) -> None:
    sample = os.environ.get("SIRIO_SAMPLE_PDF")
    if not sample or not Path(sample).is_file():
        pytest.skip("SIRIO_SAMPLE_PDF non impostato")
    store = DocumentStore(tmp_path / "documenti")
    docs = store.add_file(Path(sample).name, Path(sample).read_bytes())
    assert len(docs) >= 1
    grid = store.load_grid(docs[0].id)
    assert grid is not None and grid.detected and grid.score > 0.8
    page = store.load_page(docs[0].id)
    assert page.shape[0] > page.shape[1]  # verticale
    info = json.loads((store.root / docs[0].id / "grid.json").read_text(encoding="utf-8"))
    assert info["width"] == page.shape[1]
