"""Test dell'archivio documenti (sirio/store.py) con fogli firma sintetici."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from sirio import pdf_io
from sirio.models import Document
from sirio.store import DocumentNotFound, DocumentStore, PartialImportError
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
