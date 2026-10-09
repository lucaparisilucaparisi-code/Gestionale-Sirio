"""Test dell'API HTTP (sirio/server.py), dell'avvio (sirio/app.py) e del server di sviluppo."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
import types
import urllib.request
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sirio import app as sirio_app
from sirio import config, server
from sirio.config import Settings
from sirio.processing import Processor
from sirio.store import DocumentStore
from tests.fake_engine import FakeEngine
from tests.synthetic import make_synthetic_sheet, sheet_to_pdf

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "token-di-prova-1234567890"
BASE = "http://127.0.0.1:8123"
FAKE_KEY = "sk-ant-api03-" + "Ab1" * 20 + "WXYZ"


# ==========================================================================
# Fixture
# ==========================================================================

@pytest.fixture(scope="module")
def sheet() -> np.ndarray:
    img, _ = make_synthetic_sheet(seed=21, dpi=110)
    return img


@pytest.fixture(scope="module")
def png(sheet: np.ndarray) -> bytes:
    return cv2.imencode(".png", sheet)[1].tobytes()


@pytest.fixture(scope="module")
def pdf(sheet: np.ndarray) -> bytes:
    return sheet_to_pdf([cv2.flip(sheet, 1)], dpi=110)   # pagina diversa dal PNG


def wait_for(predicate: Callable[[], bool], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.03)
    raise AssertionError("condizione non raggiunta entro il tempo limite")


class Env:
    def __init__(self, tmp_path: Path, web_dir: Path):
        self.store = DocumentStore(config.documents_dir())
        self.engine = FakeEngine(delay=0.05)
        self.settings_engine = "claude"
        self.processor = Processor(
            self.store, lambda: config.load_settings().model_copy(update={"engine": self.settings_engine}),
            lambda _s: self.engine,
        )
        self.processor.start()
        self.events: list[str] = []
        self.app = server.create_app(
            self.store, self.processor, TOKEN,
            on_shutdown=lambda: self.events.append("shutdown"),
            on_heartbeat=lambda: self.events.append("heartbeat"),
            on_bye=lambda: self.events.append("bye"),
        )
        self.client = TestClient(self.app, base_url=BASE, headers={"X-Sirio-Token": TOKEN})
        self.anon = TestClient(self.app, base_url=BASE)
        self.web_dir = web_dir

    def upload(self, *files: tuple[str, bytes]):
        return self.client.post("/api/upload", files=[("files", (n, d, "application/octet-stream")) for n, d in files])

    def doc(self, doc_id: str) -> dict:
        r = self.client.get(f"/api/documents/{doc_id}")
        assert r.status_code == 200, r.text
        return r.json()

    def wait_status(self, doc_id: str, status: str = "completato") -> None:
        wait_for(lambda: (d := self.store.get(doc_id)) is not None and d.status == status)

    def completed_doc(self, png: bytes) -> str:
        r = self.upload(("foglio.png", png))
        assert r.status_code == 200, r.text
        doc_id = r.json()["documents"][0]["id"]
        self.wait_status(doc_id)
        return doc_id


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SIRIO_DATA_DIR", str(tmp_path / "dati"))
    monkeypatch.setenv("SIRIO_EXPORT_DIR", str(tmp_path / "export"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(config, "_keyring", lambda: None)   # niente portachiavi di sistema: ripiego su file
    web = tmp_path / "web"
    (web / "assets").mkdir(parents=True)
    (web / "index.html").write_text(
        '<!doctype html><html lang="it"><head><meta name="sirio-token" content="__SIRIO_TOKEN__">'
        '<link rel="stylesheet" href="/static/styles.css"></head><body>Sirio</body></html>',
        encoding="utf-8",
    )
    (web / "styles.css").write_text("body{color:#123}", encoding="utf-8")
    monkeypatch.setattr(server, "WEB_DIR", web)
    e = Env(tmp_path, web)
    yield e
    e.processor.stop(timeout=5)


# ==========================================================================
# Sicurezza
# ==========================================================================

def test_token_required_on_api(env: Env) -> None:
    r = env.anon.get("/api/state")
    assert r.status_code == 403
    assert "non autorizzato" in r.json()["detail"]
    r = env.anon.get("/api/state", headers={"X-Sirio-Token": "sbagliato"})
    assert r.status_code == 403
    assert env.client.get("/api/state").status_code == 200
    assert env.anon.get(f"/api/state?t={TOKEN}").status_code == 200
    r = env.anon.get("/api/health")
    assert r.status_code == 200 and r.json() == {"ok": True, "version": "1.0.0"}
    assert env.anon.post("/api/heartbeat").status_code == 403
    assert env.anon.post("/api/export", json={}).status_code == 403


def test_host_header_must_be_local(env: Env) -> None:
    for host in ("evil.example", "evil.example:8123", "127.0.0.1.evil.example", "localhost.evil:80", ""):
        r = env.client.get("/api/health", headers={"Host": host})
        assert r.status_code == 403, host
        assert "solo connessioni da questo computer" in r.json()["detail"]
    r = env.client.get("/", headers={"Host": "attacker.test"})
    assert r.status_code == 403
    for host in ("127.0.0.1:8123", "localhost:9999", "[::1]:8123", "LOCALHOST"):
        assert env.client.get("/api/health", headers={"Host": host}).status_code == 200, host


def test_index_injects_token_and_sets_cookie(env: Env) -> None:
    r = env.anon.get("/")
    assert r.status_code == 200
    assert f'content="{TOKEN}"' in r.text and "__SIRIO_TOKEN__" not in r.text
    assert r.headers["cache-control"] == "no-store"
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    # il cookie vale per le letture (immagini, download) e per heartbeat/bye...
    assert env.anon.get("/api/state").status_code == 200
    assert env.anon.post("/api/heartbeat").status_code == 200
    assert env.anon.post("/api/bye").status_code == 200
    # ...ma non per le operazioni che modificano i dati
    assert env.anon.delete("/api/documents").status_code == 403
    assert env.anon.put("/api/settings", json={"tema": "scuro"}).status_code == 403
    assert env.events == ["heartbeat", "bye"]


def test_static_files_no_cache_and_no_cors(env: Env) -> None:
    r = env.anon.get("/static/styles.css")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    r = env.client.get("/api/state", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
    r = env.client.options("/api/state", headers={
        "Origin": "http://evil.example", "Access-Control-Request-Method": "DELETE"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
    (env.web_dir.parent / "segreto.txt").write_text("SEGRETO", encoding="utf-8")
    for path in ("/static/..%2Fsegreto.txt", "/static/%2E%2E/segreto.txt", "/static/..%5Csegreto.txt",
                 "/static/%2E%2E%2F%2E%2E%2Fsegreto.txt"):
        r = env.anon.get(path)
        assert r.status_code in (403, 404), path
        assert "SEGRETO" not in r.text


def test_errors_are_italian_json(env: Env) -> None:
    r = env.client.get("/api/non-esiste")
    assert r.status_code == 404 and r.json() == {"detail": "Risorsa non trovata."}
    r = env.client.patch("/api/state")
    assert r.status_code == 405 and r.json()["detail"] == "Metodo non consentito per questa risorsa."
    r = env.client.get("/api/documents/123456789abc/crop?x0=a&y0=0&x1=1&y1=1")
    assert r.status_code == 422 and r.json()["detail"].startswith("Dati non validi")
    r = env.client.put("/api/documents/123456789abc", content=b"{rotto", headers={"content-type": "application/json"})
    assert r.status_code == 422 and isinstance(r.json()["detail"], str)


# ==========================================================================
# Importazione, stato, elaborazione
# ==========================================================================

def test_upload_pdf_and_png_and_errors(env: Env, png: bytes, pdf: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    r = env.upload(("foglio.png", png), ("scansione.PDF", pdf), ("note.txt", b"ciao"), ("vuoto.pdf", b""),
                   ("rotto.pdf", b"%PDF-1.7 rotto"))
    assert r.status_code == 200, r.text
    body = r.json()
    names = sorted(d["source_file"] for d in body["documents"])
    assert names == ["foglio.png", "scansione.PDF"]
    errors = {e["file"]: e["message"] for e in body["errors"]}
    assert set(errors) == {"note.txt", "vuoto.pdf", "rotto.pdf"}
    assert "non supportato" in errors["note.txt"]
    assert "vuoto" in errors["vuoto.pdf"]
    summary = body["documents"][0]
    for key in ("id", "display_name", "source_file", "source_page", "page_count", "status", "progress",
                "status_message", "error", "is_foglio_firma", "engine", "model", "user_verified",
                "created_at", "updated_at", "header", "totals", "cost_usd", "thumb_url"):
        assert key in summary, key
    assert set(summary["header"]) >= {"operatore", "alunno", "istituto", "ente", "mese", "anno", "ore_pei"}
    assert set(summary["totals"]) >= {"ore_riconosciute", "ore_dichiarate", "ore_calcolate",
                                      "totale_mensile_dichiarato", "differenza_totale", "giorni_lavorati",
                                      "n_errori", "n_attenzioni", "n_info", "campi_incerti",
                                      "campi_illeggibili", "stato"}
    # duplicato: nessun nuovo documento
    r = env.upload(("stesso.png", png))
    assert len(r.json()["documents"]) == 1 and r.json()["duplicates"]
    assert len(env.store.list()) == 2
    # limite di dimensione (abbassato per il test)
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 1000)
    r = env.upload(("grande.png", png))
    assert "supera il limite di 200 MB" in r.json()["errors"][0]["message"]
    # richieste non valide
    r = env.client.post("/api/upload", files=[("altro", ("a.txt", b"x", "text/plain"))])
    assert r.status_code == 400 and "Nessun file" in r.json()["detail"]
    assert env.client.post("/api/upload", data={"x": "1"}).status_code == 415
    r = env.client.post("/api/upload", json={"files": []})
    assert r.status_code == 415 and "multipart" in r.json()["detail"]


def test_processing_and_state_polling(env: Env, png: bytes) -> None:
    r = env.client.get("/api/state")
    rev = r.json()["revision"]
    assert env.client.get(f"/api/state?since={rev}").json() == {"revision": rev, "unchanged": True}
    doc_id = env.completed_doc(png)
    state = env.client.get(f"/api/state?since={rev}").json()
    assert state["revision"] > rev and "unchanged" not in state
    (summary,) = state["documents"]
    assert summary["id"] == doc_id and summary["status"] == "completato" and summary["progress"] == 1.0
    assert summary["header"]["operatore"] == "ROSSI MARIO"
    assert summary["thumb_url"] == f"/api/documents/{doc_id}/thumb"
    queue = state["queue"]
    assert set(queue) >= {"in_coda", "in_lavorazione", "completati", "errori", "motore_pronto", "messaggio"}
    assert queue["completati"] == 1 and queue["motore_pronto"] is True
    kpi = state["kpi"]
    assert set(kpi) >= {"documenti", "completati", "ore_totali", "errori", "attenzioni", "illeggibili",
                        "da_verificare", "verificati", "costo_usd"}
    assert kpi["documenti"] == 1 and kpi["completati"] == 1 and kpi["ore_totali"] > 0
    assert kpi["illeggibili"] >= 1 and kpi["da_verificare"] == 1 and kpi["costo_usd"] > 0
    assert env.client.get("/api/state?since=abc").status_code == 200


def test_get_document_and_images(env: Env, png: bytes) -> None:
    doc_id = env.completed_doc(png)
    d = env.doc(doc_id)
    assert d["id"] == doc_id and len(d["rows"]) == 31
    assert d["image_url"] == f"/api/documents/{doc_id}/image"
    grid = d["grid"]
    assert len(grid["col_x"]) == 11 and len(grid["row_y"]) == 32
    assert len(d["giorni"]) == 31 and d["giorni"][0]["tipo_giorno"] == "domenica"   # 1 febbraio 2026
    assert d["stati_intestazione"] == {"ore_pei": "incerto"}
    assert env.client.get("/api/documents/000000000000").status_code == 404

    full = env.client.get(f"/api/documents/{doc_id}/image")
    assert full.status_code == 200 and full.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(full.content, np.uint8), cv2.IMREAD_COLOR)
    assert (img.shape[1], img.shape[0]) == (grid["width"], grid["height"])
    small = env.client.get(f"/api/documents/{doc_id}/image?max=300")
    simg = cv2.imdecode(np.frombuffer(small.content, np.uint8), cv2.IMREAD_COLOR)
    assert max(simg.shape[:2]) == 300
    assert env.client.get(f"/api/documents/{doc_id}/image?max=5").status_code == 422
    thumb = env.client.get(f"/api/documents/{doc_id}/thumb")
    assert thumb.status_code == 200 and thumb.content[:2] == b"\xff\xd8"


def test_crop_bounds(env: Env, png: bytes) -> None:
    doc_id = env.completed_doc(png)
    grid = env.doc(doc_id)["grid"]
    w, h = grid["width"], grid["height"]
    r = env.client.get(f"/api/documents/{doc_id}/crop?x0=10&y0=20&x1=110&y1=70&scale=2")
    assert r.status_code == 200
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (100, 200)
    # coordinate invertite e parzialmente fuori pagina: ritaglio limitato all'immagine
    r = env.client.get(f"/api/documents/{doc_id}/crop?x0={w + 500}&y0={h + 500}&x1={w - 50}&y1={h - 40}&scale=1")
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (40, 50)
    for query in (f"x0={w}&y0=0&x1={w + 10}&y1=10", "x0=-50&y0=-50&x1=-1&y1=-1", "x0=5&y0=5&x1=5&y1=50"):
        r = env.client.get(f"/api/documents/{doc_id}/crop?{query}")
        assert r.status_code == 422 and "ritaglio" in r.json()["detail"], query
    assert env.client.get(f"/api/documents/{doc_id}/crop?x0=0&y0=0&x1=10&y1=10&scale=50").status_code == 422
    assert env.client.get(f"/api/documents/{doc_id}/crop?x0=0&y0=0").status_code == 422
    big = env.client.get(f"/api/documents/{doc_id}/crop?x0=0&y0=0&x1={w}&y1={h}&scale=6")
    bimg = cv2.imdecode(np.frombuffer(big.content, np.uint8), cv2.IMREAD_COLOR)
    assert max(bimg.shape[:2]) <= 4000


# ==========================================================================
# Modifica
# ==========================================================================

def test_put_tracks_edits_and_revalidates(env: Env, png: bytes) -> None:
    doc_id = env.completed_doc(png)
    before = env.doc(doc_id)
    row2 = before["rows"][1]
    assert row2["eff_uscita"] == "11:00" and "eff_uscita" in row2["incerti"]
    rev = env.store.revision

    r = env.client.put(f"/api/documents/{doc_id}", json={"rows": [{"giorno": 2, "eff_uscita": "11.30"}]})
    assert r.status_code == 200, r.text
    d = r.json()
    row2 = d["rows"][1]
    assert row2["eff_uscita"] == "11:30"
    assert "eff_uscita" not in row2["incerti"]
    assert d["user_edited"] == ["rows.2.eff_uscita"]
    assert d["ocr_originali"] == {"rows.2.eff_uscita": "11:00"}
    assert d["giorni"][1]["stati"] == {"eff_uscita": "corretto"}
    # rivalidato: 3 ore dichiarate contro 3,5 calcolate
    assert any(a["codice"] == "E01_ORE_NON_COERENTI" and a["giorno"] == 2 for a in d["anomalies"])
    assert d["totals"]["stato"] == "errori"
    assert env.store.revision > rev

    # seconda modifica: resta il primo valore OCR
    d = env.client.put(f"/api/documents/{doc_id}", json={"rows": [{"giorno": 2, "eff_uscita": "12:00"}]}).json()
    assert d["ocr_originali"] == {"rows.2.eff_uscita": "11:00"}
    # ritorno al valore letto: non e' piu' una correzione
    d = env.client.put(f"/api/documents/{doc_id}", json={"rows": [{"giorno": 2, "eff_uscita": "11:00"}]}).json()
    assert d["user_edited"] == [] and d["ocr_originali"] == {}
    assert not any(a["codice"] == "E01_ORE_NON_COERENTI" and a["giorno"] == 2 for a in d["anomalies"])

    # campo illeggibile compilato a mano (nota del giorno 17 nel motore finto)
    row17 = d["rows"][16]
    assert "note" in row17["illeggibili"] and row17["note"] is None
    d = env.client.put(f"/api/documents/{doc_id}", json={
        "rows": [{**row17, "note": "PONTE DI CARNEVALE"}],
        "header": {"operatore": "ROSSI MARIO", "ore_pei": "15", "mese": "febbraio", "incerti": []},
        "user_verified": True,
    }).json()
    row17 = d["rows"][16]
    assert row17["note"] == "PONTE DI CARNEVALE" and row17["illeggibili"] == []
    assert d["ocr_originali"]["rows.17.note"] is None
    assert "rows.17.note" in d["user_edited"]
    assert d["header"]["incerti"] == [] and d["header"]["mese"] == 2
    assert "header.operatore" not in d["user_edited"]   # valore invariato
    assert d["user_verified"] is True
    assert d["totals"]["campi_illeggibili"] == 0

    # booleani, trattino e ore
    d = env.client.put(f"/api/documents/{doc_id}", json={"rows": [
        {"giorno": 3, "eff_entrata": "-", "eff_uscita": "", "ore_dichiarate": "1,5", "assenza_alunno": "sì"}]}).json()
    row3 = d["rows"][2]
    assert row3["eff_entrata"] is None and row3["trattino_effettivo"] is True
    assert row3["ore_dichiarate"] == 1.5 and row3["assenza_alunno"] is True
    assert {"rows.3.eff_entrata", "rows.3.ore_dichiarate", "rows.3.assenza_alunno"} <= set(d["user_edited"])
    assert d["ocr_originali"]["rows.3.assenza_alunno"] == "false"

    # dati non validi: messaggio italiano e nessuna modifica
    saved = env.store.get(doc_id)
    for payload, needle in (
        ({"rows": [{"giorno": 4, "ore_dichiarate": "tre ore"}]}, "Giorno 4"),
        ({"rows": [{"giorno": 40, "note": "x"}]}, "Giorno non valido"),
        ({"header": {"mese": 13}}, "Mese"),
        ({"header": {"sostituzione": "forse"}}, "Sostituzione"),
        ({"rows": "tutte"}, "righe"),
    ):
        r = env.client.put(f"/api/documents/{doc_id}", json=payload)
        assert r.status_code == 422 and needle in r.json()["detail"], (payload, r.text)
    assert env.store.get(doc_id).rows == saved.rows
    assert env.client.put("/api/documents/000000000000", json={}).status_code == 404


def test_put_rejected_while_processing_and_manual_classification(env: Env, png: bytes) -> None:
    env.engine.delay = 1.0
    r = env.upload(("foglio.png", png))
    doc_id = r.json()["documents"][0]["id"]
    env.wait_status(doc_id, "in_lavorazione")
    r = env.client.put(f"/api/documents/{doc_id}", json={"user_verified": True})
    assert r.status_code == 409 and "elaborazione" in r.json()["detail"]
    env.wait_status(doc_id)
    d = env.client.put(f"/api/documents/{doc_id}", json={"is_foglio_firma": False}).json()
    assert d["status"] == "scartato" and d["anomalies"][0]["codice"] == "E06_NON_FOGLIO_FIRMA"
    d = env.client.put(f"/api/documents/{doc_id}", json={"is_foglio_firma": True}).json()
    assert d["status"] == "completato"


def test_reprocess_and_delete(env: Env, png: bytes, pdf: bytes) -> None:
    doc_id = env.completed_doc(png)
    env.client.put(f"/api/documents/{doc_id}", json={"rows": [{"giorno": 2, "eff_uscita": "12:00"}]})
    env.engine.delay = 0.4
    r = env.client.post(f"/api/documents/{doc_id}/reprocess")
    assert r.status_code == 200 and r.json()["status"] in ("in_coda", "in_lavorazione")
    env.wait_status(doc_id)
    d = env.doc(doc_id)
    assert d["user_edited"] == [] and d["rows"][1]["eff_uscita"] == "11:00"
    assert len(env.engine.calls) == 2
    assert env.client.post("/api/documents/000000000000/reprocess").status_code == 404

    other = env.upload(("altro.pdf", pdf)).json()["documents"][0]["id"]
    assert env.client.delete(f"/api/documents/{doc_id}").json() == {"ok": True}
    assert env.client.delete(f"/api/documents/{doc_id}").status_code == 404
    assert env.client.get(f"/api/documents/{doc_id}/thumb").status_code == 404
    assert [d["id"] for d in env.client.get("/api/state").json()["documents"]] == [other]
    assert env.client.delete("/api/documents").json() == {"ok": True}
    assert env.client.get("/api/state").json()["documents"] == []
    assert env.client.get("/api/documents").json() == []


def test_preview(env: Env, png: bytes) -> None:
    doc_id = env.completed_doc(png)
    p = env.client.get("/api/preview").json()
    assert [d["id"] for d in p["documenti"]] == [doc_id]
    days = {r["giorno"]: r for r in p["dettaglio"]}
    assert days[2]["data"] == "2026-02-02" and days[2]["giorno_settimana"] == "lun"
    assert days[2]["stati"] == {"eff_uscita": "incerto"}
    assert days[17]["stati"] == {"note": "illeggibile"}
    assert 1 not in days   # domenica vuota
    assert any(a["codice"] == "E05_CAMPO_ILLEGGIBILE" for a in p["anomalie"])
    full = env.client.get("/api/preview?giorni_vuoti=true").json()
    assert len(full["dettaglio"]) == 28


# ==========================================================================
# Impostazioni
# ==========================================================================

def test_settings_get_put_never_echo_key(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    kicks: list[int] = []
    real_kick = env.processor.kick
    monkeypatch.setattr(env.processor, "kick", lambda: (kicks.append(1), real_kick()))
    s = env.client.get("/api/settings").json()
    assert s["api_key_set"] is False and s["api_key_hint"] is None and s["api_key_source"] is None
    assert set(s["engines"]) == {"claude", "locale"}
    assert s["settings"]["engine"] in ("locale", "claude")
    assert [m["id"] for m in s["models"]][0] == "claude-opus-5-5"
    assert s["version"] == "1.0.0" and s["data_dir"] and s["export_dir"]

    r = env.client.put("/api/settings", json={"api_key": FAKE_KEY, "engine": "claude", "concorrenza": 5,
                                              "tema": "scuro", "sconosciuta": 1})
    assert r.status_code == 200, r.text
    assert FAKE_KEY not in r.text and FAKE_KEY[10:-4] not in r.text
    s = r.json()
    assert s["api_key_set"] is True and s["api_key_source"] == "file"
    assert s["api_key_hint"].endswith(FAKE_KEY[-4:]) and len(s["api_key_hint"]) < 16
    assert s["settings"]["concorrenza"] == 5 and s["settings"]["tema"] == "scuro"
    assert (config.data_dir() / ".chiave").read_text(encoding="utf-8") == FAKE_KEY
    assert config.load_settings().engine == "claude"
    assert FAKE_KEY not in env.client.get("/api/settings").text
    assert kicks == [1]

    for payload, needle in (
        ({"concorrenza": 0}, "Concorrenza"),
        ({"concorrenza": "molti"}, "Concorrenza"),
        ({"engine": "magico"}, "engine"),
        ({"claude_model": "gpt"}, "Modello"),
        ({"api_key": "corta"}, "chiave API"),
        ({"api_key": "sk-ant-con spazi dentro la chiave"}, "chiave API"),
    ):
        r = env.client.put("/api/settings", json=payload)
        assert r.status_code == 422 and needle in r.json()["detail"], (payload, r.text)
    assert config.load_settings().concorrenza == 5

    s = env.client.put("/api/settings", json={"settings": {"engine": "locale"}, "api_key": ""}).json()
    assert s["api_key_set"] is False and s["settings"]["engine"] == "locale"
    assert not (config.data_dir() / ".chiave").exists()


def test_settings_test_endpoint(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    r = env.client.post("/api/settings/test")
    assert r.json()["ok"] is False and "Nessuna chiave" in r.json()["message"]
    seen: list[tuple[str, str]] = []
    fake = types.ModuleType("sirio.engines.claude_engine")

    def test_api_key(api_key: str, model: str) -> tuple[bool, str]:
        seen.append((api_key, model))
        return api_key == FAKE_KEY, "Chiave valida." if api_key == FAKE_KEY else "Chiave API non valida."

    fake.test_api_key = test_api_key  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sirio.engines.claude_engine", fake)
    config.set_api_key(FAKE_KEY)
    r = env.client.post("/api/settings/test")
    assert r.json() == {"ok": True, "message": "Chiave valida."}
    r = env.client.post("/api/settings/test", json={"api_key": "sk-ant-altra-chiave-di-prova-xyz", "model": "claude-haiku-5-5"})
    assert r.json()["ok"] is False
    assert seen == [(FAKE_KEY, "claude-opus-5-5"), ("sk-ant-altra-chiave-di-prova-xyz", "claude-haiku-5-5")]
    assert FAKE_KEY not in r.text

    def broken(api_key: str, model: str) -> tuple[bool, str]:
        raise RuntimeError("rete")

    fake.test_api_key = broken  # type: ignore[attr-defined]
    r = env.client.post("/api/settings/test")
    assert r.json()["ok"] is False and "imprevisto" in r.json()["message"]


def test_settings_put_kicks_waiting_documents(env: Env, png: bytes) -> None:
    env.engine.available = False
    env.processor.kick()
    r = env.upload(("foglio.png", png))
    doc_id = r.json()["documents"][0]["id"]
    wait_for(lambda: env.store.get(doc_id).status_message == "In attesa: inserisci la chiave API nelle Impostazioni")
    assert env.client.get("/api/state").json()["queue"]["motore_pronto"] is False
    env.engine.available = True
    env.client.put("/api/settings", json={"api_key": FAKE_KEY})
    env.wait_status(doc_id)


# ==========================================================================
# Esportazione, download, apertura
# ==========================================================================

def _ensure_exporter(monkeypatch: pytest.MonkeyPatch) -> bool:
    """True se si usa il vero excel_export; altrimenti installa un sostituto minimo."""
    if importlib.util.find_spec("sirio.excel_export") is not None:
        try:
            import sirio.excel_export  # noqa: F401, PLC0415

            return True
        except Exception:  # noqa: BLE001
            pass
    from pydantic import BaseModel  # noqa: PLC0415

    fake = types.ModuleType("sirio.excel_export")

    class ExportOptions(BaseModel):
        fogli_per_documento: bool = True
        giorni_vuoti: bool = False
        titolo: str | None = None

    def default_filename(docs) -> str:
        return "Rendicontazione_2026-02_20261009-1530.xlsx"

    def export_workbook(docs, path, options=None):
        Path(path).write_bytes(b"PK\x03\x04finto")
        return Path(path)

    fake.ExportOptions, fake.default_filename, fake.export_workbook = ExportOptions, default_filename, export_workbook
    monkeypatch.setitem(sys.modules, "sirio.excel_export", fake)
    return False


def test_export_download_and_listing(env: Env, png: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_exporter(monkeypatch)
    r = env.client.post("/api/export", json={})
    assert r.status_code == 400 and "Nessun foglio firma completato" in r.json()["detail"]
    doc_id = env.completed_doc(png)
    r = env.client.post("/api/export", json={"options": {"giorni_vuoti": True}})
    assert r.status_code == 200, r.text
    out = r.json()
    path = Path(out["path"])
    assert path.is_file() and path.parent == config.export_dir()
    assert out["filename"].startswith("Rendicontazione_2026-02") and out["filename"].endswith(".xlsx")
    assert out["url"] == f"/api/exports/{out['filename']}"
    # secondo export nello stesso minuto: nome univoco
    out2 = env.client.post("/api/export", json={"ids": [doc_id]}).json()
    assert out2["filename"] != out["filename"]
    named = env.client.post("/api/export", json={"ids": [doc_id], "filename": "../../Mio report: febbraio"}).json()
    assert named["filename"] == "Mio report_ febbraio.xlsx"
    assert Path(named["path"]).parent == config.export_dir()
    assert env.client.post("/api/export", json={"ids": ["000000000000"]}).status_code == 404

    listing = env.client.get("/api/exports").json()
    assert {e["filename"] for e in listing} == {out["filename"], out2["filename"], named["filename"]}
    assert all(set(e) >= {"filename", "path", "size", "created_at", "url"} for e in listing)

    dl = env.client.get(out["url"])
    assert dl.status_code == 200 and dl.content[:2] == b"PK"
    assert "attachment" in dl.headers["content-disposition"]
    # il download funziona anche con il solo cookie (link nella pagina)
    env.anon.get("/")
    assert env.anon.get(out["url"]).status_code == 200


def test_export_path_traversal_rejected(env: Env, tmp_path: Path) -> None:
    secret = tmp_path / "segreto.xlsx"
    secret.write_bytes(b"PK segreto")
    (config.export_dir() / ".nascosto.xlsx").write_bytes(b"PK")
    (config.export_dir() / "dati.json").write_text("{}", encoding="utf-8")
    link = config.export_dir() / "collegamento.xlsx"
    try:
        link.symlink_to(secret)
    except OSError:
        link = None
    attempts = [
        "..%2Fsegreto.xlsx", "%2E%2E%2Fsegreto.xlsx", "..%5Csegreto.xlsx", "%2E%2E%5C%2E%2E%5Csegreto.xlsx",
        ".nascosto.xlsx", "dati.json", "..", "%00.xlsx", "C:%5Csegreto.xlsx", "inesistente.xlsx",
    ]
    if link is not None:
        attempts.append("collegamento.xlsx")
    for name in attempts:
        r = env.client.get(f"/api/exports/{name}")
        assert r.status_code in (400, 404), (name, r.status_code)
        assert b"segreto" not in r.content
    r = env.client.get("/api/exports/../dati/impostazioni.json")
    assert r.status_code == 404


def test_export_errors_are_reported(env: Env, png: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_exporter(monkeypatch)
    env.completed_doc(png)
    import sirio.excel_export as xe  # noqa: PLC0415

    def locked(docs, path, options=None):
        raise PermissionError("Impossibile salvare «x.xlsx»: il file è aperto in un altro programma (ad esempio Excel).")

    monkeypatch.setattr(xe, "export_workbook", locked)
    r = env.client.post("/api/export", json={})
    assert r.status_code == 500 and "aperto in un altro programma" in r.json()["detail"]
    r = env.client.post("/api/export", json={"options": {"giorni_vuoti": "forse"}})
    assert r.status_code == 422


def test_open_endpoint_restricted(env: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    opened: list[Path] = []
    monkeypatch.setattr(server, "open_path", lambda p: opened.append(Path(p)))
    (config.export_dir() / "report.xlsx").write_bytes(b"PK")
    assert env.client.post("/api/open", json={"target": "export_dir"}).json()["ok"] is True
    assert env.client.post("/api/open", json={"target": "file", "filename": "report.xlsx"}).status_code == 200
    assert opened == [config.export_dir(), (config.export_dir() / "report.xlsx").resolve()]
    for payload, code in (
        ({"target": "file", "filename": "../../etc/passwd"}, 400),
        ({"target": "file", "filename": "..\\segreto.xlsx"}, 400),
        ({"target": "file", "filename": "manca.xlsx"}, 404),
        ({"target": "file"}, 422),
        ({"target": "cartella_dati"}, 422),
        ({"target": "file", "filename": str(tmp_path / "x.xlsx")}, 400),
    ):
        r = env.client.post("/api/open", json=payload)
        assert r.status_code == code, (payload, r.text)
        assert isinstance(r.json()["detail"], str)
    assert len(opened) == 2


def test_lifecycle_endpoints(env: Env) -> None:
    assert env.client.post("/api/heartbeat").json() == {"ok": True}
    assert env.client.post("/api/bye").json() == {"ok": True}
    assert env.client.post("/api/shutdown").json() == {"ok": True}
    wait_for(lambda: "shutdown" in env.events)
    assert env.events[:2] == ["heartbeat", "bye"]


def test_safe_export_name() -> None:
    assert server.safe_export_name("Report finale") == "Report finale.xlsx"
    assert server.safe_export_name("a/b\\c:d*e?.XLSX") == "c_d_e_.xlsx"
    assert server.safe_export_name("CON") == "_CON.xlsx"
    assert server.safe_export_name("...") is None
    assert server.safe_export_name("") is None


# ==========================================================================
# Avvio (sirio/app.py)
# ==========================================================================

class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_lifecycle_rules() -> None:
    clock = Clock()
    lc = sirio_app.Lifecycle(auto_shutdown=True, clock=clock)
    assert lc.check() is None
    lc.bye()
    clock.t += 5
    assert lc.check() is None
    lc.heartbeat()               # ricaricamento della pagina: annulla la chiusura
    clock.t += 20
    assert lc.check() is None
    lc.bye()
    clock.t += 8.5
    assert lc.check() == "finestra chiusa"

    lc = sirio_app.Lifecycle(auto_shutdown=True, clock=clock)
    lc.window_exited()
    clock.t += 5
    lc.heartbeat()               # finestra passata a un altro processo del browser: ancora viva
    clock.t += 9
    assert lc.check() is None
    clock.t += 2
    assert lc.check() == "processo della finestra terminato"

    lc = sirio_app.Lifecycle(auto_shutdown=True, clock=clock)
    clock.t += 14 * 60
    assert lc.check() is None
    clock.t += 61
    assert lc.check() == "nessuna attività da 15 minuti"
    lc.resumed()
    assert lc.check() is None

    lc = sirio_app.Lifecycle(auto_shutdown=False, clock=clock)
    lc.bye()
    lc.window_exited()
    clock.t += 3600
    assert lc.check() is None
    lc.request_stop("prova")
    lc.request_stop("altro")
    assert lc.stop_event.is_set() and lc.reason == "prova"


def test_window_command_and_browser_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "browser"
    exe.write_text("", encoding="utf-8")
    cmd = sirio_app.window_command(exe, "http://127.0.0.1:5000/", tmp_path / "finestra")
    assert cmd == [str(exe), "--app=http://127.0.0.1:5000/", f"--user-data-dir={tmp_path / 'finestra'}",
                   "--window-size=1480,940", "--no-first-run", "--no-default-browser-check"]
    monkeypatch.setenv("SIRIO_BROWSER", str(exe))
    assert sirio_app.find_browser() == ("personalizzato", exe)
    monkeypatch.setenv("SIRIO_BROWSER", str(tmp_path / "manca"))
    monkeypatch.setattr(sirio_app.shutil, "which", lambda _name: None)
    if sys.platform.startswith("linux"):
        assert sirio_app.find_browser() is None
    # nessun browser: si usa il browser predefinito, senza eccezioni
    opened: list[str] = []
    monkeypatch.setattr(sirio_app, "find_browser", lambda: None)
    monkeypatch.setattr(sirio_app.webbrowser, "open", lambda url, new=0: opened.append(url) or True)
    assert sirio_app.open_window("http://127.0.0.1:5000/", tmp_path / "f") is None
    assert opened == ["http://127.0.0.1:5000/"]
    monkeypatch.setattr(sirio_app.webbrowser, "open", lambda url, new=0: (_ for _ in ()).throw(RuntimeError("x")))
    assert sirio_app.open_window("http://127.0.0.1:5000/", tmp_path / "f") is None


def test_open_window_launches_app_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class FakePopen:
        pid = 4321

        def __init__(self, cmd, **kwargs):
            calls.append(cmd)
            assert kwargs["stdout"] is subprocess.DEVNULL

    monkeypatch.setattr(sirio_app, "find_browser", lambda: ("Microsoft Edge", Path("/opt/edge/msedge")))
    monkeypatch.setattr(sirio_app.subprocess, "Popen", FakePopen)
    proc = sirio_app.open_window("http://127.0.0.1:5000/", tmp_path / "finestra")
    assert isinstance(proc, FakePopen)
    assert calls[0][1] == "--app=http://127.0.0.1:5000/"
    assert (tmp_path / "finestra").is_dir()


def test_instance_lock_is_exclusive(tmp_path: Path) -> None:
    a = sirio_app.InstanceLock(tmp_path / "istanza.lock")
    b = sirio_app.InstanceLock(tmp_path / "istanza.lock")
    assert a.acquire() is True
    assert b.acquire() is False
    a.release()
    assert b.acquire() is True
    b.release()


def test_parse_args() -> None:
    args = sirio_app.parse_args(["--porta", "8765", "--senza-finestra", "--cartella-dati", "x", "--log-level", "debug"])
    assert (args.porta, args.senza_finestra, args.cartella_dati, args.log_level) == (8765, True, "x", "DEBUG")
    with pytest.raises(SystemExit):
        sirio_app.parse_args(["--porta", "70000"])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str, token: str | None = None, timeout: float = 2.0) -> dict:
    req = urllib.request.Request(url, headers={"X-Sirio-Token": token} if token else {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_http(url: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return _get(url)
        except OSError:
            time.sleep(0.2)
    raise AssertionError(f"{url} non risponde")


@pytest.mark.skipif(sys.platform.startswith("win"), reason="usa i segnali POSIX")
def test_main_headless_single_instance_and_shutdown(tmp_path: Path) -> None:
    port = _free_port()
    data = tmp_path / "dati"
    env = {**os.environ, "SIRIO_EXPORT_DIR": str(tmp_path / "export"), "PYTHONUNBUFFERED": "1"}
    env.pop("SIRIO_DATA_DIR", None)
    cmd = [sys.executable, "-m", "sirio", "--senza-finestra", "--porta", str(port), "--cartella-dati", str(data)]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        assert _wait_http(f"http://127.0.0.1:{port}/api/health")["ok"] is True
        info = json.loads((data / "istanza.json").read_text(encoding="utf-8"))
        assert info["port"] == port and info["pid"] == proc.pid and len(info["token"]) >= 24
        state = _get(f"http://127.0.0.1:{port}/api/state", info["token"])
        assert state["documents"] == []
        # seconda istanza: rimanda alla prima ed esce subito
        second = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        assert second.returncode == 0
        assert "già in esecuzione" in second.stdout
        assert proc.poll() is None
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 0, out
    assert not (data / "istanza.json").exists()
    assert (data / "log" / "sirio.log").is_file()
    assert "Sirio OCR terminato" in (data / "log" / "sirio.log").read_text(encoding="utf-8")


def test_main_fails_cleanly_on_busy_port(tmp_path: Path) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        env = {**os.environ, "SIRIO_EXPORT_DIR": str(tmp_path / "export")}
        env.pop("SIRIO_DATA_DIR", None)
        res = subprocess.run(
            [sys.executable, "-m", "sirio", "--senza-finestra", "--porta", str(port),
             "--cartella-dati", str(tmp_path / "dati")],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
        )
    assert f"La porta {port} non è disponibile" in res.stderr
    assert not (tmp_path / "dati" / "istanza.json").exists()


# ==========================================================================
# Server di sviluppo
# ==========================================================================

def test_dev_server_runs(tmp_path: Path) -> None:
    port = _free_port()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    env.pop("SIRIO_DATA_DIR", None)
    env.pop("SIRIO_EXPORT_DIR", None)
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "tests" / "dev_server.py"), "--porta", str(port), "--documenti", "2",
         "--ritardo", "0.2", "--token", "sviluppo", "--cartella-dati", str(tmp_path / "dev")],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        _wait_http(f"http://127.0.0.1:{port}/api/health")
        deadline = time.monotonic() + 60
        state: dict = {}
        while time.monotonic() < deadline:
            state = _get(f"http://127.0.0.1:{port}/api/state", "sviluppo")
            statuses = sorted(d["status"] for d in state["documents"])
            if len(statuses) == 4 and "in_coda" not in statuses and "in_lavorazione" not in statuses:
                break
            time.sleep(0.3)
        statuses = sorted(d["status"] for d in state["documents"])
        assert statuses == ["completato", "completato", "errore", "scartato"], state
        assert state["kpi"]["ore_totali"] > 0
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
