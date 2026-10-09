"""Test della coda di elaborazione (sirio/processing.py) con il motore finto."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from sirio import anagrafica, config
from sirio.config import Settings
from sirio.engines.base import EngineError
from sirio.processing import GENERIC_ERROR, WAITING_NO_KEY, Processor, waiting_message
from sirio.store import DocumentStore
from tests.fake_engine import FakeEngine
from tests.synthetic import default_data, make_synthetic_sheet


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIRIO_DATA_DIR", str(tmp_path / "dati"))
    monkeypatch.setenv("SIRIO_EXPORT_DIR", str(tmp_path / "export"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(config, "_keyring", lambda: None)


@pytest.fixture(scope="module")
def png() -> bytes:
    img, _ = make_synthetic_sheet(seed=11, dpi=100)
    return cv2.imencode(".png", img)[1].tobytes()


def variant(png: bytes, n: int) -> bytes:
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    img[1, n % img.shape[1]] = (255 - n % 256, 10, 10)
    return cv2.imencode(".png", img)[1].tobytes()


def wait_for(predicate: Callable[[], bool], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condizione non raggiunta entro il tempo limite")


class Env:
    def __init__(self, tmp_path: Path, engine: FakeEngine, settings: Settings | None = None):
        self.store = DocumentStore(tmp_path / "documenti")
        self.engine = engine
        self.settings = settings or Settings(engine="claude", concorrenza=3)
        self.processor = Processor(self.store, lambda: self.settings, lambda _s: self.engine)

    def add(self, data: bytes, name: str = "foglio.png") -> list[str]:
        return [d.id for d in self.store.add_file(name, data, on_document=lambda d: self.processor.enqueue(d.id))]

    def status(self, doc_id: str) -> str:
        doc = self.store.get(doc_id)
        return doc.status if doc else "eliminato"


@pytest.fixture()
def make_env(tmp_path: Path):
    envs: list[Env] = []

    def factory(engine: FakeEngine, settings: Settings | None = None, start: bool = True) -> Env:
        env = Env(tmp_path, engine, settings)
        if start:
            env.processor.start()
        envs.append(env)
        return env

    yield factory
    for env in envs:
        env.processor.stop(timeout=5)


def test_document_processed_to_completion(make_env, png: bytes) -> None:
    env = make_env(FakeEngine(delay=0.2))
    (doc_id,) = env.add(png)
    wait_for(lambda: env.status(doc_id) == "completato")
    doc = env.store.get(doc_id)
    assert doc.progress == 1.0 and doc.error is None
    assert doc.status_message == "Lettura completata"
    assert doc.engine == "claude" and doc.model == "claude-opus-5-5"
    assert doc.header.operatore == "ROSSI MARIO"
    assert doc.usage.seconds > 0 and doc.usage.cost_usd > 0
    # validazione eseguita: campi incerti/illeggibili del motore finto contati e segnalati
    assert doc.totals.campi_incerti >= 1 and doc.totals.campi_illeggibili >= 1
    assert doc.totals.stato == "da_verificare"
    assert {a.codice for a in doc.anomalies} >= {"W04_CAMPO_INCERTO", "E05_CAMPO_ILLEGGIBILE"}
    assert env.processor.status()["completati"] == 1


def test_waits_without_api_key_then_kick(make_env, png: bytes) -> None:
    engine = FakeEngine(available=False, message="Chiave API mancante.")
    env = make_env(engine)
    (doc_id,) = env.add(png)
    wait_for(lambda: env.store.get(doc_id).status_message == WAITING_NO_KEY)
    time.sleep(0.3)
    doc = env.store.get(doc_id)
    assert doc.status == "in_coda"
    status = env.processor.status()
    assert status["motore_pronto"] is False and status["in_coda"] == 1
    assert engine.calls == []
    engine.available = True
    env.processor.kick()
    wait_for(lambda: env.status(doc_id) == "completato")
    assert env.processor.status()["motore_pronto"] is True


def test_waiting_message_for_local_engine(make_env, png: bytes) -> None:
    engine = FakeEngine(name="locale", available=False, message="Il modello di riconoscimento non è installato.")
    env = make_env(engine, Settings(engine="locale"))
    (doc_id,) = env.add(png)
    expected = "In attesa: il modello di riconoscimento non è installato"
    wait_for(lambda: env.store.get(doc_id).status_message == expected)
    assert env.processor.status()["messaggio"] == "Il modello di riconoscimento non è installato."
    assert waiting_message("locale", "PyTorch non è installato.") == "In attesa: PyTorch non è installato"


def test_engine_error_message_is_kept_verbatim(make_env, png: bytes) -> None:
    message = "Credito Anthropic esaurito: ricaricare il credito dalla console e premere «Rielabora»."
    env = make_env(FakeEngine(fail=EngineError(message)))
    (doc_id,) = env.add(png)
    wait_for(lambda: env.status(doc_id) == "errore")
    doc = env.store.get(doc_id)
    assert doc.error == message
    assert doc.status_message == "Lettura non riuscita"
    assert env.processor.status()["errori"] == 1


def test_unexpected_exception_generic_message_and_traceback(make_env, png: bytes,
                                                            caplog: pytest.LogCaptureFixture) -> None:
    env = make_env(FakeEngine(fail=RuntimeError("dettaglio interno")))
    with caplog.at_level(logging.ERROR, logger="sirio.processing"):
        (doc_id,) = env.add(png)
        wait_for(lambda: env.status(doc_id) == "errore")
    assert env.store.get(doc_id).error == GENERIC_ERROR
    assert any(r.exc_info and "dettaglio interno" in str(r.exc_info[1]) for r in caplog.records)


def test_not_a_timesheet_is_discarded(make_env, png: bytes) -> None:
    env = make_env(FakeEngine(unknown="scartato"))
    (doc_id,) = env.add(png, "verbale.png")
    wait_for(lambda: env.status(doc_id) == "scartato")
    doc = env.store.get(doc_id)
    assert doc.is_foglio_firma is False
    assert [a.codice for a in doc.anomalies] == ["E06_NON_FOGLIO_FIRMA"]
    assert env.processor.status()["scartati"] == 1


def test_restart_recovery(tmp_path: Path, make_env, png: bytes) -> None:
    env = make_env(FakeEngine(), start=False)
    ids = [d.id for d in env.store.add_file("a.png", png)] + [d.id for d in env.store.add_file("b.png", variant(png, 3))]

    def interrupted(d) -> None:
        d.status = "in_lavorazione"
        d.progress = 0.5
        d.status_message = "Lettura dei giorni 1–16…"

    env.store.update(ids[0], interrupted)
    # riavvio: nuovo archivio e nuova coda sulla stessa cartella
    store2 = DocumentStore(env.store.root)
    engine2 = FakeEngine(delay=0.05)
    proc2 = Processor(store2, lambda: Settings(engine="claude"), lambda _s: engine2)
    proc2.start()
    try:
        wait_for(lambda: all(store2.get(i).status == "completato" for i in ids))
        assert sorted(engine2.calls) == [("a.png", 1), ("b.png", 1)]
    finally:
        proc2.stop()


@pytest.mark.parametrize(("name", "concorrenza", "expected"), [("claude", 2, 2), ("locale", 3, 1)])
def test_concurrency_limits(make_env, png: bytes, name: str, concorrenza: int, expected: int) -> None:
    engine = FakeEngine(name=name, delay=0.4)
    env = make_env(engine, Settings(engine=name, concorrenza=concorrenza), start=False)
    ids = [env.add(variant(png, i), f"f{i}.png")[0] for i in range(4)]
    env.processor.start()
    wait_for(lambda: all(env.status(i) == "completato" for i in ids), timeout=20)
    assert engine.max_active == expected
    assert len(engine.calls) == 4


def test_progress_saves_are_throttled(make_env, png: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = FakeEngine(delay=1.2, steps=120)
    env = make_env(engine)
    saves: Counter[str] = Counter()
    progress_seen: list[float] = []
    real_save = env.store.save

    def counting_save(doc) -> None:
        saves[doc.id] += 1
        if doc.status == "in_lavorazione":
            progress_seen.append(doc.progress)
        real_save(doc)

    monkeypatch.setattr(env.store, "save", counting_save)
    (doc_id,) = env.add(png)
    wait_for(lambda: env.status(doc_id) == "completato")
    # 120 chiamate di avanzamento in 1,2 s -> al massimo ~4 salvataggi al secondo
    # (+ messa in coda, inizio, risultato finale)
    assert saves[doc_id] <= 1.2 * 4 + 5, saves[doc_id]
    assert progress_seen == sorted(progress_seen)
    assert progress_seen and max(progress_seen) <= 0.95 + 1e-9


def test_reprocess_supersedes_running_job(make_env, png: bytes) -> None:
    engine = FakeEngine(delay=0.6)
    env = make_env(engine)
    (doc_id,) = env.add(png)
    wait_for(lambda: env.status(doc_id) == "in_lavorazione")
    env.processor.enqueue(doc_id)          # "Rielabora" durante la lettura
    wait_for(lambda: len(engine.calls) == 2, timeout=10)
    wait_for(lambda: env.status(doc_id) == "completato")
    time.sleep(0.2)
    assert env.status(doc_id) == "completato"


def test_reprocess_clears_user_edits(make_env, png: bytes) -> None:
    env = make_env(FakeEngine(delay=0.05))
    (doc_id,) = env.add(png)
    wait_for(lambda: env.status(doc_id) == "completato")

    def edit(d) -> None:
        d.user_edited = ["rows.2.eff_uscita"]
        d.ocr_originali = {"rows.2.eff_uscita": "11:00"}
        d.user_verified = True

    env.store.update(doc_id, edit)
    env.processor.enqueue(doc_id)
    assert env.store.get(doc_id).status in ("in_coda", "in_lavorazione")
    wait_for(lambda: env.status(doc_id) == "completato")
    doc = env.store.get(doc_id)
    assert doc.user_edited == [] and doc.ocr_originali == {} and doc.user_verified is False


def test_delete_during_processing(make_env, png: bytes) -> None:
    engine = FakeEngine(delay=0.5)
    env = make_env(engine)
    (doc_id,) = env.add(png)
    wait_for(lambda: env.status(doc_id) == "in_lavorazione")
    env.processor.discard(doc_id)
    assert env.store.delete(doc_id)
    time.sleep(0.8)
    assert env.store.get(doc_id) is None
    assert not (env.store.root / doc_id).exists()
    assert env.processor.status()["attivi"] == 0


def test_engine_factory_failure_keeps_documents_waiting(tmp_path: Path, png: bytes) -> None:
    store = DocumentStore(tmp_path / "documenti")

    def factory(_settings: Settings):
        raise ImportError("No module named 'torch'")

    proc = Processor(store, lambda: Settings(engine="locale"), factory)
    proc.start()
    try:
        doc_id = store.add_file("x.png", png, on_document=lambda d: proc.enqueue(d.id))[0].id
        wait_for(lambda: store.get(doc_id).status_message.startswith("In attesa:"))
        assert "non è installato correttamente" in store.get(doc_id).status_message
        assert store.get(doc_id).status == "in_coda"
        ok, message = proc.engine_status("locale")
        assert ok is False and "installato" in message
    finally:
        proc.stop()


def test_engine_status_and_no_key_message(make_env) -> None:
    env = make_env(FakeEngine(available=False))
    ok, message = env.processor.engine_status("claude")
    assert ok is False and "Chiave API" in message


def test_stop_is_clean_and_idempotent(make_env, png: bytes) -> None:
    env = make_env(FakeEngine(delay=0.05))
    (doc_id,) = env.add(png)
    wait_for(lambda: env.status(doc_id) == "completato")
    started = time.monotonic()
    env.processor.stop()
    env.processor.stop()
    assert time.monotonic() - started < 3
    assert not any(t.name == "sirio-coda" and t.is_alive() for t in threading.enumerate())


def test_stop_leaves_running_document_for_next_start(make_env, png: bytes) -> None:
    engine = FakeEngine(delay=2.0)
    env = make_env(engine)
    (doc_id,) = env.add(png)
    wait_for(lambda: env.status(doc_id) == "in_lavorazione")
    env.processor.stop(timeout=0.2)
    assert env.status(doc_id) == "in_lavorazione"


# --------------------------------------------------------------------------
# Anagrafica: nomi letti male ricondotti a quelli gia' confermati
# --------------------------------------------------------------------------

def test_registry_corrects_misread_names(make_env, png: bytes) -> None:
    reg = anagrafica.anagrafica_predefinita()          # quella della cartella dei dati del test
    reg.aggiungi("operatore", "ROSSI MARIO")
    reg.aggiungi("operatore", "ESPOSITO ANNA")
    data = default_data()
    data["header"]["operatore"] = "R0SSI MAR1O"         # lettura OCR rovinata
    env = make_env(FakeEngine(delay=0.05, truths={"foglio.png": data}))
    assert env.processor.anagrafica is reg
    (doc_id,) = env.add(png)
    wait_for(lambda: env.status(doc_id) == "completato")
    doc = env.store.get(doc_id)
    assert doc.header.operatore == "ROSSI MARIO"
    assert "operatore" in doc.header.incerti            # sostituito ma da verificare (somiglianza 85%)
    assert "Operatore riconosciuto dall'anagrafica" in doc.ocr_notes
    assert "letto «R0SSI MAR1O»" in doc.ocr_notes and doc.ocr_notes.startswith("Lettura simulata")
    # la lettura originale e' nelle note, non fra le correzioni dell'utente
    assert doc.ocr_originali == {} and doc.user_edited == []
    assert any(a.codice == "W04_CAMPO_INCERTO" and a.campo == "operatore" for a in doc.anomalies)
    # nessun apprendimento dalla lettura grezza
    assert reg.valori("operatore") == ["ESPOSITO ANNA", "ROSSI MARIO"]
    assert reg.valori("alunno") == []


def test_explicit_registry_is_used(tmp_path: Path, png: bytes) -> None:
    reg = anagrafica.Anagrafica(tmp_path / "altra" / "anagrafica.json")
    reg.aggiungi("alunno", "BIANCHI LUCA")
    data = default_data()
    data["header"]["alunno"] = "BlANCHI LUKA"
    store = DocumentStore(tmp_path / "documenti")
    proc = Processor(store, lambda: Settings(engine="claude"),
                     lambda _s: FakeEngine(delay=0.02, truths={"f.png": data}), anagrafica=reg)
    proc.start()
    try:
        doc_id = store.add_file("f.png", png, on_document=lambda d: proc.enqueue(d.id))[0].id
        wait_for(lambda: store.get(doc_id).status == "completato")
        assert store.get(doc_id).header.alunno == "BIANCHI LUCA"
    finally:
        proc.stop()


def test_registry_failure_does_not_break_processing(make_env, png: bytes, monkeypatch: pytest.MonkeyPatch,
                                                   caplog: pytest.LogCaptureFixture) -> None:
    def broken(_self, _header):
        raise RuntimeError("anagrafica danneggiata")

    monkeypatch.setattr(anagrafica.Anagrafica, "correggi", broken)
    env = make_env(FakeEngine(delay=0.05))
    with caplog.at_level(logging.ERROR, logger="sirio.processing"):
        (doc_id,) = env.add(png)
        wait_for(lambda: env.status(doc_id) == "completato")
    assert env.store.get(doc_id).header.operatore == "ROSSI MARIO"
    assert any("anagrafica" in r.getMessage() for r in caplog.records)


def test_reprocess_detaches_document_but_keeps_learned_names(make_env, png: bytes) -> None:
    engine = FakeEngine(delay=0.05)
    env = make_env(engine)
    (doc_id,) = env.add(png)
    wait_for(lambda: env.status(doc_id) == "completato")
    reg = env.processor.anagrafica

    def confirm(d) -> None:
        d.user_verified = True

    reg.impara_documento(env.store.update(doc_id, confirm))
    assert reg.valori("operatore") == ["ROSSI MARIO"]
    env.processor.enqueue(doc_id)
    wait_for(lambda: len(engine.calls) == 2 and env.status(doc_id) == "completato")
    wait_for(lambda: doc_id not in json.loads(reg.path.read_text(encoding="utf-8"))["documenti"])
    doc = env.store.get(doc_id)
    assert doc.user_verified is False
    # la nuova lettura (non confermata) non ritira quanto appreso
    assert reg.impara_documento(doc) is False
    assert reg.valori("operatore") == ["ROSSI MARIO"] and reg.valori("alunno") == ["BIANCHI LUCA"]
