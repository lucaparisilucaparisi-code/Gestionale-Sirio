"""Test della sostituzione atomica con nuovi tentativi (sirio/fsutil.py) e dei suoi utilizzi.

Su Windows ``os.replace`` fallisce per qualche istante con ``PermissionError`` quando un
antivirus o l'indicizzatore ha appena aperto il file di destinazione: qui il blocco
temporaneo è simulato facendo fallire il primo tentativo di ogni sostituzione.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from sirio import anagrafica, config, fsutil, store
from sirio import app as sirio_app
from sirio.excel_export import export_workbook
from tests.genera_esempio_excel import documenti_esempio

_REAL_REPLACE = os.replace


class FlakyReplace:
    """``os.replace`` che fallisce ``failures`` volte per ogni destinazione, poi riesce."""

    def __init__(self, failures: int = 1, error: type[OSError] = PermissionError):
        self.failures = failures
        self.error = error
        self.calls: list[str] = []
        self._failed: dict[str, int] = {}

    def __call__(self, src, dst) -> None:
        key = str(dst)
        self.calls.append(key)
        if self._failed.get(key, 0) < self.failures:
            self._failed[key] = self._failed.get(key, 0) + 1
            raise self.error(errno.EACCES, "Accesso negato", key)
        _REAL_REPLACE(src, dst)


@pytest.fixture()
def flaky(monkeypatch: pytest.MonkeyPatch) -> FlakyReplace:
    fake = FlakyReplace()
    monkeypatch.setattr(os, "replace", fake)
    monkeypatch.setattr(fsutil.time, "sleep", lambda _s: None)
    return fake


def test_replace_with_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(fsutil.time, "sleep", sleeps.append)
    src, dst = tmp_path / "nuovo.tmp", tmp_path / "dati.json"
    dst.write_text("vecchio", encoding="utf-8")

    fake = FlakyReplace(failures=3)
    monkeypatch.setattr(os, "replace", fake)
    src.write_text("nuovo", encoding="utf-8")
    fsutil.replace_with_retry(src, dst)
    assert dst.read_text(encoding="utf-8") == "nuovo" and not src.exists()
    assert len(fake.calls) == 4 and sleeps == [0.1, 0.1, 0.1]

    # blocco persistente (es. file aperto in Excel): dopo i tentativi l'errore arriva al chiamante
    fake = FlakyReplace(failures=100)
    monkeypatch.setattr(os, "replace", fake)
    src.write_text("altro", encoding="utf-8")
    with pytest.raises(PermissionError):
        fsutil.replace_with_retry(src, dst, attempts=5, delay=0.2)
    assert len(fake.calls) == 5 and dst.read_text(encoding="utf-8") == "nuovo" and src.exists()

    # altri errori: nessun nuovo tentativo
    fake = FlakyReplace(failures=1, error=FileNotFoundError)
    monkeypatch.setattr(os, "replace", fake)
    with pytest.raises(FileNotFoundError):
        fsutil.replace_with_retry(src, dst)
    assert len(fake.calls) == 1


def test_settings_and_key_file_survive_a_briefly_locked_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                flaky: FlakyReplace) -> None:
    monkeypatch.setenv("SIRIO_DATA_DIR", str(tmp_path / "dati"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(config, "_keyring", lambda: None)
    config.save_settings(config.Settings(tema="scuro"))
    assert config.load_settings().tema == "scuro"
    key = "sk-ant-api03-" + "Zx9" * 20
    config.set_api_key(key)
    assert config.get_api_key() == key
    assert len(flaky.calls) == 4   # due sostituzioni, ciascuna riuscita al secondo tentativo


def test_registry_save_survives_a_briefly_locked_target(tmp_path: Path, flaky: FlakyReplace) -> None:
    path = tmp_path / "anagrafica.json"
    reg = anagrafica.Anagrafica(path)
    reg.aggiungi("operatore", "ROSSI MARIO")
    assert "ROSSI MARIO" in anagrafica.Anagrafica(path).valori("operatore")
    assert flaky.calls and set(flaky.calls) == {str(path)}


def test_excel_export_survives_a_briefly_locked_target(tmp_path: Path, flaky: FlakyReplace) -> None:
    path = export_workbook(documenti_esempio(), tmp_path / "rendiconto.xlsx")
    assert path.is_file() and path.stat().st_size > 0
    assert flaky.calls == [str(path), str(path)]
    assert not list(tmp_path.glob(".*.tmp"))


def test_excel_export_file_open_in_excel_keeps_italian_message(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "replace", FlakyReplace(failures=100))
    monkeypatch.setattr(fsutil.time, "sleep", lambda _s: None)
    with pytest.raises(PermissionError, match="è aperto in un altro programma"):
        export_workbook(documenti_esempio(), tmp_path / "rendiconto.xlsx")
    assert not list(tmp_path.iterdir())   # nessun file temporaneo lasciato nella cartella


def test_store_and_instance_file_survive_a_briefly_locked_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 flaky: FlakyReplace) -> None:
    target = tmp_path / "grid.json"
    store._atomic_write_bytes(target, b"{}")
    assert target.read_bytes() == b"{}"
    monkeypatch.setenv("SIRIO_DATA_DIR", str(tmp_path / "dati"))
    sirio_app.write_instance({"port": 5000, "pid": 1, "token": "x"})
    assert json.loads((tmp_path / "dati" / "istanza.json").read_text(encoding="utf-8"))["port"] == 5000
