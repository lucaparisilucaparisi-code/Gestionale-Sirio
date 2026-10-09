"""Percorsi, impostazioni persistenti e chiave API di Sirio OCR."""

from __future__ import annotations

import json
import logging
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import Literal

import platformdirs
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

APP_NAME = "Sirio OCR"
_APP_DIR_NAME = "SirioOCR"
_KEYRING_SERVICE = "SirioOCR"
_KEYRING_USER = "anthropic_api_key"

CLAUDE_MODELS: list[dict] = [
    {
        "id": "claude-opus-5-5",
        "label": "Claude Opus 5.5 — massima precisione (consigliato)",
        "description": "La lettura più accurata della scrittura a mano. Indicato per la rendicontazione ufficiale.",
        "input": 4.00,
        "output": 20.00,
    },
    {
        "id": "claude-sonnet-5-5",
        "label": "Claude Sonnet 5.5 — bilanciato",
        "description": "Ottima precisione a metà del costo di Opus.",
        "input": 2.00,
        "output": 10.00,
    },
    {
        "id": "claude-haiku-5-5",
        "label": "Claude Haiku 5.5 — economico",
        "description": "Il più rapido ed economico; consigliata la verifica manuale dei campi incerti.",
        "input": 0.10,
        "output": 0.50,
    },
]

_lock = threading.RLock()


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir() -> Path:
    override = os.environ.get("SIRIO_DATA_DIR")
    if override:
        return _ensure(Path(override).expanduser())
    return _ensure(Path(platformdirs.user_data_dir(_APP_DIR_NAME, appauthor=False)))


def documents_dir() -> Path:
    return _ensure(data_dir() / "documenti")


def logs_dir() -> Path:
    return _ensure(data_dir() / "log")


def export_dir() -> Path:
    override = os.environ.get("SIRIO_EXPORT_DIR")
    if override:
        return _ensure(Path(override).expanduser())
    return _ensure(Path(platformdirs.user_documents_dir()) / APP_NAME / "Export")


class Settings(BaseModel):
    # Motore predefinito: locale offline (gratuito, i documenti non lasciano il computer).
    engine: Literal["claude", "locale"] = "locale"
    claude_model: str = "claude-opus-5-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    verifica_incrociata: bool = True
    concorrenza: int = 3
    local_model: str = "microsoft/trocr-base-handwritten"
    export_fogli_per_documento: bool = True
    export_giorni_vuoti: bool = False
    tema: Literal["auto", "chiaro", "scuro"] = "auto"


def _settings_path() -> Path:
    return data_dir() / "impostazioni.json"


def _atomic_write(path: Path, text: str, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        if private:
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_settings() -> Settings:
    with _lock:
        path = _settings_path()
        if not path.exists():
            return Settings()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("Impostazioni illeggibili, uso i valori predefiniti")
            return Settings()
        # Ignora chiavi sconosciute o non valide una per una, senza perdere le altre.
        settings = Settings()
        for key, value in raw.items():
            if key not in Settings.model_fields:
                continue
            try:
                settings = Settings.model_validate({**settings.model_dump(), key: value})
            except ValidationError:
                log.warning("Impostazione non valida ignorata: %s", key)
        return settings


def save_settings(s: Settings) -> None:
    with _lock:
        _atomic_write(_settings_path(), s.model_dump_json(indent=2))


# --- Chiave API -------------------------------------------------------------

def _key_file() -> Path:
    return data_dir() / ".chiave"


def _keyring():
    try:
        import keyring  # noqa: PLC0415

        backend = keyring.get_keyring()
        # I backend "fail"/"null" (nessun portachiavi nel sistema) non salvano davvero nulla.
        module = type(backend).__module__.lower()
        if module.endswith((".fail", ".null")):
            return None
        if module.endswith(".chainer") and not getattr(backend, "backends", None):
            return None
        return keyring
    except Exception:  # noqa: BLE001 - qualunque problema del portachiavi => ripiego su file
        return None


def _read_keyring() -> str | None:
    kr = _keyring()
    if kr is None:
        return None
    try:
        return kr.get_password(_KEYRING_SERVICE, _KEYRING_USER) or None
    except Exception:  # noqa: BLE001
        return None


def _read_file_key() -> str | None:
    try:
        value = _key_file().read_text(encoding="utf-8").strip()
        return value or None
    except OSError:
        return None


def api_key_source() -> Literal["env", "keyring", "file"] | None:
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "env"
    if _read_keyring():
        return "keyring"
    if _read_file_key():
        return "file"
    return None


def get_api_key() -> str | None:
    env = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if env:
        return env
    return _read_keyring() or _read_file_key()


def set_api_key(key: str | None) -> None:
    key = (key or "").strip()
    with _lock:
        kr = _keyring()
        if not key:
            if kr is not None:
                try:
                    kr.delete_password(_KEYRING_SERVICE, _KEYRING_USER)
                except Exception:  # noqa: BLE001
                    pass
            try:
                _key_file().unlink()
            except OSError:
                pass
            return
        if kr is not None:
            try:
                kr.set_password(_KEYRING_SERVICE, _KEYRING_USER, key)
                if kr.get_password(_KEYRING_SERVICE, _KEYRING_USER) == key:
                    try:
                        _key_file().unlink()
                    except OSError:
                        pass
                    return
            except Exception:  # noqa: BLE001
                log.info("Portachiavi di sistema non disponibile, salvo la chiave su file protetto")
        _atomic_write(_key_file(), key, private=True)


def api_key_hint() -> str | None:
    key = get_api_key()
    if not key:
        return None
    if len(key) <= 12:
        return "…" + key[-2:]
    return f"{key[:7]}…{key[-4:]}"


def model_info(model_id: str) -> dict | None:
    return next((m for m in CLAUDE_MODELS if m["id"] == model_id), None)
