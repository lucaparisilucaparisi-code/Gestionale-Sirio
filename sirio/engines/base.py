"""Interfaccia comune dei motori OCR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol

import numpy as np

from sirio.models import ExtractionResult

if TYPE_CHECKING:
    from sirio.config import Settings
    from sirio.vision.grid import TableGrid

ProgressFn = Callable[[float, str], None]


def no_progress(_fraction: float, _message: str) -> None:
    return None


@dataclass
class PageInput:
    image: np.ndarray        # BGR normalizzata (vision.preprocess.normalize_page)
    grid: "TableGrid"
    source_file: str
    page: int                # 1-based


class EngineError(Exception):
    """Errore con messaggio già comprensibile per l'utente (in italiano)."""


class OCREngine(Protocol):
    name: str

    def is_available(self) -> tuple[bool, str]:
        ...

    def extract(self, page: PageInput, progress: ProgressFn) -> ExtractionResult:
        ...


def get_engine(settings: "Settings") -> OCREngine:
    """Costruisce il motore scelto nelle impostazioni (import pigri)."""
    if settings.engine == "locale":
        from sirio.engines.local_engine import LocalEngine  # noqa: PLC0415

        return LocalEngine(model_name=settings.local_model)

    from sirio.config import get_api_key  # noqa: PLC0415
    from sirio.engines.claude_engine import ClaudeEngine  # noqa: PLC0415

    return ClaudeEngine(
        api_key=get_api_key(),
        model=settings.claude_model,
        effort=settings.claude_effort,
        verify=settings.verifica_incrociata,
    )
