"""Motore OCR finto per i test e per il server di sviluppo (nessuna chiave API).

``FakeEngine.extract`` attende un breve intervallo chiamando la callback di
avanzamento e restituisce dati realistici: la "verita'" del foglio sintetico
corrispondente (``tests.synthetic``) se registrata per quel file/pagina,
altrimenti il mese d'esempio predefinito. Facoltativamente segna alcuni campi
come incerti o illeggibili, come farebbe un motore reale.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from sirio.config import model_info
from sirio.engines.base import PageInput, ProgressFn
from sirio.models import HEADER_FIELDS, DayRow, ExtractionResult, Header, Usage
from tests.synthetic import default_data

_ROW_KEYS = (
    "prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita", "ore_dichiarate",
    "assenza_alunno", "assenza_operatore", "firma", "note", "trattino_effettivo",
)


def result_from_truth(truth: dict, engine: str = "claude", model: str | None = "claude-opus-5-5",
                      flags: bool = True) -> ExtractionResult:
    """Costruisce un ``ExtractionResult`` dalla verita' di un foglio sintetico.

    Con ``flags`` alcuni campi vengono segnalati come farebbe un motore reale:
    l'orario d'uscita del primo giorno lavorato e le ore PEI come *incerti*, la
    nota dell'ultimo giorno con una nota come *illeggibile* (valore assente).
    """
    hdr = {k: v for k, v in (truth.get("header") or {}).items() if k in HEADER_FIELDS}
    header = Header(**hdr)
    by_day: dict[int, dict] = {int(r["giorno"]): r for r in truth.get("rows") or []}
    rows: list[DayRow] = []
    for g in range(1, 32):
        src = by_day.get(g, {})
        rows.append(DayRow(giorno=g, **{k: src[k] for k in _ROW_KEYS if k in src and src[k] is not None}))
    if flags:
        worked = [r for r in rows if r.eff_uscita]
        if worked:
            r = worked[0]
            r.incerti.append("eff_uscita")
            r.confidenza["eff_uscita"] = 0.62
        noted = [r for r in rows if r.note]
        if noted:
            r = noted[-1]
            r.note = None
            r.illeggibili.append("note")
        if header.ore_pei is not None:
            header.incerti.append("ore_pei")
    info = model_info(model or "") if model else None
    usage = Usage(input_tokens=9200, output_tokens=2600, seconds=0.0)
    if info:
        usage.cost_usd = round(usage.input_tokens * info["input"] / 1e6 + usage.output_tokens * info["output"] / 1e6, 4)
    return ExtractionResult(
        is_foglio_firma=True,
        header=header,
        rows=rows,
        ocr_notes="Lettura simulata (motore di prova).",
        confidence=0.93,
        engine=engine,
        model=model,
        usage=usage,
    )


class FakeEngine:
    """Motore finto compatibile con ``sirio.engines.base.OCREngine``.

    Parametri utili nei test:
      * ``delay``: durata simulata di una lettura (secondi);
      * ``truths``: verita' per ``source_file`` oppure per ``"source_file#pagina"``;
      * ``unknown``: cosa fare se non c'e' una verita' registrata:
        ``"default"`` (mese d'esempio) oppure ``"scartato"`` (non e' un foglio firma);
      * ``available``/``message``: risultato di ``is_available``;
      * ``fail``: eccezione da sollevare durante la lettura (``fail_for``: solo per alcuni file);
      * ``flags``: segnala alcuni campi come incerti/illeggibili.
    """

    def __init__(
        self,
        name: str = "claude",
        delay: float = 0.15,
        truths: dict[str, dict] | None = None,
        unknown: str = "default",
        available: bool = True,
        message: str = "",
        fail: BaseException | None = None,
        fail_for: dict[str, BaseException] | None = None,
        flags: bool = True,
        steps: int = 5,
        model: str | None = "claude-opus-5-5",
    ):
        self.name = name
        self.delay = delay
        self.truths = dict(truths or {})
        self.unknown = unknown
        self.available = available
        self.message = message or ("Motore di prova pronto." if available else "Motore di prova non disponibile.")
        self.fail = fail
        self.fail_for = dict(fail_for or {})
        self.flags = flags
        self.steps = max(1, steps)
        self.model = model
        self._lock = threading.Lock()
        self.calls: list[tuple[str, int]] = []
        self.active = 0
        self.max_active = 0

    def is_available(self) -> tuple[bool, str]:
        return self.available, self.message

    def _truth_for(self, page: PageInput) -> dict | None:
        return self.truths.get(f"{page.source_file}#{page.page}") or self.truths.get(page.source_file)

    def extract(self, page: PageInput, progress: ProgressFn) -> ExtractionResult:
        with self._lock:
            self.calls.append((page.source_file, page.page))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            messages = (
                "Analisi della pagina…",
                "Lettura dell'intestazione…",
                "Lettura dei giorni 1–16…",
                "Lettura dei giorni 16–31…",
                "Verifica incrociata delle righe dubbie…",
            )
            for i in range(self.steps):
                progress(i / self.steps, messages[i % len(messages)])
                time.sleep(self.delay / self.steps)
            failure = self.fail_for.get(page.source_file, self.fail)
            if failure is not None:
                raise failure
            progress(1.0, "Lettura completata")
            truth = self._truth_for(page)
            if truth is None and self.unknown == "scartato":
                return ExtractionResult(
                    is_foglio_firma=False,
                    engine=self.name,
                    model=self.model,
                    ocr_notes="La pagina non contiene un foglio firma.",
                    usage=Usage(input_tokens=3100, output_tokens=120),
                )
            data: dict[str, Any] = truth if truth is not None else default_data()
            return result_from_truth(data, engine=self.name, model=self.model, flags=self.flags)
        finally:
            with self._lock:
                self.active -= 1


__all__ = ["FakeEngine", "result_from_truth"]
