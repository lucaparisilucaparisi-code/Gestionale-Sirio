"""Operazioni sui file condivise dai moduli che scrivono su disco."""

from __future__ import annotations

import os
import time
from pathlib import Path

PathLike = str | os.PathLike[str] | Path


def replace_with_retry(src: PathLike, dst: PathLike, attempts: int = 10, delay: float = 0.1) -> None:
    """``os.replace`` (sostituzione atomica) che ritenta se la destinazione è bloccata.

    Su Windows la sostituzione può fallire per qualche istante con ``PermissionError``
    quando un altro processo (antivirus, indicizzatore, anteprima di Esplora risorse) ha
    appena aperto il file di destinazione: si ritenta fino a ``attempts`` volte, a
    ``delay`` secondi di distanza. Se il blocco persiste (es. file aperto in Excel) si
    propaga l'ultima ``PermissionError``, che il chiamante traduce in un messaggio in
    italiano; gli altri errori si propagano subito.
    """
    attempts = max(1, int(attempts))
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


__all__ = ["replace_with_retry"]
