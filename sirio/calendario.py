"""Calendario italiano: nomi di giorni e mesi, Pasqua, festività, settimane.

Tutte le funzioni sono pure e deterministiche (nessun accesso a file o rete).
Le festività comprendono quelle nazionali e il patrono di Napoli (San Gennaro,
19 settembre), perché il servizio è svolto nelle scuole del Comune di Napoli.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from functools import lru_cache
from typing import Literal

GIORNI: tuple[str, ...] = (
    "lunedì",
    "martedì",
    "mercoledì",
    "giovedì",
    "venerdì",
    "sabato",
    "domenica",
)
GIORNI_BREVI: tuple[str, ...] = ("lun", "mar", "mer", "gio", "ven", "sab", "dom")
MESI: tuple[str, ...] = (
    "gennaio",
    "febbraio",
    "marzo",
    "aprile",
    "maggio",
    "giugno",
    "luglio",
    "agosto",
    "settembre",
    "ottobre",
    "novembre",
    "dicembre",
)

TipoGiorno = Literal["feriale", "sabato", "domenica", "festivo", "inesistente"]

# Festività a data fissa: (mese, giorno) -> nome.
_FESTIVITA_FISSE: tuple[tuple[int, int, str], ...] = (
    (1, 1, "Capodanno"),
    (1, 6, "Epifania"),
    (4, 25, "Festa della Liberazione"),
    (5, 1, "Festa del Lavoro"),
    (6, 2, "Festa della Repubblica"),
    (8, 15, "Ferragosto"),
    (9, 19, "San Gennaro, patrono di Napoli"),
    (11, 1, "Ognissanti"),
    (12, 8, "Immacolata Concezione"),
    (12, 25, "Natale"),
    (12, 26, "Santo Stefano"),
)

# San Francesco d'Assisi (4 ottobre): festività nazionale dal 2026 (L. 8 ottobre 2025, n. 151).
_SAN_FRANCESCO_DAL = 2026


def _check_anno(anno: int) -> None:
    if not isinstance(anno, int) or isinstance(anno, bool) or not 1583 <= anno <= 9999:
        raise ValueError(f"Anno non valido: {anno!r}")


def _check_mese(mese: int) -> None:
    if not isinstance(mese, int) or isinstance(mese, bool) or not 1 <= mese <= 12:
        raise ValueError(f"Mese non valido: {mese!r} (atteso un numero da 1 a 12)")


def pasqua(anno: int) -> date:
    """Domenica di Pasqua (calendario gregoriano, algoritmo di Meeus/Jones/Butcher)."""
    _check_anno(anno)
    a = anno % 19
    b, c = divmod(anno, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    mese, giorno = divmod(h + ll - 7 * m + 114, 31)
    return date(anno, mese, giorno + 1)


@lru_cache(maxsize=64)
def _festivita(anno: int) -> tuple[tuple[date, str], ...]:
    feste: dict[date, str] = {date(anno, m, g): nome for m, g, nome in _FESTIVITA_FISSE}
    if anno >= _SAN_FRANCESCO_DAL:
        feste[date(anno, 10, 4)] = "San Francesco d'Assisi, patrono d'Italia"
    p = pasqua(anno)
    feste[p] = "Pasqua"
    feste[p + timedelta(days=1)] = "Lunedì dell'Angelo (Pasquetta)"
    return tuple(sorted(feste.items()))


def festivita(anno: int) -> dict[date, str]:
    """Festività nazionali italiane e San Gennaro (19/9) dell'anno, ordinate per data."""
    _check_anno(anno)
    return dict(_festivita(anno))


def giorni_nel_mese(anno: int, mese: int) -> int:
    _check_anno(anno)
    _check_mese(mese)
    return calendar.monthrange(anno, mese)[1]


def _valido(anno: int, mese: int, giorno: int) -> bool:
    try:
        return isinstance(giorno, int) and 1 <= giorno <= giorni_nel_mese(anno, mese)
    except ValueError:
        return False


def nome_festivita(anno: int, mese: int, giorno: int) -> str | None:
    """Nome della festività che cade nel giorno indicato, altrimenti None."""
    if not _valido(anno, mese, giorno):
        return None
    return dict(_festivita(anno)).get(date(anno, mese, giorno))


def tipo_giorno(anno: int, mese: int, giorno: int) -> TipoGiorno:
    """Classifica il giorno. Una festività prevale su sabato e domenica.

    Date impossibili (es. 30 febbraio, mese 13) restituiscono "inesistente".
    """
    if not _valido(anno, mese, giorno):
        return "inesistente"
    if nome_festivita(anno, mese, giorno) is not None:
        return "festivo"
    wd = date(anno, mese, giorno).weekday()
    if wd == 6:
        return "domenica"
    if wd == 5:
        return "sabato"
    return "feriale"


def is_lavorativo(anno: int, mese: int, giorno: int) -> bool:
    """True per i giorni feriali (lunedì-venerdì non festivi)."""
    return tipo_giorno(anno, mese, giorno) == "feriale"


def nome_giorno(anno: int, mese: int, giorno: int, breve: bool = False) -> str | None:
    """Giorno della settimana in italiano ("lunedì" / "lun"); None se la data non esiste."""
    if not _valido(anno, mese, giorno):
        return None
    wd = date(anno, mese, giorno).weekday()
    return GIORNI_BREVI[wd] if breve else GIORNI[wd]


def nome_mese(mese: int) -> str:
    """Nome del mese in minuscolo ("febbraio")."""
    _check_mese(mese)
    return MESI[mese - 1]


def etichetta_mese(mese: int, anno: int) -> str:
    """Es. "febbraio 2026"."""
    return f"{nome_mese(mese)} {anno}"


def settimane_del_mese(anno: int, mese: int) -> list[tuple[int, int]]:
    """Settimane lunedì-domenica troncate al mese: [(dal, al), ...].

    Es. febbraio 2026 (inizia di domenica): [(1, 1), (2, 8), (9, 15), (16, 22), (23, 28)].
    """
    n = giorni_nel_mese(anno, mese)
    settimane: list[tuple[int, int]] = []
    dal = 1
    while dal <= n:
        # giorni mancanti alla domenica successiva (weekday: lunedì=0 ... domenica=6)
        al = min(n, dal + 6 - date(anno, mese, dal).weekday())
        settimane.append((dal, al))
        dal = al + 1
    return settimane
