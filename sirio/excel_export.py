"""Esportazione della rendicontazione in Excel (.xlsx) con XlsxWriter.

Fogli generati, nell'ordine: "Riepilogo", "Dettaglio giornaliero", una scheda
per ogni foglio firma (replica del modulo cartaceo), "Anomalie", "Totali per
operatore", "Legenda e note".

Principi
--------
* Orari, date e ore sono veri valori Excel (frazioni di giorno, date seriali,
  numeri), mai numeri memorizzati come testo.
* Ore calcolate, differenze e totali sono formule con il risultato già
  memorizzato accanto (``write_formula(..., value=...)``): il file resta "vivo"
  e i valori sono visibili anche nei programmi che non ricalcolano.
* I documenti sono rivalidati su copie: totali e anomalie sono sempre coerenti
  con i dati esportati, senza modificare gli oggetti ricevuti.
* Celle illeggibili (rosso), incerte (ambra) e corrette a mano (blu) sono
  evidenziate e spiegate da un commento della cella.
"""

from __future__ import annotations

import logging
import math
import os
import re
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import xlsxwriter
from pydantic import BaseModel
from xlsxwriter.exceptions import XlsxWriterException
from xlsxwriter.format import Format
from xlsxwriter.utility import xl_col_to_name
from xlsxwriter.worksheet import Worksheet

from sirio import __version__, calendario
from sirio import validation as val
from sirio.models import DayRow, Document, Header

log = logging.getLogger(__name__)

__all__ = ["ExportOptions", "default_filename", "export_workbook"]


class ExportOptions(BaseModel):
    fogli_per_documento: bool = True  # una scheda per ogni foglio firma
    giorni_vuoti: bool = False  # includere nel dettaglio anche i giorni senza dati
    titolo: str | None = None  # titolo del Riepilogo (predefinito: "Rendicontazione Assistenza Specialistica")


TITOLO_PREDEFINITO = "Rendicontazione Assistenza Specialistica"
SERVIZIO = "Assistenza Specialistica all'integrazione scolastica"

# Nomi dei fogli fissi
S_RIEPILOGO = "Riepilogo"
S_DETTAGLIO = "Dettaglio giornaliero"
S_ANOMALIE = "Anomalie"
S_TOTALI = "Totali per operatore"
S_LEGENDA = "Legenda e note"
_FOGLI_FISSI = (S_RIEPILOGO, S_DETTAGLIO, S_ANOMALIE, S_TOTALI, S_LEGENDA)

# --- Palette e stili -----------------------------------------------------------

FONT = "Calibri"
NAVY = "#14213D"
NAVY_TEXT_SOFT = "#C7D2E8"  # testo secondario sulla fascia blu notte
ACCENT = "#2563EB"
ACCENT_DARK = "#1D4ED8"
ACCENT_FILL = "#EFF6FF"
INK = "#1F2937"
MUTED = "#6B7280"
SUBTLE = "#9CA3AF"
BORDER = "#D1D5DB"
BORDER_SOFT = "#E5E7EB"
PANEL = "#F3F4F6"
TILE = "#F5F7FB"
WEEKEND = "#EEF0F4"
INESISTENTE = "#DCDFE5"
TOTAL_FILL = "#E8EDF5"
WHITE = "#FFFFFF"
GREEN = "#15803D"
GREEN_FILL = "#DCFCE7"
AMBER = "#B45309"
AMBER_FILL = "#FEF3C7"
AMBER_TEXT = "#78350F"
RED = "#B91C1C"
RED_DARK = "#991B1B"
RED_FILL = "#FEE2E2"

FMT_ORE = "#,##0.00"
FMT_ORE_SEGNO = "+#,##0.00;-#,##0.00;0.00"
FMT_INT = "#,##0"
FMT_PCT = "0.0%"
FMT_ORA = "hh:mm"
FMT_DATA = "dd/mm/yyyy"
FMT_MESE = "[$-410]mmmm yyyy"
FMT_MESE_BREVE = "[$-410]mmm yyyy"

ILLEGGIBILE = "ILLEGGIBILE"
TOLLERANZA = val.TOLLERANZA
MAX_TESTO = 32000

_STATI = {"ok": "OK", "da_verificare": "Da verificare", "errori": "Errori"}
_STATO_COLORI = {"ok": (GREEN, GREEN_FILL), "da_verificare": (AMBER, AMBER_FILL), "errori": (RED, RED_FILL)}
_GRAVITA = {"errore": "Errore", "attenzione": "Attenzione", "info": "Info"}
_GRAVITA_COLORI = {
    "errore": (RED, RED_FILL),
    "attenzione": (AMBER, AMBER_FILL),
    "info": (ACCENT_DARK, ACCENT_FILL),
}
_MOTORI = {
    "claude": "Claude Vision (Anthropic)",
    "locale": "Motore locale offline (OpenCV + TrOCR)",
    "demo": "Dati dimostrativi",
}
_ESCLUSIONE = {
    "in_coda": "In attesa di elaborazione",
    "in_lavorazione": "Elaborazione in corso",
    "errore": "Elaborazione non riuscita",
    "scartato": "Scartato: la pagina non è un foglio firma",
}

_ORARI = ("prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita")
_BOOL_CAMPI = ("assenza_alunno", "assenza_operatore", "firma", "firma_coordinatore", "timbro_referente")
_RE_VUOTO = re.compile(r"^[\s\-–—−_/.]*$")
_RE_SPAZI = re.compile(r"\s+")
_RE_NOME_FOGLIO_VIETATI = re.compile(r"[\[\]:*?/\\]")
_RE_DATA = re.compile(r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{2,4})")
_RE_PREFISSO_GIORNO = re.compile(r"^Giorno \d+(?: \([^)]*\))?: ")


# --- Utilità --------------------------------------------------------------------


class _F(NamedTuple):
    """Formula con il risultato da memorizzare nella cella."""

    formula: str
    value: Any


def _vuoto(s: object) -> bool:
    return s is None or (isinstance(s, str) and bool(_RE_VUOTO.match(s)))


_RE_NUMERO_IT = re.compile(r"^-?\d{1,6}(?:,\d+)?$")
_RE_ORARIO = re.compile(r"^\d{1,2}:\d{2}$")


def _numero_testo(s: str | None) -> float | None:
    """ "3" / "1,5" (formato italiano della validazione) -> numero; altrimenti None."""
    if s is None or not _RE_NUMERO_IT.match(s.strip()):
        return None
    return float(s.strip().replace(",", "."))


def _pulisci(s: object) -> str:
    """Testo su una riga, spazi compattati, lunghezza entro i limiti di Excel."""
    if s is None:
        return ""
    return _RE_SPAZI.sub(" ", str(s)).strip()[:MAX_TESTO]


def _numero(x: object) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _q(sheet: str) -> str:
    """Nome di foglio tra apici per riferimenti e collegamenti."""
    return "'" + sheet.replace("'", "''") + "'"


def _cell(row: int, col: int, abs_: bool = False) -> str:
    d = "$" if abs_ else ""
    return f"{d}{xl_col_to_name(col)}{d}{row + 1}"


def _rng(r0: int, c0: int, r1: int, c1: int, abs_: bool = True) -> str:
    return f"{_cell(r0, c0, abs_)}:{_cell(r1, c1, abs_)}"


def _link(sheet: str, row: int = 0, col: int = 0) -> str:
    return f"internal:{_q(sheet)}!{_cell(row, col)}"


def _plurale(n: int, singolare: str, plurale: str) -> str:
    return f"{n} {singolare if n == 1 else plurale}"


def _plurale_formula(espr: str, singolare: str, plurale: str) -> str:
    """Formula testuale "n parola" con singolare/plurale corretto."""
    return f'{espr}&IF({espr}=1," {singolare}"," {plurale}")'


def _maiuscola(s: str) -> str:
    return s[:1].upper() + s[1:]


def _righe_testo(testo: str, larghezza: float) -> int:
    """Stima delle righe occupate da un testo a capo automatico (Calibri 10)."""
    caratteri = max(8.0, larghezza * 1.22)
    righe = 0
    for parte in str(testo).split("\n"):
        righe += max(1, math.ceil(len(parte) / caratteri))
    return righe


def _altezza(testo: str, larghezza: float, minima: float = 18.0, riga: float = 13.2) -> float:
    return max(minima, riga * _righe_testo(testo, larghezza) + 5)


@lru_cache(maxsize=4096)
def _ora_excel(s: str | None) -> float | None:
    """Orario scritto -> frazione di giorno (valore orario di Excel); None se non interpretabile."""
    p = val.parse_time(s)
    return None if p is None else (p[0] * 60 + p[1]) / 1440


def _data_compilazione(s: str | None) -> date | None:
    if not s:
        return None
    m = _RE_DATA.search(str(s))
    if not m:
        return None
    g, mm, a = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if a < 100:
        a += 2000
    try:
        return date(a, mm, g)
    except ValueError:
        return None


def _dt_locale(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


def _periodo(h: Header) -> tuple[int, int] | None:
    """(mese, anno) se validi, altrimenti None (stessi criteri della validazione)."""
    m, a = h.mese, h.anno
    if not isinstance(m, int) or isinstance(m, bool) or not 1 <= m <= 12:
        return None
    if not isinstance(a, int) or isinstance(a, bool) or not 2000 <= a <= 2100:
        return None
    return m, a


def _etichetta_mese(mese: int, anno: int) -> str:
    return f"{_maiuscola(calendario.nome_mese(mese))} {anno}"


def _testo_periodi(periodi: Sequence[tuple[int, int]]) -> str:
    """Es. "Febbraio 2026", "Febbraio – Marzo 2026", "Dicembre 2025 – Gennaio 2026"."""
    if not periodi:
        return "Periodo non indicato"
    ordinati = sorted(set(periodi), key=lambda p: (p[1], p[0]))
    (m0, a0), (m1, a1) = ordinati[0], ordinati[-1]
    if (m0, a0) == (m1, a1):
        return _etichetta_mese(m0, a0)
    if a0 == a1:
        return f"{_maiuscola(calendario.nome_mese(m0))} – {_etichetta_mese(m1, a1)}"
    return f"{_etichetta_mese(m0, a0)} – {_etichetta_mese(m1, a1)}"


def _valore_originale(campo: str, valore: str | None) -> str:
    """Valore OCR originale leggibile per il commento delle celle corrette."""
    if valore is None or not str(valore).strip():
        return "(vuoto)"
    testo = _pulisci(valore)
    if campo in _BOOL_CAMPI:
        vero = testo.lower() in ("true", "1", "sì", "si", "x", "yes", "presente")
        if campo.startswith("assenza"):
            return "crocetta presente" if vero else "nessuna crocetta"
        return "presente" if vero else "assente"
    if campo in _ORARI:
        return val.normalize_time(testo) or testo
    if campo in ("ore_dichiarate", "ore_pei", "totale_mensile_dichiarato"):
        ore = val.parse_hours(testo)
        return val.format_ore(ore) if ore is not None else testo
    return testo


def _partizione(larghezze: Sequence[float], k: int) -> list[tuple[int, int]]:
    """Divide colonne contigue in ``k`` gruppi di larghezza il più possibile uniforme."""
    n = len(larghezze)
    k = max(1, min(k, n))
    cum = [0.0]
    for w in larghezze:
        cum.append(cum[-1] + w)
    totale = cum[-1]
    confini = [0]
    for j in range(1, k):
        obiettivo = totale * j / k
        lo, hi = confini[-1] + 1, n - (k - j)
        confini.append(min(range(lo, hi + 1), key=lambda i: abs(cum[i] - obiettivo)))
    confini.append(n)
    return [(confini[i], confini[i + 1] - 1) for i in range(k)]


# --- Formati --------------------------------------------------------------------


class _Formati:
    """Fabbrica di formati con cache: ogni combinazione di proprietà viene creata una sola volta."""

    def __init__(self, wb: xlsxwriter.Workbook) -> None:
        self._wb = wb
        self._cache: dict[frozenset[tuple[str, Any]], Format] = {}

    def __call__(self, *strati: dict[str, Any], **extra: Any) -> Format:
        props: dict[str, Any] = {}
        for s in strati:
            props.update(s)
        props.update(extra)
        if props.get("indent") and "align" not in props:
            props["align"] = "left"  # il rientro richiede un allineamento orizzontale esplicito
        key = frozenset(props.items())  # valori semplici (str, int, float, bool): hashable
        f = self._cache.get(key)
        if f is None:
            f = self._wb.add_format(props)
            self._cache[key] = f
        return f


TXT: dict[str, Any] = {"font_name": FONT, "font_size": 10, "font_color": INK, "valign": "vcenter"}
GRID: dict[str, Any] = {"border": 1, "border_color": BORDER_SOFT}
CENTER: dict[str, Any] = {"align": "center"}
HEAD: dict[str, Any] = {
    "bold": True,
    "font_color": WHITE,
    "bg_color": NAVY,
    "align": "center",
    "valign": "vcenter",
    "text_wrap": True,
    "border": 1,
    "border_color": "#2E3B5B",
}
HEAD_CALC: dict[str, Any] = {**HEAD, "bg_color": ACCENT, "border_color": "#3B6FE0"}
LABEL: dict[str, Any] = {"bg_color": PANEL, "font_color": MUTED, "bold": True, "font_size": 9, "indent": 1}
SECTION: dict[str, Any] = {
    "bold": True,
    "font_size": 12,
    "font_color": NAVY,
    "bottom": 2,
    "bottom_color": ACCENT,
    "valign": "bottom",
}
TOTAL: dict[str, Any] = {
    "bold": True,
    "bg_color": TOTAL_FILL,
    "top": 2,
    "top_color": NAVY,
    "bottom": 1,
    "bottom_color": BORDER,
}

# Sovrapposizioni per lo stato di lettura dei campi
OV_ILLEGGIBILE: dict[str, Any] = {"bg_color": RED_FILL, "font_color": RED_DARK, "bold": True}
OV_MARCATORE: dict[str, Any] = {**OV_ILLEGGIBILE, "font_size": 8, "align": "center"}
OV_INCERTO: dict[str, Any] = {"bg_color": AMBER_FILL, "font_color": AMBER_TEXT}
OV_CORRETTO: dict[str, Any] = {"bg_color": ACCENT_FILL, "font_color": ACCENT_DARK}
OV_NON_VALIDO: dict[str, Any] = {"font_color": RED, "bold": True}

_SOVRAPPOSIZIONI = {"illeggibile": OV_ILLEGGIBILE, "incerto": OV_INCERTO, "corretto": OV_CORRETTO}


# --- Preparazione dei documenti -------------------------------------------------


@dataclass(frozen=True)
class _Calc:
    """Valori di una riga con la stessa semantica delle formule Excel."""

    prog: float | None
    calc: float | None
    dich: float | None
    ric: float | None  # None = cella vuota ("")
    diff: float | None


def _calcola(row: DayRow) -> _Calc:
    prog = val.hours_between(row.prog_entrata, row.prog_uscita)
    calc = val.hours_between(row.eff_entrata, row.eff_uscita)
    dich = _numero(row.ore_dichiarate)
    if row.assenza_operatore:
        ric: float | None = 0.0
    elif dich is not None:
        ric = dich
    else:
        ric = calc
    diff = dich - calc if dich is not None and calc is not None else None
    return _Calc(prog, calc, dich, ric, diff)


@dataclass
class _Val:
    """Contenuto di una cella che riporta un campo letto dal foglio."""

    value: Any = None
    kind: str = "text"  # time | hours | text | mark | dash | no | illeggibile | invalid | date | month
    state: str = ""  # "" | illeggibile | incerto | corretto
    comment: str | None = None


@dataclass
class _Doc:
    doc: Document
    n: int
    periodo: tuple[int, int] | None
    n_giorni: int
    operatore: str
    alunno: str
    istituto: str
    calcs: list[_Calc]
    sheet: str | None = None
    ws: Worksheet | None = None
    refs: dict[str, str] = field(default_factory=dict)
    righe_giorno: dict[int, int] = field(default_factory=dict)
    # Totali con la semantica delle formule (valori memorizzati nelle celle)
    tot: dict[str, float] = field(default_factory=dict)

    @property
    def mese_data(self) -> date | None:
        return date(self.periodo[1], self.periodo[0], 1) if self.periodo else None

    @property
    def stato(self) -> str:
        return self.doc.totals.stato

    def giorno_esiste(self, g: int) -> bool:
        return 1 <= g <= self.n_giorni

    def tipo(self, g: int) -> str:
        if self.periodo is None:
            return "feriale" if 1 <= g <= 31 else "inesistente"
        return calendario.tipo_giorno(self.periodo[1], self.periodo[0], g)

    def data(self, g: int) -> date | None:
        if self.periodo is None or not self.giorno_esiste(g):
            return None
        return date(self.periodo[1], self.periodo[0], g)

    def link(self, giorno: int | None = None) -> str | None:
        if not self.sheet:
            return None
        riga = self.righe_giorno.get(giorno, 0) if giorno is not None else 0
        return _link(self.sheet, riga, 0)


def _etichetta_persona(valore: str | None, illeggibile: bool) -> str:
    testo = _pulisci(valore)
    if testo:
        return testo
    return "(illeggibile)" if illeggibile else "(non indicato)"


def _prepara(doc: Document, n: int) -> _Doc:
    h = doc.header
    periodo = _periodo(h)
    n_giorni = calendario.giorni_nel_mese(periodo[1], periodo[0]) if periodo else 31
    calcs = [_calcola(r) for r in doc.rows]
    d = _Doc(
        doc=doc,
        n=n,
        periodo=periodo,
        n_giorni=n_giorni,
        operatore=_etichetta_persona(h.operatore, "operatore" in h.illeggibili),
        alunno=_etichetta_persona(h.alunno, "alunno" in h.illeggibili),
        istituto=_etichetta_persona(h.istituto, "istituto" in h.illeggibili),
        calcs=calcs,
    )
    validi = calcs[:n_giorni]
    d.tot = {
        "prog": math.fsum(c.prog or 0.0 for c in validi),
        "calc": math.fsum(c.calc or 0.0 for c in validi),
        "dich": math.fsum(c.dich or 0.0 for c in validi),
        "ric": math.fsum(c.ric or 0.0 for c in validi),
        "giorni": float(sum(1 for c in validi if (c.ric or 0.0) > 0)),
        "ass_al": float(sum(1 for r in doc.rows[:n_giorni] if r.assenza_alunno)),
        "ass_op": float(sum(1 for r in doc.rows[:n_giorni] if r.assenza_operatore)),
        "firme": float(sum(1 for r in doc.rows[:n_giorni] if r.firma)),
    }
    return d


def _ordina_documenti(docs: Iterable[Document]) -> list[Document]:
    def chiave(doc: Document) -> tuple:
        p = _periodo(doc.header)
        return (
            _pulisci(doc.header.operatore).casefold() or "\uffff",  # senza operatore: in fondo
            p[1] if p else 9999,
            p[0] if p else 99,
            _pulisci(doc.header.alunno).casefold(),
            doc.source_file.casefold(),
            doc.source_page,
        )

    return sorted(docs, key=chiave)


def _nome_scheda(d: _Doc, usati: set[str]) -> str:
    """Nome di foglio univoco ≤ 31 caratteri, es. "ROSSI M. 02-2026"."""
    periodo = f" {d.periodo[0]:02d}-{d.periodo[1]}" if d.periodo else ""
    spazio = 31 - len(periodo)
    parti = [p for p in _RE_NOME_FOGLIO_VIETATI.sub("-", _pulisci(d.doc.header.operatore)).split(" ") if p]
    if not parti:
        base = f"Foglio {d.n}"
    elif len(parti) == 1:
        base = parti[0]
    else:
        # "COGNOME NOME" -> "COGNOME N."; nomi lunghi: iniziali dalla fine ("DELLA VALLE S.M.G.A.")
        base = ""
        for k in range(len(parti) - 1, 0, -1):
            base = " ".join(parti[:k]) + " " + "".join(p[0] + "." for p in parti[k:])
            if len(base) <= spazio:
                break
    base = base.strip(" '")
    if len(base) > spazio:
        base = base[:spazio].rstrip(" '-")
    nome = (base + periodo).strip(" '") or f"Foglio {d.n}"
    candidato, k = nome, 2
    while candidato.casefold() in usati or candidato.casefold() == "history":
        suffisso = f" ({k})"
        candidato = nome[: 31 - len(suffisso)].rstrip(" '") + suffisso
        k += 1
    usati.add(candidato.casefold())
    return candidato


# --- Valori dei campi con stato di lettura -----------------------------------------


def _stato_campo(doc: Document, oggetto: Header | DayRow, campo: str, chiave: str) -> str:
    if campo in oggetto.illeggibili:
        return "illeggibile"
    if campo in oggetto.incerti:
        return "incerto"
    if chiave in doc.user_edited:
        return "corretto"
    return ""


def _commento(doc: Document, oggetto: Header | DayRow, campo: str, chiave: str, presente: bool) -> str | None:
    righe: list[str] = []
    etichetta = val.etichetta_campo(campo)
    if campo in oggetto.illeggibili:
        if presente:
            righe.append(
                f"«{etichetta}»: campo segnalato come illeggibile, il valore riportato non è affidabile. "
                "Verificare sull'originale."
            )
        else:
            righe.append(
                f"«{etichetta}»: campo scritto ma illeggibile. Leggere il valore sulla scansione "
                "originale e inserirlo a mano."
            )
    elif campo in oggetto.incerti:
        righe.append("Lettura incerta: verificare sull'originale.")
        conf = oggetto.confidenza.get(campo) if isinstance(oggetto, DayRow) else None
        if conf is not None and 0 <= conf <= 1:
            righe.append(f"Affidabilità della lettura: {round(conf * 100)}%.")
    if chiave in doc.user_edited:
        if chiave in doc.ocr_originali:
            righe.append(
                "Corretto manualmente – valore OCR originale: "
                f"{_valore_originale(campo, doc.ocr_originali[chiave])}"
            )
        else:
            righe.append("Corretto manualmente durante la revisione.")
    return "\n".join(righe) or None


def _prestazione(row: DayRow) -> bool:
    dich = _numero(row.ore_dichiarate)
    return not _vuoto(row.eff_entrata) or not _vuoto(row.eff_uscita) or (dich is not None and dich > TOLLERANZA)


def _valore_giorno(doc: Document, row: DayRow, campo: str, si: str) -> _Val:
    chiave = f"rows.{row.giorno}.{campo}"
    v = _Val(state=_stato_campo(doc, row, campo, chiave))
    if campo in _ORARI:
        raw = getattr(row, campo)
        if _vuoto(raw):
            if campo.startswith("eff_") and row.trattino_effettivo:
                v.value, v.kind = "–", "dash"
        else:
            ora = _ora_excel(raw)
            if ora is not None:
                v.value, v.kind = ora, "time"
            else:
                v.value, v.kind = _pulisci(raw), "invalid"
    elif campo == "ore_dichiarate":
        ore = _numero(row.ore_dichiarate)
        if ore is not None:
            v.value, v.kind = ore, "hours"
    elif campo in ("assenza_alunno", "assenza_operatore"):
        if getattr(row, campo):
            v.value, v.kind = si, "mark"
    elif campo == "firma":
        if row.firma:
            v.value, v.kind = "Sì", "mark"
        elif _prestazione(row) and not row.assenza_operatore:
            v.value, v.kind = "No", "no"
    elif campo == "note":
        testo = _pulisci(row.note)
        if testo:
            v.value = testo
    if campo in _ORARI:
        presente = not _vuoto(getattr(row, campo))
    elif campo == "ore_dichiarate":
        presente = v.value is not None
    elif campo in _BOOL_CAMPI:
        presente = bool(getattr(row, campo))
    else:
        presente = v.value is not None
    if v.state == "illeggibile" and not presente:
        v.value, v.kind = ILLEGGIBILE, "illeggibile"
    v.comment = _commento(doc, row, campo, chiave, presente)
    if v.kind == "invalid":
        nota = f"Orario non interpretabile: «{v.value}» (formato atteso HH:MM)."
        v.comment = f"{nota}\n{v.comment}" if v.comment else nota
    return v


def _valore_intestazione(doc: Document, campo: str) -> _Val:
    h = doc.header
    chiave = f"header.{campo}"
    v = _Val(state=_stato_campo(doc, h, campo, chiave))
    raw = getattr(h, campo)
    if campo in ("ore_pei", "totale_mensile_dichiarato"):
        ore = _numero(raw)
        if ore is not None:
            v.value, v.kind = ore, "hours"
    elif campo == "sostituzione":
        if raw in ("SI", "NO"):
            v.value = "Sì" if raw == "SI" else "No"
    elif campo in ("firma_coordinatore", "timbro_referente"):
        v.value, v.kind = ("Presente", "mark") if raw else ("Assente", "no")
    elif campo == "data_compilazione":
        dt = _data_compilazione(raw)
        if dt is not None:
            v.value, v.kind = dt, "date"
        elif raw:
            v.value = _pulisci(raw)
    else:
        testo = _pulisci(raw)
        if testo and campo in ("lotto", "municipalita") and testo.isdigit() and len(testo) <= 6:
            v.value, v.kind = int(testo), "int"  # codici numerici come numeri, non testo
        elif testo:
            v.value = testo
    presente = v.value is not None and not (campo in ("firma_coordinatore", "timbro_referente") and not raw)
    if v.state == "illeggibile" and not presente:
        v.value, v.kind = ILLEGGIBILE, "illeggibile"
    v.comment = _commento(doc, h, campo, chiave, presente)
    return v


def _valore_mese(doc: Document, periodo: tuple[int, int] | None) -> _Val:
    """Mese/anno di riferimento (due campi del modello, una sola cella)."""
    h = doc.header
    stati = [_stato_campo(doc, h, c, f"header.{c}") for c in ("mese", "anno")]
    ordine = ("illeggibile", "incerto", "corretto")
    stato = next((s for s in ordine if s in stati), "")
    v = _Val(state=stato)
    if periodo is not None:
        v.value, v.kind = date(periodo[1], periodo[0], 1), "month"
    elif h.mese is not None or h.anno is not None:
        mese = f"{h.mese:02d}" if isinstance(h.mese, int) else "??"
        v.value = f"{mese}/{h.anno if h.anno is not None else '????'}"
    if stato == "illeggibile" and v.value is None:
        v.value, v.kind = ILLEGGIBILE, "illeggibile"
    commenti = [
        c
        for c in (
            _commento(doc, h, campo, f"header.{campo}", getattr(h, campo) is not None)
            for campo in ("mese", "anno")
        )
        if c
    ]
    v.comment = "\n".join(commenti) or None
    return v


# --- Scrittura ------------------------------------------------------------------


class _Esportatore:
    def __init__(
        self,
        wb: xlsxwriter.Workbook,
        docs: list[_Doc],
        esclusi: list[tuple[Document, str]],
        options: ExportOptions,
        adesso: datetime,
    ) -> None:
        self.wb = wb
        self.f = _Formati(wb)
        self.docs = docs
        self.esclusi = esclusi
        self.opt = options
        self.adesso = adesso
        self.titolo = _pulisci(options.titolo) or TITOLO_PREDEFINITO
        self.testo_periodo = _testo_periodi([d.periodo for d in docs if d.periodo is not None])
        self.generato = f"Generato il {adesso:%d/%m/%Y} alle {adesso:%H:%M}"
        self.det: dict[str, Any] = {}

    # -- elementi comuni --

    def _imposta_foglio(
        self,
        ws: Worksheet,
        larghezze: Sequence[float],
        zoom: int,
        tab: str,
        righe_ripetute: tuple[int, int] | None = None,
        verticale: bool = False,
    ) -> None:
        for c, w in enumerate(larghezze):
            ws.set_column(c, c, w)
        ws.hide_gridlines(2)
        ws.set_zoom(zoom)
        ws.set_tab_color(tab)
        ws.set_default_row(18)
        ws.set_paper(9)
        if verticale:
            ws.set_portrait()
        else:
            ws.set_landscape()
        ws.fit_to_pages(1, 0)
        ws.center_horizontally()
        ws.set_margins(left=0.35, right=0.35, top=0.6, bottom=0.6)
        ws.set_header(
            f'&L&"{FONT},Regular"&8&K6B7280Sirio OCR · {self.titolo.replace("&", "&&")}'
            f'&R&"{FONT},Regular"&8&K6B7280&A',
            {"margin": 0.25},
        )
        ws.set_footer(
            f'&L&"{FONT},Regular"&8&K6B7280{self.generato}&R&"{FONT},Regular"&8&K6B7280Pagina &P di &N',
            {"margin": 0.25},
        )
        if righe_ripetute:
            ws.repeat_rows(*righe_ripetute)

    def _fascia(
        self, ws: Worksheet, ultima_col: int, titolo: str, sottotitolo: str, nota: str | None = None
    ) -> None:
        """Fascia del titolo blu notte (righe 0-1) con sottile linea d'accento (riga 2)."""
        f = self.f
        ws.set_row(0, 34)
        ws.set_row(1, 20)
        ws.set_row(2, 4)
        self._unisci(
            ws,
            0,
            0,
            0,
            ultima_col,
            titolo,
            f(TXT, bg_color=NAVY, font_color=WHITE, bold=True, font_size=16, indent=1),
        )
        self._unisci(
            ws,
            1,
            0,
            1,
            ultima_col,
            sottotitolo,
            f(TXT, bg_color=NAVY, font_color=NAVY_TEXT_SOFT, font_size=10, indent=1),
        )
        for c in range(ultima_col + 1):
            ws.write_blank(2, c, None, f(bg_color=ACCENT))
        if nota:
            ws.set_row(3, 20)
            self._unisci(
                ws, 3, 0, 3, ultima_col, nota, f(TXT, font_color=MUTED, italic=True, font_size=9, indent=1)
            )

    def _unisci(self, ws: Worksheet, r0: int, c0: int, r1: int, c1: int, valore: Any, fmt: Format) -> None:
        """Celle unite (o cella singola) con un valore tipizzato nella prima cella."""
        if (r0, c0) != (r1, c1):
            ws.merge_range(r0, c0, r1, c1, "", fmt)
        self._scrivi(ws, r0, c0, valore, fmt)

    def _scrivi(self, ws: Worksheet, r: int, c: int, valore: Any, fmt: Format | None = None) -> None:
        if isinstance(valore, _F):
            risultato = valore.value
            if isinstance(risultato, float) and not math.isfinite(risultato):
                risultato = 0
            ws.write_formula(r, c, valore.formula, fmt, "" if risultato is None else risultato)
        elif valore is None or valore == "":
            ws.write_blank(r, c, None, fmt)
        elif isinstance(valore, bool):
            ws.write_string(r, c, "Sì" if valore else "No", fmt)
        elif isinstance(valore, (int, float)):
            if math.isfinite(float(valore)):
                ws.write_number(r, c, valore, fmt)
            else:
                ws.write_blank(r, c, None, fmt)
        elif isinstance(valore, (date, datetime)):
            ws.write_datetime(r, c, valore, fmt)
        else:
            ws.write_string(r, c, str(valore)[:MAX_TESTO], fmt)

    def _commenta(self, ws: Worksheet, r: int, c: int, testo: str | None) -> None:
        if not testo:
            return
        righe = _righe_testo(testo, 38)
        ws.write_comment(
            r,
            c,
            testo[:MAX_TESTO],
            {
                "author": "Sirio OCR",
                "font_name": FONT,
                "font_size": 9,
                "x_scale": 2.2,
                "y_scale": max(0.9, 0.36 * righe + 0.3),
            },
        )

    def _fmt_valore(self, base: dict[str, Any], v: _Val, num_format: str | None = None) -> Format:
        """Formato di una cella-campo: base + tipo + sovrapposizione dello stato di lettura."""
        props: dict[str, Any] = dict(base)
        if v.kind == "time":
            props["num_format"] = FMT_ORA
        elif v.kind == "hours":
            props["num_format"] = num_format or FMT_ORE
        elif v.kind == "date":
            props["num_format"] = FMT_DATA
        elif v.kind == "month":
            props["num_format"] = FMT_MESE
        elif v.kind == "int":
            props["num_format"] = "0"
        elif v.kind == "dash":
            props["font_color"] = SUBTLE
        elif v.kind == "no":
            props.update(font_color=RED, bold=True)
        elif v.kind == "invalid":
            props.update(OV_NON_VALIDO)
        if v.kind == "illeggibile":
            props.update(OV_MARCATORE)
        elif v.state:
            props.update(_SOVRAPPOSIZIONI[v.state])
            if v.state == "illeggibile":
                props["font_color"] = RED_DARK
        return self.f(props)

    def _scrivi_valore(
        self, ws: Worksheet, r: int, c: int, v: _Val, base: dict[str, Any], c1: int | None = None
    ) -> None:
        fmt = self._fmt_valore(base, v)
        if c1 is not None and c1 != c:
            self._unisci(ws, r, c, r, c1, v.value, fmt)
        else:
            self._scrivi(ws, r, c, v.value, fmt)
        self._commenta(ws, r, c, v.comment)

    def _titolo_sezione(self, ws: Worksheet, r: int, c0: int, c1: int, testo: str, altezza: float = 26) -> None:
        ws.set_row(r, altezza)
        self._unisci(ws, r, c0, r, c1, testo, self.f(TXT, SECTION))

    # -- costruzione --

    def build(self) -> None:
        wb = self.wb
        self.ws_riep = wb.add_worksheet(S_RIEPILOGO)
        self.ws_det = wb.add_worksheet(S_DETTAGLIO)
        if self.opt.fogli_per_documento:
            usati = {s.casefold() for s in _FOGLI_FISSI}
            for d in self.docs:
                d.sheet = _nome_scheda(d, usati)
                d.ws = wb.add_worksheet(d.sheet)
        self.ws_anom = wb.add_worksheet(S_ANOMALIE)
        self.ws_tot = wb.add_worksheet(S_TOTALI)
        self.ws_leg = wb.add_worksheet(S_LEGENDA)

        if self.opt.fogli_per_documento:
            for d in self.docs:
                self._scheda(d)
        self._dettaglio()
        self._riepilogo()
        self._anomalie()
        self._totali()
        self._legenda()
        self.ws_riep.activate()

    # ------------------------------------------------------------------ Riepilogo

    R_RIEP_HEAD = 10
    _R_COLS: tuple[tuple[str, float], ...] = (
        ("N.", 5),
        ("Operatore", 24),
        ("Alunno", 24),
        ("Istituto", 22),
        ("Mese", 13),
        ("Ore PEI sett.", 8.5),
        ("Ore programmate", 10.5),
        ("Ore calcolate", 10),
        ("Ore dichiarate", 10),
        ("Ore riconosciute", 10.5),
        ("Totale mensile dichiarato", 11),
        ("Differenza totale", 10),
        ("Giorni lavorati", 8.5),
        ("Assenze alunno", 8.5),
        ("Assenze operatore", 9),
        ("Firme mancanti", 8.5),
        ("Campi incerti", 8.5),
        ("Campi illeggibili", 9.5),
        ("Errori", 7.5),
        ("Attenzioni", 9),
        ("Stato", 13.5),
        ("Verificato", 9.5),
        ("File di origine", 30),
    )

    def _riepilogo(self) -> None:
        ws, f = self.ws_riep, self.f
        larghezze = [w for _, w in self._R_COLS]
        last_c = len(larghezze) - 1
        R_HEAD = self.R_RIEP_HEAD
        r_first = R_HEAD + 1
        r_last = r_first + len(self.docs) - 1
        r_tot = r_last + 1
        self._imposta_foglio(ws, larghezze, 90, NAVY, righe_ripetute=(R_HEAD, R_HEAD))

        enti = sorted({_pulisci(d.doc.header.ente) for d in self.docs if _pulisci(d.doc.header.ente)})
        ente = f" · {enti[0]}" if len(enti) == 1 else ""
        self._fascia(
            ws,
            last_c,
            self.titolo,
            f"Comune di Napoli · {SERVIZIO} · {self.testo_periodo}{ente}",
        )
        ws.set_row(3, 18)
        self._unisci(
            ws,
            3,
            0,
            3,
            last_c,
            f"{self.generato} con Sirio OCR {__version__} · "
            f"{_plurale(len(self.docs), 'foglio firma', 'fogli firma')}",
            f(TXT, font_color=MUTED, font_size=9, indent=1),
        )

        # Colonne (lettere) usate nelle formule
        C = {nome: i for i, (nome, _) in enumerate(self._R_COLS)}

        def col_rng(nome: str) -> str:
            c = C[nome]
            return _rng(r_first, c, r_last, c)

        def tot_ref(nome: str) -> str:
            return _cell(r_tot, C[nome], True)

        # Valori complessivi (per i risultati memorizzati)
        somma = {
            k: math.fsum(d.tot[k] for d in self.docs)
            for k in ("prog", "calc", "dich", "ric", "giorni", "ass_al", "ass_op")
        }
        n_inc = sum(d.doc.totals.campi_incerti for d in self.docs)
        n_ill = sum(d.doc.totals.campi_illeggibili for d in self.docs)
        n_err = sum(d.doc.totals.n_errori for d in self.docs)
        n_att = sum(d.doc.totals.n_attenzioni for d in self.docs)
        n_ver = sum(1 for d in self.docs if d.doc.user_verified)
        n_doc_err = sum(1 for d in self.docs if d.stato == "errori")

        # KPI
        tiles: list[tuple[str, _F, str, str | _F, str]] = [
            (
                "FOGLI FIRMA",
                _F(f"SUBTOTAL(103,{col_rng('Operatore')})", len(self.docs)),
                FMT_INT,
                _F(
                    _plurale_formula(f'COUNTIF({col_rng("Verificato")},"Sì")', "verificato", "verificati")
                    + '&" · "&'
                    + f'COUNTIF({col_rng("Stato")},"Errori")&" con errori"',
                    f"{_plurale(n_ver, 'verificato', 'verificati')} · {n_doc_err} con errori",
                ),
                NAVY,
            ),
            (
                "ORE RICONOSCIUTE",
                _F(tot_ref("Ore riconosciute"), somma["ric"]),
                FMT_ORE,
                "ore effettive ammesse",
                NAVY,
            ),
            (
                "ORE PROGRAMMATE",
                _F(tot_ref("Ore programmate"), somma["prog"]),
                FMT_ORE,
                "da orario programmato",
                NAVY,
            ),
            (
                "GIORNI LAVORATI",
                _F(tot_ref("Giorni lavorati"), somma["giorni"]),
                FMT_INT,
                "giornate con ore riconosciute",
                NAVY,
            ),
            ("ASSENZE ALUNNO", _F(tot_ref("Assenze alunno"), somma["ass_al"]), FMT_INT, "giornate", NAVY),
            ("ASSENZE OPERATORE", _F(tot_ref("Assenze operatore"), somma["ass_op"]), FMT_INT, "giornate", NAVY),
            (
                "CAMPI ILLEGGIBILI",
                _F(tot_ref("Campi illeggibili"), n_ill),
                FMT_INT,
                _F(
                    '"più "&' + _plurale_formula(tot_ref("Campi incerti"), "lettura incerta", "letture incerte"),
                    f"più {_plurale(n_inc, 'lettura incerta', 'letture incerte')}",
                ),
                RED if n_ill else NAVY,
            ),
            (
                "ANOMALIE DA ESAMINARE",
                _F(f"{tot_ref('Errori')}+{tot_ref('Attenzioni')}", n_err + n_att),
                FMT_INT,
                _F(
                    _plurale_formula(tot_ref("Errori"), "errore", "errori")
                    + '&" · "&'
                    + _plurale_formula(tot_ref("Attenzioni"), "attenzione", "attenzioni"),
                    f"{_plurale(n_err, 'errore', 'errori')} · {_plurale(n_att, 'attenzione', 'attenzioni')}",
                ),
                RED if n_err else (AMBER if n_att else NAVY),
            ),
        ]
        ws.set_row(4, 22)
        ws.set_row(5, 34)
        ws.set_row(6, 18)
        tile = {"bg_color": TILE, "left": 5, "right": 5, "left_color": WHITE, "right_color": WHITE}
        for (c0, c1), (etichetta, valore, numfmt, didascalia, colore) in zip(
            _partizione(larghezze, len(tiles)), tiles
        ):
            self._unisci(
                ws,
                4,
                c0,
                4,
                c1,
                etichetta,
                f(
                    TXT,
                    tile,
                    font_size=8,
                    bold=True,
                    font_color=MUTED,
                    indent=1,
                    top=2,
                    top_color=ACCENT,
                    valign="vcenter",
                ),
            )
            self._unisci(
                ws,
                5,
                c0,
                5,
                c1,
                valore,
                f(
                    TXT,
                    tile,
                    font_size=20,
                    bold=True,
                    font_color=colore,
                    indent=1,
                    num_format=numfmt,
                    align="left",
                ),
            )
            self._unisci(
                ws, 6, c0, 6, c1, didascalia, f(TXT, tile, font_size=8.5, font_color=MUTED, indent=1, valign="top")
            )

        # Avviso campi illeggibili/incerti
        ws.set_row(7, 8)
        ws.set_row(8, 24)
        if n_ill or n_inc:
            parti = []
            if n_ill:
                parti.append(_plurale(n_ill, "campo illeggibile", "campi illeggibili"))
            if n_inc:
                parti.append(_plurale(n_inc, "lettura incerta", "letture incerte"))
            testo = (
                f"Da verificare sugli originali: {' e '.join(parti)}. Le celle sono evidenziate in rosso "
                "(illeggibile) e in ambra (incerto) nelle schede dei fogli firma e nel Dettaglio giornaliero; "
                "l'elenco completo è nel foglio Anomalie."
            )
            colore, sfondo = (RED_DARK, RED_FILL) if n_ill else (AMBER_TEXT, AMBER_FILL)
        else:
            testo = "Tutti i campi sono stati letti con sicurezza: nessun campo illeggibile o incerto."
            colore, sfondo = GREEN, GREEN_FILL
        self._unisci(
            ws,
            8,
            0,
            8,
            last_c,
            testo,
            f(TXT, font_color=colore, bg_color=sfondo, bold=True, indent=1, left=5, left_color=colore),
        )
        ws.set_row(9, 10)

        # Tabella Excel
        head_fmt = f(TXT, HEAD, font_size=9)
        head_calc = f(TXT, HEAD_CALC, font_size=9)
        calcolate = {"Ore programmate", "Ore calcolate", "Ore riconosciute", "Differenza totale"}
        ws.set_row(R_HEAD, 36)
        ws.add_table(
            R_HEAD,
            0,
            max(r_last, r_first),
            last_c,
            {
                "name": "Riepilogo_fogli",
                "style": "Table Style Medium 2",
                "autofilter": True,
                "banded_rows": True,
                "columns": [
                    {"header": nome, "header_format": head_calc if nome in calcolate else head_fmt}
                    for nome, _ in self._R_COLS
                ],
            },
        )
        cella = {**TXT, **GRID}
        testo_a_capo = {**cella, "text_wrap": True}
        num = {**cella, "num_format": FMT_ORE}
        intero = {**cella, "num_format": FMT_INT, "align": "center"}
        larg = dict(self._R_COLS)
        for i, d in enumerate(self.docs):
            r = r_first + i
            ws.set_row(
                r,
                max(
                    20.0,
                    *(
                        _altezza(testo, larg[nome] * 0.95, 20, 13)
                        for nome, testo in (
                            ("Operatore", d.operatore),
                            ("Alunno", d.alunno),
                            ("Istituto", d.istituto),
                        )
                    ),
                ),
            )
            doc, t = d.doc, d.doc.totals
            self._scrivi(ws, r, C["N."], d.n, f(intero, font_color=MUTED))
            # Operatore con collegamento alla scheda
            v_op = _valore_intestazione(doc, "operatore")
            fmt_op = self._fmt_valore({**testo_a_capo, "bold": True}, _Val(state=v_op.state))
            if d.sheet:
                ws.write_url(
                    r,
                    C["Operatore"],
                    d.link() or "",
                    self._fmt_valore(
                        {**testo_a_capo, "bold": True, "font_color": ACCENT_DARK, "underline": 1},
                        _Val(state=v_op.state),
                    ),
                    string=d.operatore,
                    tip="Apri la scheda del foglio firma",
                )
            else:
                self._scrivi(ws, r, C["Operatore"], d.operatore, fmt_op)
            self._commenta(ws, r, C["Operatore"], v_op.comment)
            for nome, campo, testo in (("Alunno", "alunno", d.alunno), ("Istituto", "istituto", d.istituto)):
                v = _valore_intestazione(doc, campo)
                self._scrivi(ws, r, C[nome], testo, self._fmt_valore(testo_a_capo, _Val(state=v.state)))
                self._commenta(ws, r, C[nome], v.comment)
            v_mese = _valore_mese(doc, d.periodo)
            self._scrivi_valore(ws, r, C["Mese"], v_mese, {**cella, "align": "left"})
            self._scrivi_valore(
                ws, r, C["Ore PEI sett."], _valore_intestazione(doc, "ore_pei"), {**cella, "num_format": FMT_ORE}
            )

            # Valori collegati alla scheda (o al Dettaglio se le schede non sono generate)
            for nome, chiave in (
                ("Ore programmate", "prog"),
                ("Ore calcolate", "calc"),
                ("Ore dichiarate", "dich"),
                ("Ore riconosciute", "ric"),
            ):
                ws.write_formula(
                    r, C[nome], self._rif_doc(d, chiave), f(num, bold=nome == "Ore riconosciute"), d.tot[chiave]
                )
            v_tot = _valore_intestazione(doc, "totale_mensile_dichiarato")
            if d.sheet:
                ref = d.refs["tot_dich"]
                cached: Any = v_tot.value if v_tot.value is not None else ""
                self._scrivi(
                    ws,
                    r,
                    C["Totale mensile dichiarato"],
                    _F(f'IF(ISBLANK({ref}),"",{ref})', cached),
                    self._fmt_valore(num, v_tot),
                )
            else:
                self._scrivi(ws, r, C["Totale mensile dichiarato"], v_tot.value, self._fmt_valore(num, v_tot))
            self._commenta(ws, r, C["Totale mensile dichiarato"], v_tot.comment)
            tot_c = _cell(r, C["Totale mensile dichiarato"])
            dich_c = _cell(r, C["Ore dichiarate"])
            tot_val = _numero(v_tot.value) if v_tot.kind == "hours" else None
            ws.write_formula(
                r,
                C["Differenza totale"],
                f'IF(ISNUMBER({tot_c}),{dich_c}-{tot_c},"")',
                f(cella, num_format=FMT_ORE_SEGNO),
                "" if tot_val is None else d.tot["dich"] - tot_val,
            )
            for nome, chiave in (
                ("Giorni lavorati", "giorni"),
                ("Assenze alunno", "ass_al"),
                ("Assenze operatore", "ass_op"),
            ):
                ws.write_formula(r, C[nome], self._rif_doc(d, chiave), f(intero), d.tot[chiave])
            for nome, conteggio in (
                ("Firme mancanti", t.firme_mancanti),
                ("Campi incerti", t.campi_incerti),
                ("Campi illeggibili", t.campi_illeggibili),
                ("Errori", t.n_errori),
                ("Attenzioni", t.n_attenzioni),
            ):
                self._scrivi(ws, r, C[nome], conteggio, f(intero))
            self._scrivi(ws, r, C["Stato"], _STATI[d.stato], f(cella, bold=True, align="center"))
            self._scrivi(ws, r, C["Verificato"], "Sì" if doc.user_verified else "No", f(cella, align="center"))
            origine = doc.source_file + (f" · pag. {doc.source_page}" if doc.page_count > 1 else "")
            self._scrivi(ws, r, C["File di origine"], origine, f(cella, font_color=MUTED, font_size=9))

        # Riga dei totali (SUBTOTAL: rispetta i filtri)
        ws.set_row(r_tot, 24)
        tot_fmt = {**TXT, **TOTAL}
        self._scrivi(ws, r_tot, 0, None, f(tot_fmt))
        self._scrivi(ws, r_tot, C["Operatore"], "Totale", f(tot_fmt))
        self._scrivi(
            ws,
            r_tot,
            C["Alunno"],
            _F(f'SUBTOTAL(103,{col_rng("Operatore")})&" fogli firma"', f"{len(self.docs)} fogli firma"),
            f(tot_fmt, font_color=MUTED, bold=False, font_size=9),
        )
        for nome in ("Istituto", "Mese", "Ore PEI sett.", "Stato", "Verificato", "File di origine"):
            self._scrivi(ws, r_tot, C[nome], None, f(tot_fmt))
        somme_ore = {
            "Ore programmate": somma["prog"],
            "Ore calcolate": somma["calc"],
            "Ore dichiarate": somma["dich"],
            "Ore riconosciute": somma["ric"],
            "Totale mensile dichiarato": math.fsum(
                _numero(d.doc.header.totale_mensile_dichiarato) or 0.0 for d in self.docs
            ),
            "Differenza totale": math.fsum(
                d.tot["dich"] - tot
                for d in self.docs
                if (tot := _numero(d.doc.header.totale_mensile_dichiarato)) is not None
            ),
        }
        for nome, somma_ore in somme_ore.items():
            numfmt = FMT_ORE_SEGNO if nome == "Differenza totale" else FMT_ORE
            self._scrivi(
                ws, r_tot, C[nome], _F(f"SUBTOTAL(109,{col_rng(nome)})", somma_ore), f(tot_fmt, num_format=numfmt)
            )
        conteggi = {
            "Giorni lavorati": somma["giorni"],
            "Assenze alunno": somma["ass_al"],
            "Assenze operatore": somma["ass_op"],
            "Firme mancanti": sum(d.doc.totals.firme_mancanti for d in self.docs),
            "Campi incerti": n_inc,
            "Campi illeggibili": n_ill,
            "Errori": n_err,
            "Attenzioni": n_att,
        }
        for nome, n_tot in conteggi.items():
            self._scrivi(
                ws,
                r_tot,
                C[nome],
                _F(f"SUBTOTAL(109,{col_rng(nome)})", n_tot),
                f(tot_fmt, num_format=FMT_INT, align="center"),
            )

        # Formattazione condizionale
        cs = C["Stato"]
        for testo, (colore, sfondo) in (
            ("OK", _STATO_COLORI["ok"]),
            ("Da verificare", _STATO_COLORI["da_verificare"]),
            ("Errori", _STATO_COLORI["errori"]),
        ):
            ws.conditional_format(
                r_first,
                cs,
                r_last,
                cs,
                {
                    "type": "cell",
                    "criteria": "==",
                    "value": f'"{testo}"',
                    "format": f(font_color=colore, bg_color=sfondo, bold=True),
                },
            )
        cv = C["Verificato"]
        ws.conditional_format(
            r_first,
            cv,
            r_last,
            cv,
            {"type": "cell", "criteria": "==", "value": '"Sì"', "format": f(font_color=GREEN, bold=True)},
        )
        for nome, colore, sfondo in (
            ("Campi illeggibili", RED_DARK, RED_FILL),
            ("Errori", RED_DARK, RED_FILL),
            ("Campi incerti", AMBER_TEXT, AMBER_FILL),
            ("Attenzioni", AMBER_TEXT, AMBER_FILL),
            ("Firme mancanti", AMBER_TEXT, AMBER_FILL),
        ):
            c = C[nome]
            ws.conditional_format(
                r_first,
                c,
                r_last,
                c,
                {
                    "type": "cell",
                    "criteria": ">",
                    "value": 0,
                    "format": f(font_color=colore, bg_color=sfondo, bold=True),
                },
            )
        cd = C["Differenza totale"]
        ws.conditional_format(
            r_first,
            cd,
            r_last,
            cd,
            {
                "type": "formula",
                "criteria": f"=AND(ISNUMBER({_cell(r_first, cd)}),ABS({_cell(r_first, cd)})>{TOLLERANZA})",
                "format": f(font_color=RED_DARK, bg_color=RED_FILL, bold=True),
            },
        )

        ws.data_validation(
            r_first,
            cv,
            r_last,
            cv,
            {
                "validate": "list",
                "source": ["Sì", "No"],
                "error_title": "Valore non ammesso",
                "error_message": "Scegliere «Sì» oppure «No».",
            },
        )

        # Note sotto la tabella
        r_note = r_tot + 2
        note = [
            (
                "Fare clic sul nome dell'operatore per aprire la scheda del foglio firma. Le colonne in blu sono "
                "calcolate con formule; i totali considerano solo le righe visibili quando si usano i filtri."
            ),
            (
                "Stato: «Errori» = almeno un controllo non superato; «Da verificare» = segnalazioni o campi "
                "incerti/illeggibili; «OK» = nessun rilievo. Il significato dei colori e le regole di calcolo sono "
                "nel foglio «Legenda e note»."
            ),
        ]
        if not self.opt.fogli_per_documento:
            note[0] = (
                "Le colonne in blu sono calcolate con formule dal foglio «Dettaglio giornaliero»; "
                "i totali considerano solo le righe visibili quando si usano i filtri."
            )
        for k, testo in enumerate(note):
            self._unisci(
                ws,
                r_note + k,
                0,
                r_note + k,
                last_c,
                testo,
                f(TXT, font_color=MUTED, font_size=9, italic=True, indent=1),
            )
        ws.freeze_panes(r_first, 2)

    def _rif_doc(self, d: _Doc, chiave: str) -> str:
        """Riferimento a un totale del documento: scheda se presente, altrimenti formule sul Dettaglio."""
        if d.sheet:
            return d.refs[chiave]
        rng = self.det["rng"]
        if chiave == "giorni":
            return f'COUNTIFS({rng["doc"]},{d.n},{rng["ric"]},">0")'
        if chiave in ("ass_al", "ass_op"):
            colonna = "aa" if chiave == "ass_al" else "ao"
            return f'COUNTIFS({rng["doc"]},{d.n},{rng[colonna]},"Sì")'
        return f"SUMIFS({rng[chiave]},{rng['doc']},{d.n})"

    # ---------------------------------------------------------- Scheda documento

    # Colonne della scheda: le prime 12 replicano il modulo, le ultime 5 sono calcolate.
    _S_WIDTHS = (7, 11, 6, 8.5, 8.5, 8.5, 8.5, 9.5, 9.5, 9.5, 9.5, 22, 10, 10, 10.5, 9.5, 44)
    S_G, S_DATA, S_WD, S_PE, S_PU, S_EE, S_EU, S_OD, S_AA, S_AO, S_FI, S_NO, S_OP, S_OC, S_OR, S_DF, S_ES = range(
        17
    )
    S_R_TH1, S_R_TH2, S_R_D1 = 11, 12, 13
    S_R_TOT = 44

    def _scheda(self, d: _Doc) -> None:
        ws = d.ws
        assert ws is not None
        f = self.f
        doc, h = d.doc, d.doc.header
        last_c = len(self._S_WIDTHS) - 1
        colore_stato, sfondo_stato = _STATO_COLORI[d.stato]
        self._imposta_foglio(ws, self._S_WIDTHS, 100, colore_stato, righe_ripetute=(self.S_R_TH1, self.S_R_TH2))

        # Fascia del titolo con collegamento al Riepilogo
        anno_scol = _pulisci(h.anno_scolastico)
        if not anno_scol and d.periodo:
            mese_rif, anno_rif = d.periodo
            anno_scol = f"{anno_rif}/{anno_rif + 1}" if mese_rif >= 9 else f"{anno_rif - 1}/{anno_rif}"
        sotto = ["Foglio firma mensile", SERVIZIO + " destinato agli alunni disabili"]
        if anno_scol:
            sotto.append(f"Anno scolastico {anno_scol}")
        ws.set_row(0, 34)
        ws.set_row(1, 20)
        ws.set_row(2, 4)
        band = {**TXT, "bg_color": NAVY}
        self._unisci(
            ws,
            0,
            0,
            0,
            last_c - 1,
            "COMUNE DI NAPOLI  ·  Assistenza Specialistica",
            f(band, font_color=WHITE, bold=True, font_size=16, indent=1),
        )
        self._scrivi(
            ws,
            0,
            last_c,
            f"Foglio {d.n} di {len(self.docs)}",
            f(band, font_color=NAVY_TEXT_SOFT, align="right", indent=1),
        )
        self._unisci(ws, 1, 0, 1, last_c, "  ·  ".join(sotto), f(band, font_color=NAVY_TEXT_SOFT, indent=1))
        for c in range(last_c + 1):
            ws.write_blank(2, c, None, f(bg_color=ACCENT))
        ws.set_row(3, 18)
        ws.write_url(
            3,
            last_c,
            _link(S_RIEPILOGO),
            f(TXT, font_color=ACCENT_DARK, underline=1, font_size=9, align="right"),
            string="‹ Torna al Riepilogo",
            tip="Torna al foglio Riepilogo",
        )

        # Intestazione del modulo
        lab = f(TXT, LABEL, border=1, border_color=WHITE)
        val_base = {
            **TXT,
            "bold": True,
            "bottom": 1,
            "bottom_color": BORDER,
            "indent": 1,
            "align": "left",
            "shrink": True,
        }
        sinistra = [
            ("ente", "Ente"),
            ("istituto", "Istituto scolastico"),
            ("operatore", "Operatore"),
            ("alunno", "Alunno"),
            ("lotto", "Lotto"),
            ("municipalita", "Municipalità"),
        ]
        destra = [
            ("mese", "Mese/anno di riferimento"),
            ("ore_pei", "Ore da PEI (settimanali)"),
            ("sostituzione", "Sostituzione"),
            ("data_compilazione", "Data di compilazione"),
            ("firma_coordinatore", "Firma coordinatore ente"),
            ("timbro_referente", "Timbro referente scolastico"),
        ]
        larghezza_sx = sum(self._S_WIDTHS[3:8])
        larghezza_dx = sum(self._S_WIDTHS[11:14])

        def adatta(base: dict[str, Any], v: _Val, larghezza: float) -> tuple[dict[str, Any], float]:
            """Testi lunghi (es. nomi di istituto) vanno a capo invece di rimpicciolirsi."""
            if not isinstance(v.value, str):
                return base, 21.0
            scala = 10 / base.get("font_size", 10) * 0.92  # grassetto: caratteri più larghi
            righe = _righe_testo(v.value, larghezza * scala)
            if righe <= 1:
                return base, 21.0
            senza_riduzione = {k: x for k, x in base.items() if k != "shrink"}
            return {**senza_riduzione, "text_wrap": True}, 14.0 * righe + 6

        for i in range(6):
            r = 4 + i
            campo, etichetta = sinistra[i]
            self._unisci(ws, r, 0, r, 2, etichetta, lab)
            v = _valore_intestazione(doc, campo)
            base = {**val_base, "font_size": 11} if campo in ("operatore", "alunno") else val_base
            base, altezza_sx = adatta(base, v, larghezza_sx)
            self._scrivi_valore(ws, r, 3, v, base, c1=7)
            campo, etichetta = destra[i]
            self._unisci(ws, r, 8, r, 10, etichetta, lab)
            v = _valore_mese(doc, d.periodo) if campo == "mese" else _valore_intestazione(doc, campo)
            if campo in ("firma_coordinatore", "timbro_referente") and v.kind == "mark":
                base = {**val_base, "font_color": GREEN}
            else:
                base = val_base
            base, altezza_dx = adatta(base, v, larghezza_dx)
            self._scrivi_valore(ws, r, 11, v, base, c1=13)
            ws.set_row(r, max(altezza_sx, altezza_dx))
            if campo == "ore_pei":
                d.refs["ore_pei_cell"] = _cell(r, 11, True)

        # Riquadro dello stato
        card = {**TXT, "bg_color": sfondo_stato, "left": 5, "left_color": colore_stato, "indent": 1}
        t = doc.totals
        corretti = len(doc.user_edited)
        self._unisci(
            ws, 4, 14, 4, last_c, "STATO DEL DOCUMENTO", f(card, font_size=8, bold=True, font_color=MUTED)
        )
        self._unisci(
            ws,
            5,
            14,
            6,
            last_c,
            _STATI[d.stato].upper(),
            f(card, font_size=18, bold=True, font_color=colore_stato),
        )
        self._unisci(
            ws,
            7,
            14,
            7,
            last_c,
            f"{_plurale(t.n_errori, 'errore', 'errori')} · "
            f"{_plurale(t.n_attenzioni, 'attenzione', 'attenzioni')} · "
            f"{_plurale(t.n_info, 'informazione', 'informazioni')}",
            f(card, font_color=INK),
        )
        self._unisci(
            ws,
            8,
            14,
            8,
            last_c,
            f"{_plurale(t.campi_illeggibili, 'campo illeggibile', 'campi illeggibili')} · "
            f"{_plurale(t.campi_incerti, 'lettura incerta', 'letture incerte')} · "
            f"{_plurale(corretti, 'correzione manuale', 'correzioni manuali')}",
            f(card, font_color=INK),
        )
        self._unisci(
            ws,
            9,
            14,
            9,
            last_c,
            "Confermato in revisione" if doc.user_verified else "Non ancora confermato in revisione",
            f(
                card,
                font_color=GREEN if doc.user_verified else MUTED,
                bold=doc.user_verified,
                italic=not doc.user_verified,
            ),
        )
        ws.set_row(10, 16)
        self._unisci(
            ws,
            10,
            self.S_OP,
            10,
            last_c,
            "Calcoli e controlli di Sirio OCR",
            f(TXT, font_size=8, bold=True, font_color=ACCENT, align="center", valign="bottom"),
        )

        # Intestazione della tabella (due righe, come sul modulo)
        ws.set_row(self.S_R_TH1, 30)
        ws.set_row(self.S_R_TH2, 18)
        hf = f(TXT, HEAD, font_size=9)
        hc = f(TXT, HEAD_CALC, font_size=9)
        r1, r2 = self.S_R_TH1, self.S_R_TH2
        for c, testo in (
            (self.S_G, "Giorno"),
            (self.S_DATA, "Data"),
            (self.S_WD, "Gg."),
            (self.S_OD, "Tot. ore effettive"),
            (self.S_AA, "Assenza alunno"),
            (self.S_AO, "Assenza operatore"),
            (self.S_FI, "Firma operatore"),
            (self.S_NO, "Note"),
        ):
            self._unisci(ws, r1, c, r2, c, testo, hf)
        self._unisci(ws, r1, self.S_PE, r1, self.S_PU, "Orario programmato", hf)
        self._unisci(ws, r1, self.S_EE, r1, self.S_EU, "Orario effettivo", hf)
        for c, testo in (
            (self.S_PE, "Entrata"),
            (self.S_PU, "Uscita"),
            (self.S_EE, "Entrata"),
            (self.S_EU, "Uscita"),
        ):
            self._scrivi(ws, r2, c, testo, hf)
        for c, testo in (
            (self.S_OP, "Ore programmate"),
            (self.S_OC, "Ore calcolate"),
            (self.S_OR, "Ore riconosciute"),
            (self.S_DF, "Differenza dich. − calc."),
            (self.S_ES, "Esito dei controlli"),
        ):
            self._unisci(ws, r1, c, r2, c, testo, hc)

        # Righe dei giorni
        A = {k: xl_col_to_name(getattr(self, k)) for k in ("S_PE", "S_PU", "S_EE", "S_EU", "S_OD", "S_AO", "S_OC")}
        e01 = {a.giorno for a in doc.anomalies if a.codice == val.E01_ORE_NON_COERENTI and a.giorno}
        for g in range(1, 32):
            r = self.S_R_D1 + g - 1
            n = r + 1
            d.righe_giorno[g] = r
            row = doc.rows[g - 1]
            calc = d.calcs[g - 1]
            tipo = d.tipo(g)
            esiste = d.giorno_esiste(g)
            fill: dict[str, Any] = {}
            if not esiste:
                fill = {"bg_color": INESISTENTE}
            elif tipo in ("sabato", "domenica", "festivo"):
                fill = {"bg_color": WEEKEND}
            base = {**TXT, **GRID, **CENTER, **fill}
            muted = {**base, "font_color": MUTED}
            self._scrivi(ws, r, self.S_G, g, f(base, bold=True))
            data = d.data(g)
            self._scrivi(ws, r, self.S_DATA, data, f(muted, num_format=FMT_DATA))
            wd = calendario.GIORNI_BREVI[data.weekday()] if data else ""
            self._scrivi(
                ws,
                r,
                self.S_WD,
                wd,
                f(
                    muted,
                    bold=tipo in ("domenica", "festivo"),
                    font_color=RED if tipo in ("domenica", "festivo") else MUTED,
                ),
            )
            for c, campo in (
                (self.S_PE, "prog_entrata"),
                (self.S_PU, "prog_uscita"),
                (self.S_EE, "eff_entrata"),
                (self.S_EU, "eff_uscita"),
                (self.S_OD, "ore_dichiarate"),
                (self.S_AA, "assenza_alunno"),
                (self.S_AO, "assenza_operatore"),
                (self.S_FI, "firma"),
            ):
                b = (
                    {**base, "bold": True}
                    if campo in ("ore_dichiarate", "assenza_alunno", "assenza_operatore")
                    else base
                )
                self._scrivi_valore(ws, r, c, _valore_giorno(doc, row, campo, "X"), b)
            v_note = _valore_giorno(doc, row, "note", "X")
            self._scrivi_valore(
                ws, r, self.S_NO, v_note, {**base, "align": "left", "font_size": 9, "indent": 1, "text_wrap": True}
            )
            testo_note = v_note.value if isinstance(v_note.value, str) else ""

            ore = {**base, "num_format": FMT_ORE}
            if esiste:
                self._scrivi(
                    ws,
                    r,
                    self.S_OP,
                    _F(
                        f"IF(AND(ISNUMBER({A['S_PE']}{n}),ISNUMBER({A['S_PU']}{n})),"
                        f'IF({A["S_PU"]}{n}>{A["S_PE"]}{n},({A["S_PU"]}{n}-{A["S_PE"]}{n})*24,""),"")',
                        calc.prog,
                    ),
                    f(ore, font_color=MUTED),
                )
                self._scrivi(
                    ws,
                    r,
                    self.S_OC,
                    _F(
                        f"IF(AND(ISNUMBER({A['S_EE']}{n}),ISNUMBER({A['S_EU']}{n})),"
                        f'IF({A["S_EU"]}{n}>{A["S_EE"]}{n},({A["S_EU"]}{n}-{A["S_EE"]}{n})*24,""),"")',
                        calc.calc,
                    ),
                    f(ore),
                )
                self._scrivi(
                    ws,
                    r,
                    self.S_OR,
                    _F(
                        f'IF({A["S_AO"]}{n}="X",0,IF(ISNUMBER({A["S_OD"]}{n}),{A["S_OD"]}{n},'
                        f'IF(ISNUMBER({A["S_OC"]}{n}),{A["S_OC"]}{n},"")))',
                        calc.ric,
                    ),
                    f(ore, bold=True, font_color=NAVY),
                )
                diff_fmt = {**base, "num_format": FMT_ORE_SEGNO}
                if g in e01:
                    diff_fmt.update(OV_NON_VALIDO)
                else:
                    diff_fmt["font_color"] = MUTED
                self._scrivi(
                    ws,
                    r,
                    self.S_DF,
                    _F(
                        f'IF(AND(ISNUMBER({A["S_OD"]}{n}),ISNUMBER({A["S_OC"]}{n})),{A["S_OD"]}{n}-{A["S_OC"]}{n},"")',
                        calc.diff,
                    ),
                    f(diff_fmt),
                )
            else:
                for c in (self.S_OP, self.S_OC, self.S_OR, self.S_DF):
                    self._scrivi(ws, r, c, None, f(base))

            # Esito
            esito = _testo_esito(val.esito_riga(row, doc.anomalies)) if esiste or row.has_content() else ""
            festa = calendario.nome_festivita(d.periodo[1], d.periodo[0], g) if d.periodo and esiste else None
            es_fmt = {**base, "align": "left", "indent": 1, "font_size": 9}
            if not esiste and not esito:
                esito = "Giorno inesistente nel mese"
                es_fmt.update(font_color=SUBTLE, italic=True)
            elif esito.startswith("Errore"):
                es_fmt.update(font_color=RED, bold=True)
            elif esito.startswith("Da verificare"):
                es_fmt.update(font_color=AMBER, bold=True)
            elif esito == "OK":
                es_fmt.update(font_color=GREEN, bold=True)
            elif esito.startswith("Assenza"):
                es_fmt.update(font_color=ACCENT_DARK)
            else:
                es_fmt.update(font_color=MUTED, italic=True)
            if festa:
                esito = f"{esito} · {festa}" if esito else festa
                if esito == festa:
                    es_fmt.update(font_color=MUTED, italic=True, bold=False)
            elif tipo in ("sabato", "domenica") and not esito:
                esito = _maiuscola(tipo)
                es_fmt.update(font_color=SUBTLE, italic=True, bold=False)
            # esiti lunghi (più campi illeggibili) vanno a capo invece di uscire dalla tabella
            ws.set_row(
                r,
                max(
                    _altezza(esito, self._S_WIDTHS[self.S_ES] * 1.15, 18, 12.5),
                    _altezza(testo_note, self._S_WIDTHS[self.S_NO] * 1.1, 18, 12.5),
                ),
            )
            self._scrivi(ws, r, self.S_ES, esito, f(es_fmt, text_wrap=True))

        # Riga dei totali
        rt = self.S_R_TOT
        r_a = self.S_R_D1
        r_b = self.S_R_D1 + d.n_giorni - 1
        ws.set_row(rt, 24)
        tf = {**TXT, **TOTAL, "align": "center"}
        self._unisci(
            ws,
            rt,
            0,
            rt,
            self.S_EU,
            "Totale ore effettive mensili (somma dei giorni)",
            f(tf, align="right", indent=1),
        )

        def somma(c: int) -> str:
            return f"SUM({_rng(r_a, c, r_b, c, False)})"

        def conta(c: int, criterio: str) -> str:
            return f'COUNTIF({_rng(r_a, c, r_b, c, False)},"{criterio}")'

        tot = d.tot
        self._scrivi(ws, rt, self.S_OD, _F(somma(self.S_OD), tot["dich"]), f(tf, num_format=FMT_ORE))
        self._scrivi(ws, rt, self.S_AA, _F(conta(self.S_AA, "X"), tot["ass_al"]), f(tf, num_format=FMT_INT))
        self._scrivi(ws, rt, self.S_AO, _F(conta(self.S_AO, "X"), tot["ass_op"]), f(tf, num_format=FMT_INT))
        self._scrivi(ws, rt, self.S_FI, _F(conta(self.S_FI, "Sì"), tot["firme"]), f(tf, num_format=FMT_INT))
        self._scrivi(ws, rt, self.S_NO, None, f(tf))
        self._scrivi(ws, rt, self.S_OP, _F(somma(self.S_OP), tot["prog"]), f(tf, num_format=FMT_ORE))
        self._scrivi(ws, rt, self.S_OC, _F(somma(self.S_OC), tot["calc"]), f(tf, num_format=FMT_ORE))
        self._scrivi(
            ws, rt, self.S_OR, _F(somma(self.S_OR), tot["ric"]), f(tf, num_format=FMT_ORE, font_color=NAVY)
        )
        self._scrivi(ws, rt, self.S_DF, None, f(tf))
        self._scrivi(ws, rt, self.S_ES, "", f(tf))
        sheet = _q(d.sheet or "")
        d.refs.update(
            {
                "dich": f"{sheet}!{_cell(rt, self.S_OD, True)}",
                "ass_al": f"{sheet}!{_cell(rt, self.S_AA, True)}",
                "ass_op": f"{sheet}!{_cell(rt, self.S_AO, True)}",
                "prog": f"{sheet}!{_cell(rt, self.S_OP, True)}",
                "calc": f"{sheet}!{_cell(rt, self.S_OC, True)}",
                "ric": f"{sheet}!{_cell(rt, self.S_OR, True)}",
            }
        )

        # Confronto con il totale dichiarato (A:J) e riepilogo settimanale (L:Q)
        rs = rt + 2
        self._titolo_sezione(ws, rs, 0, 9, "Confronto con il totale mensile dichiarato")
        self._titolo_sezione(ws, rs, 11, last_c, "Ore settimanali rispetto alle ore da PEI")
        lab2 = f(TXT, LABEL, border=1, border_color=WHITE, font_size=9.5)
        vb = {
            **TXT,
            "bold": True,
            "align": "right",
            "indent": 1,
            "bottom": 1,
            "bottom_color": BORDER_SOFT,
            "shrink": True,
        }
        v_tot = _valore_intestazione(doc, "totale_mensile_dichiarato")
        r0 = rs + 1
        tot_c = _cell(r0, 7)
        somma_c = _cell(r0 + 1, 7)
        diff_c = _cell(r0 + 2, 7)
        tot_val = _numero(v_tot.value) if v_tot.kind == "hours" else None
        diff_val = tot["dich"] - tot_val if tot_val is not None else None
        if tot_val is None:
            esito_tot = "Totale non leggibile" if v_tot.value is not None else "Totale non indicato"
        elif abs(diff_val or 0.0) <= TOLLERANZA:
            esito_tot = "Corrisponde"
        else:
            esito_tot = "Non corrisponde"
        righe_conf: list[tuple[str, Any, dict[str, Any] | None]] = [
            ("Totale ore effettive mensili dichiarato sul foglio", None, None),
            (
                "Somma delle ore giornaliere (colonna «Tot. ore effettive»)",
                _F(_cell(rt, self.S_OD), tot["dich"]),
                {"num_format": FMT_ORE},
            ),
            (
                "Differenza (somma dei giorni − totale dichiarato)",
                _F(f'IF(ISNUMBER({tot_c}),{somma_c}-{tot_c},"")', diff_val),
                {"num_format": FMT_ORE_SEGNO},
            ),
            (
                "Esito del confronto",
                _F(
                    f"IF(ISNUMBER({tot_c}),IF(ABS({diff_c})<={TOLLERANZA},"
                    '"Corrisponde","Non corrisponde"),'
                    f'IF(ISBLANK({tot_c}),"Totale non indicato","Totale non leggibile"))',
                    esito_tot,
                ),
                {},
            ),
            ("Ore programmate", _F(_cell(rt, self.S_OP), tot["prog"]), {"num_format": FMT_ORE}),
            (
                "Ore calcolate dagli orari effettivi",
                _F(_cell(rt, self.S_OC), tot["calc"]),
                {"num_format": FMT_ORE},
            ),
            (
                "Ore riconosciute",
                _F(_cell(rt, self.S_OR), tot["ric"]),
                {"num_format": FMT_ORE, "font_color": NAVY, "font_size": 11, "bg_color": ACCENT_FILL},
            ),
            (
                "Giorni lavorati (con ore riconosciute)",
                _F(f'COUNTIF({_rng(r_a, self.S_OR, r_b, self.S_OR, False)},">0")', tot["giorni"]),
                {"num_format": FMT_INT},
            ),
            ("Assenze alunno", _F(_cell(rt, self.S_AA), tot["ass_al"]), {"num_format": FMT_INT}),
            ("Assenze operatore", _F(_cell(rt, self.S_AO), tot["ass_op"]), {"num_format": FMT_INT}),
        ]
        for k, (etichetta, valore, extra) in enumerate(righe_conf):
            r = r0 + k
            ws.set_row(r, 20)
            self._unisci(ws, r, 0, r, 6, etichetta, lab2)
            if k == 0:
                ws.set_row(r, 30)  # condivisa con l'intestazione su due righe del riepilogo settimanale
                self._scrivi_valore(ws, r, 7, v_tot, {**vb, "font_size": 12}, c1=9)
                continue
            self._unisci(ws, r, 7, r, 9, valore, f(vb, extra or {}))
        d.refs["tot_dich"] = f"{sheet}!{_cell(r0, 7, True)}"
        d.refs["diff"] = f"{sheet}!{_cell(r0 + 2, 7, True)}"
        d.refs["giorni"] = f"{sheet}!{_cell(r0 + 7, 7, True)}"
        # colori dell'esito del confronto
        esito_rng = _rng(r0 + 3, 7, r0 + 3, 7, False)
        ws.conditional_format(
            esito_rng,
            {"type": "text", "criteria": "begins with", "value": "Corrisponde", "format": f(font_color=GREEN)},
        )
        ws.conditional_format(
            esito_rng,
            {"type": "text", "criteria": "begins with", "value": "Non corrisponde", "format": f(font_color=RED)},
        )
        ws.conditional_format(
            esito_rng,
            {
                "type": "text",
                "criteria": "begins with",
                "value": "Totale non",
                "format": f(font_color=AMBER),
            },
        )
        diff_rng = _rng(r0 + 2, 7, r0 + 2, 7, False)
        ws.conditional_format(
            diff_rng,
            {
                "type": "formula",
                "criteria": f"=AND(ISNUMBER({diff_c}),ABS({diff_c})>{TOLLERANZA})",
                "format": f(font_color=RED, bg_color=RED_FILL),
            },
        )

        # Settimane
        sh = rs + 1
        ws_cols = (11, 12, 13, 14, 15, 16)
        for c, testo in zip(
            ws_cols,
            (
                "Settimana",
                "Ore riconosciute",
                "Ore PEI attese",
                "Limite PEI settimanale",
                "Differenza ore − attese",
                "Esito",
            ),
        ):
            self._scrivi(ws, sh, c, testo, f(TXT, HEAD, font_size=9))
        settimane = doc.totals.settimane
        pei_ref = d.refs.get("ore_pei_cell", "")
        pei_val = _numero(h.ore_pei)
        pei_val = pei_val if pei_val is not None and pei_val > 0 else None
        cella = {**TXT, **GRID, "align": "center"}
        if not settimane:
            self._unisci(
                ws,
                sh + 1,
                11,
                sh + 1,
                last_c,
                "Riepilogo settimanale non disponibile: mese o anno di riferimento non indicati.",
                f(TXT, font_color=MUTED, italic=True, indent=1),
            )
        for k, w in enumerate(settimane):
            r = sh + 1 + k
            ws.set_row(r, 20)
            ra, rb = self.S_R_D1 + w.dal - 1, self.S_R_D1 + w.al - 1
            dal, al = d.data(w.dal), d.data(w.al)
            if dal and al:
                etichetta = f"{w.settimana}ª · {dal:%d/%m}" + (f" – {al:%d/%m}" if w.al != w.dal else "")
            else:
                etichetta = f"{w.settimana}ª · giorni {w.dal}–{w.al}"
            self._scrivi(ws, r, 11, etichetta, f(cella, align="left", indent=1))
            ore_val = math.fsum(d.calcs[g - 1].ric or 0.0 for g in range(w.dal, w.al + 1))
            ore_c, att_c, lim_c = _cell(r, 12), _cell(r, 13), _cell(r, 14)
            self._scrivi(
                ws,
                r,
                12,
                _F(f"SUM({_rng(ra, self.S_OR, rb, self.S_OR, False)})", ore_val),
                f(cella, num_format=FMT_ORE, bold=True),
            )
            self._scrivi(ws, r, 13, w.ore_pei, f(cella, num_format=FMT_ORE, font_color=MUTED))
            self._scrivi(
                ws,
                r,
                14,
                _F(f'IF(ISNUMBER({pei_ref}),{pei_ref},"")', pei_val) if pei_ref else pei_val,
                f(cella, num_format=FMT_ORE, font_color=MUTED),
            )
            self._scrivi(
                ws,
                r,
                15,
                _F(
                    f'IF(ISNUMBER({att_c}),{ore_c}-{att_c},"")',
                    ore_val - w.ore_pei if w.ore_pei is not None else None,
                ),
                f(cella, num_format=FMT_ORE_SEGNO),
            )
            if pei_val is None:
                es = "Ore PEI non indicate"
            elif ore_val > pei_val + TOLLERANZA:
                es = "Oltre il limite PEI"
            else:
                es = "Entro il limite PEI"
            self._scrivi(
                ws,
                r,
                16,
                _F(
                    f'IF(NOT(ISNUMBER({lim_c})),"Ore PEI non indicate",IF({ore_c}>{lim_c}+{TOLLERANZA},'
                    f'"Oltre il limite PEI","Entro il limite PEI"))',
                    es,
                ),
                f(cella, align="left", indent=1),
            )
        if settimane:
            r_wa, r_wb = sh + 1, sh + len(settimane)
            col_es = _rng(r_wa, 16, r_wb, 16, False)
            ws.conditional_format(
                col_es,
                {
                    "type": "text",
                    "criteria": "begins with",
                    "value": "Oltre",
                    "format": f(font_color=RED, bg_color=RED_FILL, bold=True),
                },
            )
            ws.conditional_format(
                col_es,
                {"type": "text", "criteria": "begins with", "value": "Entro", "format": f(font_color=GREEN)},
            )
            ws.conditional_format(
                col_es,
                {"type": "text", "criteria": "begins with", "value": "Ore PEI non", "format": f(font_color=AMBER)},
            )
            r = r_wb + 1
            ws.set_row(r, 22)
            tfw = {**TXT, **TOTAL, "align": "center"}
            self._scrivi(ws, r, 11, "Totale mese", f(tfw, align="left", indent=1))
            self._scrivi(
                ws, r, 12, _F(f"SUM({_rng(r_wa, 12, r_wb, 12, False)})", tot["ric"]), f(tfw, num_format=FMT_ORE)
            )
            att_tot = math.fsum(w.ore_pei or 0.0 for w in settimane) if pei_val is not None else None
            self._scrivi(
                ws,
                r,
                13,
                _F(f"SUM({_rng(r_wa, 13, r_wb, 13, False)})", att_tot) if att_tot is not None else None,
                f(tfw, num_format=FMT_ORE),
            )
            self._scrivi(ws, r, 14, None, f(tfw))
            self._scrivi(
                ws,
                r,
                15,
                _F(
                    f'IF(ISNUMBER({_cell(r, 13)}),{_cell(r, 12)}-{_cell(r, 13)},"")',
                    tot["ric"] - att_tot if att_tot is not None else None,
                )
                if att_tot is not None
                else None,
                f(tfw, num_format=FMT_ORE_SEGNO),
            )
            self._scrivi(ws, r, 16, None, f(tfw))
            nota_r = r + 1
            self._unisci(
                ws,
                nota_r,
                11,
                nota_r,
                last_c,
                "Ore PEI attese: limite settimanale proporzionato ai giorni scolastici della settimana "
                "compresi nel mese.",
                f(TXT, font_size=8.5, italic=True, font_color=MUTED, indent=1),
            )

        # Anomalie del documento
        ra0 = max(r0 + len(righe_conf), sh + len(settimane) + 3) + 1
        self._titolo_sezione(ws, ra0, 0, last_c, f"Anomalie e segnalazioni ({len(doc.anomalies)})")
        r = ra0 + 1
        if doc.anomalies:
            for c0, c1, testo in (
                (0, 1, "Gravità"),
                (2, 2, "Giorno"),
                (3, 6, "Controllo"),
                (7, last_c, "Descrizione"),
            ):
                self._unisci(ws, r, c0, r, c1, testo, f(TXT, HEAD, font_size=9))
            ws.set_row(r, 20)
            larghezza_descr = sum(self._S_WIDTHS[7:])
            for a in doc.anomalies:
                r += 1
                colore, sfondo = _GRAVITA_COLORI[a.gravita]
                testo = a.messaggio
                ws.set_row(r, _altezza(testo, larghezza_descr))
                cel = {**TXT, **GRID}
                self._unisci(
                    ws,
                    r,
                    0,
                    r,
                    1,
                    _GRAVITA[a.gravita],
                    f(cel, font_color=colore, bg_color=sfondo, bold=True, align="center"),
                )
                if a.giorno and a.giorno in d.righe_giorno and d.sheet:
                    # il giorno porta alla riga corrispondente della tabella; la cella resta numerica
                    fmt_link = f(cel, align="center", font_color=ACCENT_DARK, underline=1)
                    ws.write_url(
                        r,
                        2,
                        _link(d.sheet, d.righe_giorno[a.giorno], 0),
                        fmt_link,
                        string=str(a.giorno),
                        tip=f"Vai al giorno {a.giorno}",
                    )
                    ws.write_number(r, 2, a.giorno, fmt_link)
                else:
                    self._scrivi(ws, r, 2, a.giorno, f(cel, align="center"))
                self._unisci(ws, r, 3, r, 6, _titolo_codice(a.codice), f(cel, font_size=9, indent=1))
                self._unisci(ws, r, 7, r, last_c, testo, f(cel, text_wrap=True, indent=1))
        else:
            ws.set_row(r, 22)
            self._unisci(
                ws,
                r,
                0,
                r,
                last_c,
                "Nessuna anomalia rilevata: il foglio firma ha superato tutti i controlli.",
                f(TXT, font_color=GREEN, bg_color=GREEN_FILL, bold=True, indent=1),
            )

        # Piede: fonte e motore
        r += 2
        info = [
            f"Fonte: {doc.source_file}"
            + (f", pagina {doc.source_page} di {doc.page_count}" if doc.page_count > 1 else "")
        ]
        motore = _MOTORI.get(doc.engine or "", doc.engine or "non indicato")
        info.append(f"Lettura: {motore}" + (f" – modello {doc.model}" if doc.model else ""))
        if doc.confidence is not None and 0 <= doc.confidence <= 1:
            info.append(f"affidabilità complessiva {round(doc.confidence * 100)}%")
        agg = _dt_locale(doc.updated_at) or _dt_locale(doc.created_at)
        if agg:
            info.append(f"ultimo aggiornamento {agg:%d/%m/%Y %H:%M}")
        info.append(f"rif. documento {doc.id}")
        piede = f(TXT, font_size=8.5, font_color=MUTED, italic=True, indent=1, text_wrap=True, valign="top")
        larghezza = sum(self._S_WIDTHS)
        testo = " · ".join(info)
        ws.set_row(r, _altezza(testo, larghezza, 16, 12))
        self._unisci(ws, r, 0, r, last_c, testo, piede)
        if doc.ocr_notes and _pulisci(doc.ocr_notes):
            r += 1
            testo = f"Note di lettura: {_pulisci(doc.ocr_notes)}"
            ws.set_row(r, _altezza(testo, larghezza, 16, 12))
            self._unisci(ws, r, 0, r, last_c, testo, piede)
        r += 1
        self._unisci(
            ws,
            r,
            0,
            r,
            last_c,
            "Celle evidenziate: rosso = illeggibile · ambra = lettura incerta · blu = corretto a mano · "
            "grigio = sabato, domenica o festivo. Passare con il mouse sulla cella per il dettaglio.",
            piede,
        )
        r += 1
        ws.write_url(
            r,
            0,
            _link(S_RIEPILOGO),
            f(TXT, font_color=ACCENT_DARK, underline=1, font_size=9, indent=1),
            string="‹ Torna al Riepilogo",
        )
        ws.freeze_panes(self.S_R_D1, 0)

    # --------------------------------------------------------- Dettaglio giornaliero

    _D_COLS: tuple[tuple[str, float], ...] = (
        ("Doc.", 6),
        ("Operatore", 24),
        ("Alunno", 26),
        ("Istituto", 25),
        ("Mese", 10),
        ("Data", 11),
        ("Giorno", 7),
        ("Tipo giorno", 9.5),
        ("Entrata progr.", 9.5),
        ("Uscita progr.", 9.5),
        ("Entrata eff.", 9.5),
        ("Uscita eff.", 9.5),
        ("Ore progr.", 9),
        ("Ore calcolate", 9.5),
        ("Ore dichiarate", 9.5),
        ("Ore riconosciute", 10.5),
        ("Differenza", 9.5),
        ("Assenza alunno", 9.5),
        ("Assenza operatore", 9.5),
        ("Firma", 9.5),
        ("Note", 20),
        ("Esito", 38),
        ("Anomalie del giorno", 64),
    )

    def _dettaglio(self) -> None:
        ws, f = self.ws_det, self.f
        nomi = [n for n, _ in self._D_COLS]
        C = {n: i for i, n in enumerate(nomi)}
        larghezze = [w for _, w in self._D_COLS]
        larg = dict(self._D_COLS)
        last_c = len(nomi) - 1
        R_HEAD = 4
        r = R_HEAD + 1

        righe: list[tuple[_Doc, int]] = []
        for d in self.docs:
            for g in range(1, d.n_giorni + 1):
                if self.opt.giorni_vuoti or d.doc.rows[g - 1].has_content():
                    righe.append((d, g))
        r_first = R_HEAD + 1
        r_last = max(r_first, R_HEAD + len(righe))
        L = {n: xl_col_to_name(i) for n, i in C.items()}

        def rng(nome: str) -> str:
            return f"{_q(S_DETTAGLIO)}!{_rng(r_first, C[nome], r_last, C[nome])}"

        self.det = {
            "rng": {
                "doc": rng("Doc."),
                "op": rng("Operatore"),
                "al": rng("Alunno"),
                "ist": rng("Istituto"),
                "mese": rng("Mese"),
                "prog": rng("Ore progr."),
                "calc": rng("Ore calcolate"),
                "dich": rng("Ore dichiarate"),
                "ric": rng("Ore riconosciute"),
                "aa": rng("Assenza alunno"),
                "ao": rng("Assenza operatore"),
            },
            "righe": righe,
        }

        self._imposta_foglio(ws, larghezze, 90, ACCENT, righe_ripetute=(R_HEAD, R_HEAD))
        self._fascia(
            ws,
            last_c,
            "Dettaglio giornaliero",
            f"{self.testo_periodo} · una riga per ogni giornata "
            + ("del mese" if self.opt.giorni_vuoti else "con dati")
            + " · ore in formato decimale (1,50 = un'ora e mezza)",
        )

        # Totali delle righe visibili (sopra l'intestazione: non interferiscono con filtri e ordinamenti)
        r_sub = 3
        ws.set_row(r_sub, 22)
        sub = {**TXT, "bold": True, "bg_color": TOTAL_FILL, "top": 1, "top_color": BORDER}
        self._unisci(
            ws, r_sub, 0, r_sub, C["Uscita eff."], "Totale delle righe visibili", f(sub, align="right", indent=1)
        )
        somme = {
            "Ore progr.": math.fsum(d.calcs[g - 1].prog or 0.0 for d, g in righe),
            "Ore calcolate": math.fsum(d.calcs[g - 1].calc or 0.0 for d, g in righe),
            "Ore dichiarate": math.fsum(d.calcs[g - 1].dich or 0.0 for d, g in righe),
            "Ore riconosciute": math.fsum(d.calcs[g - 1].ric or 0.0 for d, g in righe),
        }
        for nome, valore in somme.items():
            self._scrivi(
                ws,
                r_sub,
                C[nome],
                _F(f"SUBTOTAL(109,{_rng(r_first, C[nome], r_last, C[nome])})", valore),
                f(sub, num_format=FMT_ORE, align="center"),
            )
        for nome in ("Differenza", "Assenza alunno", "Assenza operatore", "Firma", "Note", "Anomalie del giorno"):
            self._scrivi(ws, r_sub, C[nome], None, f(sub))
        self._scrivi(
            ws,
            r_sub,
            C["Esito"],
            _F(
                f'SUBTOTAL(103,{_rng(r_first, C["Doc."], r_last, C["Doc."])})&" giornate visibili"',
                f"{len(righe)} giornate visibili",
            ),
            f(sub, font_color=MUTED, bold=False, indent=1),
        )

        # Intestazione
        ws.set_row(R_HEAD, 34)
        calcolate = {
            "Ore progr.",
            "Ore calcolate",
            "Ore riconosciute",
            "Differenza",
            "Esito",
            "Anomalie del giorno",
        }
        for c, nome in enumerate(nomi):
            self._scrivi(ws, R_HEAD, c, nome, f(TXT, HEAD_CALC if nome in calcolate else HEAD, font_size=9))

        cella = {**TXT, **GRID}
        for d, g in righe:
            doc = d.doc
            row = doc.rows[g - 1]
            calc = d.calcs[g - 1]
            n = r + 1
            tipo = d.tipo(g)
            fill = {"bg_color": WEEKEND} if tipo in ("sabato", "domenica", "festivo") else {}
            base = {**cella, **fill}
            centro = {**base, "align": "center"}
            link = d.link(g)
            if link:
                ws.write_url(
                    r,
                    C["Doc."],
                    link,
                    f(centro, font_color=ACCENT_DARK, underline=1),
                    string=str(d.n),
                    tip=f"Apri la scheda: {d.sheet}, giorno {g}",
                )
                # il collegamento resta, ma il valore della cella torna numerico (filtri e SOMMA.PIÙ.SE)
                ws.write_number(r, C["Doc."], d.n, f(centro, font_color=ACCENT_DARK, underline=1))
            else:
                self._scrivi(ws, r, C["Doc."], d.n, f(centro, font_color=MUTED))
            for nome, campo, testo in (
                ("Operatore", "operatore", d.operatore),
                ("Alunno", "alunno", d.alunno),
                ("Istituto", "istituto", d.istituto),
            ):
                stato = _stato_campo(doc, doc.header, campo, f"header.{campo}")
                self._scrivi(ws, r, C[nome], testo, self._fmt_valore(base, _Val(state=stato)))
            self._scrivi(ws, r, C["Mese"], d.mese_data, f(base, num_format=FMT_MESE_BREVE, align="center"))
            data = d.data(g)
            self._scrivi(ws, r, C["Data"], data, f(centro, num_format=FMT_DATA))
            self._scrivi(
                ws,
                r,
                C["Giorno"],
                calendario.GIORNI_BREVI[data.weekday()] if data else g,
                f(centro, font_color=RED if tipo in ("domenica", "festivo") else MUTED),
            )
            if d.periodo:
                testo_tipo = {
                    "feriale": "Feriale",
                    "sabato": "Sabato",
                    "domenica": "Domenica",
                    "festivo": "Festivo",
                }.get(tipo, "")
                self._scrivi(ws, r, C["Tipo giorno"], testo_tipo, f(centro, font_color=MUTED, font_size=9))
                festa = calendario.nome_festivita(d.periodo[1], d.periodo[0], g)
                if festa:
                    self._commenta(ws, r, C["Tipo giorno"], f"Festività: {festa}.")
            else:
                self._scrivi(ws, r, C["Tipo giorno"], None, f(centro))
            for nome, campo in (
                ("Entrata progr.", "prog_entrata"),
                ("Uscita progr.", "prog_uscita"),
                ("Entrata eff.", "eff_entrata"),
                ("Uscita eff.", "eff_uscita"),
                ("Ore dichiarate", "ore_dichiarate"),
                ("Assenza alunno", "assenza_alunno"),
                ("Assenza operatore", "assenza_operatore"),
                ("Firma", "firma"),
            ):
                self._scrivi_valore(ws, r, C[nome], _valore_giorno(doc, row, campo, "Sì"), centro)
            v_note = _valore_giorno(doc, row, "note", "Sì")
            self._scrivi_valore(ws, r, C["Note"], v_note, {**base, "font_size": 9, "text_wrap": True})
            ore = {**centro, "num_format": FMT_ORE}
            pe, pu, ee, eu = L["Entrata progr."], L["Uscita progr."], L["Entrata eff."], L["Uscita eff."]
            od, oc, ao = L["Ore dichiarate"], L["Ore calcolate"], L["Assenza operatore"]
            self._scrivi(
                ws,
                r,
                C["Ore progr."],
                _F(
                    f'IF(AND(ISNUMBER({pe}{n}),ISNUMBER({pu}{n})),IF({pu}{n}>{pe}{n},({pu}{n}-{pe}{n})*24,""),"")',
                    calc.prog,
                ),
                f(ore, font_color=MUTED),
            )
            self._scrivi(
                ws,
                r,
                C["Ore calcolate"],
                _F(
                    f'IF(AND(ISNUMBER({ee}{n}),ISNUMBER({eu}{n})),IF({eu}{n}>{ee}{n},({eu}{n}-{ee}{n})*24,""),"")',
                    calc.calc,
                ),
                f(ore),
            )
            self._scrivi(
                ws,
                r,
                C["Ore riconosciute"],
                _F(f'IF({ao}{n}="Sì",0,IF(ISNUMBER({od}{n}),{od}{n},IF(ISNUMBER({oc}{n}),{oc}{n},"")))', calc.ric),
                f(ore, bold=True, font_color=NAVY),
            )
            self._scrivi(
                ws,
                r,
                C["Differenza"],
                _F(f'IF(AND(ISNUMBER({od}{n}),ISNUMBER({oc}{n})),{od}{n}-{oc}{n},"")', calc.diff),
                f(centro, num_format=FMT_ORE_SEGNO, font_color=MUTED),
            )
            esito = _testo_esito(val.esito_riga(row, doc.anomalies))
            if not esito:
                festa = calendario.nome_festivita(d.periodo[1], d.periodo[0], g) if d.periodo else None
                esito = festa or ""
            self._scrivi(
                ws, r, C["Esito"], esito, f(base, font_size=9, indent=1, font_color=MUTED, text_wrap=True)
            )
            messaggi = [
                _maiuscola(_RE_PREFISSO_GIORNO.sub("", a.messaggio))
                for a in val.anomalies_for_day(doc.anomalies, g)
            ]
            testo_anomalie = " • ".join(messaggi)
            self._scrivi(
                ws,
                r,
                C["Anomalie del giorno"],
                testo_anomalie,
                f(base, font_size=9, font_color=MUTED, text_wrap=True),
            )
            ws.set_row(
                r,
                max(
                    _altezza(esito, larg["Esito"] * 1.08),
                    _altezza(testo_anomalie, larg["Anomalie del giorno"] * 1.08),
                    _altezza(v_note.value if isinstance(v_note.value, str) else "", larg["Note"] * 1.08),
                ),
            )
            r += 1

        if not righe:
            self._unisci(
                ws,
                r_first,
                0,
                r_first,
                last_c,
                "Nessuna giornata con dati nei fogli firma esportati.",
                f(TXT, font_color=MUTED, italic=True, indent=1),
            )
            return

        # Filtri, riquadri bloccati e formattazione condizionale
        ws.autofilter(R_HEAD, 0, r_last, last_c)
        ws.freeze_panes(R_HEAD + 1, 3)
        ce = C["Esito"]
        for testo, colore, sfondo in (("Errore", RED_DARK, RED_FILL), ("Da verificare", AMBER_TEXT, AMBER_FILL)):
            ws.conditional_format(
                r_first,
                ce,
                r_last,
                ce,
                {
                    "type": "text",
                    "criteria": "begins with",
                    "value": testo,
                    "format": f(font_color=colore, bg_color=sfondo, bold=True),
                },
            )
        ws.conditional_format(
            r_first,
            ce,
            r_last,
            ce,
            {"type": "cell", "criteria": "==", "value": '"OK"', "format": f(font_color=GREEN, bold=True)},
        )
        ws.conditional_format(
            r_first,
            ce,
            r_last,
            ce,
            {"type": "text", "criteria": "begins with", "value": "Assenza", "format": f(font_color=ACCENT_DARK)},
        )
        cd = C["Differenza"]
        first_diff = _cell(r_first, cd)
        first_aa = _cell(r_first, C["Assenza alunno"])
        ws.conditional_format(
            r_first,
            cd,
            r_last,
            cd,
            {
                "type": "formula",
                "criteria": f'=AND(ISNUMBER({first_diff}),ABS({first_diff})>{TOLLERANZA},{first_aa}<>"Sì")',
                "format": f(font_color=RED, bold=True),
            },
        )
        cf = C["Firma"]
        ws.conditional_format(
            r_first,
            cf,
            r_last,
            cf,
            {"type": "cell", "criteria": "==", "value": '"No"', "format": f(font_color=RED, bold=True)},
        )

    # ------------------------------------------------------------------ Anomalie

    _A_COLS: tuple[tuple[str, float], ...] = (
        ("N.", 5.5),
        ("Operatore", 22),
        ("Alunno", 20),
        ("Mese", 10),
        ("Giorno", 7.5),
        ("Data", 11),
        ("Gravità", 11),
        ("Codice", 30),
        ("Controllo", 26),
        ("Campo", 20),
        ("Descrizione", 72),
        ("Valore letto", 16),
        ("Valore atteso", 16),
    )

    def _anomalie(self) -> None:
        ws, f = self.ws_anom, self.f
        nomi = [n for n, _ in self._A_COLS]
        C = {n: i for i, n in enumerate(nomi)}
        larghezze = [w for _, w in self._A_COLS]
        last_c = len(nomi) - 1
        R_HEAD = 4
        tutte = [(d, a) for d in self.docs for a in d.doc.anomalies]
        conta = {g: sum(1 for _, a in tutte if a.gravita == g) for g in ("errore", "attenzione", "info")}
        self._imposta_foglio(ws, larghezze, 90, RED, righe_ripetute=(R_HEAD, R_HEAD))
        self._fascia(
            ws,
            last_c,
            "Anomalie e segnalazioni",
            f"{self.testo_periodo} · {_plurale(len(tutte), 'segnalazione', 'segnalazioni')}: "
            f"{_plurale(conta['errore'], 'errore', 'errori')}, "
            f"{_plurale(conta['attenzione'], 'attenzione', 'attenzioni')}, "
            f"{_plurale(conta['info'], 'informazione', 'informazioni')}",
            "Fare clic sul nome dell'operatore per aprire la scheda del foglio firma al giorno indicato. "
            "Usare i filtri dell'intestazione per selezionare gravità o codice (es. E05 = campi illeggibili).",
        )
        ws.set_row(R_HEAD, 30)
        for c, nome in enumerate(nomi):
            self._scrivi(ws, R_HEAD, c, nome, f(TXT, HEAD, font_size=9))
        r = R_HEAD + 1
        top = {**TXT, **GRID}
        larg = dict(self._A_COLS)
        for k, (d, a) in enumerate(tutte, start=1):
            controllo = _maiuscola(val.CODICI.get(a.codice, {}).get("titolo", a.codice))
            campo = val.etichetta_campo(a.campo) if a.campo else ""
            ws.set_row(
                r,
                max(
                    _altezza(a.messaggio, larg["Descrizione"]),
                    _altezza(controllo, larg["Controllo"]),
                    _altezza(campo, larg["Campo"] * 1.1),
                    _altezza(a.valore_letto or "", larg["Valore letto"]),
                    _altezza(a.valore_atteso or "", larg["Valore atteso"]),
                ),
            )
            self._scrivi(ws, r, C["N."], k, f(top, align="center", font_color=MUTED))
            link = d.link(a.giorno)
            if link:
                ws.write_url(
                    r,
                    C["Operatore"],
                    link,
                    f(top, font_color=ACCENT_DARK, underline=1),
                    string=d.operatore,
                    tip="Apri la scheda del foglio firma",
                )
            else:
                self._scrivi(ws, r, C["Operatore"], d.operatore, f(top))
            self._scrivi(ws, r, C["Alunno"], d.alunno, f(top))
            self._scrivi(ws, r, C["Mese"], d.mese_data, f(top, num_format=FMT_MESE_BREVE, align="center"))
            self._scrivi(ws, r, C["Giorno"], a.giorno, f(top, align="center"))
            data = d.data(a.giorno) if a.giorno else None
            self._scrivi(ws, r, C["Data"], data, f(top, align="center", num_format=FMT_DATA))
            self._scrivi(ws, r, C["Gravità"], _GRAVITA[a.gravita], f(top, align="center", bold=True))
            self._scrivi(ws, r, C["Codice"], a.codice, f(top, font_size=9, font_color=MUTED))
            self._scrivi(ws, r, C["Controllo"], controllo, f(top, text_wrap=True))
            self._scrivi(ws, r, C["Campo"], campo, f(top, font_color=MUTED, text_wrap=True, font_size=9))
            self._scrivi(ws, r, C["Descrizione"], a.messaggio, f(top, text_wrap=True))
            for nome, valore in (("Valore letto", a.valore_letto), ("Valore atteso", a.valore_atteso)):
                # numeri e orari singoli come valori Excel; intervalli e testi restano testo
                numero = _numero_testo(valore)
                ora = _ora_excel(valore) if valore and _RE_ORARIO.match(valore.strip()) else None
                if numero is not None:
                    self._scrivi(ws, r, C[nome], numero, f(top, align="center"))
                elif ora is not None:
                    self._scrivi(ws, r, C[nome], ora, f(top, align="center", num_format=FMT_ORA))
                else:
                    self._scrivi(ws, r, C[nome], valore or "", f(top, align="center", text_wrap=True))
            r += 1
        if not tutte:
            self._unisci(
                ws,
                R_HEAD + 1,
                0,
                R_HEAD + 1,
                last_c,
                "Nessuna anomalia: tutti i fogli firma esportati hanno superato i controlli.",
                f(TXT, font_color=GREEN, bg_color=GREEN_FILL, bold=True, indent=1),
            )
            return
        r_first, r_last = R_HEAD + 1, r - 1
        ws.autofilter(R_HEAD, 0, r_last, last_c)
        ws.freeze_panes(R_HEAD + 1, 2)
        cg = C["Gravità"]
        for g, (colore, sfondo) in _GRAVITA_COLORI.items():
            ws.conditional_format(
                r_first,
                cg,
                r_last,
                cg,
                {
                    "type": "cell",
                    "criteria": "==",
                    "value": f'"{_GRAVITA[g]}"',
                    "format": f(font_color=colore, bg_color=sfondo, bold=True),
                },
            )

    # ------------------------------------------------------------ Totali per operatore

    def _totali(self) -> None:
        ws, f = self.ws_tot, self.f
        det = self.det["rng"]
        righe: list[tuple[_Doc, int]] = self.det["righe"]
        mesi = sorted({d.periodo for d in self.docs if d.periodo}, key=lambda p: (p[1], p[0]))
        colonne_mese: list[tuple[str, tuple[int, int] | None]] = [(_etichetta_mese(m, a), (m, a)) for m, a in mesi]
        if any(d.periodo is None for d in self.docs):
            colonne_mese.append(("Mese non indicato", None))
        # Con un solo mese la colonna del mese coinciderebbe con il totale: non si ripete.
        if len(colonne_mese) < 2:
            colonne_mese = []
        n_mesi = len(colonne_mese)
        larghezze = [32, 8] + [12] * n_mesi + [12.5, 12, 11, 10, 10, 10, 34]
        last_c = len(larghezze) - 1
        self._imposta_foglio(ws, larghezze, 90, ACCENT)
        self._fascia(
            ws,
            last_c,
            "Totali per operatore, alunno e istituto",
            f"{self.testo_periodo} · ore e giornate calcolate con formule (SOMMA.PIÙ.SE, CONTA.PIÙ.SE) "
            f"sul foglio «{S_DETTAGLIO}»",
        )
        cella = {**TXT, **GRID}
        tf = {**TXT, **TOTAL}
        riep_first = self.R_RIEP_HEAD + 1
        riep_last = riep_first + len(self.docs) - 1
        sezioni = (
            ("Ore per operatore", "operatore", "Operatore", "op", 1, "Alunni seguiti"),
            ("Ore per alunno", "alunno", "Alunno", "al", 2, "Operatori"),
            ("Ore per istituto", "istituto", "Istituto", "ist", 3, "Alunni"),
        )
        r = 4
        for titolo, attr, intestazione, chiave_rng, col_riep, etichetta_altri in sezioni:
            gruppi: dict[str, tuple[str, list[_Doc]]] = {}
            for d in self.docs:
                nome = getattr(d, attr)
                gruppi.setdefault(nome.casefold(), (nome, []))[1].append(d)
            elenco = sorted(gruppi.values(), key=lambda x: (x[0].startswith("("), x[0].casefold()))

            self._titolo_sezione(ws, r, 0, last_c, titolo)
            r += 1
            intestazioni = [intestazione, "Fogli"] + [f"Ore {nome.lower()}" for nome, _ in colonne_mese]
            intestazioni += [
                "Totale ore riconosciute" if n_mesi else "Ore riconosciute",
                "Ore programmate",
                "% ore su programmate",
                "Giorni lavorati",
                "Assenze alunno",
                "Assenze operatore",
                etichetta_altri,
            ]
            c_ric = 2 + n_mesi
            c_prog, c_pct, c_gg, c_aa, c_ao, c_altri = range(c_ric + 1, c_ric + 7)
            ws.set_row(r, 32)
            for c, testo in enumerate(intestazioni):
                calcolata = 2 <= c <= c_pct
                self._scrivi(ws, r, c, testo, f(TXT, HEAD_CALC if calcolata else HEAD, font_size=9))
            r += 1
            r_a = r
            somme: dict[int, float] = {}
            for nome, ds in elenco:
                ws.set_row(r, 20)
                ids = {id(d) for d in ds}
                proprie = [(d, g) for d, g in righe if id(d) in ids]
                crit = f"$A{r + 1}"
                valori: dict[int, float | None] = {
                    1: float(len(ds)),
                    c_ric: math.fsum(d.calcs[g - 1].ric or 0.0 for d, g in proprie),
                    c_prog: math.fsum(d.calcs[g - 1].prog or 0.0 for d, g in proprie),
                    c_gg: float(sum(1 for d, g in proprie if (d.calcs[g - 1].ric or 0.0) > 0)),
                    c_aa: float(sum(1 for d, g in proprie if d.doc.rows[g - 1].assenza_alunno)),
                    c_ao: float(sum(1 for d, g in proprie if d.doc.rows[g - 1].assenza_operatore)),
                }
                for k, (_, mese) in enumerate(colonne_mese):
                    valori[2 + k] = math.fsum(d.calcs[g - 1].ric or 0.0 for d, g in proprie if d.periodo == mese)
                prog = valori[c_prog] or 0.0
                valori[c_pct] = (valori[c_ric] or 0.0) / prog if prog > 0 else None
                for c, v in valori.items():
                    if c != c_pct and v is not None:
                        somme[c] = somme.get(c, 0.0) + v

                rng_crit = det[chiave_rng]
                riep = f"{_q(S_RIEPILOGO)}!{_rng(riep_first, col_riep, riep_last, col_riep)}"
                self._scrivi(ws, r, 0, nome, f(cella, bold=True, indent=1))
                self._scrivi(
                    ws,
                    r,
                    1,
                    _F(f"COUNTIF({riep},{crit})", valori[1]),
                    f(cella, align="center", num_format=FMT_INT),
                )
                for k, (_, mese) in enumerate(colonne_mese):
                    cond_mese = '""' if mese is None else f"DATE({mese[1]},{mese[0]},1)"
                    self._scrivi(
                        ws,
                        r,
                        2 + k,
                        _F(f"SUMIFS({det['ric']},{rng_crit},{crit},{det['mese']},{cond_mese})", valori[2 + k]),
                        f(cella, num_format=FMT_ORE),
                    )
                self._scrivi(
                    ws,
                    r,
                    c_ric,
                    _F(f"SUMIFS({det['ric']},{rng_crit},{crit})", valori[c_ric]),
                    f(cella, num_format=FMT_ORE, bold=True, font_color=NAVY),
                )
                self._scrivi(
                    ws,
                    r,
                    c_prog,
                    _F(f"SUMIFS({det['prog']},{rng_crit},{crit})", valori[c_prog]),
                    f(cella, num_format=FMT_ORE),
                )
                self._scrivi(
                    ws,
                    r,
                    c_pct,
                    _F(f'IF({_cell(r, c_prog)}>0,{_cell(r, c_ric)}/{_cell(r, c_prog)},"")', valori[c_pct]),
                    f(cella, num_format=FMT_PCT, align="center"),
                )
                for c, formula in (
                    (c_gg, f'COUNTIFS({rng_crit},{crit},{det["ric"]},">0")'),
                    (c_aa, f'COUNTIFS({rng_crit},{crit},{det["aa"]},"Sì")'),
                    (c_ao, f'COUNTIFS({rng_crit},{crit},{det["ao"]},"Sì")'),
                ):
                    self._scrivi(ws, r, c, _F(formula, valori[c]), f(cella, align="center", num_format=FMT_INT))
                altri_attr = "operatore" if attr == "alunno" else "alunno"
                altri = sorted({getattr(d, altri_attr) for d in ds}, key=str.casefold)
                self._scrivi(ws, r, c_altri, ", ".join(altri), f(cella, font_size=9, font_color=MUTED, indent=1))
                r += 1
            r_b = r - 1

            # Riga del totale della sezione
            ws.set_row(r, 22)
            self._scrivi(ws, r, 0, "Totale", f(tf, indent=1))
            for c in range(1, c_altri + 1):
                if c == c_pct:
                    prog, ric = somme.get(c_prog, 0.0), somme.get(c_ric, 0.0)
                    self._scrivi(
                        ws,
                        r,
                        c,
                        _F(
                            f'IF({_cell(r, c_prog)}>0,{_cell(r, c_ric)}/{_cell(r, c_prog)},"")',
                            ric / prog if prog > 0 else None,
                        ),
                        f(tf, num_format=FMT_PCT, align="center"),
                    )
                elif c == c_altri:
                    self._scrivi(ws, r, c, None, f(tf))
                else:
                    intero = c in (1, c_gg, c_aa, c_ao)
                    self._scrivi(
                        ws,
                        r,
                        c,
                        _F(f"SUM({_rng(r_a, c, r_b, c, False)})", somme.get(c, 0.0)),
                        f(tf, num_format=FMT_INT if intero else FMT_ORE, align="center" if intero else "right"),
                    )
            r += 2

        ws.set_row(r, 30)
        self._unisci(
            ws,
            r,
            0,
            r,
            last_c,
            f"Le ore e le giornate sono sommate dal foglio «{S_DETTAGLIO}»: correggendo un valore nel dettaglio "
            "questi totali si aggiornano automaticamente. "
            "% ore su programmate = ore riconosciute ÷ ore programmate.",
            f(TXT, font_size=9, italic=True, font_color=MUTED, indent=1, text_wrap=True),
        )
        ws.freeze_panes(4, 1)

    # -------------------------------------------------------------- Legenda e note

    def _legenda(self) -> None:
        ws, f = self.ws_leg, self.f
        larghezze = [3, 18, 16, 16, 16, 16, 16, 16, 16]
        last_c = len(larghezze) - 1
        self._imposta_foglio(ws, larghezze, 100, SUBTLE, verticale=False)
        self._fascia(ws, last_c, "Legenda e note", f"{self.titolo} · {self.testo_periodo} · {self.generato}")
        r = 4
        testo_fmt = f(TXT, text_wrap=True, valign="vcenter", indent=1)
        larghezza_testo = sum(larghezze[2:])

        def riga_testo(testo: str, campione: tuple[Any, Format] | None = None) -> None:
            nonlocal r
            ws.set_row(r, _altezza(testo, larghezza_testo, 22))
            if campione is not None:
                self._scrivi(ws, r, 1, campione[0], campione[1])
            self._unisci(ws, r, 2, r, last_c, testo, testo_fmt)
            r += 1

        # Colori
        self._titolo_sezione(ws, r, 1, last_c, "Significato dei colori")
        r += 1
        cb = {**TXT, "border": 1, "border_color": BORDER, "align": "center"}
        campioni: list[tuple[Any, Format, str]] = [
            (
                ILLEGGIBILE,
                f(cb, OV_MARCATORE),
                (
                    "Campo illeggibile: scritto sul foglio ma non leggibile. Il valore va letto sulla scansione "
                    "originale e inserito a mano; il commento della cella (triangolino rosso) spiega il problema. "
                    "Se il campo incide sulle ore è segnalato come errore E05."
                ),
            ),
            (
                8 / 24,
                f(cb, OV_INCERTO, num_format=FMT_ORA),
                "Lettura incerta: il valore è stato letto ma va verificato sull'originale.",
            ),
            (
                8.5 / 24,
                f(cb, OV_CORRETTO, num_format=FMT_ORA),
                (
                    "Corretto manualmente durante la revisione: il commento riporta il valore letto "
                    "originariamente dall'OCR."
                ),
            ),
            (
                "sab",
                f(cb, bg_color=WEEKEND, font_color=MUTED),
                "Sabato, domenica o giorno festivo (festività nazionali e San Gennaro, patrono di Napoli).",
            ),
            (
                31,
                f(cb, bg_color=INESISTENTE, font_color=MUTED),
                "Giorno inesistente nel mese di riferimento (es. 30 febbraio): escluso dai totali.",
            ),
            (
                "–",
                f(cb, font_color=SUBTLE),
                (
                    "Trattino scritto negli orari effettivi: prestazione non svolta in orario "
                    "(ad es. per assenza dell'alunno)."
                ),
            ),
            (
                "OK",
                f(cb, font_color=GREEN, bg_color=GREEN_FILL, bold=True),
                "Stato OK: nessun rilievo sul foglio firma.",
            ),
            (
                "Da verificare",
                f(cb, font_color=AMBER, bg_color=AMBER_FILL, bold=True),
                "Stato da verificare: segnalazioni di attenzione o campi incerti/illeggibili.",
            ),
            (
                "Errori",
                f(cb, font_color=RED, bg_color=RED_FILL, bold=True),
                "Stato errori: almeno un controllo non superato (es. ore non coerenti, totale diverso).",
            ),
            (
                "Intestazione",
                f(cb, HEAD, font_size=9),
                "Colonne che riportano i dati scritti sul modulo cartaceo.",
            ),
            ("Intestazione", f(cb, HEAD_CALC, font_size=9), "Colonne calcolate da Sirio OCR con formule Excel."),
        ]
        for campione, fmt, spiegazione in campioni:
            riga_testo(spiegazione, (campione, fmt))
        r += 1

        # Regole di calcolo
        self._titolo_sezione(ws, r, 1, last_c, "Regole di calcolo")
        r += 1
        tolleranza = val.format_ore(TOLLERANZA)
        regole = [
            (
                "Ore programmate e ore calcolate = (uscita − entrata) × 24, dagli orari programmati ed effettivi; "
                "gli orari sono veri valori orari di Excel e le ore sono in formato decimale (1,50 = un'ora e mezza)."
            ),
            (
                "Ore riconosciute = ore scritte nella colonna «Tot. ore effettive»; se non scritte, ore calcolate "
                "dall'orario effettivo; zero in caso di assenza dell'operatore. Con l'assenza dell'alunno sono "
                "ammesse ore parziali (es. 1,5 su 3)."
            ),
            (
                f"Differenza giornaliera = ore dichiarate − ore calcolate. Differenze oltre {tolleranza} ore sono "
                "segnalate come errore E01, salvo le ore parziali riconosciute per assenza dell'alunno."
            ),
            (
                "Differenza totale = somma delle ore giornaliere dichiarate − «Totale ore effettive mensili» "
                "scritto sul foglio (errore E02 se diversa da zero)."
            ),
            (
                "Settimane da lunedì a domenica, troncate al mese. Le ore PEI attese di ogni settimana sono "
                "proporzionate ai giorni scolastici compresi (lunedì–venerdì non festivi, più il sabato se nel "
                "foglio risultano attività di sabato); il superamento del limite settimanale del PEI è segnalato "
                "con W06."
            ),
            "Giorni lavorati = giornate con ore riconosciute maggiori di zero.",
            "Stato, esiti e anomalie riflettono i controlli eseguiti da Sirio OCR al momento della generazione del file.",
        ]
        if self.opt.fogli_per_documento:
            regole.append(
                "Le colonne calcolate e i totali sono formule: correggendo un valore nella scheda di un foglio "
                "firma si aggiornano i totali della scheda e del Riepilogo; i «Totali per operatore» si basano "
                "sul «Dettaglio giornaliero»."
            )
        else:
            regole.append(
                "Le colonne calcolate e i totali sono formule basate sul «Dettaglio giornaliero»: correggendo un "
                "valore nel dettaglio si aggiornano il Riepilogo e i Totali per operatore."
            )
        for testo in regole:
            riga_testo(testo, ("•", f(TXT, align="right", font_color=ACCENT, bold=True)))
        r += 1

        # Codici dei controlli
        self._titolo_sezione(ws, r, 1, last_c, "Controlli eseguiti")
        r += 1
        ws.set_row(r, 22)
        self._scrivi(ws, r, 1, "Codice", f(TXT, HEAD, font_size=9))
        self._unisci(ws, r, 2, r, 3, "Controllo", f(TXT, HEAD, font_size=9))
        self._scrivi(ws, r, 4, "Gravità", f(TXT, HEAD, font_size=9))
        self._unisci(ws, r, 5, r, last_c, "Descrizione", f(TXT, HEAD, font_size=9))
        r += 1
        cella = {**TXT, **GRID}
        for codice, info in val.CODICI.items():
            descr = info["descrizione"]
            ws.set_row(r, _altezza(descr, sum(larghezze[5:]), 20))
            colore, sfondo = _GRAVITA_COLORI.get(info["gravita"], (INK, PANEL))
            self._scrivi(ws, r, 1, codice[:3], f(cella, bold=True, align="center"))
            self._unisci(ws, r, 2, r, 3, _maiuscola(info["titolo"]), f(cella, indent=1))
            gravita = _GRAVITA.get(info["gravita"], info["gravita"])
            if codice in (val.E05_CAMPO_ILLEGGIBILE, val.W03_GIORNO_FESTIVO):
                gravita += " *"
            self._scrivi(
                ws, r, 4, gravita, f(cella, font_color=colore, bg_color=sfondo, bold=True, align="center")
            )
            self._unisci(ws, r, 5, r, last_c, descr, f(cella, text_wrap=True, indent=1))
            r += 1
        self._unisci(
            ws,
            r,
            1,
            r,
            last_c,
            "* La gravità dipende dal caso, come indicato nella descrizione.",
            f(TXT, font_size=8.5, italic=True, font_color=MUTED),
        )
        r += 2

        # Motori OCR e informazioni sul file
        self._titolo_sezione(ws, r, 1, last_c, "Lettura OCR e generazione del file")
        r += 1
        motori: dict[tuple[str, str], list[_Doc]] = {}
        for d in self.docs:
            motori.setdefault((d.doc.engine or "", d.doc.model or ""), []).append(d)
        info_righe: list[tuple[str, str | int]] = []
        for (motore, modello), ds in sorted(motori.items()):
            nome = _MOTORI.get(motore, motore or "Non indicato")
            if modello:
                nome += f" – modello {modello}"
            conf = [x.doc.confidence for x in ds if x.doc.confidence is not None and 0 <= x.doc.confidence <= 1]
            dettaglio = _plurale(len(ds), "foglio firma", "fogli firma")
            if conf:
                dettaglio += f" · affidabilità media {round(sum(conf) / len(conf) * 100)}%"
            info_righe.append((nome, dettaglio))
        info_righe += [
            ("Data e ora di generazione", f"{self.adesso:%d/%m/%Y %H:%M}"),
            ("Programma", f"Sirio OCR {__version__}"),
            ("Periodo di riferimento", self.testo_periodo),
            ("Fogli firma inclusi", len(self.docs)),
            ("Giorni senza dati nel dettaglio", "inclusi" if self.opt.giorni_vuoti else "esclusi"),
        ]
        for etichetta, valore in info_righe:
            ws.set_row(r, 20)
            self._unisci(ws, r, 1, r, 3, etichetta, f(TXT, LABEL, border=1, border_color=WHITE))
            self._unisci(
                ws, r, 4, r, last_c, valore, f(TXT, bold=True, indent=1, bottom=1, bottom_color=BORDER_SOFT)
            )
            r += 1
        r += 1

        # Documenti non inclusi
        self._titolo_sezione(ws, r, 1, last_c, f"Documenti non inclusi ({len(self.esclusi)})")
        r += 1
        if not self.esclusi:
            ws.set_row(r, 20)
            self._unisci(
                ws,
                r,
                1,
                r,
                last_c,
                "Tutti i documenti selezionati sono stati inclusi nella rendicontazione.",
                f(TXT, font_color=MUTED, italic=True, indent=1),
            )
            r += 1
        else:
            ws.set_row(r, 22)
            self._unisci(ws, r, 1, r, 3, "File di origine", f(TXT, HEAD, font_size=9))
            self._scrivi(ws, r, 4, "Pagina", f(TXT, HEAD, font_size=9))
            self._unisci(ws, r, 5, r, last_c, "Motivo dell'esclusione", f(TXT, HEAD, font_size=9))
            r += 1
            for doc, motivo in self.esclusi:
                ws.set_row(r, _altezza(motivo, sum(larghezze[5:]), 20))
                self._unisci(ws, r, 1, r, 3, doc.source_file, f(cella, indent=1))
                self._scrivi(ws, r, 4, doc.source_page, f(cella, align="center"))
                self._unisci(ws, r, 5, r, last_c, motivo, f(cella, text_wrap=True, indent=1))
                r += 1
        r += 1
        ws.write_url(
            r,
            1,
            _link(S_RIEPILOGO),
            f(TXT, font_color=ACCENT_DARK, underline=1, font_size=9),
            string="‹ Torna al Riepilogo",
        )


def _testo_esito(esito: str) -> str:
    """ "Errore: illeggibile: …" -> "Errore – illeggibile: …" (più leggibile in tabella)."""
    for prefisso in ("Errore: ", "Da verificare: "):
        if esito.startswith(prefisso):
            return f"{prefisso[:-2]} – {esito[len(prefisso) :]}"
    return esito


def _titolo_codice(codice: str) -> str:
    titolo = val.CODICI.get(codice, {}).get("titolo", codice)
    return f"{codice[:3]} · {_maiuscola(titolo)}"


# --- API pubblica ---------------------------------------------------------------


def _adesso() -> datetime:
    """Ora locale (senza fuso) usata per il nome del file e le date di generazione."""
    return datetime.now().astimezone().replace(tzinfo=None)


def _includibile(doc: Document) -> bool:
    return doc.status == "completato" and doc.is_foglio_firma


def _motivo_esclusione(doc: Document) -> str:
    if doc.status == "completato" and not doc.is_foglio_firma:
        return "Non riconosciuto come foglio firma"
    motivo = _ESCLUSIONE.get(doc.status, f"Stato «{doc.status}»")
    if doc.status == "errore" and doc.error:
        motivo += f": {_pulisci(doc.error)}"
    return motivo


def default_filename(docs: list[Document]) -> str:
    """Nome predefinito del file, es. "Rendicontazione_2026-02_20261009-1530.xlsx"."""
    candidati = [d for d in docs if _includibile(d)] or list(docs)
    periodi = sorted({p for p in (_periodo(d.header) for d in candidati) if p}, key=lambda p: (p[1], p[0]))
    parti = ["Rendicontazione"]
    if periodi:
        (m0, a0), (m1, a1) = periodi[0], periodi[-1]
        parti.append(f"{a0}-{m0:02d}" if (m0, a0) == (m1, a1) else f"{a0}-{m0:02d}_{a1}-{m1:02d}")
    parti.append(_adesso().strftime("%Y%m%d-%H%M"))
    return "_".join(parti) + ".xlsx"


def export_workbook(docs: list[Document], path: Path, options: ExportOptions | None = None) -> Path:
    """Genera il file Excel della rendicontazione e ne restituisce il percorso.

    I documenti sono rivalidati su copie; quelli non completati o non
    riconosciuti come fogli firma sono esclusi ed elencati nel foglio
    "Legenda e note". Solleva ``ValueError`` se non c'è nulla da esportare e
    ``PermissionError``/``OSError`` (con messaggio in italiano) se il file non
    può essere scritto.
    """
    options = options or ExportOptions()
    path = Path(path)
    if path.suffix.lower() != ".xlsx":
        path = path.with_name(path.name + ".xlsx")
    if not docs:
        raise ValueError("Nessun documento da esportare.")

    inclusi: list[Document] = []
    esclusi: list[tuple[Document, str]] = []
    for doc in docs:
        if _includibile(doc):
            copia = doc.model_copy(deep=True)
            val.validate_document(copia)
            inclusi.append(copia)
        else:
            esclusi.append((doc, _motivo_esclusione(doc)))
    if not inclusi:
        raise ValueError(
            "Nessun foglio firma completato da esportare: attendere la fine dell'elaborazione "
            "oppure selezionare documenti riconosciuti come fogli firma."
        )

    prepared = [_prepara(doc, i) for i, doc in enumerate(_ordina_documenti(inclusi), start=1)]
    adesso = _adesso()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Impossibile creare la cartella «{path.parent}»: {exc.strerror or exc}") from exc
    # File temporaneo nella stessa cartella (sostituzione atomica), creato da XlsxWriter con i permessi
    # predefiniti dell'utente (mkstemp lo renderebbe leggibile solo dal proprietario).
    tmp = path.parent / f".{path.stem}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        wb = xlsxwriter.Workbook(
            str(tmp),
            {
                "default_date_format": FMT_DATA,
                "strings_to_numbers": False,
                "strings_to_formulas": False,
                "strings_to_urls": False,
                "default_format_properties": {"font_name": FONT, "font_size": 10},
            },
        )
        enti = sorted({_pulisci(d.doc.header.ente) for d in prepared if _pulisci(d.doc.header.ente)})
        titolo = _pulisci(options.titolo) or TITOLO_PREDEFINITO
        wb.set_properties(
            {
                "title": f"{titolo} – {_testo_periodi([d.periodo for d in prepared if d.periodo])}",
                "subject": f"{SERVIZIO} – Comune di Napoli",
                "author": "Sirio OCR",
                "company": enti[0] if len(enti) == 1 else "",
                "keywords": "fogli firma, rendicontazione, assistenza specialistica",
                "comments": f"Generato da Sirio OCR {__version__} il {adesso:%d/%m/%Y %H:%M}",
                "created": datetime.now(timezone.utc),
            }
        )
        _Esportatore(wb, prepared, esclusi, options, adesso).build()
        try:
            wb.close()
        except XlsxWriterException as exc:
            raise OSError(f"Impossibile scrivere il file Excel: {exc}") from exc
        try:
            os.replace(tmp, path)
        except PermissionError as exc:
            raise PermissionError(
                f"Impossibile salvare «{path.name}»: il file è aperto in un altro programma (ad esempio Excel). "
                "Chiuderlo e riprovare."
            ) from exc
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                log.warning("File temporaneo non rimosso: %s", tmp)
    log.info("Excel generato: %s (%d fogli firma, %d esclusi)", path, len(prepared), len(esclusi))
    return path
