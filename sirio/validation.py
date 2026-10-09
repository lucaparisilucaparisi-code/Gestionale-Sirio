"""Parsing di orari e ore, controlli di coerenza e totali dei fogli firma.

Funzioni pure e deterministiche (nessun I/O): ricevono il modello dati di
``sirio.models`` e restituiscono anomalie (``Anomaly``) e totali (``Totals``).

Regole principali
-----------------
* Ore riconosciute per riga: le ore dichiarate se presenti, altrimenti quelle
  calcolate dall'orario effettivo; 0 in caso di assenza dell'operatore.
* Il totale mensile dichiarato si confronta con la somma della colonna
  "Tot. ore effettive" (``Totals.differenza_totale = ore_dichiarate - totale``).
* Settimane lunedì-domenica troncate al mese; le ore PEI attese di ogni
  settimana sono proporzionate ai giorni scolastici (lunedì-venerdì non
  festivi, più il sabato se nel foglio risultano attività di sabato) compresi
  nella settimana. ``W06`` confronta invece le ore della settimana con il
  valore settimanale pieno del PEI.
* Senza mese/anno di riferimento validi si saltano i controlli di calendario
  (giorni inesistenti, festivi, settimane) ma si eseguono tutti gli altri.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from sirio import calendario
from sirio.models import (
    DAY_FIELDS,
    HEADER_FIELDS,
    Anomaly,
    DayRow,
    Document,
    Header,
    Severity,
    Totals,
    WeekTotal,
)

TOLLERANZA = 0.01
MAX_ORE_GIORNO = 24.0

# --- Codici anomalia ----------------------------------------------------------

E01_ORE_NON_COERENTI = "E01_ORE_NON_COERENTI"
E02_TOTALE_MENSILE_DIVERSO = "E02_TOTALE_MENSILE_DIVERSO"
E03_ORARIO_NON_VALIDO = "E03_ORARIO_NON_VALIDO"
E04_GIORNO_INESISTENTE = "E04_GIORNO_INESISTENTE"
E05_CAMPO_ILLEGGIBILE = "E05_CAMPO_ILLEGGIBILE"
E06_NON_FOGLIO_FIRMA = "E06_NON_FOGLIO_FIRMA"
W01_FIRMA_MANCANTE = "W01_FIRMA_MANCANTE"
W02_ORE_SENZA_ORARIO = "W02_ORE_SENZA_ORARIO"
W03_GIORNO_FESTIVO = "W03_GIORNO_FESTIVO"
W04_CAMPO_INCERTO = "W04_CAMPO_INCERTO"
W05_ASSENZA_OPERATORE_CON_ORE = "W05_ASSENZA_OPERATORE_CON_ORE"
W06_ORE_PEI_SUPERATE = "W06_ORE_PEI_SUPERATE"
W07_INTESTAZIONE_INCOMPLETA = "W07_INTESTAZIONE_INCOMPLETA"
W08_FIRMA_COORDINATORE_MANCANTE = "W08_FIRMA_COORDINATORE_MANCANTE"
W09_TIMBRO_REFERENTE_MANCANTE = "W09_TIMBRO_REFERENTE_MANCANTE"
W10_TOTALE_MENSILE_ASSENTE = "W10_TOTALE_MENSILE_ASSENTE"
W11_DOPPIA_ASSENZA = "W11_DOPPIA_ASSENZA"
W12_ORE_NON_INDICATE = "W12_ORE_NON_INDICATE"
I01_ASSENZA_ALUNNO = "I01_ASSENZA_ALUNNO"
I02_ASSENZA_OPERATORE = "I02_ASSENZA_OPERATORE"
I03_ORARIO_DIVERSO = "I03_ORARIO_DIVERSO"
I04_ORE_SENZA_PROGRAMMATO = "I04_ORE_SENZA_PROGRAMMATO"
I05_NON_SVOLTO = "I05_NON_SVOLTO"

# Descrizione dei codici (per legende di interfaccia ed Excel).
# "gravita" e' quella predefinita; E05 e W03 la adattano al caso.
CODICI: dict[str, dict[str, str]] = {
    E01_ORE_NON_COERENTI: {
        "gravita": "errore",
        "titolo": "ore non coerenti con l'orario",
        "descrizione": "Le ore dichiarate non corrispondono alla differenza tra uscita ed entrata effettive.",
    },
    E02_TOTALE_MENSILE_DIVERSO: {
        "gravita": "errore",
        "titolo": "totale mensile diverso dalla somma",
        "descrizione": "Il totale mensile dichiarato non corrisponde alla somma delle ore giornaliere.",
    },
    E03_ORARIO_NON_VALIDO: {
        "gravita": "errore",
        "titolo": "orario non valido",
        "descrizione": "Orario non interpretabile o incompleto, oppure uscita non successiva all'entrata.",
    },
    E04_GIORNO_INESISTENTE: {
        "gravita": "errore",
        "titolo": "giorno inesistente nel mese",
        "descrizione": "Dati scritti in un giorno che non esiste nel mese di riferimento (es. 30 febbraio).",
    },
    E05_CAMPO_ILLEGGIBILE: {
        "gravita": "errore",
        "titolo": "campo illeggibile",
        "descrizione": "Campo scritto ma non leggibile: errore se incide sulle ore (orari, ore, totale), "
        "altrimenti attenzione.",
    },
    E06_NON_FOGLIO_FIRMA: {
        "gravita": "errore",
        "titolo": "non è un foglio firma",
        "descrizione": "La pagina non è stata riconosciuta come foglio firma dell'Assistenza Specialistica.",
    },
    W01_FIRMA_MANCANTE: {
        "gravita": "attenzione",
        "titolo": "firma operatore mancante",
        "descrizione": "Ore o orari effettivi presenti senza la firma dell'operatore.",
    },
    W02_ORE_SENZA_ORARIO: {
        "gravita": "attenzione",
        "titolo": "ore senza orario effettivo",
        "descrizione": "Ore dichiarate senza orario effettivo di entrata e uscita e senza assenza dell'alunno.",
    },
    W03_GIORNO_FESTIVO: {
        "gravita": "attenzione",
        "titolo": "prestazione in giorno non lavorativo",
        "descrizione": "Prestazione registrata di domenica o in un giorno festivo (attenzione) "
        "oppure di sabato (informazione).",
    },
    W04_CAMPO_INCERTO: {
        "gravita": "attenzione",
        "titolo": "lettura incerta",
        "descrizione": "Valore letto con incertezza: verificarlo sulla scansione.",
    },
    W05_ASSENZA_OPERATORE_CON_ORE: {
        "gravita": "attenzione",
        "titolo": "assenza operatore con ore",
        "descrizione": "Segnalata l'assenza dell'operatore ma sono presenti ore o orari effettivi.",
    },
    W06_ORE_PEI_SUPERATE: {
        "gravita": "attenzione",
        "titolo": "ore PEI superate",
        "descrizione": "Le ore della settimana superano le ore settimanali previste dal PEI.",
    },
    W07_INTESTAZIONE_INCOMPLETA: {
        "gravita": "attenzione",
        "titolo": "intestazione incompleta",
        "descrizione": "Manca (o non è valido) l'operatore, l'alunno, il mese o l'anno di riferimento.",
    },
    W08_FIRMA_COORDINATORE_MANCANTE: {
        "gravita": "attenzione",
        "titolo": "firma coordinatore mancante",
        "descrizione": "Manca la firma del coordinatore dell'ente.",
    },
    W09_TIMBRO_REFERENTE_MANCANTE: {
        "gravita": "attenzione",
        "titolo": "timbro referente mancante",
        "descrizione": "Mancano il timbro e la firma del referente scolastico.",
    },
    W10_TOTALE_MENSILE_ASSENTE: {
        "gravita": "attenzione",
        "titolo": "totale mensile assente",
        "descrizione": "Il totale delle ore effettive mensili non è scritto sul foglio.",
    },
    W11_DOPPIA_ASSENZA: {
        "gravita": "attenzione",
        "titolo": "doppia assenza",
        "descrizione": "Crocette in entrambe le colonne di assenza (alunno e operatore).",
    },
    W12_ORE_NON_INDICATE: {
        "gravita": "attenzione",
        "titolo": "ore non indicate",
        "descrizione": "Orario effettivo presente ma ore effettive non scritte: le ore sono calcolate dall'orario.",
    },
    I01_ASSENZA_ALUNNO: {
        "gravita": "info",
        "titolo": "assenza alunno",
        "descrizione": "Assenza dell'alunno, con le ore riconosciute e la percentuale sul programmato.",
    },
    I02_ASSENZA_OPERATORE: {
        "gravita": "info",
        "titolo": "assenza operatore",
        "descrizione": "Assenza dell'operatore (con l'eventuale nota, es. 104 = permesso Legge 104/1992).",
    },
    I03_ORARIO_DIVERSO: {
        "gravita": "info",
        "titolo": "orario diverso dal programmato",
        "descrizione": "L'orario effettivo differisce da quello programmato.",
    },
    I04_ORE_SENZA_PROGRAMMATO: {
        "gravita": "info",
        "titolo": "orario senza programmato",
        "descrizione": "Orario effettivo presente senza orario programmato.",
    },
    I05_NON_SVOLTO: {
        "gravita": "info",
        "titolo": "prestazione non svolta",
        "descrizione": "Prestazione programmata non svolta (nessun orario effettivo, nessuna ora, nessuna assenza).",
    },
}

# Etichette dei campi (colonne della tabella e campi d'intestazione).
ETICHETTE_CAMPI: dict[str, str] = {
    "giorno": "Giorno",
    "prog_entrata": "Entrata programmata",
    "prog_uscita": "Uscita programmata",
    "eff_entrata": "Entrata effettiva",
    "eff_uscita": "Uscita effettiva",
    "ore_dichiarate": "Tot. ore effettive",
    "assenza_alunno": "Assenza alunno",
    "assenza_operatore": "Assenza operatore",
    "firma": "Firma operatore",
    "note": "Note",
    "trattino_effettivo": "Trattino negli orari effettivi",
    "anno_scolastico": "Anno scolastico",
    "lotto": "Lotto",
    "municipalita": "Municipalità",
    "ente": "Ente",
    "istituto": "Istituto scolastico",
    "operatore": "Operatore",
    "alunno": "Alunno",
    "mese": "Mese di riferimento",
    "anno": "Anno di riferimento",
    "ore_pei": "Ore da PEI",
    "sostituzione": "Sostituzione",
    "data_compilazione": "Data di compilazione",
    "firma_coordinatore": "Firma coordinatore dell'ente",
    "timbro_referente": "Timbro e firma referente scolastico",
    "totale_mensile_dichiarato": "Totale ore effettive mensili",
}

# Campi che incidono sul calcolo delle ore: se illeggibili sono errori.
CAMPI_ORE_GIORNO: frozenset[str] = frozenset(
    {"prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita", "ore_dichiarate"}
)
CAMPI_ORE_INTESTAZIONE: frozenset[str] = frozenset({"totale_mensile_dichiarato"})

_RANGO_GRAVITA: dict[str, int] = {"errore": 0, "attenzione": 1, "info": 2}


def etichetta_campo(campo: str) -> str:
    return ETICHETTE_CAMPI.get(campo, campo.replace("_", " ").capitalize())


# --- Formattazione ------------------------------------------------------------


def format_ore(x: float | None) -> str:
    """Numero di ore all'italiana: 3 -> "3", 1.5 -> "1,5", 2.25 -> "2,25"; None -> ""."""
    if x is None or isinstance(x, bool):
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(v):
        return ""
    v = round(v, 2) + 0.0  # + 0.0 elimina lo zero negativo
    if v.is_integer():
        return str(int(v))
    return f"{v:.2f}".rstrip("0").rstrip(".").replace(".", ",")


def hours_label(x: float | None) -> str:
    """Ore con unità: "1 ora", "3 ore", "1,5 ore"; None -> ""."""
    testo = format_ore(x)
    if not testo:
        return ""
    return "1 ora" if testo == "1" else f"{testo} ore"


def _ore_participio(x: float, participio_plurale: str) -> str:
    """Ore con participio concordato: "3 ore programmate", "1 ora programmata"."""
    if format_ore(x) == "1":
        return f"1 ora {participio_plurale[:-1]}a"
    return f"{hours_label(x)} {participio_plurale}"


def _maiuscola(testo: str) -> str:
    return testo[:1].upper() + testo[1:]


def format_percentuale(x: float) -> str:
    """50.0 -> "50%", 33.333 -> "33,3%"."""
    v = round(x, 1) + 0.0
    testo = str(int(v)) if v.is_integer() else f"{v:.1f}".replace(".", ",")
    return f"{testo}%"


def format_intervallo(entrata: str | None, uscita: str | None) -> str:
    """Intervallo "08:00–11:00" (orari normalizzati se interpretabili, altrimenti come scritti)."""
    return f"{_testo_orario(entrata)}–{_testo_orario(uscita)}"


def _testo_orario(s: str | None) -> str:
    if _vuoto(s):
        return "?"
    return normalize_time(s) or str(s).strip()


def _testo(s: object) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _dal_al(dal: int, al: int, mese: int | None = None) -> str:
    """Periodo con preposizioni articolate: "dal 2 all'8 febbraio", "dall'1 al 7"."""

    def art(prep: str, n: int) -> str:
        # elisione davanti a numeri che iniziano per vocale: 1 (uno), 8 (otto), 11 (undici)
        return f"{prep}l'{n}" if n in (1, 8, 11) else f"{prep}l {n}"

    if dal == al:
        testo = str(dal)
    else:
        testo = f"{art('da', dal)} {art('al', al)}"
    if mese is not None and 1 <= mese <= 12:
        testo += f" {calendario.nome_mese(mese)}"
    return testo


# --- Parsing di orari e ore ---------------------------------------------------

# Caratteri che l'OCR confonde con le cifre (dopo la conversione in minuscolo).
_CONFUSABILI = str.maketrans({"o": "0", "l": "1", "i": "1", "|": "1", "!": "1"})
_RE_VUOTO = re.compile(r"^[\s\-–—−_/.]*$")
_RE_SEP_ORARIO = re.compile(r"^(\d{1,2})(?:\s*[:.,;'’`´]+\s*|\s+)(\d{1,2})(?:\s*[:.]\s*\d{2})?$")
_RE_SOLO_CIFRE = re.compile(r"^\d{1,4}$")


def _vuoto(s: object) -> bool:
    """Cella vuota o con solo trattini/segni."""
    return s is None or (isinstance(s, str) and bool(_RE_VUOTO.match(s)))


def parse_time(s: str | None) -> tuple[int, int] | None:
    """Interpreta un orario scritto a mano/letto dall'OCR -> (ore, minuti), None se non valido.

    Accetta "8:00", "8.00", "8,00", "8;00", "800", "0800", "8", "08:00", "8h",
    "ore 8", "8:00h", " 8 : 00 " e le confusioni tipiche dell'OCR (o/O -> 0,
    l/I -> 1). Rifiuta ore > 23 e minuti > 59.
    """
    if s is None or isinstance(s, bool):
        return None
    t = unicodedata.normalize("NFKC", str(s)).strip().lower()
    if not t:
        return None
    t = re.sub(r"^(?:ore|alle|h)\s*(?=\d|[oli|!])", "", t)
    t = re.sub(r"\b(?:ore|alle)\b", " ", t)
    t = t.translate(_CONFUSABILI)
    t = re.sub(r"(?<=\d)\s*h\s*$", "", t)  # "8h", "8:00h", "8:00 h"
    t = re.sub(r"(?<=\d)\s*h\s*(?=\d)", ":", t)  # "8h30"
    t = re.sub(r"\bh\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" .,;:'’`´")
    if not t:
        return None
    m = _RE_SEP_ORARIO.match(t)
    if m:
        ore = int(m.group(1))
        minuti_txt = m.group(2)
        minuti = int(minuti_txt) * (10 if len(minuti_txt) == 1 else 1)  # "8:3" -> 8:30
    elif _RE_SOLO_CIFRE.match(t):
        if len(t) <= 2:
            ore, minuti = int(t), 0
        elif len(t) == 3:
            ore, minuti = int(t[0]), int(t[1:])
        else:
            ore, minuti = int(t[:2]), int(t[2:])
    else:
        return None
    if not (0 <= ore <= 23 and 0 <= minuti <= 59):
        return None
    return ore, minuti


def normalize_time(s: str | None) -> str | None:
    """Orario normalizzato "HH:MM" oppure None."""
    p = parse_time(s)
    return None if p is None else f"{p[0]:02d}:{p[1]:02d}"


_FRAZIONI = {"½": 0.5, "¼": 0.25, "¾": 0.75}
_FRAZIONI_PAROLE = {"mezza": 0.5, "mezzo": 0.5, "un quarto": 0.25, "tre quarti": 0.75}
_RE_ORE_FRAZIONE = re.compile(r"^(\d{1,2})?\s*(?:e\s*)?([½¼¾])$")
_RE_ORE_E_PAROLE = re.compile(r"^(\d{1,2})\s+e\s+(mezza|mezzo|un quarto|tre quarti)$")
_RE_ORE_E_MINUTI = re.compile(r"^(\d{1,2})\s*e\s*(\d{1,2})$")
_RE_ORE_DUEPUNTI = re.compile(r"^(\d{1,2})\s*:\s*(\d{2})$")
_RE_ORE_DECIMALI = re.compile(r"^(?:\d{1,3}(?:[.,]\d+)?|[.,]\d+)$")
_RE_SOLO_MINUTI = re.compile(r"^(\d{1,4})\s*(?:minuti|minuto|min|')$")


def _ore_valide(v: float) -> float | None:
    if not math.isfinite(v) or v < 0 or v > MAX_ORE_GIORNO:
        return None
    return v + 0.0


def parse_hours(s: str | float | None) -> float | None:
    """Interpreta un numero di ore -> float, None se assente o non plausibile (< 0 o > 24).

    Accetta "1,5", "1.5", "3", "3h", "3 ore", "2:30" (2,5), "2h30", "1 e 30" (1,5),
    "1 e mezza", "1½", "mezz'ora", "30 min"; i numeri passano invariati.
    """
    if s is None or isinstance(s, bool):
        return None
    if isinstance(s, (int, float)):
        return _ore_valide(float(s))
    t = str(s).strip().lower().replace("’", "'")
    if not t:
        return None
    m = _RE_ORE_FRAZIONE.match(t)
    if m:
        return _ore_valide(int(m.group(1) or 0) + _FRAZIONI[m.group(2)])
    t = unicodedata.normalize("NFKC", t)
    if t in ("mezz'ora", "mezzora", "mezza ora"):
        return 0.5
    m = _RE_SOLO_MINUTI.match(t)
    if m:
        return _ore_valide(int(m.group(1)) / 60)
    t = re.sub(r"(?<=\d)\s*h\s*(?=\d)", ":", t)  # "2h30" -> "2:30"
    t = re.sub(r"(?<=\d)\s*(?:minuti|minuto|min|m|')\s*$", "", t)  # "2:30 min", "1 e 30'"
    t = re.sub(r"(?<=\d)\s*(?:ore|ora|hrs|hr|hh|h)\b", " ", t)  # "3h", "3ore", "1 ora e 30"
    t = re.sub(r"\b(?:ore|ora|hrs|hr|hh|h)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" ;:")
    t = t.rstrip(".,") if not _RE_ORE_DECIMALI.match(t) else t
    if not t:
        return None
    m = _RE_ORE_E_PAROLE.match(t)
    if m:
        return _ore_valide(int(m.group(1)) + _FRAZIONI_PAROLE[m.group(2)])
    m = _RE_ORE_E_MINUTI.match(t)
    if m:
        minuti = int(m.group(2))
        return _ore_valide(int(m.group(1)) + minuti / 60) if minuti <= 59 else None
    for candidato in (t, t.translate(_CONFUSABILI)):
        m = _RE_ORE_DUEPUNTI.match(candidato)
        if m:
            minuti = int(m.group(2))
            return _ore_valide(int(m.group(1)) + minuti / 60) if minuti <= 59 else None
        if _RE_ORE_DECIMALI.match(candidato):
            return _ore_valide(float(candidato.replace(",", ".")))
    return None


def _minuti(s: str | None) -> int | None:
    p = parse_time(s)
    return None if p is None else p[0] * 60 + p[1]


def hours_between(a: str | None, b: str | None) -> float | None:
    """Ore tra entrata ``a`` e uscita ``b``; None se mancante, non valido o non positivo."""
    ma, mb = _minuti(a), _minuti(b)
    if ma is None or mb is None or mb <= ma:
        return None
    return (mb - ma) / 60


def _ore_dichiarate_valide(row: DayRow) -> float | None:
    d = row.ore_dichiarate
    if d is None or isinstance(d, bool):
        return None
    try:
        return _ore_valide(float(d))
    except (TypeError, ValueError):
        return None


def ore_riconosciute(row: DayRow) -> float:
    """Dichiarate se presenti, altrimenti calcolate dall'orario effettivo (0 se assenza operatore)."""
    if row.assenza_operatore:
        return 0.0
    dichiarate = _ore_dichiarate_valide(row)
    if dichiarate is not None:
        return dichiarate
    return hours_between(row.eff_entrata, row.eff_uscita) or 0.0


# --- Analisi di una riga ------------------------------------------------------


@dataclass(frozen=True)
class _Coppia:
    """Orario di entrata/uscita (programmato o effettivo) di una riga."""

    entrata: str | None
    uscita: str | None
    entrata_illeggibile: bool
    uscita_illeggibile: bool

    @property
    def entrata_scritta(self) -> bool:
        return not _vuoto(self.entrata)

    @property
    def uscita_scritta(self) -> bool:
        return not _vuoto(self.uscita)

    @property
    def scritta(self) -> bool:
        """Almeno un orario scritto (anche se illeggibile)."""
        return (
            self.entrata_scritta or self.uscita_scritta or self.entrata_illeggibile or self.uscita_illeggibile
        )

    @property
    def leggibile_vuota(self) -> bool:
        """Nessun orario scritto e nessun orario illeggibile."""
        return not self.scritta

    @property
    def ore(self) -> float | None:
        return hours_between(self.entrata, self.uscita)

    @property
    def valida(self) -> bool:
        return self.ore is not None

    def intervallo(self) -> str:
        return format_intervallo(self.entrata, self.uscita)

    def normalizzata(self) -> tuple[str | None, str | None]:
        return normalize_time(self.entrata), normalize_time(self.uscita)


def _coppie(row: DayRow) -> tuple[_Coppia, _Coppia]:
    ill = set(row.illeggibili)
    prog = _Coppia(row.prog_entrata, row.prog_uscita, "prog_entrata" in ill, "prog_uscita" in ill)
    eff = _Coppia(row.eff_entrata, row.eff_uscita, "eff_entrata" in ill, "eff_uscita" in ill)
    return prog, eff


def _unici(valori: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(v for v in valori if isinstance(v, str) and v))


def _ordina_campi(campi: Iterable[str], ordine: Sequence[str]) -> list[str]:
    elenco = _unici(campi)
    pos = {c: i for i, c in enumerate(ordine)}
    return sorted(elenco, key=lambda c: (pos.get(c, len(ordine)), elenco.index(c)))


def _campi_incerti(incerti: Iterable[str], illeggibili: Iterable[str], ordine: Sequence[str]) -> list[str]:
    ill = set(illeggibili)
    return [c for c in _ordina_campi(incerti, ordine) if c not in ill]


def _valore_testo(campo: str, valore: object) -> str:
    """Valore leggibile per messaggi e colonna "valore letto"."""
    if campo in ("firma", "firma_coordinatore", "timbro_referente"):
        return "presente" if valore else "assente"
    if campo in ("assenza_alunno", "assenza_operatore"):
        return "crocetta presente" if valore else "nessuna crocetta"
    if valore is None or (isinstance(valore, str) and not valore.strip()):
        return "campo vuoto"
    if isinstance(valore, (int, float)) and not isinstance(valore, bool):
        return format_ore(valore) or _testo(valore)
    if isinstance(valore, str) and campo in ("prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita"):
        return _testo_orario(valore)
    return _testo(valore)


def _nota(row: DayRow) -> str | None:
    if row.note is None:
        return None
    testo = _testo(row.note)
    return testo or None


_RE_104 = re.compile(r"(?<!\d)104(?!\d)")


def _descrizione_nota(nota: str) -> str:
    testo = f"nota «{nota}»"
    if _RE_104.search(nota):
        testo += " (permesso Legge 104/1992)"
    return testo


def _anomalia(
    codice: str,
    messaggio: str,
    giorno: int | None = None,
    campo: str | None = None,
    valore_letto: str | None = None,
    valore_atteso: str | None = None,
    gravita: Severity | None = None,
) -> Anomaly:
    return Anomaly(
        codice=codice,
        gravita=gravita or cast(Severity, CODICI[codice]["gravita"]),
        messaggio=messaggio,
        giorno=giorno,
        campo=campo,
        valore_letto=valore_letto,
        valore_atteso=valore_atteso,
    )


@dataclass(frozen=True)
class _Periodo:
    mese: int
    anno: int


def _periodo(header: Header) -> _Periodo | None:
    mese, anno = header.mese, header.anno
    if not isinstance(mese, int) or isinstance(mese, bool) or not 1 <= mese <= 12:
        return None
    if not isinstance(anno, int) or isinstance(anno, bool) or not 2000 <= anno <= 2100:
        return None
    return _Periodo(mese, anno)


def _controlli_campi(
    giorno: int | None,
    incerti: Iterable[str],
    illeggibili: Iterable[str],
    valori: object,
    ordine: Sequence[str],
    campi_ore: frozenset[str],
) -> list[Anomaly]:
    """E05 (campi illeggibili) e W04 (campi incerti, esclusi gli illeggibili)."""
    out: list[Anomaly] = []
    prefisso = f"Giorno {giorno}: " if giorno is not None else ""
    ill = _ordina_campi(illeggibili, ordine)
    for campo in ill:
        grave = campo in campi_ore
        seguito = "da leggere sulla scansione e inserire a mano" if grave else "da verificare sulla scansione"
        out.append(
            _anomalia(
                E05_CAMPO_ILLEGGIBILE,
                _maiuscola(f"{prefisso}campo «{etichetta_campo(campo)}» illeggibile: {seguito}."),
                giorno=giorno,
                campo=campo,
                gravita="errore" if grave else "attenzione",
            )
        )
    for campo in _campi_incerti(incerti, ill, ordine):
        letto = _valore_testo(campo, getattr(valori, campo, None))
        out.append(
            _anomalia(
                W04_CAMPO_INCERTO,
                _maiuscola(
                    f"{prefisso}lettura incerta del campo «{etichetta_campo(campo)}» "
                    f"(valore letto: {letto}); verificare sulla scansione."
                ),
                giorno=giorno,
                campo=campo,
                valore_letto=letto,
            )
        )
    return out


def _controlli_orari(g: int, coppia: _Coppia, tipo: str) -> list[Anomaly]:
    """E03 per una coppia entrata/uscita ("programmato" o "effettivo")."""
    out: list[Anomaly] = []
    pref = "prog" if tipo == "programmato" else "eff"
    agg = "programmata" if tipo == "programmato" else "effettiva"
    validi: dict[str, bool] = {}
    for verso, valore, illeggibile in (
        ("entrata", coppia.entrata, coppia.entrata_illeggibile),
        ("uscita", coppia.uscita, coppia.uscita_illeggibile),
    ):
        campo = f"{pref}_{verso}"
        if _vuoto(valore):
            validi[verso] = False
            continue
        if parse_time(valore) is None:
            validi[verso] = False
            if illeggibile:  # già segnalato da E05
                continue
            letto = _testo(valore)
            out.append(
                _anomalia(
                    E03_ORARIO_NON_VALIDO,
                    f"Giorno {g}: {verso} {agg} non interpretabile come orario («{letto}»).",
                    giorno=g,
                    campo=campo,
                    valore_letto=letto,
                    valore_atteso="HH:MM",
                )
            )
        else:
            validi[verso] = True
    if validi["entrata"] and validi["uscita"]:
        if coppia.ore is None:  # uscita <= entrata
            ent, usc = coppia.normalizzata()
            out.append(
                _anomalia(
                    E03_ORARIO_NON_VALIDO,
                    f"Giorno {g}: l'uscita {agg} ({usc}) non è successiva all'entrata ({ent}).",
                    giorno=g,
                    campo=f"{pref}_uscita",
                    valore_letto=f"{ent}–{usc}",
                )
            )
    else:
        # una sola delle due scritta (e l'altra non illeggibile): orario incompleto
        for presente, mancante, ill_mancante in (
            ("entrata", "uscita", coppia.uscita_illeggibile),
            ("uscita", "entrata", coppia.entrata_illeggibile),
        ):
            if validi[presente] and _vuoto(getattr(coppia, mancante)) and not ill_mancante:
                valore = normalize_time(getattr(coppia, presente))
                out.append(
                    _anomalia(
                        E03_ORARIO_NON_VALIDO,
                        f"Giorno {g}: orario {tipo} incompleto, manca l'{mancante} ({presente} {valore}).",
                        giorno=g,
                        campo=f"{pref}_{mancante}",
                        valore_letto=valore,
                    )
                )
    return out


def _controlli_riga(row: DayRow, periodo: _Periodo | None) -> list[Anomaly]:
    g = row.giorno
    out: list[Anomaly] = []
    ill = set(row.illeggibili)
    inc = set(row.incerti) - ill
    prog, eff = _coppie(row)
    nota = _nota(row)

    out += _controlli_orari(g, prog, "programmato")
    out += _controlli_orari(g, eff, "effettivo")

    # Ore dichiarate
    dich_raw = row.ore_dichiarate
    dich = _ore_dichiarate_valide(row)
    if dich_raw is not None and dich is None:
        letto = format_ore(dich_raw) or _testo(dich_raw)
        out.append(
            _anomalia(
                E01_ORE_NON_COERENTI,
                f"Giorno {g}: ore dichiarate non plausibili ({letto}): "
                f"il valore deve essere compreso tra 0 e {format_ore(MAX_ORE_GIORNO)}.",
                giorno=g,
                campo="ore_dichiarate",
                valore_letto=letto,
            )
        )
    ore_eff = eff.ore
    ore_prog = prog.ore
    eff_scritto = eff.scritta
    ore_illeggibili = "ore_dichiarate" in ill

    # E01: ore dichiarate diverse dall'orario effettivo
    # Con assenza dell'alunno sono ammesse ore riconosciute parziali (inferiori all'orario).
    if (
        dich is not None
        and ore_eff is not None
        and not row.assenza_operatore
        and abs(dich - ore_eff) > TOLLERANZA
        and not (row.assenza_alunno and dich < ore_eff)
    ):
        out.append(
            _anomalia(
                E01_ORE_NON_COERENTI,
                f"Giorno {g}: dichiarate {hours_label(dich)} ma l'orario effettivo "
                f"{eff.intervallo()} corrisponde a {hours_label(ore_eff)}.",
                giorno=g,
                campo="ore_dichiarate",
                valore_letto=format_ore(dich),
                valore_atteso=format_ore(ore_eff),
            )
        )

    # W12: orario effettivo senza ore scritte -> ore calcolate
    if ore_eff is not None and dich_raw is None and not ore_illeggibili and not row.assenza_operatore:
        out.append(
            _anomalia(
                W12_ORE_NON_INDICATE,
                f"Giorno {g}: ore effettive non indicate sul foglio; riconosciute "
                f"{hours_label(ore_eff)} calcolate dall'orario {eff.intervallo()}.",
                giorno=g,
                campo="ore_dichiarate",
                valore_atteso=format_ore(ore_eff),
            )
        )

    # W02: ore senza orario effettivo (e senza assenze)
    if (
        dich is not None
        and dich > TOLLERANZA
        and eff.leggibile_vuota
        and not row.assenza_alunno
        and not row.assenza_operatore
    ):
        out.append(
            _anomalia(
                W02_ORE_SENZA_ORARIO,
                f"Giorno {g}: dichiarate {hours_label(dich)} senza orario effettivo di entrata e uscita.",
                giorno=g,
                campo="ore_dichiarate",
                valore_letto=format_ore(dich),
            )
        )

    ore_scritte = dich if dich is not None and dich > TOLLERANZA else None
    ha_ore = ore_scritte is not None
    prestazione = eff_scritto or ha_ore

    # W01: firma dell'operatore mancante
    if (
        prestazione
        and not row.firma
        and not row.assenza_operatore
        and "firma" not in ill
        and "firma" not in inc
    ):
        dettaglio = f"orario {eff.intervallo()}" if eff.entrata_scritta or eff.uscita_scritta else ""
        if ore_scritte is not None:
            ore_txt = _ore_participio(ore_scritte, "dichiarate")
            dettaglio = f"{dettaglio}, {ore_txt}" if dettaglio else ore_txt
        dettaglio = f" ({dettaglio})" if dettaglio else ""
        out.append(
            _anomalia(
                W01_FIRMA_MANCANTE,
                f"Giorno {g}: manca la firma dell'operatore{dettaglio}.",
                giorno=g,
                campo="firma",
                valore_letto="assente",
            )
        )

    # W03: prestazione in giorno non lavorativo
    if periodo is not None and prestazione and not row.assenza_operatore:
        tipo = calendario.tipo_giorno(periodo.anno, periodo.mese, g)
        nome = calendario.nome_giorno(periodo.anno, periodo.mese, g)
        if tipo == "festivo":
            festa = calendario.nome_festivita(periodo.anno, periodo.mese, g)
            out.append(
                _anomalia(
                    W03_GIORNO_FESTIVO,
                    f"Giorno {g} ({nome}, {festa}): prestazione registrata in un giorno festivo.",
                    giorno=g,
                    gravita="attenzione",
                )
            )
        elif tipo == "domenica":
            out.append(
                _anomalia(
                    W03_GIORNO_FESTIVO,
                    f"Giorno {g} (domenica): prestazione registrata di domenica.",
                    giorno=g,
                    gravita="attenzione",
                )
            )
        elif tipo == "sabato":
            out.append(
                _anomalia(
                    W03_GIORNO_FESTIVO,
                    f"Giorno {g} (sabato): prestazione registrata di sabato; verificare che sia "
                    "prevista dal calendario scolastico.",
                    giorno=g,
                    gravita="info",
                )
            )

    # W05: assenza operatore con ore/orari effettivi
    if row.assenza_operatore and prestazione:
        parti = []
        if eff_scritto:
            parti.append(f"l'orario effettivo {eff.intervallo()}")
        if ore_scritte is not None:
            parti.append(hours_label(ore_scritte))
        out.append(
            _anomalia(
                W05_ASSENZA_OPERATORE_CON_ORE,
                f"Giorno {g}: segnalata l'assenza dell'operatore ma sono indicati {' e '.join(parti)}; "
                "le ore non vengono riconosciute.",
                giorno=g,
                campo="assenza_operatore",
            )
        )

    # W11: doppia assenza
    if row.assenza_alunno and row.assenza_operatore:
        out.append(
            _anomalia(
                W11_DOPPIA_ASSENZA,
                f"Giorno {g}: crocette sia su «Assenza alunno» sia su «Assenza operatore».",
                giorno=g,
                campo="assenza_alunno",
            )
        )

    # I01: assenza alunno
    if row.assenza_alunno:
        ric = ore_riconosciute(row)
        if ric > TOLLERANZA:
            verbo = "riconosciuta" if format_ore(ric) == "1" else "riconosciute"
            testo = f"{verbo} {hours_label(ric)}"
            if ore_prog:
                testo += (
                    f" su {format_ore(ore_prog)} programmate ({format_percentuale(ric / ore_prog * 100)})"
                )
        else:
            testo = "nessuna ora riconosciuta"
            if ore_prog:
                testo += f" ({_ore_participio(ore_prog, 'programmate')})"
        msg = f"Giorno {g}: assenza dell'alunno; {testo}."
        if nota:
            msg += f" {_maiuscola(_descrizione_nota(nota))}."
        out.append(
            _anomalia(
                I01_ASSENZA_ALUNNO,
                msg,
                giorno=g,
                campo="assenza_alunno",
                valore_letto=format_ore(ric),
                valore_atteso=format_ore(ore_prog) if ore_prog else None,
            )
        )

    # I02: assenza operatore
    if row.assenza_operatore:
        msg = f"Giorno {g}: assenza dell'operatore"
        if nota:
            msg += f" – {_descrizione_nota(nota)}"
        if ore_prog:
            non_svolte = "non svolta" if format_ore(ore_prog) == "1" else "non svolte"
            msg += f"; {_ore_participio(ore_prog, 'programmate')} {non_svolte}"
        out.append(
            _anomalia(
                I02_ASSENZA_OPERATORE,
                msg + ".",
                giorno=g,
                campo="assenza_operatore",
                valore_letto=nota,
            )
        )

    # I03 / I04: confronto con il programmato
    if eff.valida and not row.assenza_operatore:
        if prog.valida:
            if prog.normalizzata() != eff.normalizzata():
                out.append(
                    _anomalia(
                        I03_ORARIO_DIVERSO,
                        f"Giorno {g}: orario effettivo {eff.intervallo()} diverso dal programmato "
                        f"{prog.intervallo()}.",
                        giorno=g,
                        campo="eff_entrata"
                        if prog.normalizzata()[0] != eff.normalizzata()[0]
                        else "eff_uscita",
                        valore_letto=eff.intervallo(),
                        valore_atteso=prog.intervallo(),
                    )
                )
        elif prog.leggibile_vuota:
            out.append(
                _anomalia(
                    I04_ORE_SENZA_PROGRAMMATO,
                    f"Giorno {g}: orario effettivo {eff.intervallo()} senza orario programmato.",
                    giorno=g,
                    campo="prog_entrata",
                    valore_letto=eff.intervallo(),
                )
            )

    # I05: prestazione programmata non svolta (nessun orario effettivo, nessuna ora, nessuna assenza)
    nessuna_ora = dich_raw is None or (dich is not None and dich <= TOLLERANZA)
    if (
        (prog.entrata_scritta or prog.uscita_scritta)
        and eff.leggibile_vuota
        and nessuna_ora
        and not ore_illeggibili
        and not row.assenza_alunno
        and not row.assenza_operatore
    ):
        msg = f"Giorno {g}: prestazione programmata non svolta (orario programmato {prog.intervallo()})"
        if nota:
            msg += f" – {_descrizione_nota(nota)}"
        out.append(
            _anomalia(
                I05_NON_SVOLTO,
                msg + ".",
                giorno=g,
                campo="eff_entrata",
                valore_letto=nota,
            )
        )

    # E05 / W04 per ultimi: a parità di gravità i problemi concreti precedono le letture dubbie.
    out += _controlli_campi(g, row.incerti, row.illeggibili, row, DAY_FIELDS, CAMPI_ORE_GIORNO)
    return out


def _controlli_intestazione(header: Header, periodo: _Periodo | None) -> list[Anomaly]:
    out: list[Anomaly] = []
    ill = set(header.illeggibili)
    inc = set(header.incerti) - ill

    # W07: intestazione incompleta (i campi illeggibili sono già segnalati da E05)
    mancanti = {
        "operatore": "manca il nome dell'operatore",
        "alunno": "manca il nome dell'alunno",
        "mese": "manca il mese di riferimento",
        "anno": "manca l'anno di riferimento",
    }
    for campo, testo in mancanti.items():
        if campo in ill:
            continue
        valore = getattr(header, campo)
        if isinstance(valore, str):
            valore = valore.strip() or None
        if valore is None:
            out.append(
                _anomalia(W07_INTESTAZIONE_INCOMPLETA, f"Intestazione incompleta: {testo}.", campo=campo)
            )
        elif campo in ("mese", "anno") and periodo is None:
            corretto = (campo == "mese" and isinstance(valore, int) and 1 <= valore <= 12) or (
                campo == "anno" and isinstance(valore, int) and 2000 <= valore <= 2100
            )
            if not corretto:
                out.append(
                    _anomalia(
                        W07_INTESTAZIONE_INCOMPLETA,
                        f"Intestazione: {etichetta_campo(campo).lower()} non valido («{_testo(valore)}»).",
                        campo=campo,
                        valore_letto=_testo(valore),
                    )
                )

    out += _controlli_campi(
        None, header.incerti, header.illeggibili, header, HEADER_FIELDS, CAMPI_ORE_INTESTAZIONE
    )

    if not header.firma_coordinatore and not {"firma_coordinatore"} & (ill | inc):
        out.append(
            _anomalia(
                W08_FIRMA_COORDINATORE_MANCANTE,
                "Manca la firma del coordinatore dell'ente accanto al totale mensile.",
                campo="firma_coordinatore",
                valore_letto="assente",
            )
        )
    if not header.timbro_referente and not {"timbro_referente"} & (ill | inc):
        out.append(
            _anomalia(
                W09_TIMBRO_REFERENTE_MANCANTE,
                "Mancano il timbro e la firma del referente scolastico in calce al foglio.",
                campo="timbro_referente",
                valore_letto="assente",
            )
        )
    return out


def _ore_pei(header: Header) -> float | None:
    v = header.ore_pei
    if v is None or isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def _r(x: float) -> float:
    return round(x, 2) + 0.0


def _settimane(periodo: _Periodo, righe: Sequence[DayRow], ore_pei: float | None) -> list[WeekTotal]:
    ric: dict[int, float] = {}
    for r in righe:
        ric[r.giorno] = ric.get(r.giorno, 0.0) + ore_riconosciute(r)
    # Scuola aperta anche il sabato se nel foglio risultano orari o ore di sabato.
    sabato_attivo = any(
        calendario.tipo_giorno(periodo.anno, periodo.mese, r.giorno) == "sabato"
        and (ric[r.giorno] > 0 or any(c.scritta for c in _coppie(r)))
        for r in righe
    )
    giorni_settimana = 6 if sabato_attivo else 5
    out: list[WeekTotal] = []
    for n, (dal, al) in enumerate(calendario.settimane_del_mese(periodo.anno, periodo.mese), start=1):
        ore = _r(math.fsum(ric.get(g, 0.0) for g in range(dal, al + 1)))
        pei_sett: float | None = None
        diff: float | None = None
        if ore_pei is not None:
            tipi_scolastici = ("feriale", "sabato") if sabato_attivo else ("feriale",)
            scolastici = sum(
                1
                for g in range(dal, al + 1)
                if calendario.tipo_giorno(periodo.anno, periodo.mese, g) in tipi_scolastici
            )
            pei_sett = _r(ore_pei * scolastici / giorni_settimana)
            diff = _r(ore - pei_sett)
        out.append(WeekTotal(settimana=n, dal=dal, al=al, ore=ore, ore_pei=pei_sett, differenza=diff))
    return out


def _ordina(anomalie: Iterable[Anomaly]) -> list[Anomaly]:
    """Prima le anomalie del documento, poi per giorno; a parità per gravità (ordinamento stabile)."""
    return sorted(
        anomalie,
        key=lambda a: (a.giorno is not None, a.giorno or 0, _RANGO_GRAVITA.get(a.gravita, 3)),
    )


def _conta(header: Header, righe: Sequence[DayRow]) -> tuple[int, int]:
    """(campi incerti, campi illeggibili) su intestazione e righe; incerti esclusi gli illeggibili."""
    illeggibili = len(_unici(header.illeggibili)) + sum(len(_unici(r.illeggibili)) for r in righe)
    incerti = len(_campi_incerti(header.incerti, header.illeggibili, HEADER_FIELDS)) + sum(
        len(_campi_incerti(r.incerti, r.illeggibili, DAY_FIELDS)) for r in righe
    )
    return incerti, illeggibili


def _finalizza(anomalie: list[Anomaly], totals: Totals) -> tuple[list[Anomaly], Totals]:
    anomalie = _ordina(anomalie)
    totals.n_errori = sum(1 for a in anomalie if a.gravita == "errore")
    totals.n_attenzioni = sum(1 for a in anomalie if a.gravita == "attenzione")
    totals.n_info = sum(1 for a in anomalie if a.gravita == "info")
    totals.firme_mancanti = sum(1 for a in anomalie if a.codice == W01_FIRMA_MANCANTE)
    if totals.n_errori:
        totals.stato = "errori"
    elif totals.n_attenzioni or totals.campi_incerti or totals.campi_illeggibili:
        totals.stato = "da_verificare"
    else:
        totals.stato = "ok"
    return anomalie, totals


def validate(header: Header, rows: list[DayRow]) -> tuple[list[Anomaly], Totals]:
    """Esegue tutti i controlli e calcola i totali. Non modifica gli oggetti ricevuti."""
    periodo = _periodo(header)
    n_giorni = calendario.giorni_nel_mese(periodo.anno, periodo.mese) if periodo else 31
    anomalie: list[Anomaly] = _controlli_intestazione(header, periodo)
    righe: list[DayRow] = []

    for row in sorted(rows, key=lambda r: r.giorno):
        g = row.giorno
        if not 1 <= g <= n_giorni:
            if row.has_content():
                if periodo is not None and 1 <= g <= 31:
                    msg = (
                        f"Giorno {g}: il giorno non esiste nel mese di "
                        f"{calendario.etichetta_mese(periodo.mese, periodo.anno)} ma la riga contiene dati; "
                        "i valori non sono conteggiati nei totali."
                    )
                else:
                    msg = f"Giorno {g}: giorno non valido ma la riga contiene dati; i valori non sono conteggiati."
                anomalie.append(_anomalia(E04_GIORNO_INESISTENTE, msg, giorno=g))
            continue
        righe.append(row)
        anomalie += _controlli_riga(row, periodo)

    # Totali
    coppie = [_coppie(r) for r in righe]
    dichiarate = [d for d in (_ore_dichiarate_valide(r) for r in righe) if d is not None]
    incerti, illeggibili = _conta(header, righe)
    ore_pei = _ore_pei(header)
    totals = Totals(
        giorni_programmati=sum(1 for prog, _ in coppie if prog.entrata_scritta or prog.uscita_scritta),
        giorni_lavorati=sum(1 for r in righe if not r.assenza_operatore and ore_riconosciute(r) > 0),
        ore_programmate=_r(math.fsum(prog.ore or 0.0 for prog, _ in coppie)),
        ore_calcolate=_r(math.fsum(eff.ore or 0.0 for _, eff in coppie)),
        ore_dichiarate=_r(math.fsum(dichiarate)),
        ore_riconosciute=_r(math.fsum(ore_riconosciute(r) for r in righe)),
        assenze_alunno=sum(1 for r in righe if r.assenza_alunno),
        assenze_operatore=sum(1 for r in righe if r.assenza_operatore),
        campi_incerti=incerti,
        campi_illeggibili=illeggibili,
        settimane=_settimane(periodo, righe, ore_pei) if periodo else [],
    )

    # Totale mensile (E02 / W10)
    totale = header.totale_mensile_dichiarato
    if totale is not None and not isinstance(totale, bool) and math.isfinite(float(totale)):
        totals.totale_mensile_dichiarato = float(totale)
        totals.differenza_totale = _r(totals.ore_dichiarate - float(totale))
        if abs(totals.differenza_totale) > TOLLERANZA:
            anomalie.append(
                _anomalia(
                    E02_TOTALE_MENSILE_DIVERSO,
                    f"Totale mensile dichiarato {hours_label(float(totale))}, somma dei giorni "
                    f"{hours_label(totals.ore_dichiarate)} (differenza {format_ore(abs(totals.differenza_totale))}).",
                    campo="totale_mensile_dichiarato",
                    valore_letto=format_ore(float(totale)),
                    valore_atteso=format_ore(totals.ore_dichiarate),
                )
            )
    elif "totale_mensile_dichiarato" not in set(header.illeggibili):
        anomalie.append(
            _anomalia(
                W10_TOTALE_MENSILE_ASSENTE,
                "Totale ore effettive mensili non indicato sul foglio "
                f"(somma dei giorni: {hours_label(totals.ore_dichiarate)}).",
                campo="totale_mensile_dichiarato",
                valore_atteso=format_ore(totals.ore_dichiarate),
            )
        )

    # W06: ore settimanali oltre il PEI
    if ore_pei is not None and periodo is not None:
        for w in totals.settimane:
            if w.ore > ore_pei + TOLLERANZA:
                eccedenza = _r(w.ore - ore_pei)
                anomalie.append(
                    _anomalia(
                        W06_ORE_PEI_SUPERATE,
                        f"Settimana {w.settimana} ({_dal_al(w.dal, w.al, periodo.mese)}): {hours_label(w.ore)}, "
                        f"oltre il limite settimanale del PEI ({hours_label(ore_pei)}; "
                        f"eccedenza {hours_label(eccedenza)}).",
                        campo="ore_pei",
                        valore_letto=format_ore(w.ore),
                        valore_atteso=format_ore(ore_pei),
                    )
                )

    return _finalizza(anomalie, totals)


def validate_document(doc: Document) -> Document:
    """Imposta ``doc.anomalies`` e ``doc.totals`` e restituisce ``doc``."""
    if doc.is_foglio_firma:
        doc.anomalies, doc.totals = validate(doc.header, doc.rows)
        return doc
    e06 = _anomalia(
        E06_NON_FOGLIO_FIRMA,
        "La pagina non è stata riconosciuta come foglio firma dell'Assistenza Specialistica: "
        "verificare il file oppure inserire i dati manualmente.",
    )
    if any(r.has_content() for r in doc.rows):
        anomalie, totals = validate(doc.header, doc.rows)
        doc.anomalies, doc.totals = _finalizza([e06, *anomalie], totals)
    else:
        incerti, illeggibili = _conta(doc.header, doc.rows)
        doc.anomalies, doc.totals = _finalizza(
            [e06], Totals(campi_incerti=incerti, campi_illeggibili=illeggibili)
        )
    return doc


# --- Aiuti per interfaccia ed Excel -------------------------------------------


def anomalies_for_day(anomalies: Iterable[Anomaly], giorno: int) -> list[Anomaly]:
    """Anomalie riferite al giorno indicato (nell'ordine ricevuto)."""
    return [a for a in anomalies if a.giorno == giorno]


def anomalies_for_document(anomalies: Iterable[Anomaly]) -> list[Anomaly]:
    """Anomalie che riguardano l'intero documento (giorno = None)."""
    return [a for a in anomalies if a.giorno is None]


def stato_campo(oggetto: Header | DayRow, campo: str) -> Literal["illeggibile", "incerto", ""]:
    """Stato di lettura di un campo: "illeggibile" prevale su "incerto"; "" se letto con sicurezza."""
    if campo in oggetto.illeggibili:
        return "illeggibile"
    if campo in oggetto.incerti:
        return "incerto"
    return ""


def _sintesi(anomalie: Sequence[Anomaly]) -> str:
    """Titoli brevi senza ripetizioni; per E05/W04 elenca i campi."""
    gruppi: dict[str, list[str]] = {}
    for a in anomalie:
        campi = gruppi.setdefault(a.codice, [])
        if a.codice in (E05_CAMPO_ILLEGGIBILE, W04_CAMPO_INCERTO) and a.campo:
            etichetta = etichetta_campo(a.campo).lower()
            if etichetta not in campi:
                campi.append(etichetta)
    parti: list[str] = []
    for codice, campi in gruppi.items():
        if codice == E05_CAMPO_ILLEGGIBILE and campi:
            parti.append(f"illeggibile: {', '.join(campi)}")
        elif codice == W04_CAMPO_INCERTO and campi:
            parti.append(f"lettura incerta: {', '.join(campi)}")
        else:
            parti.append(CODICI.get(codice, {}).get("titolo", codice))
    return "; ".join(parti)


def esito_riga(row: DayRow, anomalies: Iterable[Anomaly]) -> str:
    """Esito sintetico di una riga giornaliera.

    "Errore: …", "Da verificare: …", "Assenza operatore", "Assenza alunno",
    "Non svolto", "OK" oppure "" per i giorni vuoti. ``anomalies`` può essere
    l'elenco completo del documento: si considerano solo quelle del giorno.
    """
    proprie = anomalies_for_day(anomalies, row.giorno)
    errori = [a for a in proprie if a.gravita == "errore"]
    if errori:
        return f"Errore: {_sintesi(errori)}"
    attenzioni = [a for a in proprie if a.gravita == "attenzione"]
    if attenzioni:
        return f"Da verificare: {_sintesi(attenzioni)}"
    if row.assenza_operatore:
        return "Assenza operatore"
    if row.assenza_alunno:
        return "Assenza alunno"
    if not row.has_content():
        return ""
    if any(a.codice == I05_NON_SVOLTO for a in proprie):
        return "Non svolto"
    _, eff = _coppie(row)
    if ore_riconosciute(row) > 0 or eff.scritta:
        return "OK"
    return "Non svolto"
