"""Anagrafica appresa: nomi già confermati dall'utente e correzione delle letture.

Gli stessi operatori, alunni, istituti ed enti ricorrono ogni mese: i nomi letti
male dal riconoscimento della scrittura (es. «TOMBREU ALESSIA») vengono
ricondotti ai nomi già confermati nei fogli precedenti («TOMBERLI ALESSIA»).
Il modulo è indipendente dal motore OCR: ``processing`` lo applica al risultato
di ``engine.extract`` prima della validazione.

Apprendimento (solo da dati affidabili, mai dalla lettura OCR grezza)
---------------------------------------------------------------------
* documento confermato (``user_verified``): tutti i campi anagrafici
  dell'intestazione e l'associazione operatore → {alunno, istituto, ente,
  lotto, municipalità, ore PEI} (un operatore può seguire più alunni);
* campo anagrafico corretto a mano (``user_edited`` contiene ``header.<campo>``):
  il nuovo valore; il valore OCR sostituito non viene appreso;
* aggiunta manuale (``aggiungi``, ``POST /api/anagrafica``).

Il contributo di ogni documento è registrato: ricalcolarlo a ogni salvataggio è
idempotente e una correzione successiva *sostituisce* il valore appreso prima
dallo stesso documento (un errore di battitura poi corretto non resta
nell'anagrafica; togliere la conferma ritira ciò che non è stato scritto a mano).
Quando il documento viene eliminato o riletto, il contributo viene scollegato:
quanto appreso resta.

Riconoscimento (``Anagrafica.correggi``)
----------------------------------------
I testi sono normalizzati (maiuscole, senza accenti e punteggiatura, spazi
compattati) e confrontati con :func:`confronta`: media fra ``SequenceMatcher`` e
una distanza di modifica pesata (scambi di lettere adiacenti e confusioni tipiche
della scrittura a mano come M/H, U/N, O/0 costano meno), sul miglior ordine dei
nomi («NOME COGNOME» / «COGNOME NOME»); in più la *copertura*: ogni parola di un
nome deve ritrovarsi, anche approssimativamente, nell'altro (distingue «ESPOSITO
MARIA» da «ESPOSITO ANNA», dove la sola somiglianza globale sarebbe alta).

Regole di decisione (calibrate su letture reali del motore locale e su
corruzioni simulate, vedi ``SOGLIE``):

* lettura identica a una voce (a meno di maiuscole, accenti, punteggiatura e
  ordine dei nomi): si usa la voce e il campo non è più incerto;
* punteggio (somiglianza + eventuale bonus d'associazione) ≥ soglia di
  sostituzione del campo, copertura sufficiente e distacco ≥ ``MARGINE`` dal
  secondo candidato: la lettura viene sostituita dalla voce (lettura originale
  annotata in ``ocr_notes``); il campo resta incerto se la somiglianza è sotto
  ``CERTEZZA`` o la copertura sotto ``COPERTURA_CERTA``;
* più candidati vicini (ambiguità), un candidato poco simile, nomi che
  differiscono solo per la vocale finale (MARIO/MARIA, GIOVANNI/GIOVANNA: persone
  diverse) o istituti con numeri diversi («IC 9» / «IC 10»): si tiene la lettura,
  il campo diventa incerto e i candidati sono indicati nelle note;
* lettura poco simile a tutte le voci (sotto la soglia di suggerimento): è un
  nome nuovo, nessuna modifica;
* associazioni: riconosciuto l'operatore, alunni/istituti/enti a lui associati
  ricevono un bonus (maggiore se anche l'alunno è riconosciuto); un campo
  illeggibile viene proposto (e resta incerto) solo se l'associazione è univoca,
  es. alunno illeggibile e un solo alunno noto per quell'operatore.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import tempfile
import threading
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from sirio import config
from sirio.models import Document, ExtractionResult, Header

log = logging.getLogger(__name__)

FILE_NAME = "anagrafica.json"
VERSIONE = 1

#: Campi anagrafici dell'intestazione gestiti dall'anagrafica.
CAMPI: tuple[str, ...] = ("operatore", "alunno", "istituto", "ente")
#: Valori memorizzati per ogni associazione operatore/alunno.
CAMPI_ASSOCIAZIONE: tuple[str, ...] = ("alunno", "istituto", "ente", "lotto", "municipalita", "ore_pei")

ETICHETTE: dict[str, str] = {
    "operatore": "Operatore",
    "alunno": "Alunno",
    "istituto": "Istituto",
    "ente": "Ente",
    "lotto": "Lotto",
    "municipalita": "Municipalità",
    "ore_pei": "Ore da PEI",
}

MAX_LUNGHEZZA = 200
MAX_DOCUMENTI = 20_000          # contributi per documento conservati (i più vecchi vengono scollegati)


# ==========================================================================
# Soglie (calibrate: vedi docstring del modulo e tests/test_anagrafica.py)
# ==========================================================================

@dataclass(frozen=True)
class Soglie:
    sostituzione: float          # punteggio minimo (somiglianza + bonus d'associazione) per sostituire
    copertura: float             # copertura minima delle parole per sostituire
    copertura_associata: float   # ... se la voce è associata all'operatore/alunno riconosciuti
    suggerimento: float          # somiglianza minima per segnalare un candidato nelle note


# Calibrazione (letture reali del motore locale + 1.700 letture simulate con errori tipici
# della scrittura a mano, su 110 nomi con cognomi ricorrenti): con queste soglie i nomi
# letti male vengono ricondotti alla voce giusta nel ~72% dei casi senza alcuna
# sostituzione errata; una persona nuova viene sostituita (restando incerta) in meno
# dell'1% dei casi.
SOGLIE: dict[str, Soglie] = {
    # nomi di persona: brevi, con omonimie parziali frequenti (stesso cognome)
    "operatore": Soglie(sostituzione=0.72, copertura=0.60, copertura_associata=0.40, suggerimento=0.70),
    "alunno": Soglie(sostituzione=0.72, copertura=0.60, copertura_associata=0.40, suggerimento=0.70),
    # denominazioni: più lunghe e lette peggio; contano i numeri («IC 9» non è «IC 10»)
    "istituto": Soglie(sostituzione=0.62, copertura=0.35, copertura_associata=0.25, suggerimento=0.55),
    # enti: prefissi comuni («COOPERATIVA SOCIALE ...»): senza associazione serve più somiglianza
    "ente": Soglie(sostituzione=0.70, copertura=0.55, copertura_associata=0.30, suggerimento=0.55),
}
CERTEZZA = 0.90                 # somiglianza oltre la quale la voce sostituita non resta incerta
COPERTURA_CERTA = 0.80          # ... purché ogni parola sia ben ritrovata (e i numeri coincidano)
MARGINE = 0.08                  # distacco minimo dal secondo candidato
BONUS_OPERATORE = 0.10          # voce associata all'operatore (o all'alunno) riconosciuto
BONUS_COPPIA = 0.15             # voce associata alla coppia operatore + alunno riconosciuta
MAX_CANDIDATI_NOTE = 3


# ==========================================================================
# Normalizzazione e somiglianza
# ==========================================================================

def pulisci(testo: Any) -> str | None:
    """Testo con gli spazi compattati (come lo ha scritto l'utente), o None se vuoto."""
    if testo is None:
        return None
    text = " ".join(str(testo).split())
    return text or None


def normalizza(testo: str | None) -> str:
    """Maiuscole, senza accenti né punteggiatura, spazi compattati: «D'Angelo  Lucà» -> «D ANGELO LUCA»."""
    if not testo:
        return ""
    out: list[str] = []
    for ch in unicodedata.normalize("NFKD", str(testo)):
        if unicodedata.combining(ch):
            continue
        out.append(ch.upper() if ch.isalnum() else " ")
    return " ".join("".join(out).split())


def chiave(testo: str | None) -> str:
    """Chiave di confronto indipendente dall'ordine delle parole («NOME COGNOME» = «COGNOME NOME»)."""
    return " ".join(sorted(normalizza(testo).split()))


def valido(testo: str | None) -> bool:
    """True se il testo può essere una voce d'anagrafica (almeno due lettere)."""
    return sum(ch.isalpha() for ch in normalizza(testo)) >= 2


# Coppie di caratteri che la scrittura a mano (e TrOCR) confonde spesso: costano metà.
_SIMILI_COPPIE = (
    "O0", "OQ", "OD", "OU", "I1", "IL", "L1", "IJ", "S5", "B8", "Z2", "G6", "G9", "Q9", "T7",
    "UV", "VY", "MN", "MH", "NH", "UN", "UH", "AO", "CE", "RN", "LU", "EI",
)
_SIMILI: frozenset[tuple[str, str]] = frozenset(
    pair for p in _SIMILI_COPPIE for pair in ((p[0], p[1]), (p[1], p[0]))
)
COSTO_SIMILE = 0.5


def _costo(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return COSTO_SIMILE if (a, b) in _SIMILI else 1.0


def distanza(a: str, b: str) -> float:
    """Distanza di modifica pesata (Damerau ristretta): inserimenti, cancellazioni e
    scambi di lettere adiacenti costano 1, le sostituzioni 1 oppure ``COSTO_SIMILE``
    per i caratteri che si confondono facilmente."""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return float(la or lb)
    prev2: list[float] | None = None
    prev = [float(j) for j in range(lb + 1)]
    for i in range(1, la + 1):
        ca = a[i - 1]
        cur = [float(i)] + [0.0] * lb
        for j in range(1, lb + 1):
            cb = b[j - 1]
            v = prev[j - 1] + _costo(ca, cb)
            if prev[j] + 1.0 < v:
                v = prev[j] + 1.0
            if cur[j - 1] + 1.0 < v:
                v = cur[j - 1] + 1.0
            if prev2 is not None and j > 1 and ca != cb and ca == b[j - 2] and a[i - 2] == cb:
                v = min(v, prev2[j - 2] + 1.0)
            cur[j] = v
        prev2, prev = prev, cur
    return prev[lb]


def _distanza_interna(parola: str, testo: str) -> float:
    """Distanza minima fra ``parola`` e una qualunque porzione contigua di ``testo``."""
    if not parola:
        return 0.0
    if not testo:
        return float(len(parola))
    n = len(testo)
    prev = [0.0] * (n + 1)              # inizio libero nel testo
    for i, cp in enumerate(parola, start=1):
        cur = [float(i)] + [0.0] * n
        for j in range(1, n + 1):
            v = prev[j - 1] + _costo(cp, testo[j - 1])
            if prev[j] + 1.0 < v:
                v = prev[j] + 1.0
            if cur[j - 1] + 1.0 < v:
                v = cur[j - 1] + 1.0
            cur[j] = v
        prev = cur
    return min(prev)                    # fine libera nel testo


def _copertura(parole: Iterable[str], testo: str) -> float:
    """Quanto ogni parola (di almeno 3 caratteri) si ritrova nel testo: minimo su 0..1."""
    valori = [1.0 - _distanza_interna(p, testo) / len(p) for p in parole if len(p) >= 3]
    return max(0.0, min(valori)) if valori else 1.0


def _ordinamenti(parole: list[str]) -> list[list[str]]:
    """Rotazioni dei nomi: «MARIO ROSSI» <-> «ROSSI MARIO», «ROSSI ANNA MARIA» <-> «ANNA MARIA ROSSI»."""
    n = len(parole)
    return [parole[i:] + parole[:i] for i in range(min(n, 6))]


_VOCALI_FINALI = frozenset("AEIO")


def _solo_desinenza(letto: list[str], noto: list[str]) -> bool:
    """True se i nomi differiscono solo per la vocale finale di qualche parola
    (MARIO/MARIA, GIOVANNI/GIOVANNA): di solito sono persone diverse."""
    if len(letto) != len(noto):
        return False
    diverse = [(a, b) for a, b in zip(letto, noto) if a != b]
    return bool(diverse) and all(
        len(a) == len(b) >= 3 and a[:-1] == b[:-1] and a[-1] in _VOCALI_FINALI and b[-1] in _VOCALI_FINALI
        for a, b in diverse
    )


def _numeri(parole: list[str]) -> frozenset[str]:
    return frozenset(p.lstrip("0") or "0" for p in parole if p.isdigit())


@dataclass(frozen=True)
class Somiglianza:
    """Esito del confronto fra una lettura e una voce dell'anagrafica."""

    valore: float                 # 0..1 somiglianza dell'intero testo (miglior ordine dei nomi)
    copertura: float              # 0..1 parola peggio ritrovata (in entrambe le direzioni)
    sospetta: bool = False        # probabilmente un'altra persona/scuola (MARIO/MARIA, «IC 9»/«IC 10»)
    stessi_numeri: bool = True    # i numeri (es. dell'istituto) coincidono


def confronta(letto: str | None, noto: str | None) -> Somiglianza:
    """Somiglianza fra una lettura OCR e un valore noto (vedi docstring del modulo)."""
    ta, tb = normalizza(letto).split(), normalizza(noto).split()
    if not ta or not tb:
        return Somiglianza(0.0, 0.0)
    if sorted(ta) == sorted(tb):
        return Somiglianza(1.0, 1.0)
    a = "".join(ta)
    best_seq, best_order = -1.0, tb
    for order in _ordinamenti(tb):
        seq = difflib.SequenceMatcher(None, a, "".join(order), autojunk=False).ratio()
        if seq > best_seq:
            best_seq, best_order = seq, order
    b = "".join(best_order)
    lev = max(0.0, 1.0 - distanza(a, b) / max(len(a), len(b)))
    valore = round(0.5 * best_seq + 0.5 * lev, 4)
    copertura = round(min(_copertura(tb, a), _copertura(ta, b)), 4)
    na, nb = _numeri(ta), _numeri(tb)
    sospetta = _solo_desinenza(ta, best_order) or bool(na and nb and not na & nb)
    return Somiglianza(valore, copertura, sospetta, na == nb)


# ==========================================================================
# Esiti del riconoscimento
# ==========================================================================

Esito = Literal["identico", "sostituito", "ambiguo", "simile", "nuovo", "dedotto"]


@dataclass
class Candidato:
    valore: str
    somiglianza: float
    copertura: float
    bonus: float = 0.0
    sospetta: bool = False
    stessi_numeri: bool = True

    @property
    def punteggio(self) -> float:
        return self.somiglianza + self.bonus


@dataclass
class Riconoscimento:
    """Decisione presa per un campo dell'intestazione."""

    campo: str
    letto: str | None
    esito: Esito
    valore: str | float | None = None    # valore scelto (None = lettura invariata; ore PEI: numero)
    somiglianza: float = 0.0
    incerto: bool = False
    candidati: list[Candidato] = field(default_factory=list)
    associato: bool = False              # deciso grazie a un'associazione
    era_incerto: bool = False            # il motore aveva segnalato il campo come incerto

    @property
    def riconosciuto(self) -> bool:
        return self.esito in ("identico", "sostituito")


def _pct(x: float) -> str:
    return f"{round(100 * x):d}%"


def _testo(valore: str | float | None) -> str:
    if isinstance(valore, float):
        return f"{valore:g}".replace(".", ",")
    return str(valore)


def _elenco(candidati: list[Candidato]) -> str:
    parti = []
    for c in candidati[:MAX_CANDIDATI_NOTE]:
        extra = ", associazione nota" if c.bonus > 0 else ""
        parti.append(f"«{c.valore}» ({_pct(c.somiglianza)}{extra})")
    return ", ".join(parti)


def _nota(r: Riconoscimento) -> str | None:
    etichetta = ETICHETTE.get(r.campo, r.campo)
    if r.esito == "identico":
        if r.valore is not None and r.letto != r.valore:
            return f"{etichetta} uniformato all'anagrafica: letto «{r.letto}», usato «{r.valore}»."
        if r.era_incerto:
            return f"{etichetta} «{r.valore}» presente nell'anagrafica: lettura confermata."
        return None
    if r.esito == "sostituito":
        dettagli = [f"somiglianza {_pct(r.somiglianza)}"]
        if r.associato:
            dettagli.append("associazione nota")
        if r.incerto:
            dettagli.append("da verificare")
        verbo = "proposto" if r.incerto else "usato"
        return (f"{etichetta} riconosciuto dall'anagrafica ({', '.join(dettagli)}): "
                f"letto «{r.letto}», {verbo} «{r.valore}».")
    if r.esito == "ambiguo":
        return (f"{etichetta} letto «{r.letto}»: più voci simili nell'anagrafica ({_elenco(r.candidati)}), "
                "da verificare.")
    if r.esito == "simile":
        return (f"{etichetta} letto «{r.letto}»: possibile corrispondenza nell'anagrafica "
                f"{_elenco(r.candidati)}, da verificare.")
    if r.esito == "dedotto":
        return (f"{etichetta} non leggibile: proposto «{_testo(r.valore)}» dall'anagrafica "
                "(unico valore associato), da verificare.")
    return None


# ==========================================================================
# Anagrafica
# ==========================================================================

def _adesso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _vuoto() -> dict[str, Any]:
    return {"versione": VERSIONE, "campi": {c: {} for c in CAMPI}, "associazioni": {}, "documenti": {}}


def _chiave_associazione(op_key: str, al_key: str | None) -> str:
    return f"{op_key}|{al_key or ''}"


def _valore_associazione(campo: str, value: Any) -> Any:
    if campo == "ore_pei":
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return v if v == v and 0 < v <= 200 else None   # niente NaN
    return pulisci(value)


class Anagrafica:
    """Anagrafica persistente e thread-safe (file JSON con scrittura atomica)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data = self._carica()

    # ------------------------------------------------------------ persistenza
    def _carica(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return _vuoto()
        except (OSError, ValueError):
            log.warning("Anagrafica illeggibile (%s): ne conservo una copia e riparto da zero", self.path,
                        exc_info=True)
            self._metti_da_parte()
            return _vuoto()
        data = _vuoto()
        if not isinstance(raw, dict):
            self._metti_da_parte()
            return data
        campi = raw.get("campi") if isinstance(raw.get("campi"), dict) else {}
        for campo in CAMPI:
            voci = campi.get(campo) if isinstance(campi.get(campo), dict) else {}
            for key, voce in voci.items():
                if not isinstance(voce, dict) or not valido(voce.get("valore")):
                    continue
                valore = pulisci(voce.get("valore"))
                data["campi"][campo][chiave(valore)] = {
                    "valore": valore,
                    "conteggio": max(0, int(voce.get("conteggio") or 0)),
                    "ultimo_uso": str(voce.get("ultimo_uso") or ""),
                    "manuale": bool(voce.get("manuale")),
                }
        assoc = raw.get("associazioni") if isinstance(raw.get("associazioni"), dict) else {}
        for key, rec in assoc.items():
            if isinstance(rec, dict) and valido(rec.get("operatore")) and isinstance(key, str):
                data["associazioni"][key] = {
                    "operatore": pulisci(rec.get("operatore")),
                    **{c: _valore_associazione(c, rec.get(c)) for c in CAMPI_ASSOCIAZIONE},
                    "conteggio": max(0, int(rec.get("conteggio") or 0)),
                    "ultimo_uso": str(rec.get("ultimo_uso") or ""),
                }
        docs = raw.get("documenti") if isinstance(raw.get("documenti"), dict) else {}
        for doc_id, rec in docs.items():
            if isinstance(rec, dict) and isinstance(rec.get("campi"), dict):
                data["documenti"][str(doc_id)] = {
                    "campi": {c: str(k) for c, k in rec["campi"].items() if c in CAMPI and k},
                    "associazione": rec.get("associazione") if isinstance(rec.get("associazione"), str) else None,
                    "aggiornato": str(rec.get("aggiornato") or ""),
                }
        return data

    def _metti_da_parte(self) -> None:
        try:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            os.replace(self.path, self.path.with_name(f"{self.path.name}.illeggibile-{stamp}"))
        except OSError:
            pass

    def _salva(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self._data, ensure_ascii=False, indent=1, sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------ consultazione
    def valori(self, campo: str) -> list[str]:
        """Valori noti di un campo (ordinati per uso decrescente)."""
        with self._lock:
            voci = list(self._data["campi"].get(campo, {}).values())
        voci.sort(key=lambda v: (-v["conteggio"], v["valore"]))
        return [v["valore"] for v in voci]

    def esporta(self) -> dict[str, Any]:
        """Contenuto per ``GET /api/anagrafica``."""
        with self._lock:
            campi = {
                campo: sorted(
                    ({"valore": v["valore"], "conteggio": v["conteggio"], "ultimo_uso": v["ultimo_uso"] or None,
                      "manuale": v["manuale"]} for v in self._data["campi"][campo].values()),
                    key=lambda v: normalizza(v["valore"]),
                )
                for campo in CAMPI
            }
            associazioni = sorted(
                ({"operatore": r["operatore"], **{c: r.get(c) for c in CAMPI_ASSOCIAZIONE},
                  "conteggio": r["conteggio"], "ultimo_uso": r["ultimo_uso"] or None}
                 for r in self._data["associazioni"].values()),
                key=lambda r: (normalizza(r["operatore"]), normalizza(r.get("alunno"))),
            )
        return {"campi": campi, "associazioni": associazioni}

    def __len__(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._data["campi"].values())

    # ------------------------------------------------------------ modifiche manuali
    def aggiungi(self, campo: str, valore: str) -> dict[str, Any]:
        """Aggiunge (o segna come manuale) una voce. ``ValueError`` con messaggio italiano se non valida."""
        if campo not in CAMPI:
            raise ValueError(f"Campo dell'anagrafica sconosciuto: «{campo}» (ammessi: {', '.join(CAMPI)}).")
        text = pulisci(valore)
        if not text or not valido(text):
            raise ValueError("Valore non valido: indicare un nome di almeno due lettere.")
        if len(text) > MAX_LUNGHEZZA:
            raise ValueError(f"Valore troppo lungo (massimo {MAX_LUNGHEZZA} caratteri).")
        with self._lock:
            voci = self._data["campi"][campo]
            key = chiave(text)
            voce = voci.get(key)
            if voce is None:
                voce = voci[key] = {"valore": text, "conteggio": 0, "ultimo_uso": _adesso(), "manuale": True}
            else:
                voce.update(valore=text, manuale=True, ultimo_uso=_adesso())
            self._salva()
            return dict(voce)

    def rimuovi(self, campo: str, valore: str) -> bool:
        """Elimina una voce errata (e le associazioni che la usano). False se non esiste."""
        if campo not in CAMPI:
            raise ValueError(f"Campo dell'anagrafica sconosciuto: «{campo}» (ammessi: {', '.join(CAMPI)}).")
        key = chiave(valore)
        with self._lock:
            if not key or key not in self._data["campi"][campo]:
                return False
            del self._data["campi"][campo][key]
            assoc = self._data["associazioni"]
            for akey in list(assoc):
                rec = assoc[akey]
                if campo == "operatore" and chiave(rec["operatore"]) == key:
                    del assoc[akey]
                elif campo != "operatore" and chiave(rec.get(campo)) == key:
                    if campo == "alunno":
                        del assoc[akey]
                    else:
                        rec[campo] = None
            for rec in self._data["documenti"].values():
                if rec["campi"].get(campo) == key:
                    del rec["campi"][campo]
                if rec.get("associazione") and rec["associazione"] not in assoc:
                    rec["associazione"] = None
            self._salva()
            return True

    # ------------------------------------------------------------ apprendimento
    def _incrementa(self, campo: str, key: str, valore: str, adesso: str) -> None:
        voci = self._data["campi"][campo]
        voce = voci.get(key)
        if voce is None:
            voci[key] = {"valore": valore, "conteggio": 1, "ultimo_uso": adesso, "manuale": False}
        else:
            voce.update(valore=valore, conteggio=voce["conteggio"] + 1, ultimo_uso=adesso)

    def _decrementa(self, campo: str, key: str) -> None:
        voci = self._data["campi"][campo]
        voce = voci.get(key)
        if voce is None:
            return
        voce["conteggio"] = max(0, voce["conteggio"] - 1)
        if voce["conteggio"] == 0 and not voce["manuale"]:
            del voci[key]

    def impara_documento(self, doc: Document) -> bool:
        """Registra i valori affidabili del documento (vedi docstring del modulo).

        Idempotente: può essere chiamata a ogni salvataggio. True se l'anagrafica è cambiata."""
        if not doc.is_foglio_firma:
            return False
        header = doc.header
        if doc.user_verified:
            fidati = set(CAMPI)
        else:
            fidati = {c for c in CAMPI if f"header.{c}" in doc.user_edited}
        voluti: dict[str, tuple[str, str]] = {}
        for campo in fidati:
            text = pulisci(getattr(header, campo))
            if text and valido(text) and len(text) <= MAX_LUNGHEZZA:
                voluti[campo] = (chiave(text), text)
        assoc_voluta: tuple[str, dict[str, Any]] | None = None
        operatore = voluti.get("operatore")
        if doc.user_verified and operatore:
            valori = {c: _valore_associazione(c, getattr(header, c, None)) for c in CAMPI_ASSOCIAZIONE}
            for c in ("alunno", "istituto", "ente"):
                if c not in voluti:
                    valori[c] = None
            alunno = voluti.get("alunno")
            assoc_voluta = (_chiave_associazione(operatore[0], alunno[0] if alunno else None),
                            {"operatore": operatore[1], **valori})
        with self._lock:
            rec = self._data["documenti"].get(doc.id) or {"campi": {}, "associazione": None, "aggiornato": ""}
            adesso = _adesso()
            changed = False
            for campo in CAMPI:
                old = rec["campi"].get(campo)
                new = voluti.get(campo)
                if new is not None and old == new[0]:
                    voce = self._data["campi"][campo].get(old)
                    if voce is not None and voce["valore"] != new[1]:
                        voce["valore"] = new[1]          # stessa voce scritta diversamente: vale l'ultima
                        changed = True
                    elif voce is None:                    # voce eliminata a mano e poi riconfermata
                        self._incrementa(campo, new[0], new[1], adesso)
                        changed = True
                    continue
                if old is None and new is None:
                    continue
                if old is not None:
                    self._decrementa(campo, old)
                    del rec["campi"][campo]
                if new is not None:
                    self._incrementa(campo, new[0], new[1], adesso)
                    rec["campi"][campo] = new[0]
                changed = True
            changed |= self._aggiorna_associazione(rec, assoc_voluta, adesso)
            if not changed:
                return False
            if rec["campi"] or rec["associazione"]:
                rec["aggiornato"] = adesso
                self._data["documenti"][doc.id] = rec
                self._limita_documenti()
            else:
                self._data["documenti"].pop(doc.id, None)
            self._salva()
            return True

    def _aggiorna_associazione(self, rec: dict[str, Any], voluta: tuple[str, dict[str, Any]] | None,
                               adesso: str) -> bool:
        assoc = self._data["associazioni"]
        old = rec.get("associazione")
        if voluta is not None and old == voluta[0] and old in assoc:
            current = assoc[old]
            if all(current.get(k) == v for k, v in voluta[1].items()):
                return False
            current.update(voluta[1])            # valori più recenti (es. cambio di istituto o di ore PEI)
            current["ultimo_uso"] = adesso
            return True
        if old is None and voluta is None:
            return False
        if old is not None and old in assoc and old != (voluta[0] if voluta else None):
            assoc[old]["conteggio"] = max(0, assoc[old]["conteggio"] - 1)
            if assoc[old]["conteggio"] == 0:
                del assoc[old]
        rec["associazione"] = None
        if voluta is not None:
            key, valori = voluta
            current = assoc.get(key)
            if current is None:
                assoc[key] = {**valori, "conteggio": 1, "ultimo_uso": adesso}
            else:
                current.update(valori)
                current["conteggio"] += 1
                current["ultimo_uso"] = adesso
            rec["associazione"] = key
        return True

    def _limita_documenti(self) -> None:
        docs = self._data["documenti"]
        if len(docs) <= MAX_DOCUMENTI:
            return
        for doc_id in sorted(docs, key=lambda d: docs[d].get("aggiornato", ""))[: len(docs) - MAX_DOCUMENTI]:
            del docs[doc_id]

    def scollega_documento(self, doc_id: str) -> None:
        """Il documento è stato eliminato o riletto: quanto appreso resta, ma non è più
        legato al documento (una sua lettura successiva non lo ritira)."""
        with self._lock:
            if self._data["documenti"].pop(doc_id, None) is not None:
                self._salva()

    def scollega_tutti(self) -> None:
        with self._lock:
            if self._data["documenti"]:
                self._data["documenti"] = {}
                self._salva()

    # ------------------------------------------------------------ riconoscimento
    def _istantanea(self) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
        with self._lock:
            voci = {c: {k: v["valore"] for k, v in self._data["campi"][c].items()} for c in CAMPI}
            assoc = [dict(r) for r in self._data["associazioni"].values()]
        return voci, assoc

    def correggi(self, header: Header) -> list[Riconoscimento]:
        """Riconduce i campi anagrafici dell'intestazione alle voci note (modifica ``header``).

        Restituisce le decisioni prese; le note per l'utente si ottengono con :func:`note`."""
        voci, assoc = self._istantanea()
        if not any(voci.values()):
            return []
        decisioni: dict[str, Riconoscimento] = {}

        def risolto(campo: str) -> str | None:
            r = decisioni.get(campo)
            return chiave(str(r.valore)) if r is not None and r.riconosciuto and r.valore else None

        def bonus(campo: str) -> dict[str, float]:
            op, al = risolto("operatore"), risolto("alunno")
            out: dict[str, float] = {}
            for rec in assoc:
                rec_op, rec_al = chiave(rec["operatore"]), chiave(rec.get("alunno"))
                if campo == "operatore":
                    peso = BONUS_OPERATORE if al and rec_al == al else 0.0
                    target = rec["operatore"]
                else:
                    if op and rec_op == op:
                        peso = BONUS_COPPIA if (al and rec_al == al and campo != "alunno") else BONUS_OPERATORE
                    elif al and rec_al == al and campo != "alunno":
                        peso = BONUS_OPERATORE
                    else:
                        peso = 0.0
                    target = rec.get(campo)
                if peso and target:
                    k = chiave(target)
                    out[k] = max(out.get(k, 0.0), peso)
            return out

        def abbina(campo: str) -> Riconoscimento | None:
            return self._abbina(campo, getattr(header, campo), voci[campo], bonus(campo))

        for campo in ("operatore", "alunno"):
            if (r := abbina(campo)) is not None:
                decisioni[campo] = r
        # operatore non riconosciuto ma alunno sì: seconda prova con le associazioni dell'alunno
        if risolto("alunno") and not risolto("operatore"):
            if (r := abbina("operatore")) is not None and r.riconosciuto:
                decisioni["operatore"] = r
        for campo in ("istituto", "ente"):
            if (r := abbina(campo)) is not None:
                decisioni[campo] = r
        for r in decisioni.values():
            r.era_incerto = r.campo in header.incerti
            _applica(header, r)
        for r in self._deduci(header, assoc, risolto("operatore"), risolto("alunno")):
            decisioni[r.campo] = r
            _applica(header, r)
        ordine = ("operatore", "alunno", "istituto", "ente", "lotto", "municipalita", "ore_pei")
        return [decisioni[c] for c in ordine if c in decisioni]

    def _abbina(self, campo: str, letto: str | None, voci: dict[str, str],
                bonus: dict[str, float]) -> Riconoscimento | None:
        text = pulisci(letto)
        if not text or not voci or not valido(text):
            return None
        key = chiave(text)
        if key in voci:
            return Riconoscimento(campo, text, "identico", valore=voci[key], somiglianza=1.0)
        soglie = SOGLIE[campo]
        candidati: list[Candidato] = []
        for k, valore in voci.items():
            s = confronta(text, valore)
            candidati.append(Candidato(valore, s.valore, s.copertura, bonus.get(k, 0.0), s.sospetta,
                                       s.stessi_numeri))
        candidati.sort(key=lambda c: (-c.punteggio, -c.somiglianza, c.valore))
        best = candidati[0]
        second = candidati[1].punteggio if len(candidati) > 1 else 0.0
        vicini = [c for c in candidati
                  if c.somiglianza >= soglie.suggerimento or c.punteggio >= soglie.sostituzione
                  or (c.bonus > 0 and c.punteggio >= soglie.suggerimento)]
        copertura = soglie.copertura_associata if best.bonus > 0 else soglie.copertura
        if (best.punteggio >= soglie.sostituzione and best.copertura >= copertura
                and best.punteggio - second >= MARGINE and not best.sospetta):
            certo = (best.somiglianza >= CERTEZZA and best.copertura >= COPERTURA_CERTA
                     and best.stessi_numeri)
            return Riconoscimento(campo, text, "sostituito", valore=best.valore, somiglianza=best.somiglianza,
                                  incerto=not certo, candidati=vicini, associato=best.bonus > 0)
        if not vicini:
            return Riconoscimento(campo, text, "nuovo")
        ambiguo = len(vicini) > 1 and vicini[1].punteggio >= best.punteggio - MARGINE
        return Riconoscimento(campo, text, "ambiguo" if ambiguo else "simile", somiglianza=best.somiglianza,
                              incerto=True, candidati=vicini)

    def _deduci(self, header: Header, assoc: list[dict[str, Any]], op: str | None,
                al: str | None) -> list[Riconoscimento]:
        """Campi illeggibili proposti da un'associazione univoca con l'operatore/alunno riconosciuti."""
        out: list[Riconoscimento] = []
        if not op and not al:
            return out
        for campo in ("operatore", "alunno", "istituto", "ente", "lotto", "municipalita", "ore_pei"):
            if getattr(header, campo) is not None or campo not in header.illeggibili:
                continue
            if campo == "operatore":
                pertinenti = [r for r in assoc if al and chiave(r.get("alunno")) == al]
            elif campo == "alunno":
                pertinenti = [r for r in assoc if op and chiave(r["operatore"]) == op]
            else:
                pertinenti = [r for r in assoc
                              if (not op or chiave(r["operatore"]) == op) and (not al or chiave(r.get("alunno")) == al)]
            valori = {}
            for r in pertinenti:
                v = r.get(campo)
                if v is not None:
                    valori[chiave(str(v)) if isinstance(v, str) else v] = v
            if len(valori) != 1 or len(pertinenti) != sum(1 for r in pertinenti if r.get(campo) is not None):
                continue
            (valore,) = valori.values()
            out.append(Riconoscimento(campo, None, "dedotto", valore=valore, incerto=True, associato=True))
        return out


def _applica(header: Header, r: Riconoscimento) -> None:
    campo = r.campo
    if r.esito in ("identico", "sostituito", "dedotto") and r.valore is not None:
        setattr(header, campo, r.valore)
        if campo in header.illeggibili:
            header.illeggibili.remove(campo)
    if r.incerto:
        if campo not in header.incerti:
            header.incerti.append(campo)
    elif r.riconosciuto and campo in header.incerti:
        header.incerti.remove(campo)


def note(decisioni: Iterable[Riconoscimento]) -> list[str]:
    """Frasi in italiano che spiegano le decisioni (per ``ocr_notes``)."""
    return [n for r in decisioni if (n := _nota(r))]


def applica_al_risultato(result: ExtractionResult, anagrafica: Anagrafica) -> list[Riconoscimento]:
    """Applica l'anagrafica al risultato di un motore OCR (intestazione e ``ocr_notes``)."""
    if not result.is_foglio_firma:
        return []
    decisioni = anagrafica.correggi(result.header)
    righe = note(decisioni)
    if righe:
        result.ocr_notes = _unisci(result.ocr_notes, righe)
    return decisioni


def _unisci(esistenti: str | None, righe: list[str]) -> str:
    parti = [esistenti.strip()] if esistenti and esistenti.strip() else []
    return " ".join(parti + righe)


# ==========================================================================
# Istanza condivisa (una per cartella dei dati)
# ==========================================================================

_ISTANZE: dict[Path, Anagrafica] = {}
_ISTANZE_LOCK = threading.Lock()


def percorso_predefinito() -> Path:
    return config.data_dir() / FILE_NAME


def anagrafica_predefinita() -> Anagrafica:
    """Anagrafica della cartella dei dati corrente (la stessa istanza per server e coda)."""
    path = percorso_predefinito().resolve()
    with _ISTANZE_LOCK:
        inst = _ISTANZE.get(path)
        if inst is None:
            inst = _ISTANZE[path] = Anagrafica(path)
        return inst


__all__ = [
    "CAMPI",
    "CAMPI_ASSOCIAZIONE",
    "CERTEZZA",
    "MARGINE",
    "SOGLIE",
    "Anagrafica",
    "Candidato",
    "Riconoscimento",
    "Somiglianza",
    "anagrafica_predefinita",
    "applica_al_risultato",
    "chiave",
    "confronta",
    "distanza",
    "normalizza",
    "note",
    "percorso_predefinito",
    "pulisci",
]
