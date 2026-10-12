"""Test del motore OCR locale (TrOCR + analisi dell'inchiostro).

Quasi tutti i test girano senza modello: il riconoscitore TrOCR e' sostituito da
uno finto che "legge" i valori veri del foglio sintetico (``tests/synthetic.py``)
o letture costruite a mano. I test con il modello reale richiedono il modello
gia' scaricato nella cache indicata da ``SIRIO_HF_CACHE`` (nessun download
durante i test); quello di accuratezza richiede anche ``SIRIO_SAMPLE_PDF`` e
``SIRIO_SAMPLE_TRUTH``.
"""

from __future__ import annotations

import dataclasses
import errno
import functools
import json
import math
import os
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest

from sirio.engines import local_engine as le
from sirio.engines.base import EngineError, PageInput
from sirio.models import DAY_FIELDS
from sirio.vision.grid import TableGrid, detect_grid, template_grid
from sirio.vision.preprocess import normalize_page
from tests.synthetic import make_synthetic_sheet, photograph

TIME_FIELDS = ("prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita")


# --------------------------------------------------------------------------
# Riconoscitore finto
# --------------------------------------------------------------------------

def _fmt_hours(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}".replace(".", ",")


class FakeRecognizer:
    """Restituisce le letture "vere" del foglio sintetico; registra le richieste."""

    def __init__(self, truth: dict, overrides: dict[str, le.Reading] | None = None):
        self.truth = truth
        self.overrides = overrides or {}
        self.keys: list[str] = []
        self.calls = 0

    def _reading(self, key: str) -> le.Reading:
        if key in self.overrides:
            return self.overrides[key]
        hdr = self.truth["header"]
        parts = key.split(".")
        if parts[0] == "rows":
            row = self.truth["rows"][int(parts[1]) - 1]
            f = parts[2]
            v = row[f]
            if v is None:                      # cella vuota nella verita': nessuna scrittura leggibile
                return le.Reading([], "", -math.inf, 0)
            if f in TIME_FIELDS:
                h, m = v.split(":")
                return le.Reading([(v, -0.05)], f"{int(h)}:{m}", -0.08, 4)
            if f == "ore_dichiarate":
                return le.Reading([(float(v), -0.05)], _fmt_hours(float(v)), -0.06, 2)
            if f == "note":
                canon = le.notes_lexicon().value(v)
                cands = [(canon, -0.3)] if canon else []
                return le.Reading(cands, v.title(), -0.3, 6)
        if parts[0] == "header":
            name = parts[1]
            if name in ("operatore", "alunno", "ente", "istituto"):
                return le.Reading([], str(hdr[name]).title(), -0.4, 6)
            if name in ("lotto", "municipalita"):
                return le.Reading([(str(hdr[name]), -0.05)], str(hdr[name]), -0.05, 1)
            if name == "mese_anno":
                text = f"{hdr['mese']:02d}/{hdr['anno']}"
                return le.Reading([((hdr["mese"], hdr["anno"]), -0.1)], text, -0.1, 4)
            if name == "ore_pei":
                return le.Reading([(float(hdr["ore_pei"]), -0.05)], _fmt_hours(hdr["ore_pei"]), -0.05, 1)
        if key == "footer.totale":
            tot = float(hdr["totale_mensile_dichiarato"])
            return le.Reading([(tot, -0.05)], _fmt_hours(tot), -0.05, 3)
        return le.Reading([], "", -math.inf, 0)

    def read(self, requests, progress=None):
        self.calls += 1
        out = []
        for i, req in enumerate(requests):
            assert req.image.ndim == 2 and req.image.dtype == np.uint8 and req.image.size > 0
            self.keys.append(req.key)
            out.append(self._reading(req.key))
            if progress is not None:
                progress(i + 1, len(requests))
        return out


@functools.cache
def _synthetic(seed: int = 0, skew: float = 0.0, dpi: int = 200) -> tuple[np.ndarray, TableGrid, dict]:
    img, truth = make_synthetic_sheet(seed=seed, skew_deg=skew, dpi=dpi)
    page = normalize_page(img)
    return page, detect_grid(page), truth


def _page(seed: int = 0, skew: float = 0.0, dpi: int = 200) -> tuple[PageInput, dict]:
    img, grid, truth = _synthetic(seed, skew, dpi)
    return PageInput(image=img, grid=grid, source_file="sintetico.pdf", page=1), truth


# --------------------------------------------------------------------------
# Lessici e letture libere
# --------------------------------------------------------------------------

def test_time_lexicon_forms_and_tolerances():
    lex = le.time_lexicon()
    assert lex.value("8:00") == "08:00"
    assert lex.value("08.00") == "08:00"
    assert lex.value("8,30") == "08:30"
    assert lex.value("1100") == "11:00"
    assert lex.value("11'00") == "11:00"
    assert lex.value("8") == "08:00"
    # confusioni tipiche della lettura delle cifre e segni spuri ai bordi
    assert lex.value("U.00") == "11:00"
    assert lex.value("Il:00") == "11:00"
    assert lex.value("8.000") == "08:00"
    assert lex.value(".8:00.") == "08:00"
    # fuori dai vincoli del modulo
    assert lex.value("5:00") is None          # prima delle 06:00
    assert lex.value("21:00") is None         # dopo le 20:00
    assert lex.value("8:10") == "08:10"        # minuti non a quarti d'ora (orari sfalsati)
    assert lex.value("8:07") is None          # minuti non multipli di 5
    assert lex.value("Atlas") is None
    assert lex.is_prefix("8:") and lex.is_prefix("1") and not lex.is_prefix("8:7")
    assert lex.signature("8.00") == lex.signature("8:00") == lex.signature("8:OO")


def test_time_lexicon_real_world_surface_forms():
    """Scritture viste sui fogli di operatori diversi: virgola, nessun separatore,
    "h" come separatore o in coda."""
    lex = le.time_lexicon()
    for text, value in (("10,30", "10:30"), ("9,00", "09:00"), ("800", "08:00"), ("1100", "11:00"),
                        ("1300", "13:00"), ("1000", "10:00"), ("830", "08:30"), ("8:30h", "08:30"),
                        ("8h30", "08:30"), ("8h", "08:00"), ("13:30", "13:30"), ("12,30", "12:30")):
        assert lex.value(text) == value, text


def test_hours_and_other_lexicons():
    ore = le.hours_lexicon()
    assert ore.value("3") == 3.0
    assert ore.value("1,5") == 1.5 and ore.value("1.5") == 1.5 and ore.value("I,5") == 1.5
    assert ore.value("3h") == 3.0
    assert ore.value("8,5") is None and ore.value("9") is None
    # ore e minuti, quarti d'ora, zero
    for text, value in (("2h30'", 2.5), ("1h15'", 1.25), ("1h45'", 1.75), ("4h30'", 4.5), ("2h", 2.0),
                        ("2,5", 2.5), ("2:30", 2.5), ("2,30", 2.5), ("30'", 0.5), ("0", 0.0), ("1:15", 1.25)):
        assert ore.value(text) == value, text
    # quarti in forma decimale esclusi: "2,5" letto "2.25" non deve diventare un valore ammesso
    assert ore.value("2.25") is None and ore.value("1,75") is None
    assert le.total_lexicon().value("49,5") == 49.5
    assert le.pei_lexicon().value("15") == 15.0
    assert le.small_int_lexicon(10, roman=True).value("II") == "2"
    my = le.month_year_lexicon((2025, 2026))
    assert my.value("02/2026") == (2, 2026)
    assert my.value("2/26") == (2, 2026)
    assert my.value("02,/2026") == (2, 2026)
    assert my.value("febbraio 2026") == (2, 2026)
    assert le.notes_lexicon().value("Ponte di carnevale") == "PONTE DI CARNEVALE"
    assert le.notes_lexicon().value("L. 104") == "104"
    # la nota resta come e' scritta (nessuna frase "canonica" diversa)
    assert le.notes_lexicon().value("scuola chiusa") == "SCUOLA CHIUSA"


def test_free_value():
    t = le.time_lexicon()
    assert le.free_value("8.000", t, "time") == "08:00"
    assert le.free_value("8:10", t, "time") == "08:10"      # orario valido fuori lessico
    assert le.free_value("Atlas .", t, "time") is None
    assert le.free_value("3 3", le.hours_lexicon(), "hours") == 3.0
    assert le.free_value("", t, "time") is None


def test_evidence_coverage():
    t = le.time_lexicon()
    good = le.evidence(le.Reading([("08:00", -0.2), ("08:30", -3.0)], "8.00", -0.3, 3), t, "time")
    assert good.top == "08:00" and good.coverage > 0.99 and good.p("08:00") > 0.9
    bad = le.evidence(le.Reading([("08:00", -15.0)], "Atlas .", -0.7, 3), t, "time")
    assert bad.coverage < 1e-3
    assert le.evidence(None, t, "time").options == []


# --------------------------------------------------------------------------
# Riconciliazione di riga
# --------------------------------------------------------------------------

def test_row_consistency_prefers_coherent_values():
    ok = {"prog_entrata": "08:00", "prog_uscita": "11:00", "eff_entrata": "08:00", "eff_uscita": "11:00",
          "ore_dichiarate": 3.0}
    wrong = dict(ok, eff_uscita="14:00")
    assert le.row_consistency(ok) > le.row_consistency(wrong) + 4
    # uscita prima dell'entrata: fortemente penalizzata
    assert le.row_consistency({"eff_entrata": "11:00", "eff_uscita": "08:00"}) <= le.W_ORDER
    # con assenza dell'alunno le ore riconosciute possono essere inferiori all'orario
    partial = {"eff_entrata": "08:00", "eff_uscita": "11:00", "ore_dichiarate": 1.5}
    assert le.row_consistency(partial, assenza_alunno=True) > le.row_consistency(partial)


def test_reconcile_row_uses_hours_to_disambiguate():
    lg = math.log
    options = {
        "eff_entrata": [("08:00", lg(0.95)), ("08:30", lg(0.05))],
        "eff_uscita": [("14:00", lg(0.55)), ("11:00", lg(0.45))],   # grafia ambigua 11/14
        "ore_dichiarate": [(3.0, lg(0.97)), (5.0, lg(0.03))],
    }
    chosen = le.reconcile_row(options)
    assert chosen["eff_uscita"][0] == "11:00"
    assert chosen["eff_entrata"][0] == "08:00"
    assert le.reconcile_row({}) == {}


def test_unreliable_grid_flags_every_value():
    full = {f: le.TESTO for f in le.TEXT_FIELDS} | {"firma": le.FIRMA}
    plan = _plan_with({2: full})
    readings = _row_readings(2)
    res = le.assemble(plan, readings, _grid(reliable=False))
    r = res.rows[1]
    assert r.eff_uscita == "11:00"
    assert set(le.TEXT_FIELDS) | {"firma"} <= set(r.incerti)
    res = le.assemble(plan, readings, _grid())
    assert not res.rows[1].incerti


def test_time_prior_is_soft_and_plausibility_based():
    # minuti: preferenza debole (una lettura chiara di 10:30 non diventa 10:00)
    assert le._time_prior("10:00") - le._time_prior("10:30") < 1.0
    # fuori dall'orario scolastico: penalita' forte (12,30 letto 19,30)
    assert le._time_prior("12:30") - le._time_prior("19:30") >= 2.0
    assert le._time_prior("06:00") < le._time_prior("07:00") < le._time_prior("08:00")


def test_row_doubts_flag_inconsistency_without_changing_values():
    v = {"prog_entrata": "09:00", "prog_uscita": "13:30", "eff_entrata": "09:00", "eff_uscita": "13:30",
         "ore_dichiarate": 2.0}
    d = le.row_doubts(dict(v))
    # ore diverse dall'orario effettivo: tutta la riga e' da verificare
    assert set(d) == set(v) and all("incoerente" in why for why in d.values())
    ok = dict(v, ore_dichiarate=4.5)
    assert le.row_doubts(ok) == {}
    # alunno assente: ore inferiori all'orario ammesse
    assert le.row_doubts(dict(v, ore_dichiarate=1.0), assenza_alunno=True) == {}
    # orari fuori dalla fascia scolastica e ore giornaliere eccessive
    late = le.row_doubts({"prog_entrata": "11:30", "prog_uscita": "19:30"})
    assert "fuori_orario" in late["prog_uscita"]
    assert "incoerente" in le.row_doubts({"eff_entrata": "08:00", "eff_uscita": "16:00"})["eff_uscita"]
    # senza orari effettivi: ore oltre l'orario programmato
    d = le.row_doubts({"prog_entrata": "08:00", "prog_uscita": "10:00", "ore_dichiarate": 3.0})
    assert "incoerente" in d["ore_dichiarate"]
    # un solo orario effettivo
    assert "incompleto" in le.row_doubts({"eff_entrata": "08:00"})["eff_entrata"]


def test_free_reading_digits_and_short_forms():
    r = le.Reading([("10:00", -4.8), ("10:30", -8.2)], "10/3", -1.3, 3, {"10:00": "10,"})
    assert le._free_digits_conflict("eff_entrata", "10:00", r)          # 1 0 3 -> 10:30
    assert not le._free_digits_conflict("eff_entrata", "10:30", r)
    assert not le._free_digits_conflict("eff_entrata", "16:00",
                                        le.Reading([], "16.60", -1.0, 3))   # 16:60 non e' un orario
    short = le.Reading([("10:00", -0.8), ("10:30", -5.5)], "10.", -0.8, 2, {"10:00": "10."})
    assert le._short_form("eff_uscita", "10:00", short, aspect=1.6)
    assert not le._short_form("eff_uscita", "10:00", short, aspect=1.0)     # ritaglio stretto: "10" e' plausibile
    assert not le._short_form("eff_uscita", "10:00", r, aspect=1.6)         # cifre in piu': ci pensa il confronto
    assert le._hour_conflict("eff_uscita", "11:00", le.Reading([], "13 w.", -0.3, 3))
    assert not le._hour_conflict("eff_entrata", "09:00", le.Reading([], "933", -1.0, 2))


def test_notes_are_conservative():
    def note(text, lp=-0.5, n=4, cands=()):
        return le._note_field(le.Reading(list(cands), text, lp, n), aspect=2.0)

    # code di firme, tratti, timbri: letture brevi fuori vocabolario non sono note
    for garbage in ("DE", "49", "WE", "R", "no"):
        fld = note(garbage, -0.2, 1)
        assert fld.value is None and not fld.illeggibile, garbage
    assert note("104", -0.1, 1).value == "104"
    # testo in parole note e letto con fiducia: sicuro
    fld = note("Scuola chiusa", -1.0, 4)
    assert fld.value == "SCUOLA CHIUSA" and not fld.incerto
    # parole sconosciute (allucinazioni del modello): riportate ma da verificare
    fld = note("The bodies so", -1.0, 4)
    assert fld.value and fld.incerto
    # abbreviazioni di parole note
    fld = note("Ponte carn.", -1.0, 4)
    assert fld.value == "PONTE CARN." and not fld.incerto
    # frase del lessico confermata dalla lettura libera
    fld = note("Ponte di Carnevale", -0.9, 6, [("PONTE DI CARNEVALE", -1.0)])
    assert fld.value == "PONTE DI CARNEVALE" and not fld.incerto
    # frase del lessico NON confermata: si riporta la lettura, da verificare
    fld = note("Cent of", -1.5, 2, [("CARNEVALE", -37.0)])
    assert fld.value != "CARNEVALE" and fld.incerto


# --------------------------------------------------------------------------
# Assemblaggio: incerti, illeggibili, valori dedotti
# --------------------------------------------------------------------------

def _grid(reliable: bool = True) -> TableGrid:
    """Griglia del modulo standard considerata rilevata dalle linee reali (o no)."""
    return dataclasses.replace(template_grid(1654, 2339), detected=reliable, score=0.9 if reliable else 0.05)


def _plan_with(cells: dict[int, dict[str, str]]) -> le.PagePlan:
    plan = le.PagePlan()
    for g in range(1, 32):
        st = {f: le.VUOTA for f in (*le.TEXT_FIELDS, "assenza_alunno", "assenza_operatore", "firma", "note")}
        st.update(cells.get(g, {}))
        plan.cells[g] = st
    return plan


def _time(v: str, lp: float = -0.1) -> le.Reading:
    h, m = v.split(":")
    return le.Reading([(v, lp)], f"{int(h)}:{m}", lp - 0.05, 4)


def test_assemble_flags_uncertain_and_illegible_cells():
    full = {f: le.TESTO for f in le.TEXT_FIELDS} | {"firma": le.FIRMA}
    plan = _plan_with({2: full, 3: full, 4: {"prog_entrata": le.TESTO}})
    readings = {
        "rows.2.prog_entrata": _time("08:00"), "rows.2.prog_uscita": _time("11:00"),
        "rows.2.eff_entrata": _time("08:00"),
        # uscita effettiva illeggibile per il modello, ma determinata da entrata + ore
        "rows.2.eff_uscita": le.Reading([("08:00", -16.0), ("06:00", -17.0)], "Atlas .", -0.7, 3),
        "rows.2.ore_dichiarate": le.Reading([(3.0, -0.1)], "3", -0.1, 1),
        # riga 3: lettura ambigua 11:00 / 14:00 risolta dalle ore
        "rows.3.prog_entrata": _time("08:00"), "rows.3.prog_uscita": _time("11:00"),
        "rows.3.eff_entrata": _time("08:00"),
        "rows.3.eff_uscita": le.Reading([("14:00", -0.6), ("11:00", -0.8)], "14:00", -0.6, 4),
        "rows.3.ore_dichiarate": le.Reading([(3.0, -0.1)], "3", -0.1, 1),
        # riga 4: scrittura presente ma senza alcun contesto -> illeggibile
        "rows.4.prog_entrata": le.Reading([("19:30", -18.0)], "xiss .", -1.0, 3),
    }
    grid = _grid()
    res = le.assemble(plan, readings, grid)
    r2, r3, r4 = res.rows[1], res.rows[2], res.rows[3]
    assert (r2.prog_entrata, r2.prog_uscita, r2.eff_entrata, r2.ore_dichiarate) == ("08:00", "11:00", "08:00", 3.0)
    assert r2.eff_uscita == "11:00" and "eff_uscita" in r2.incerti and not r2.illeggibili
    assert "dedotti" in (res.ocr_notes or "")
    assert r3.eff_uscita == "11:00" and "eff_uscita" in r3.incerti
    assert r3.firma and not r3.illeggibili
    assert r4.prog_entrata is None and r4.illeggibili == ["prog_entrata"]
    assert res.engine == "locale" and 0.0 <= (res.confidence or 0) <= 1.0
    for row in res.rows:
        assert set(row.incerti) <= set(DAY_FIELDS) and set(row.illeggibili) <= set(DAY_FIELDS)


def test_assemble_column_pattern_flags_unusual_values():
    full = {f: le.TESTO for f in TIME_FIELDS}
    plan = _plan_with({g: full for g in range(2, 8)})
    readings = {}
    for g in range(2, 8):
        readings |= {f"rows.{g}.prog_entrata": _time("08:00"), f"rows.{g}.prog_uscita": _time("11:00"),
                     f"rows.{g}.eff_entrata": _time("08:00"), f"rows.{g}.eff_uscita": _time("11:00")}
    # il giorno 7 sembra 09:00 ma l'08:00 e' fra le alternative: va segnalato
    readings["rows.7.prog_entrata"] = le.Reading([("09:00", -0.3), ("08:00", -1.8)], "9:00", -0.3, 3)
    res = le.assemble(plan, readings, _grid())
    assert "prog_entrata" in res.rows[6].incerti
    assert all(not res.rows[g - 1].incerti for g in range(2, 7))


def _row_readings(g: int, a: str = "08:00", b: str = "11:00", ore: float = 3.0) -> dict[str, le.Reading]:
    """Riga completa letta con sicurezza (programmato = effettivo, ore coerenti)."""
    return {f"rows.{g}.prog_entrata": _time(a), f"rows.{g}.prog_uscita": _time(b),
            f"rows.{g}.eff_entrata": _time(a), f"rows.{g}.eff_uscita": _time(b),
            f"rows.{g}.ore_dichiarate": le.Reading([(ore, -0.05)], _fmt_hours(ore), -0.06, 2)}


def _faint(value: str, alt: str, free: str = "thios .") -> le.Reading:
    """Scrittura poco leggibile: fra i valori ammessi la lettura preferisce nettamente
    ``value`` (circa 90%), ma la lettura libera e' una parola molto piu' probabile."""
    return le.Reading([(value, -4.0), (alt, -6.3)], free, -0.3, 3)


FULL_ROW = {f: le.TESTO for f in le.TEXT_FIELDS} | {"firma": le.FIRMA}


def test_assemble_confirms_faint_reading_implied_by_solid_cells():
    plan = _plan_with({g: FULL_ROW for g in (2, 3, 4)})
    readings = {**_row_readings(2), **_row_readings(3), **_row_readings(4)}
    # giorno 2: uscita effettiva poco leggibile ma letta come 11:00, che e' anche quanto
    # impongono l'uscita programmata e entrata + ore (celle sicure) -> confermata
    readings["rows.2.eff_uscita"] = _faint("11:00", "11:30")
    # giorno 3: stessa scrittura poco leggibile, ma la lettura preferisce 14:00: il valore
    # scelto (11:00) viene solo dal contesto e resta da verificare
    readings["rows.3.eff_uscita"] = _faint("14:00", "11:00")
    # giorno 4: due celle poco leggibili che si confermerebbero solo a vicenda
    readings["rows.4.eff_entrata"] = _faint("08:00", "08:30")
    readings["rows.4.eff_uscita"] = _faint("11:00", "11:30")
    readings["rows.4.prog_entrata"] = _faint("08:00", "09:00")
    readings["rows.4.prog_uscita"] = _faint("11:00", "12:00")
    res = le.assemble(plan, readings, _grid())
    r2, r3, r4 = res.rows[1], res.rows[2], res.rows[3]
    assert r2.eff_uscita == "11:00" and "eff_uscita" not in r2.incerti
    assert r2.confidenza["eff_uscita"] >= le.CONF_CONFERMA
    assert "confermati dalla coerenza" in (res.ocr_notes or "")
    assert r3.eff_uscita == "11:00" and "eff_uscita" in r3.incerti
    assert {"prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita"} <= set(r4.incerti)
    assert (r4.prog_entrata, r4.prog_uscita, r4.eff_entrata, r4.eff_uscita) == ("08:00", "11:00", "08:00", "11:00")


def test_assemble_confirmation_respects_hours_inconsistency():
    plan = _plan_with({2: FULL_ROW})
    readings = _row_readings(2)
    readings["rows.2.ore_dichiarate"] = le.Reading([(2.5, -0.05)], "2,5", -0.06, 3)
    # uscita poco leggibile letta come 11:00 = uscita programmata, ma entrata 8:00 + 2,5 ore
    # (celle sicure) darebbero 10:30: la relazione orari/ore e' violata -> resta incerta
    readings["rows.2.eff_uscita"] = _faint("11:00", "10:30")
    res = le.assemble(plan, readings, _grid())
    assert "eff_uscita" in res.rows[1].incerti


def test_assemble_flags_programmed_effective_mismatch_with_plausible_misreading():
    plan = _plan_with({2: FULL_ROW, 3: FULL_ROW})
    readings = {**_row_readings(2, "09:00", "13:00", 4.0), **_row_readings(3, "08:00", "11:00", 3.0)}
    # "13:00" letto con sicurezza come 13:30 nell'orario programmato (13:00 fra le
    # alternative), mentre effettivo e ore dicono 13:00: va segnalato
    readings["rows.2.prog_uscita"] = le.Reading([("13:30", -0.1), ("13:00", -3.0)], "13.30", -0.12, 4)
    # ritardo vero: entrata effettiva 8:30, giustificata da uscita - ore; l'8:00
    # programmato non ha 8:30 fra le letture -> nessuna segnalazione
    readings["rows.3.eff_entrata"] = le.Reading([("08:30", -0.05), ("08:00", -3.5)], "8:30", -0.06, 4)
    readings["rows.3.ore_dichiarate"] = le.Reading([(2.5, -0.05)], "2,5", -0.06, 3)
    readings["rows.3.prog_entrata"] = le.Reading([("08:00", -0.05), ("08:30", -6.0)], "8:00", -0.06, 4)
    res = le.assemble(plan, readings, _grid())
    r2, r3 = res.rows[1], res.rows[2]
    assert r2.prog_uscita == "13:30" and "prog_uscita" in r2.incerti
    assert r2.eff_uscita == "13:00" and "eff_uscita" not in r2.incerti
    assert (r3.prog_entrata, r3.eff_entrata, r3.ore_dichiarate) == ("08:00", "08:30", 2.5)
    assert not r3.incerti


def test_programmed_time_confirmed_by_weekly_schedule():
    prog_only = {"prog_entrata": le.TESTO, "prog_uscita": le.TESTO, "note": le.TESTO}
    days = (2, 9, 16, 23)          # stesso giorno della settimana
    readings = {}
    for g in days:
        readings[f"rows.{g}.prog_entrata"] = _time("08:00")
        readings[f"rows.{g}.prog_uscita"] = _time("11:00")
        readings[f"rows.{g}.note"] = le.Reading([("FESTIVO", -0.2)], "Festivo", -0.2, 3)
    # lettura incerta (copertura ~0,27: "8.80"), ma fra gli orari ammessi preferisce 08:00
    readings["rows.16.prog_entrata"] = le.Reading([("08:00", -1.6), ("09:00", -3.8)], "8.80", -0.6, 4)
    plan = _plan_with({g: prog_only for g in days})
    res = le.assemble(plan, readings, _grid())
    assert res.rows[15].prog_entrata == "08:00" and "prog_entrata" not in res.rows[15].incerti
    # un giorno della stessa settimana con un orario diverso: nessuna conferma
    readings["rows.9.prog_entrata"] = _time("09:00")
    res = le.assemble(plan, readings, _grid())
    assert "prog_entrata" in res.rows[15].incerti


def test_free_reading_outside_form_rules_does_not_block_confirmation():
    plan = _plan_with({2: FULL_ROW, 3: FULL_ROW})
    readings = {**_row_readings(2), **_row_readings(3)}
    # lettura libera "11.10" (minuti non a quarti d'ora: cifra letta male) -> confermabile
    readings["rows.2.eff_uscita"] = le.Reading([("11:00", -0.2), ("11:30", -2.5)], "11.10", -0.15, 4)
    # lettura libera "11.30" (orario ammesso diverso) -> resta da verificare
    readings["rows.3.eff_uscita"] = le.Reading([("11:00", -0.2), ("11:30", -2.5)], "11.30", -0.15, 4)
    res = le.assemble(plan, readings, _grid())
    assert res.rows[1].eff_uscita == "11:00" and "eff_uscita" not in res.rows[1].incerti
    assert res.rows[2].eff_uscita == "11:00" and "eff_uscita" in res.rows[2].incerti


def test_notes_snap_to_recurring_phrases_and_neighbours():
    plan = _plan_with({16: {"note": le.TESTO}, 17: {"note": le.TESTO}, 18: {"note": le.TESTO}})
    readings = {
        "rows.16.note": le.Reading([("PONTE DI CARNEVALE", -1.0)], "Ponte di Carnevale", -0.9, 6),
        "rows.17.note": le.Reading([], "Ponte di Casarevali", -2.5, 7),       # simile a una formula
        "rows.18.note": le.Reading([], "Barrie in Glenlevale", -6.0, 7),      # simile alla nota vicina
    }
    res = le.assemble(plan, readings, _grid())
    assert res.rows[15].note == "PONTE DI CARNEVALE" and "note" not in res.rows[15].incerti
    assert res.rows[16].note == "PONTE DI CARNEVALE" and "note" in res.rows[16].incerti
    assert res.rows[17].note == "PONTE DI CARNEVALE" and "note" in res.rows[17].incerti


def test_month_year_uses_calendar_consistency():
    # giorni compilati: feriali di febbraio 2026 (in febbraio 2025 cadrebbero di domenica)
    worked = [d for d in range(1, 29) if date(2026, 2, d).weekday() < 5]
    ev = le.evidence(le.Reading([((2, 2025), -1.0), ((2, 2026), -1.6)], "02/2025", -1.0, 4), None, "month")
    fld = le._choose_month(ev, worked, today=date(2026, 10, 9))
    assert fld.value == (2, 2026) and fld.incerto
    assert le.calendar_penalty((2, 2026), worked) == 0.0
    assert le.calendar_penalty((2, 2025), worked) < 0
    assert le.calendar_penalty((2, 2026), [30]) < 0          # 30 febbraio inesistente


def test_month_name_read_clearly_and_year_fixed_by_calendar_is_not_flagged():
    """"Febbrano '96": il mese si legge, l'anno no; i 20 giorni feriali compilati cadono
    di domenica in ogni altro anno passato (il 2027 e' futuro): mese/anno sicuri."""
    worked = [d for d in range(1, 29) if date(2026, 2, d).weekday() < 5]
    today = date(2026, 10, 9)
    reading = le.Reading([((2, 2024), -0.5), ((2, 2025), -1.5), ((2, 2026), -3.0)], "Febbrano '96", -2.0, 6)
    fld = le._choose_month(le.evidence(reading, None, "month"), worked, today=today, free_text=reading.free_text)
    assert fld.value == (2, 2026) and not fld.incerto
    # pochi giorni compilati, nome del mese non riconoscibile o anno scritto diverso: resta da verificare
    for days, text in ((worked[:4], "Febbrano '96"), (worked, "fessario"), (worked, "Marzo 2026"),
                       (worked, "Febbraio 2025")):
        r = le.Reading(reading.candidates, text, -2.0, 6)
        fld = le._choose_month(le.evidence(r, None, "month"), days, today=today, free_text=text)
        assert fld.incerto, text


# --------------------------------------------------------------------------
# Immagini: ritagli puliti, timbro, foglio sintetico completo
# --------------------------------------------------------------------------

def test_clean_crop_removes_grid_lines():
    import cv2

    img = np.full((300, 400), 245, np.uint8)
    for y in (100, 200):
        cv2.line(img, (0, y), (399, y), 20, 3)
    for x in (100, 300):
        cv2.line(img, (x, 0), (x, 299), 20, 3)
    box = (100, 100, 300, 200)
    assert le.clean_crop(img, box, 100.0, rows=(100, 200), cols=(100, 300)) is None
    cv2.putText(img, "8:00", (130, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 30, 4)
    crop = le.clean_crop(img, box, 100.0, rows=(100, 200), cols=(100, 300))
    assert crop is not None and crop.dtype == np.uint8
    h, w = crop.shape
    assert w < 200 and h < 100                     # solo la scrittura, senza la cella intera
    assert crop[0].min() > 200 and crop[:, 0].min() > 200   # bordi bianchi: nessuna linea rimasta
    assert crop.min() < 80                          # la scritta c'e'


def test_plan_page_classifies_synthetic_sheet():
    img, grid, truth = _synthetic(0)
    plan = le.plan_page(img, grid)
    for t in truth["rows"]:
        st = plan.cells[t["giorno"]]
        for f in le.TEXT_FIELDS:
            expected = le.TESTO if t[f] is not None else (
                le.TRATTINO if f.startswith("eff") and t["trattino_effettivo"] else le.VUOTA)
            assert st[f] == expected, (t["giorno"], f)
        assert (st["assenza_alunno"] == le.CROCETTA) == t["assenza_alunno"]
        assert (st["assenza_operatore"] == le.CROCETTA) == t["assenza_operatore"]
        assert (st["firma"] == le.FIRMA) == t["firma"]
        assert (st["note"] == le.TESTO) == bool(t["note"])
    assert all(plan.header_ink.values())
    assert plan.totale_ink and plan.firma_coordinatore and plan.timbro_referente
    assert not plan.data_ink


def test_extract_on_synthetic_sheet_with_fake_recognizer():
    page, truth = _page(0)
    fake = FakeRecognizer(truth)
    steps: list[tuple[float, str]] = []
    engine = le.LocalEngine(recognizer=fake)
    assert engine.name == "locale" and engine.is_available()[0]
    res = engine.extract(page, lambda f, m: steps.append((f, m)))
    assert res.is_foglio_firma and res.engine == "locale"
    # si leggono solo le celle con scrittura (niente OCR su celle vuote o trattini)
    expected_keys = {f"rows.{t['giorno']}.{f}" for t in truth["rows"] for f in (*le.TEXT_FIELDS, "note")
                     if t[f] is not None}
    assert {k for k in fake.keys if k.startswith("rows.")} == expected_keys
    for t, r in zip(truth["rows"], res.rows):
        for f in ("prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita", "ore_dichiarate",
                  "assenza_alunno", "assenza_operatore", "firma", "note", "trattino_effettivo"):
            assert getattr(r, f) == t[f], (t["giorno"], f, getattr(r, f), t[f])
        assert not r.incerti and not r.illeggibili, (t["giorno"], r.incerti, r.illeggibili)
    h, th = res.header, truth["header"]
    for f in ("lotto", "municipalita", "ente", "istituto", "operatore", "alunno", "mese", "anno", "ore_pei",
              "totale_mensile_dichiarato", "anno_scolastico", "firma_coordinatore", "timbro_referente"):
        assert getattr(h, f) == th[f], f
    assert not h.illeggibili
    # avanzamento: frazioni crescenti in [0, 1], messaggi in italiano, chiusura a 1
    fracs = [f for f, _ in steps]
    assert fracs == sorted(fracs) and fracs[-1] == 1.0 and all(0.0 <= f <= 1.0 for f in fracs)
    assert any("Lettura della scrittura a mano" in m for _, m in steps)


def _degraded_page(**kw) -> tuple[PageInput, dict]:
    img, truth = make_synthetic_sheet(**kw)
    page = normalize_page(img)
    return PageInput(image=page, grid=detect_grid(page), source_file="sintetico.pdf", page=1), truth


def test_extract_ignores_crossed_out_rows_and_signature_overflow():
    """Righe non usate barrate con lunghe diagonali, firme che sconfinano nelle
    note, linea a penna sugli orari effettivi, orari con la virgola: nessun valore
    inventato, nessuna nota dalle code delle firme, "non svolta" riconosciuta."""
    data = {"rows": [{"giorno": 18, "prog_entrata": "08:00", "prog_uscita": "11:00", "linea_effettivo": True,
                      "trattino_effettivo": True, "note": "SCIOPERO"}]}
    page, truth = _degraded_page(data=data, seed=5, time_style="comma", crossed_rows=((29, 31),),
                                 long_signatures=True)
    fake = FakeRecognizer(truth)
    res = le.LocalEngine(recognizer=fake).extract(page, lambda *_: None)
    read_rows = {int(k.split(".")[1]) for k in fake.keys if k.startswith("rows.")}
    assert not read_rows & {29, 30, 31}, sorted(read_rows)
    for r in res.rows[28:31]:
        assert not r.has_content() or set(r.incerti) | set(r.illeggibili), r
    # nessuna nota letta dove non c'e' (le firme lunghe finiscono nella colonna Note)
    noted = {int(k.split(".")[1]) for k in fake.keys if k.endswith(".note")}
    assert noted == {t["giorno"] for t in truth["rows"] if t["note"]}, sorted(noted)
    r18 = res.rows[17]
    assert r18.trattino_effettivo and r18.eff_entrata is None and r18.eff_uscita is None
    assert not r18.assenza_alunno and not r18.assenza_operatore and not r18.firma
    for t, r in zip(truth["rows"], res.rows):
        for f in (*le.TEXT_FIELDS, "assenza_alunno", "assenza_operatore", "firma"):
            if getattr(r, f) != t[f]:
                assert f in r.incerti or f in r.illeggibili, (t["giorno"], f, getattr(r, f), t[f])


def test_slash_in_unworked_row_is_not_a_value():
    page, truth = _page(0)
    img = page.image.copy()
    g = 7            # sabato: riga vuota
    x0, y0, x1, y1 = page.grid.cell(g, "prog_entrata")
    import cv2
    cv2.line(img, (x0 + (x1 - x0) // 3, y1 - (y1 - y0) // 4), (x0 + 2 * (x1 - x0) // 3, y0 + (y1 - y0) // 4),
             (40, 40, 40), 3, cv2.LINE_AA)
    plan = le.plan_page(img, page.grid)
    assert plan.cells[g]["prog_entrata"] == le.NULLO
    assert not any(r.key.startswith(f"rows.{g}.") for r in plan.requests)


def test_extract_rejects_pages_without_table():
    blank = np.full((2339, 1654, 3), 250, np.uint8)
    grid = detect_grid(blank)
    fake = FakeRecognizer({"header": {}, "rows": []})
    res = le.LocalEngine(recognizer=fake).extract(PageInput(blank, grid, "vuoto.png", 1), lambda *_: None)
    assert not res.is_foglio_firma and fake.calls == 0
    notes = res.ocr_notes or ""
    # il messaggio vale anche per un foglio firma fotografato male: spiega come rimediare
    assert "non sembra un foglio firma" in notes and "foto" in notes and "scanner" in notes


def test_extract_on_photo_taken_at_an_angle():
    """Foto di sbieco su un tavolo: la prospettiva e' corretta e ogni valore
    finisce nel giorno giusto (prima: dati spostati di alcuni giorni, non segnalati)."""
    img, truth = make_synthetic_sheet(seed=0)
    page = normalize_page(photograph(img, keystone=0.06))
    grid = detect_grid(page)
    fake = FakeRecognizer(truth)
    res = le.LocalEngine(recognizer=fake).extract(PageInput(page, grid, "foto.jpg", 1), lambda *_: None)
    assert res.is_foglio_firma
    for t, r in zip(truth["rows"], res.rows):
        for f in ("prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita", "ore_dichiarate",
                  "assenza_alunno", "assenza_operatore", "firma", "note", "trattino_effettivo"):
            assert getattr(r, f) == t[f], (t["giorno"], f, getattr(r, f), t[f])
    assert res.header.totale_mensile_dichiarato == truth["header"]["totale_mensile_dichiarato"]
    assert (res.header.mese, res.header.anno) == (truth["header"]["mese"], truth["header"]["anno"])


def test_extract_with_stamp_over_notes_column():
    import cv2

    img, grid, _truth = _synthetic(0)
    img = img.copy()
    rh = grid.row_height
    cx = int((grid.col_x[9] + grid.col_x[10]) / 2)
    cy = int(grid.row_y[29])
    cv2.circle(img, (cx, cy), int(1.4 * rh), (110, 60, 90), 3)
    cv2.circle(img, (cx, cy), int(1.0 * rh), (110, 60, 90), 3)
    cv2.putText(img, "SCUOLA", (cx - int(0.8 * rh), cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 60, 90), 2)
    plan = le.plan_page(img, grid)
    assert le.stamp_boxes(le.to_gray(img), grid)
    for g in (28, 29, 30, 31):
        assert plan.cells[g]["note"] in (le.VUOTA, le.TIMBRO)
    assert not any(r.key.endswith(".note") and int(r.key.split(".")[1]) >= 28 for r in plan.requests)


# --------------------------------------------------------------------------
# Disponibilita' e gestione del modello
# --------------------------------------------------------------------------

def _fake_snapshot(root: Path, model: str = le.DEFAULT_MODEL) -> Path:
    repo = root / ("models--" + model.replace("/", "--"))
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    for name in ("config.json", "preprocessor_config.json", "model.safetensors", "vocab.json", "merges.txt"):
        (snap / name).write_text("{}", encoding="utf-8")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123", encoding="utf-8")
    return snap


def test_is_available_without_offline_components(monkeypatch, tmp_path):
    real_find_spec = le.importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in ("torch", "transformers"):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(le, "_IMPORT_STATE", {})
    monkeypatch.setattr(le.importlib.util, "find_spec", fake_find_spec)
    ok, msg = le.LocalEngine(cache_dir=tmp_path).is_available()
    assert not ok and msg.startswith("Componenti offline non installati")
    with pytest.raises(EngineError, match="Componenti offline"):
        le.LocalEngine(cache_dir=tmp_path).extract(_page(0)[0], lambda *_: None)


def test_is_available_reports_model_download_state(monkeypatch, tmp_path):
    monkeypatch.setattr(le, "_IMPORT_STATE", {"ok": True})
    ok, msg = le.LocalEngine(cache_dir=tmp_path).is_available()
    assert ok and "verrà scaricato" in msg and "1,3 GB" in msg
    snap = _fake_snapshot(tmp_path)
    assert le.local_model_path(le.DEFAULT_MODEL, tmp_path) == snap
    ok, msg = le.LocalEngine(cache_dir=tmp_path).is_available()
    assert ok and "già scaricato" in msg
    # cache con la struttura di HF_HOME (sottocartella "hub")
    other = tmp_path / "home"
    snap2 = _fake_snapshot(other / "hub")
    assert le.local_model_path(le.DEFAULT_MODEL, other) == snap2
    # un modello incompleto (senza pesi) non conta come scaricato
    (snap / "model.safetensors").unlink()
    assert le.local_model_path(le.DEFAULT_MODEL, tmp_path) is None


def test_default_cache_dir_is_under_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SIRIO_DATA_DIR", str(tmp_path))
    assert le.LocalEngine().cache_dir == tmp_path / "modelli"


def test_download_errors_are_explained_in_italian(monkeypatch, tmp_path):
    hub = pytest.importorskip("huggingface_hub")

    def offline(*_a, **_k):
        raise ConnectionError("Failed to resolve 'huggingface.co'")

    monkeypatch.setattr(hub.HfApi, "model_info", offline)
    with pytest.raises(EngineError, match="nessuna connessione a Internet"):
        le.download_model(le.DEFAULT_MODEL, tmp_path)
    err = le._download_error(OSError(errno.ENOSPC, "No space left on device"), le.DEFAULT_MODEL, 1340)
    assert "Spazio su disco insufficiente" in str(err)


def test_extract_reports_download_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(le, "_IMPORT_STATE", {"ok": True})

    def fail(*_a, **_k):
        raise EngineError("Impossibile scaricare il modello per il riconoscimento offline: nessuna connessione.")

    monkeypatch.setattr(le, "download_model", fail)
    with pytest.raises(EngineError, match="Impossibile scaricare"):
        le.LocalEngine(cache_dir=tmp_path).extract(_page(0)[0], lambda *_: None)


def test_download_checks_free_disk_space_first(monkeypatch, tmp_path):
    """Disco quasi pieno: messaggio chiaro prima di scaricare (durante il download
    hf_xet produce solo errori generici in inglese)."""
    hub = pytest.importorskip("huggingface_hub")
    from types import SimpleNamespace

    files = [SimpleNamespace(rfilename="model.safetensors", size=1_333_384_464),
             SimpleNamespace(rfilename="config.json", size=4_000)]
    monkeypatch.setattr(hub.HfApi, "model_info", lambda *_a, **_k: SimpleNamespace(siblings=files, sha="abc"))

    def no_download(*_a, **_k):
        raise AssertionError("download avviato senza spazio sufficiente")

    monkeypatch.setattr(hub, "snapshot_download", no_download)
    monkeypatch.setattr(le.shutil, "disk_usage", lambda _p: SimpleNamespace(total=10**11, used=10**11, free=30e6))
    with pytest.raises(EngineError, match="Spazio su disco insufficiente") as info:
        le.download_model(le.DEFAULT_MODEL, tmp_path)
    assert "1,7 GB" in str(info.value)                     # 1,33 GB + 10% + margine


def test_disk_full_during_download_is_recognized(monkeypatch, tmp_path):
    xet = RuntimeError("Task error: File reconstruction error: Internal Writer Error: Background writer channel "
                       "closed")
    from types import SimpleNamespace

    monkeypatch.setattr(le.shutil, "disk_usage", lambda _p: SimpleNamespace(total=10**11, used=10**11, free=1e6))
    err = le._download_error(xet, le.DEFAULT_MODEL, 1340, tmp_path, 1.2e9)
    assert "Spazio su disco insufficiente" in str(err) and "Writer" not in str(err)
    for exc in (OSError(errno.ENOSPC, "No space left on device"),
                RuntimeError("IO error: No space left on device (os error 28)")):
        assert "Spazio su disco insufficiente" in str(le._download_error(exc, le.DEFAULT_MODEL, 1340))
    # con spazio sufficiente lo stesso errore non e' attribuito al disco
    monkeypatch.setattr(le.shutil, "disk_usage", lambda _p: SimpleNamespace(total=10**12, used=0, free=5e11))
    assert "Spazio su disco" not in str(le._download_error(xet, le.DEFAULT_MODEL, 1340, tmp_path, 1.2e9))


@pytest.mark.parametrize("exc", [
    MemoryError(),
    OSError(errno.ENOMEM, "Cannot allocate memory"),
    RuntimeError("[enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: "
                 "you tried to allocate 10000000000000 bytes. Error code 12 (Cannot allocate memory)"),
    OSError("unable to mmap 1333384464 bytes from file </modelli/model.safetensors>: Cannot allocate memory "
            "(os error 12)"),
    RuntimeError("Il file di paging è troppo piccolo per completare l'operazione. (os error 1455)"),
], ids=["MemoryError", "ENOMEM", "torch", "mmap", "windows"])
def test_out_of_memory_is_recognized(exc):
    assert le._is_oom(exc)
    wrapped = ValueError("caricamento non riuscito")
    wrapped.__cause__ = exc
    assert le._is_oom(wrapped)


def test_damaged_files_are_not_out_of_memory():
    for exc in (RuntimeError("Error while deserializing header: incomplete metadata, file not fully covered"),
                ValueError("Error while initializing BPE: EOF while parsing a value at line 1 column 0"),
                OSError(errno.ENOENT, "No such file or directory")):
        assert not le._is_oom(exc)


class _RaisingRecognizer:
    def __init__(self, exc: BaseException):
        self.exc = exc

    def read(self, requests, progress=None):
        raise self.exc


def test_out_of_memory_while_reading_has_clear_message():
    page, _truth = _page(0)
    oom = RuntimeError("[enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory")
    with pytest.raises(EngineError) as info:
        le.LocalEngine(recognizer=_RaisingRecognizer(oom)).extract(page, lambda *_: None)
    assert str(info.value) == le.OOM_MSG and "alloc_cpu" not in str(info.value)


def test_out_of_memory_while_loading_keeps_the_model(monkeypatch, tmp_path):
    """Memoria insufficiente al caricamento: nessun invito a eliminare (e riscaricare) il modello."""
    transformers = pytest.importorskip("transformers")
    snap = _fake_snapshot(tmp_path)

    def oom(*_a, **_k):
        raise OSError(errno.ENOMEM, "Cannot allocate memory (os error 12)")

    monkeypatch.setattr(transformers.TrOCRProcessor, "from_pretrained", lambda *_a, **_k: object())
    monkeypatch.setattr(transformers.VisionEncoderDecoderModel, "from_pretrained", oom)
    with pytest.raises(EngineError) as info:
        le._load_recognizer(snap, le.DEFAULT_MODEL, "cpu")
    assert str(info.value) == le.OOM_MSG and not isinstance(info.value, le.ModelDamagedError)

    def no_download(*_a, **_k):
        raise AssertionError("nessun nuovo download per mancanza di memoria")

    monkeypatch.setattr(le, "_IMPORT_STATE", {"ok": True})
    monkeypatch.setattr(le, "download_model", no_download)
    with pytest.raises(EngineError, match="Memoria insufficiente"):
        le.LocalEngine(cache_dir=tmp_path).extract(_page(0)[0], lambda *_: None)
    assert snap.is_dir()


def test_unreadable_model_files_are_reported_as_damaged(tmp_path):
    pytest.importorskip("transformers")
    snap = _fake_snapshot(tmp_path)              # file presenti ma non validi
    with pytest.raises(le.ModelDamagedError) as info:
        le._load_recognizer(snap, le.DEFAULT_MODEL, "cpu")
    assert "{}" not in str(info.value)


@pytest.fixture
def _session_replacements(monkeypatch):
    monkeypatch.setattr(le, "_REPLACED", set())
    monkeypatch.setattr(le, "_MODELS", {})
    monkeypatch.setattr(le, "_IMPORT_STATE", {"ok": True})


def test_damaged_model_is_downloaded_again(monkeypatch, tmp_path, _session_replacements):
    """Copia locale danneggiata: eliminata e scaricata di nuovo automaticamente
    (prima: ogni documento falliva con l'errore tecnico finche' l'utente non
    trovava ed eliminava la cartella)."""
    snap = _fake_snapshot(tmp_path)
    repo = snap.parent.parent
    (snap / "model.safetensors").chmod(0o444)               # come nella cache di huggingface_hub
    page, truth = _page(0)
    fake = FakeRecognizer(truth)
    loads: list[Path] = []
    downloads: list[str | None] = []

    def load(path, *_a):
        loads.append(path)
        if len(loads) == 1:
            raise le.ModelDamagedError("Impossibile caricare il modello di riconoscimento offline.")
        return fake

    def download(name, cache, progress, message=None):
        assert not repo.exists()                            # la copia danneggiata e' stata eliminata
        downloads.append(message)
        progress(0.5, f"{message}: 600 MB di 1.334 MB…")
        return _fake_snapshot(cache)

    monkeypatch.setattr(le, "_load_recognizer", load)
    monkeypatch.setattr(le, "download_model", download)
    steps: list[str] = []
    res = le.LocalEngine(cache_dir=tmp_path).extract(page, lambda _f, m: steps.append(m))
    assert res.is_foglio_firma and fake.calls == 1
    assert len(loads) == 2 and downloads and "danneggiato" in downloads[0]
    assert any("Modello danneggiato: nuovo download" in m for m in steps)


def test_damaged_model_is_downloaded_again_only_once(monkeypatch, tmp_path, _session_replacements):
    _fake_snapshot(tmp_path)
    downloads: list[int] = []

    def load(*_a):
        raise le.ModelDamagedError("Impossibile caricare il modello di riconoscimento offline.")

    def download(name, cache, progress, message=None):
        downloads.append(1)
        return _fake_snapshot(cache)

    monkeypatch.setattr(le, "_load_recognizer", load)
    monkeypatch.setattr(le, "download_model", download)
    engine = le.LocalEngine(cache_dir=tmp_path)
    for _ in range(3):                                      # tre documenti in coda
        with pytest.raises(EngineError) as info:
            engine.extract(_page(0)[0], lambda *_: None)
        msg = str(info.value)
        assert "danneggiato" in msg and "models--microsoft--trocr-base-handwritten" in msg
        assert "Error while" not in msg
    assert len(downloads) == 1


def test_damaged_model_in_user_folder_is_not_deleted(monkeypatch, tmp_path, _session_replacements):
    folder = tmp_path / "mio-modello"
    folder.mkdir()
    for name in ("config.json", "preprocessor_config.json", "model.safetensors", "vocab.json"):
        (folder / name).write_text("{}", encoding="utf-8")

    def load(*_a):
        raise le.ModelDamagedError("Impossibile caricare il modello di riconoscimento offline.")

    monkeypatch.setattr(le, "_load_recognizer", load)
    with pytest.raises(EngineError, match="mio-modello"):
        le.LocalEngine(model_name=str(folder), cache_dir=tmp_path / "cache").extract(_page(0)[0], lambda *_: None)
    assert (folder / "model.safetensors").exists()


def test_windows_dll_failure_is_explained(monkeypatch):
    real_import = le.importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise OSError('[WinError 126] Impossibile trovare il modulo specificato. Error loading '
                          '"C:\\Sirio\\.venv\\Lib\\site-packages\\torch\\lib\\c10.dll" or one of its '
                          'dependencies.')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(le, "_IMPORT_STATE", {})
    monkeypatch.setattr(le.importlib, "import_module", fake_import)
    for _ in range(2):                                      # anche dal risultato memorizzato
        ok, msg = le.LocalEngine().is_available()
        assert not ok and "Visual C++ Redistributable" in msg and "programma di avvio" not in msg


def test_low_resolution_page_is_refused():
    """Righe di circa 19 pixel (scansione a ~70 dpi): TrOCR inventerebbe valori nelle celle vuote."""
    import cv2

    img, truth = make_synthetic_sheet(seed=0)
    small = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3), interpolation=cv2.INTER_AREA)
    page = normalize_page(small)
    grid = detect_grid(page)
    assert grid.detected and grid.row_height < le.MIN_ROW_HEIGHT
    fake = FakeRecognizer(truth)
    with pytest.raises(EngineError, match="almeno 150 dpi"):
        le.LocalEngine(recognizer=fake).extract(PageInput(page, grid, "piccola.jpg", 1), lambda *_: None)
    assert fake.calls == 0


def test_get_engine_builds_local_engine():
    from sirio.config import Settings
    from sirio.engines.base import get_engine

    engine = get_engine(Settings(engine="locale", local_model="microsoft/trocr-base-handwritten"))
    assert isinstance(engine, le.LocalEngine) and engine.model_name == "microsoft/trocr-base-handwritten"


# --------------------------------------------------------------------------
# Test con il modello reale (solo se gia' presente nella cache)
# --------------------------------------------------------------------------

def _hf_cache() -> Path | None:
    raw = os.environ.get("SIRIO_HF_CACHE")
    if not raw:
        return None
    path = Path(raw)
    return path if le.local_model_path(le.DEFAULT_MODEL, path) is not None else None


needs_model = pytest.mark.skipif(_hf_cache() is None,
                                 reason="modello TrOCR non presente nella cache SIRIO_HF_CACHE")


@functools.cache
def _real_recognizer() -> le.TrOCRRecognizer:
    return le._load_recognizer(le.local_model_path(le.DEFAULT_MODEL, _hf_cache()), le.DEFAULT_MODEL, "cpu")


@needs_model
def test_batched_decoding_matches_reference_search():
    """Le ricerche fatte avanzare insieme per piu' ritagli danno gli stessi punteggi
    della decodifica di riferimento (un ritaglio alla volta, senza cache)."""
    pytest.importorskip("torch")
    import torch

    img, grid, _truth = _synthetic(0)
    plan = le.plan_page(img, grid)
    reqs = [r for r in plan.requests if r.key.endswith(("prog_entrata", "ore_dichiarate"))][:3]
    rec = _real_recognizer()
    with torch.inference_mode():
        enc = rec.encode([r.image for r in reqs])
        cross = rec._cross_kv(enc)
        n = len(reqs)
        start = torch.full((n,), rec.start_id, dtype=torch.long)
        lps, cache = rec._step(start, 0, None, cross, torch.arange(n))
        jobs = [rec._new_job(i, r, True) for i, r in enumerate(reqs)]
        jobs += [rec._new_job(i, r, False) for i, r in enumerate(reqs)]
        rec._run(jobs, cache, lps, cross)
        for i, req in enumerate(reqs):
            lex_job, free_job = jobs[i], jobs[n + i]
            slow = rec._search_slow(enc[i:i + 1], lex_job.width, lex_job.max_steps, req.lexicon, None)
            slow_scores = {(key, ids): score for key, ids, score in slow}
            common = [(f, slow_scores[(f[0], f[1])]) for f in lex_job.finished if (f[0], f[1]) in slow_scores]
            assert common, req.key
            for (_key, _ids, score), ref in common:
                assert abs(score - ref) < 1e-3
            best_fast = max(lex_job.finished, key=lambda f: f[2])
            best_slow = max(slow, key=lambda f: f[2])
            assert req.lexicon.complete(best_fast[0]) == req.lexicon.complete(best_slow[0])
            # lettura libera "golosa": identica alla ricerca di riferimento con un solo fascio
            slow_free = rec._search_slow(enc[i:i + 1], 1, req.max_tokens, None, req.charset)
            best = max(free_job.finished, key=lambda f: rec._rank(f, True))
            ref = max(slow_free, key=lambda f: rec._rank(f, True))
            assert best[1] == ref[1] and abs(best[2] - ref[2]) < 1e-3


@needs_model
def test_fast_path_self_check_covers_cached_steps(monkeypatch):
    """Il controllo della decodifica a lotti confronta piu' passi (cache della
    self-attention, posizioni) e non solo il primo."""
    pytest.importorskip("torch")
    import torch

    img, grid, _truth = _synthetic(0)
    req = next(r for r in le.plan_page(img, grid).requests if r.key.endswith("prog_entrata"))
    rec = _real_recognizer()
    with torch.inference_mode():
        enc = rec.encode([req.image])
        assert rec._verify_fast_path(enc)
        original = rec._step

        def shifted(tokens, pos, cache, cross, row_crop):
            # posizioni sbagliate dal secondo passo (es. cambio di API di transformers)
            return original(tokens, pos + 1 if pos else pos, cache, cross, row_crop)

        monkeypatch.setattr(rec, "_step", shifted)
        assert not rec._verify_fast_path(enc)


@needs_model
def test_truncated_weights_are_reported_as_damaged(tmp_path):
    """Pesi troncati (download interrotto, disco danneggiato): errore di "danno", non tecnico."""
    real = le.local_model_path(le.DEFAULT_MODEL, _hf_cache())
    snap = tmp_path / "modello"
    snap.mkdir()
    for f in real.iterdir():
        if f.name != "model.safetensors":
            (snap / f.name).write_bytes(f.read_bytes())
    with (real / "model.safetensors").open("rb") as src:
        (snap / "model.safetensors").write_bytes(src.read(5_000_000))
    with pytest.raises(le.ModelDamagedError) as info:
        le._load_recognizer(snap, le.DEFAULT_MODEL, "cpu")
    assert "deserializ" not in str(info.value).lower()


@needs_model
def test_readings_do_not_depend_on_batch_composition():
    """Ordine dei risultati preservato e lettura di un ritaglio indipendente dagli
    altri ritagli letti insieme (pesi in float32, nessuna quantizzazione per lotto)."""
    img, grid, _truth = _synthetic(0)
    plan = le.plan_page(img, grid)
    reqs = [r for r in plan.requests if r.key.startswith("rows.")][:4]
    reqs += [r for r in plan.requests if r.key == "header.lotto"]
    rec = _real_recognizer()
    together = rec.read(reqs)
    alone = [rec.read([r])[0] for r in reversed(reqs)][::-1]
    for req, a, b in zip(reqs, together, alone):
        assert [v for v, _ in a.candidates[:3]] == [v for v, _ in b.candidates[:3]], req.key
        for (_va, la), (_vb, lb) in zip(a.candidates[:3], b.candidates[:3]):
            assert abs(la - lb) < 1e-3, req.key
        assert a.free_text == b.free_text, req.key


# Soglie minime sul foglio reale (misurate: vedi docstring del modulo del motore;
# un errore in piu' di quelli misurati e' tollerato per le differenze di arrotondamento
# fra CPU diverse). Il motore non e' piu' tarato su questo solo foglio (preferenze
# "effettivo = programmato" e "orari tondi" ridotte a spareggi, controlli che segnalano
# invece di correggere): qui legge 2 orari e un'ora in meno di prima, ma su fogli di
# altri operatori sbaglia molto meno, e ogni errore resta segnalato.
MIN_ACCURACY = {
    "orari": 0.91,           # misurato 67/72
    "ore": 0.88,             # misurato 16/17
    "assenze": 1.0,
    "firme": 1.0,
    "trattini": 1.0,
    "note": 0.90,            # misurato 31/31
    "intestazione": 0.85,    # misurato 10/10
}
# Orari corretti ma segnalati come incerti (lavoro di verifica inutile per l'utente):
# misurati 14 su 67.
MAX_FALSI_INCERTI_ORARI = 16


@needs_model
def test_accuracy_on_real_sample():
    pdf_path = os.environ.get("SIRIO_SAMPLE_PDF")
    truth_path = os.environ.get("SIRIO_SAMPLE_TRUTH")
    if not pdf_path or not truth_path:
        pytest.skip("SIRIO_SAMPLE_PDF / SIRIO_SAMPLE_TRUTH non impostate")
    from sirio.pdf_io import iter_pages

    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    _, img = next(iter_pages(Path(pdf_path).read_bytes(), Path(pdf_path).name))
    page = normalize_page(img)
    res = le.LocalEngine(cache_dir=_hf_cache()).extract(
        PageInput(page, detect_grid(page), Path(pdf_path).name, 1), lambda *_: None)
    assert res.is_foglio_firma

    score: dict[str, list[int]] = {k: [0, 0] for k in MIN_ACCURACY}
    unflagged: list[tuple] = []          # errori su orari/ore non segnalati
    false_doubts = 0                     # orari corretti segnalati come incerti

    def add(cat: str, ok: bool) -> None:
        score[cat][0] += int(ok)
        score[cat][1] += 1

    for t, r in zip(truth["rows"], res.rows):
        doubtful = set(r.incerti) | set(r.illeggibili)
        for f in (*TIME_FIELDS, "ore_dichiarate"):
            if t[f] is None and getattr(r, f) is None and f not in r.illeggibili:
                continue
            ok = getattr(r, f) == t[f]
            add("ore" if f == "ore_dichiarate" else "orari", ok)
            if not ok and f not in doubtful:
                unflagged.append((t["giorno"], f, getattr(r, f), t[f]))
            false_doubts += int(ok and f in doubtful and f != "ore_dichiarate")
        for f in ("assenza_alunno", "assenza_operatore"):
            add("assenze", getattr(r, f) == t[f])
        add("firme", r.firma == t["firma"])
        add("trattini", r.trattino_effettivo == t["trattino_effettivo"])
        add("note", (r.note or "").upper() == (t["note"] or "").upper())
    for f in ("lotto", "municipalita", "mese", "anno", "ore_pei", "firma_coordinatore", "timbro_referente",
              "totale_mensile_dichiarato", "anno_scolastico", "sostituzione"):
        add("intestazione", getattr(res.header, f) == truth["header"].get(f))
    for cat, (ok, n) in score.items():
        assert n and ok / n >= MIN_ACCURACY[cat], (cat, ok, n)
    # ogni errore residuo su orari e ore deve essere segnalato (incerto o illeggibile)
    assert not unflagged, unflagged
    assert false_doubts <= MAX_FALSI_INCERTI_ORARI, false_doubts


# --------------------------------------------------------------------------
# Segni a mano (due rilevatori), avvisi sulla griglia, secondo parere
# --------------------------------------------------------------------------

def test_cross_inside_crossed_out_row_is_flagged_not_trusted():
    """Una crocetta (o un pezzo di barratura che le somiglia) in una riga barrata non
    diventa mai un'assenza sicura: resta segnalata."""
    import cv2
    page, _truth = _degraded_page(seed=6, crossed_rows=((28, 31),))
    img = page.image.copy()
    x0, y0, x1, y1 = page.grid.cell(29, "assenza_alunno")
    m = (x1 - x0) // 4
    cv2.line(img, (x0 + m, y0 + m // 2), (x1 - m, y1 - m // 2), (30, 30, 30), 3, cv2.LINE_AA)
    cv2.line(img, (x0 + m, y1 - m // 2), (x1 - m, y0 + m // 2), (30, 30, 30), 3, cv2.LINE_AA)
    plan = le.plan_page(img, page.grid)
    marks = le.PageMarks(le.to_gray(img), img, page.grid)
    assert {29, 30, 31} <= marks.struck_rows
    assert plan.cells[29]["assenza_alunno"] != le.CROCETTA
    assert any("Righe barrate" in n for n in plan.notes)


def test_unreliable_grid_produces_page_warning():
    full = {f: le.TESTO for f in le.TEXT_FIELDS} | {"firma": le.FIRMA}
    plan = _plan_with({2: full})
    res = le.assemble(plan, _row_readings(2), _grid(reliable=False))
    assert [a.codice for a in res.avvisi] == ["W14_GRIGLIA_INCERTA"]
    assert all(set(DAY_FIELDS) <= set(r.incerti) for r in res.rows)
    assert "tutti i valori della pagina" in res.avvisi[0].messaggio
    assert (res.ocr_notes or "").startswith("Griglia incerta")
    # griglia affidabile ma una riga non allineata: avviso per quella riga soltanto
    g = dataclasses.replace(_grid(), alignment=0.95, row_quality=[1.0] + [0.2] + [1.0] * 29)
    res = le.assemble(plan, _row_readings(2), g)
    assert [(a.codice, a.giorno) for a in res.avvisi] == [("W14_GRIGLIA_INCERTA", 2)]
    # tutti i campi della riga, anche quelli vuoti (assenze, nota)
    assert set(DAY_FIELDS) <= set(res.rows[1].incerti)
    assert not res.rows[0].incerti and not res.header.incerti
    res = le.assemble(plan, _row_readings(2), _grid())
    assert not res.avvisi


class FakeSecondOpinion:
    """Secondo parere finto: legge in ogni ritaglio il valore indicato per il tipo di cella."""

    def __init__(self, values: dict[str, object], conf: float = 0.95):
        from sirio.engines.secondo_parere import Parere

        self.parere = Parere
        self.values = values
        self.conf = conf
        self.kinds: list[str] = []

    def read_many(self, requests, progress=None):
        out = []
        for crop, kind in requests:
            assert crop.ndim == 3 and crop.size > 0
            self.kinds.append(kind)
            v = self.values.get(kind)
            out.append(self.parere(testo=str(v), valore=v, stato="valore" if v is not None else "vuota",
                                   confidenza=self.conf))
        return out


def test_second_opinion_confirms_only_with_agreement_and_exact_relation():
    page, truth = _page(0)
    # giorno 2 (8:00-11:00, 3 ore): uscita effettiva letta 11:00, ma la lettura libera dice 11:30
    overrides = {"rows.2.eff_uscita": le.Reading([("11:00", -0.5), ("11:30", -1.2)], "11:30", -0.45, 4)}
    res0 = le.LocalEngine(recognizer=FakeRecognizer(truth, overrides)).extract(page, lambda *_: None)
    assert res0.rows[1].eff_uscita == "11:00" and "eff_uscita" in res0.rows[1].incerti
    # il secondo parere legge 11:00 come il motore e l'uscita programmata (sicura) lo conferma
    sp = FakeSecondOpinion({"orario": "11:00"})
    res = le.LocalEngine(recognizer=FakeRecognizer(truth, overrides), second_opinion_reader=sp).extract(
        page, lambda *_: None)
    r = res.rows[1]
    assert r.eff_uscita == "11:00" and "eff_uscita" not in r.incerti
    assert r.confidenza["eff_uscita"] >= le.CONF_SECONDO_PARERE
    assert sp.kinds == ["orario"]                     # solo le celle dubbie vengono rilette
    assert "Secondo parere" in (res.ocr_notes or "")
    # se il secondo parere legge un altro valore la cella resta segnalata
    sp2 = FakeSecondOpinion({"orario": "11:30"})
    res2 = le.LocalEngine(recognizer=FakeRecognizer(truth, overrides), second_opinion_reader=sp2).extract(
        page, lambda *_: None)
    assert "eff_uscita" in res2.rows[1].incerti


def test_second_opinion_never_clears_structural_doubts():
    page, truth = _page(0)
    overrides = {"rows.2.eff_uscita": _faint("14:00", "11:00")}
    # riga 2 non allineata alla griglia: nessuna rilettura, nessuna conferma
    grid = dataclasses.replace(page.grid, alignment=0.95, row_quality=[1.0] + [0.2] + [1.0] * 29)
    page2 = PageInput(image=page.image, grid=grid, source_file="x.pdf", page=1)
    sp = FakeSecondOpinion({"orario": "11:00", "ore": 3.0})
    res = le.LocalEngine(recognizer=FakeRecognizer(truth, overrides), second_opinion_reader=sp).extract(
        page2, lambda *_: None)
    assert "eff_uscita" in res.rows[1].incerti and not sp.kinds
    assert any(a.codice == "W14_GRIGLIA_INCERTA" for a in res.avvisi)


def test_second_opinion_unavailable_keeps_flags_and_explains(monkeypatch, tmp_path):
    from sirio.engines import secondo_parere

    monkeypatch.setattr(le, "_SP_FAILED", {})
    monkeypatch.setattr(secondo_parere.SecondoParere, "is_available",
                        lambda self: (False, "Secondo parere offline non attivo: servono almeno 12 GB di RAM."))
    page, truth = _page(0)
    overrides = {"rows.2.eff_uscita": _faint("14:00", "11:00")}
    eng = le.LocalEngine(recognizer=FakeRecognizer(truth, overrides), second_opinion=True, cache_dir=tmp_path)
    res = eng.extract(page, lambda *_: None)
    assert "eff_uscita" in res.rows[1].incerti
    assert "Secondo parere non usato" in (res.ocr_notes or "") and "12 GB" in (res.ocr_notes or "")
    # disattivato: nessun tentativo
    calls = []
    monkeypatch.setattr(secondo_parere.SecondoParere, "is_available", lambda self: calls.append(1) or (True, ""))
    le.LocalEngine(recognizer=FakeRecognizer(truth, overrides), cache_dir=tmp_path).extract(page, lambda *_: None)
    assert not calls


def test_second_opinion_failure_while_reading_keeps_the_result():
    page, truth = _page(0)

    class Broken:
        def read_many(self, requests, progress=None):
            raise RuntimeError("guasto")

    overrides = {"rows.2.eff_uscita": _faint("14:00", "11:00")}
    res = le.LocalEngine(recognizer=FakeRecognizer(truth, overrides), second_opinion_reader=Broken()).extract(
        page, lambda *_: None)
    assert res.rows[1].eff_uscita == "11:00" and "eff_uscita" in res.rows[1].incerti
    assert "Secondo parere non riuscito" in (res.ocr_notes or "")


def test_get_engine_passes_second_opinion_setting():
    from sirio.config import Settings
    from sirio.engines.base import get_engine

    assert Settings().secondo_parere is False
    assert get_engine(Settings()).second_opinion is False
    assert get_engine(Settings(secondo_parere=True)).second_opinion is True


# --------------------------------------------------------------------------
# Regole di prudenza (nessun valore sbagliato mostrato come sicuro)
# --------------------------------------------------------------------------

def test_strong_rival_reading_keeps_value_flagged():
    """"14:00" scritto in modo da sembrare "16:00": la lettura preferisce 16:00 ma 14:00
    resta molto probabile e nessuna relazione esatta decide: va verificato."""
    plan = _plan_with({2: {"prog_entrata": le.TESTO, "prog_uscita": le.TESTO, "ore_dichiarate": le.TESTO}})
    readings = {"rows.2.prog_entrata": _time("12:00"),
                "rows.2.prog_uscita": le.Reading([("16:00", -1.0), ("14:00", -1.9)], "16:00", -1.0, 4),
                "rows.2.ore_dichiarate": le.Reading([(1.0, -0.05)], "1", -0.06, 1)}
    res = le.assemble(plan, readings, _grid())
    assert res.rows[1].prog_uscita == "16:00" and "prog_uscita" in res.rows[1].incerti


def test_twin_overridden_reading_flags_the_other_cell():
    """Effettivo poco leggibile riportato al programmato (9:30-13:30) mentre la sua lettura
    dice 9:00-13:00: il programmato non e' confermato da quell'uguaglianza."""
    plan = _plan_with({3: FULL_ROW})
    readings = {"rows.3.prog_entrata": le.Reading([("09:30", -0.7), ("09:00", -2.6)], "9,30", -0.7, 4),
                "rows.3.prog_uscita": le.Reading([("13:30", -0.9), ("13:00", -4.0)], "13.30", -0.9, 4),
                "rows.3.eff_entrata": le.Reading([("09:00", -9.0), ("08:00", -12.0)], "3,os .", -1.0, 4),
                "rows.3.eff_uscita": le.Reading([("13:00", -8.0), ("13:30", -10.5)], "13 w.", -1.0, 4),
                "rows.3.ore_dichiarate": le.Reading([(4.0, -0.9)], "4", -0.9, 1)}
    res = le.assemble(plan, readings, _grid())
    r = res.rows[2]
    for f in ("prog_entrata", "prog_uscita"):
        if getattr(r, f) not in ("09:00", "13:00"):
            assert f in r.incerti, (f, getattr(r, f), r.incerti)


def test_programmed_time_differing_from_same_weekday_is_flagged():
    """Orario programmato settimanale: 9:00 in un solo martedi' contro 8:00 negli altri."""
    days = (3, 10, 17, 24)
    plan = _plan_with({g: {"prog_entrata": le.TESTO, "prog_uscita": le.TESTO} for g in days})
    readings = {}
    for g in days:
        readings |= {f"rows.{g}.prog_entrata": _time("08:00"), f"rows.{g}.prog_uscita": _time("11:00")}
    readings["rows.10.prog_entrata"] = _time("09:00", -0.01)
    res = le.assemble(plan, readings, _grid())
    assert res.rows[9].prog_entrata == "09:00" and "prog_entrata" in res.rows[9].incerti
    assert all("prog_entrata" not in res.rows[g - 1].incerti for g in (3, 17, 24))


def test_programmed_time_without_its_pair_is_flagged():
    plan = _plan_with({17: {"prog_entrata": le.TESTO, "prog_uscita": le.TESTO}})
    readings = {"rows.17.prog_entrata": _time("13:30", -0.02),
                "rows.17.prog_uscita": le.Reading([("10:30", -20.0)], "103.00", -1.0, 4)}   # illeggibile
    res = le.assemble(plan, readings, _grid())
    r = res.rows[16]
    assert r.prog_uscita is None and "prog_uscita" in r.illeggibili
    assert r.prog_entrata == "13:30" and "prog_entrata" in r.incerti


def test_monthly_total_different_from_sum_is_flagged():
    ev = le.evidence(le.Reading([(45.0, -0.01)], "45", -0.01, 2), le.total_lexicon(), "hours")
    assert le._choose_total(ev, 43.0).incerto
    assert not le._choose_total(ev, 45.0).incerto
    # la differenza sta tutta in ore giornaliere illeggibili: nessun dubbio aggiunto
    assert not le._choose_total(ev, 43.5, (43.5, 51.5)).incerto


def test_short_note_close_to_vocabulary_is_proposed_flagged():
    r = le.Reading([("104", math.log(0.25))], "1oh .", -3.0, 3)
    fld = le._note_field(r, 2.0)
    assert fld.value == "104" and fld.incerto
    for junk in ("DE", "49", "R"):
        fld = le._note_field(le.Reading([("104", math.log(0.25))], junk, -3.0, 3), 2.0)
        assert fld.value is None


def test_operator_absence_contradicted_by_worked_row_is_flagged():
    """Crocetta nella colonna "assenza operatore" ma riga con orari effettivi e firma
    (es. l'inizio di una firma che sconfina): non e' un'assenza sicura."""
    import cv2
    page, _truth = _page(0)
    img = page.image.copy()
    g = 2            # riga lavorata e firmata
    x0, y0, x1, y1 = page.grid.cell(g, "assenza_operatore")
    m = (x1 - x0) // 4
    cv2.line(img, (x0 + m, y0 + m // 2), (x1 - m, y1 - m // 2), (30, 30, 30), 3, cv2.LINE_AA)
    cv2.line(img, (x0 + m, y1 - m // 2), (x1 - m, y0 + m // 2), (30, 30, 30), 3, cv2.LINE_AA)
    plan = le.plan_page(img, page.grid)
    assert plan.cells[g]["assenza_operatore"] == le.SEGNO


def test_page_marks_own_crop_for_fallback_reading():
    page, truth = _page(0)
    marks = le.PageMarks(le.to_gray(page.image), page.image, page.grid)
    crop = marks.own_crop(2, "eff_entrata")            # "8:00" scritto
    assert crop is not None and crop.ndim == 2 and crop.dtype == np.uint8 and (crop < 128).any()
    assert marks.own_crop(7, "eff_entrata") is None    # sabato: cella vuota
    assert not marks.struck_rows


# --------------------------------------------------------------------------
# Pieghe, timbri del pie' di pagina, note non lette, orari programmati implausibili
# --------------------------------------------------------------------------

class _SplitNoteRecognizer(FakeRecognizer):
    """Come ``FakeRecognizer``; le note lette riga per riga ("#1", "#2") non leggibili
    salvo letture indicate in ``overrides``."""

    def _reading(self, key: str) -> le.Reading:
        if key in self.overrides:
            return self.overrides[key]
        if "#" in key:
            return le.Reading([], "", -math.inf, 0)
        return super()._reading(key)


def _wrong_unflagged(truth: dict, res) -> list[tuple]:
    out = []
    for t, r in zip(truth["rows"], res.rows):
        for f in (*le.TEXT_FIELDS, "assenza_alunno", "assenza_operatore", "firma", "note", "trattino_effettivo"):
            if getattr(r, f) != t[f] and f not in r.incerti and f not in r.illeggibili:
                out.append((t["giorno"], f, getattr(r, f), t[f]))
    return out


@pytest.mark.parametrize("column,darkness,width", [
    ("assenza_operatore", 110, 5), ("assenza_operatore", 60, 7), ("assenza_alunno", 110, 5),
    ("firma", 110, 5), ("ore_dichiarate", 110, 5), ("note", 60, 7),
])
def test_fold_crease_is_not_read_as_marks_or_text(column, darkness, width):
    """Foto di un foglio piegato in verticale: l'ombra della piega attraversa tutta la
    colonna. Prima: crocetta di assenza dell'operatore letta in ~28 righe (alcune non
    segnalate); ora la piega e' tolta come le barrature in ogni colonna: nessun segno
    inventato, nessun ritaglio letto dove c'e' solo la piega, nessun errore non segnalato."""
    from tests.synthetic import COLUMN_NAMES, fold_crease

    img, truth = make_synthetic_sheet(seed=1)
    gx = truth["griglia"]["col_x"]
    j = COLUMN_NAMES.index(column)
    img = fold_crease(img, gx[j] + 0.45 * (gx[j + 1] - gx[j]), darkness=darkness, width=width, seed=1)
    page = normalize_page(img)
    fake = FakeRecognizer(truth)
    res = le.LocalEngine(recognizer=fake).extract(PageInput(page, detect_grid(page), "piega.jpg", 1),
                                                  lambda *_: None)
    assert "Piega del foglio" in (res.ocr_notes or "")
    for t, r in zip(truth["rows"], res.rows):
        for f in ("assenza_alunno", "assenza_operatore", "firma"):
            assert getattr(r, f) == t[f], (t["giorno"], f)       # nessun segno inventato dalla piega
    assert not _wrong_unflagged(truth, res)
    expected = {f"rows.{t['giorno']}.{f}" for t in truth["rows"] for f in (*le.TEXT_FIELDS, "note")
                if t[f] is not None}
    assert {k for k in fake.keys if k.startswith("rows.")} <= expected


def test_fold_crease_through_written_cells_flags_them():
    """La piega che taglia la scrittura (ore) la rende da verificare; una piega che passa
    nella parte vuota della cella no."""
    from tests.synthetic import fold_crease

    img, truth = make_synthetic_sheet(seed=1)
    gx = truth["griglia"]["col_x"]
    j = 5          # ore: la cifra e' centrata nella cella
    page = normalize_page(fold_crease(img, (gx[j] + gx[j + 1]) / 2.0, darkness=110, width=5, seed=1))
    grid = detect_grid(page)
    plan = le.plan_page(page, grid)
    worked = [t["giorno"] for t in truth["rows"] if t["ore_dichiarate"] is not None]
    assert all(plan.flags.get(g, {}).get("ore_dichiarate") == "piega" for g in worked)
    # piega nella parte destra (vuota) delle celle dell'entrata effettiva
    page = normalize_page(fold_crease(img, gx[3] + 0.88 * (gx[4] - gx[3]), darkness=110, width=5, seed=1))
    plan = le.plan_page(page, detect_grid(page))
    assert not any(plan.flags.get(g, {}).get("eff_entrata") == "piega" for g in range(1, 32))


def test_stamp_and_footer_signature_over_last_rows_are_not_operator_signatures():
    """Timbro tondo della scuola e firma del referente appoggiati sull'angolo in basso a
    destra della tabella: sconfinano nelle celle "Firma operatore" delle ultime righe
    (vuote). Prima: firma=True non segnalata nei giorni 30 e 31."""
    page, truth = _degraded_page(seed=2, footer_over_table=True)
    plan = le.plan_page(page.image, page.grid)
    for g in (30, 31):
        assert plan.cells[g]["firma"] in (le.VUOTA, le.TIMBRO), (g, plan.cells[g]["firma"])
    res = le.LocalEngine(recognizer=FakeRecognizer(truth)).extract(page, lambda *_: None)
    assert not _wrong_unflagged(truth, res)
    for r in res.rows[28:]:
        assert not r.firma or "firma" in r.incerti, r
    # le firme delle righe lavorate restano firme sicure
    signed = [t["giorno"] for t in truth["rows"] if t["firma"] and t["giorno"] < 28]
    assert all(res.rows[g - 1].firma and "firma" not in res.rows[g - 1].incerti for g in signed)


def test_foreign_ink_keeps_touching_row_signatures():
    """Firme di righe vicine che si toccano (struttura alta piu' righe dentro la tabella)
    restano firme delle righe: solo cio' che scende nel pie' di pagina e' estraneo."""
    img, grid, truth = _synthetic(0)
    foreign = le.foreign_ink(le.to_gray(img), grid)
    for t in truth["rows"]:
        if t["firma"]:
            assert foreign is None or foreign.share(grid.cell(t["giorno"], "firma")) < le.QUOTA_DUBBIA


def test_isolated_signature_or_absence_mark_is_never_sure():
    """Firma (o crocetta) in una riga senza orari, ore e note: segno isolato (timbro,
    firma del pie' di pagina, macchia), da verificare; nelle righe compilate no."""
    plan = _plan_with({7: {"firma": le.FIRMA}, 8: {"assenza_alunno": le.CROCETTA},
                       9: {**FULL_ROW, "firma": le.FIRMA},
                       10: {"prog_entrata": le.TESTO, "prog_uscita": le.TESTO, "assenza_operatore": le.CROCETTA,
                            "firma": le.FIRMA}})
    readings = {**_row_readings(9), "rows.10.prog_entrata": _time("08:00"), "rows.10.prog_uscita": _time("11:00")}
    res = le.assemble(plan, readings, _grid())
    assert res.rows[6].firma and "firma" in res.rows[6].incerti
    assert res.rows[7].assenza_alunno and "assenza_alunno" in res.rows[7].incerti
    assert not res.rows[8].incerti
    assert res.rows[9].assenza_operatore and not ({"firma", "assenza_operatore"} & set(res.rows[9].incerti))


def test_signature_on_row_with_only_an_uncertain_note_is_flagged():
    """Firma del pie' di pagina che sconfina in una riga vuota, insieme a un pezzo di
    timbro letto come nota incerta: la nota incerta non rende la riga "compilata",
    quindi la firma resta da verificare. Con una nota letta con sicurezza no."""
    from sirio.models import DayRow

    st = {"note": le.TESTO, "firma": le.FIRMA}
    dubbia = DayRow(giorno=30, firma=True, note="11 ALL MUN.", incerti=["note"])
    assert le._lonely_marks(dubbia, st) == ["firma"]
    sicura = DayRow(giorno=30, firma=True, note="SCIOPERO")
    assert le._lonely_marks(sicura, st) == []
    compilata = DayRow(giorno=30, firma=True, prog_entrata="08:00", note="X", incerti=["note"])
    assert le._lonely_marks(compilata, {**st, "prog_entrata": le.TESTO}) == []


def test_marks_on_a_row_with_only_doubtful_values_are_flagged():
    """Riga "annullata" con una barra in ogni cella: le barre nelle colonne di orari
    sono lette come valori dubbi, quella nella colonna della firma sembra una firma.
    Senza almeno un valore sicuro nella riga, firma e crocette sono da verificare."""
    from sirio.models import DayRow

    st = {"prog_entrata": le.NULLO, "ore_dichiarate": le.TESTO, "firma": le.FIRMA,
          "assenza_operatore": le.CROCETTA}
    barrata = DayRow(giorno=14, ore_dichiarate=1.0, firma=True, assenza_operatore=True,
                     incerti=["ore_dichiarate"])
    assert le._lonely_marks(barrata, st) == ["firma", "assenza_operatore"]
    solo_trattini = DayRow(giorno=15, firma=True, trattino_effettivo=True)
    assert le._lonely_marks(solo_trattini, {"eff_entrata": le.TRATTINO, "firma": le.FIRMA}) == ["firma"]
    lavorata = DayRow(giorno=16, prog_entrata="08:00", eff_entrata="09:00", firma=True,
                      incerti=["eff_entrata"])
    assert le._lonely_marks(lavorata, {"prog_entrata": le.TESTO, "firma": le.FIRMA}) == []
    # riga lavorata ma con tutti gli orari dubbi (scrittura difficile): la firma che ha la
    # forma di una firma resta sicura, un tratto solo no
    difficile = DayRow(giorno=17, prog_entrata="08:00", prog_uscita="11:00", firma=True,
                       incerti=["prog_entrata", "prog_uscita"])
    st2 = {"prog_entrata": le.TESTO, "prog_uscita": le.TESTO, "firma": le.FIRMA}
    assert le._lonely_marks(difficile, st2, firma_vera=True) == []
    assert le._lonely_marks(difficile, st2, firma_vera=False) == ["firma"]
    # ... ma se la "scrittura" degli orari sono barre (riga annullata con "/ / /"), no
    barre = {"prog_entrata": "barretta", "prog_uscita": "barretta"}
    assert le._lonely_marks(difficile, st2, firma_vera=True, flags=barre) == ["firma"]
    # riga senza scrittura nelle celle di orari: anche una firma vera e' isolata
    assert le._lonely_marks(DayRow(giorno=30, firma=True), {"firma": le.FIRMA}, firma_vera=True) == ["firma"]


def test_minutes_off_the_quarter_hour_are_read_when_the_page_uses_them():
    """Chi lavora "8:10 - 13:10": i minuti 10 sono ammessi e, se ricorrono sulla pagina,
    una lettura libera "13.10" contro un 13:30 scelto e' un orario diverso (segnalato).
    Su una pagina dove nessuno scrive i minuti 10, "11.10" resta una cifra letta male."""
    prior = le.page_minute_prior([10, 10, 10, 10, 0, 0])
    assert 10 in prior.in_uso and prior("08:10") > le.MINUTI_MIN_LOG_ALTRI
    generic = le.page_minute_prior([10, 0, 0])            # una sola lettura: non "in uso"
    assert 10 not in generic.in_uso and generic("08:10") <= le.MINUTI_MIN_LOG_ALTRI
    assert generic("08:00") > generic("08:30") > generic("08:10")
    r = le.Reading([("13:30", -0.1)], "13.10", -0.2, 4)
    assert le._free_digits_conflict("prog_uscita", "13:30", r, prior.in_uso)
    assert not le._free_digits_conflict("prog_uscita", "13:30", r, generic.in_uso)


def _glyph_image(text: str, seed: int) -> np.ndarray:
    """Ritaglio di un orario "scritto" con un tratto leggermente diverso a ogni seme."""
    rng = np.random.default_rng(seed)
    img = np.full((60, 200), 255, np.uint8)
    cv2.putText(img, text, (8 + int(rng.integers(0, 6)), 45 + int(rng.integers(-3, 3))),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.3, 0, 2 + int(rng.integers(0, 2)))
    return img


def test_minutes_written_like_the_others_but_read_differently_are_flagged():
    """Stessa grafia dei minuti di tutte le altre celle lette ":00", ma letta ":30"
    (zeri a ricciolo letti come "30"): segnalata anche se la riga e' coerente; i
    minuti ":30" scritti davvero diversi non lo sono."""
    plan = _plan_with({})
    states = {}
    for g in range(2, 12):
        for f in ("prog_entrata", "prog_uscita"):
            vero = "11:00" if f == "prog_entrata" else "13:00"
            letto = "13:30" if (g, f) == (5, "prog_uscita") else vero
            plan.requests.append(le.ReadRequest(f"rows.{g}.{f}", _glyph_image(vero.replace(":", "."), g)))
            ev = le.Evidence([(letto, 0.0)], 1.0)
            states[(g, f)] = le._CellState(g, f, letto, ev, set())
    flagged = le._flag_minute_lookalikes(plan, states)
    assert flagged == [(5, "prog_uscita")] and "grafia_minuti" in states[(5, "prog_uscita")].reasons
    # pagina con mezz'ore vere (scritte "30"): nessun segnale
    plan2 = _plan_with({})
    states2 = {}
    for g in range(2, 12):
        v = "11:30" if g % 2 else "11:00"
        plan2.requests.append(le.ReadRequest(f"rows.{g}.prog_entrata", _glyph_image(v.replace(":", "."), g)))
        states2[(g, "prog_entrata")] = le._CellState(g, "prog_entrata", v, le.Evidence([(v, 0.0)], 1.0), set())
    assert le._flag_minute_lookalikes(plan2, states2) == []


def test_hours_without_any_time_in_the_row_are_flagged():
    """Ore diverse da zero in una riga senza alcun orario (riga annullata con "/" e lo
    "0" letto "3"): nessun riscontro possibile, da verificare. Lo zero no."""
    plan = _plan_with({5: {"ore_dichiarate": le.TESTO, "firma": le.FIRMA},
                       6: {"ore_dichiarate": le.TESTO, "firma": le.FIRMA}, 7: FULL_ROW})
    readings = {"rows.5.ore_dichiarate": le.Reading([(3.0, -0.02)], "3", -0.03, 2),
                "rows.6.ore_dichiarate": le.Reading([(0.0, -0.02)], "0", -0.03, 2), **_row_readings(7)}
    res = le.assemble(plan, readings, _grid())
    assert res.rows[4].ore_dichiarate == 3.0 and "ore_dichiarate" in res.rows[4].incerti
    assert res.rows[5].ore_dichiarate == 0.0 and "ore_dichiarate" not in res.rows[5].incerti
    assert not res.rows[6].incerti


def test_page_with_almost_every_value_doubtful_flags_the_rest():
    """Se quasi tutti gli orari e le ore della pagina sono dubbi, la sicurezza del modello
    sui pochi rimasti non vale: si segnalano anche quelli (con una nota)."""
    from sirio.models import DayRow

    rows = []
    for g in range(1, 7):
        r = DayRow(giorno=g, prog_entrata="08:00", prog_uscita="11:00", ore_dichiarate=3.0)
        if g > 1:
            r.incerti = ["prog_entrata", "prog_uscita", "ore_dichiarate"]
        rows.append(r)
    rows += [DayRow(giorno=g, prog_entrata="08:00", prog_uscita="11:00", eff_entrata="08:00",
                    eff_uscita="11:00", ore_dichiarate=3.0, incerti=list(le.TEXT_FIELDS)) for g in range(7, 10)]
    assert le._flag_difficult_page(rows)
    assert set(rows[0].incerti) == {"prog_entrata", "prog_uscita", "ore_dichiarate"}
    # pagina letta per lo piu' con sicurezza: nessun cambiamento
    sure = [DayRow(giorno=g, prog_entrata="08:00", prog_uscita="11:00", ore_dichiarate=3.0) for g in range(1, 11)]
    sure[0].incerti = ["prog_entrata"]
    assert not le._flag_difficult_page(sure) and not sure[1].incerti


def test_two_time_slots_in_one_cell_are_read_line_by_line():
    """Cella con due orari uno sopra l'altro: due righe di scrittura separate."""
    crop = np.full((120, 220), 255, np.uint8)
    cv2.putText(crop, "9.00", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    cv2.putText(crop, "13.00", (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    lines = le.split_text_lines(crop, 120.0, min_width=le.ORARI_DUE_RIGHE_LARGHEZZA)
    assert len(lines) == 2
    single = np.full((120, 220), 255, np.uint8)
    cv2.putText(single, "10.30", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
    assert len(le.split_text_lines(single, 120.0, min_width=le.ORARI_DUE_RIGHE_LARGHEZZA)) == 1


def test_monthly_total_written_on_the_day_31_row():
    """Casella del totale vuota e "28" nelle ore del giorno 31 (impossibile come ore di un
    giorno): e' il totale mensile scritto una riga sopra, riportato come totale e da
    verificare; un valore giornaliero plausibile resta invece le ore del giorno 31."""
    plan = _plan_with({31: {"ore_dichiarate": le.TESTO}})
    readings = {"rows.31.ore_dichiarate": le.Reading([], "28", -0.1, 2),
                le.TOTALE_RIGA_31: le.Reading([(28.0, -0.05)], "28", -0.06, 2)}
    res = le.assemble(plan, readings, _grid())
    assert res.header.totale_mensile_dichiarato == 28.0 and "totale_mensile_dichiarato" in res.header.incerti
    assert res.rows[30].ore_dichiarate is None and "ore_dichiarate" not in res.rows[30].illeggibili
    plan2 = _plan_with({31: {"ore_dichiarate": le.TESTO}})
    readings2 = {"rows.31.ore_dichiarate": le.Reading([(3.0, -0.05)], "3", -0.06, 2),
                 le.TOTALE_RIGA_31: le.Reading([(3.0, -0.05)], "3", -0.06, 2)}
    res2 = le.assemble(plan2, readings2, _grid())
    assert res2.header.totale_mensile_dichiarato is None and res2.rows[30].ore_dichiarate == 3.0


def test_header_number_read_as_split_digits_is_flagged():
    """"13" letto come due segni staccati "1 2": il numero (12) non e' sicuro."""
    plan = _plan_with({})
    plan.header_ink["ore_pei"] = True
    res = le.assemble(plan, {"header.ore_pei": le.Reading([(12.0, -0.001)], "1 2", -0.01, 3)}, _grid())
    assert res.header.ore_pei == 12.0 and "ore_pei" in res.header.incerti
    res = le.assemble(plan, {"header.ore_pei": le.Reading([(13.0, -0.001)], "13", -0.01, 2)}, _grid())
    assert res.header.ore_pei == 13.0 and "ore_pei" not in res.header.incerti


def test_hours_with_only_doubtful_times_are_flagged():
    """Ore lette con sicurezza ma orari della riga tutti dubbi: le ore non hanno riscontro
    (orari letti male che per caso tornano con le ore lette male)."""
    plan = _plan_with({9: FULL_ROW})
    readings = {**_row_readings(9, "16:00", "18:00", 2.0)}
    for f in le.TIME_FIELDS:
        readings[f"rows.9.{f}"] = _faint(readings[f"rows.9.{f}"].candidates[0][0], "11:00")
    res = le.assemble(plan, readings, _grid())
    row = res.rows[8]
    assert all(f in row.incerti for f in le.TIME_FIELDS)
    assert row.ore_dichiarate == 2.0 and "ore_dichiarate" in row.incerti


def test_halftone_dots_do_not_join_writing():
    """Fondino retinato delle fotocopie: i puntini isolati si tolgono prima di ricucire i
    tratti (altrimenti firma, nota e piè di pagina diventano un'unica struttura); i
    tratti veri restano."""
    m = np.zeros((200, 400), bool)
    m[::6, ::6] = True                                   # retino
    m[100:104, 50:350] = True                            # tratto
    out = le._drop_specks(m, 60.0)
    assert out[100:104, 50:350].all()
    assert out.sum() == m[100:104, 50:350].sum()


def test_halftone_photocopy_is_descreened_before_the_analysis():
    """Fotocopia con il fondino retinato (puntini fitti su tutta la tabella): riconosciuta e
    "deretinata", le righe vuote restano vuote e quelle scritte restano scritte; una pagina
    normale non viene toccata."""
    page, grid, truth = _synthetic(0)
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    assert not le.is_halftone(gray, grid)
    screened = page.copy()
    step = max(4, int(round(grid.row_height / 9)))
    ys, xs = np.mgrid[0:page.shape[0]:step, 0:page.shape[1]:step]
    for y, x in zip(ys.ravel(), xs.ravel()):
        cv2.circle(screened, (int(x), int(y)), 1, (90, 90, 90), -1)
    assert le.is_halftone(cv2.cvtColor(screened, cv2.COLOR_BGR2GRAY), grid)
    plan = le.plan_page(screened, grid)
    assert any("retinato" in n for n in plan.notes)
    rows = {r["giorno"]: r for r in truth["rows"]}
    empty = [g for g, r in rows.items() if not any(r.get(f) for f in (*le.TEXT_FIELDS, "firma"))]
    written = [g for g, r in rows.items() if r.get("prog_entrata") and r.get("firma")]
    assert empty and written
    for g in empty:
        assert plan.cells[g]["firma"] == le.VUOTA and plan.cells[g]["prog_entrata"] == le.VUOTA, g
    for g in written:
        assert plan.cells[g]["prog_entrata"] == le.TESTO and plan.cells[g]["firma"] != le.VUOTA, g


def test_note_under_the_stamp_on_a_filled_row_is_flagged():
    """Nota scritta sopra il timbro della scuola in una riga compilata (crocetta, firma):
    non si legge ma non sparisce in silenzio: illeggibile. In una riga vuota no."""
    plan = _plan_with({31: {"note": le.TIMBRO, "assenza_alunno": le.CROCETTA, "firma": le.FIRMA},
                       30: {"note": le.TIMBRO}})
    plan.flags[31] = {"note": "sotto_timbro"}
    res = le.assemble(plan, {}, _grid())
    assert "note" in res.rows[30].illeggibili and res.rows[30].note is None
    assert "note" not in res.rows[29].illeggibili


def test_time_far_outside_the_sheet_range_is_flagged_not_changed():
    """Un orario sicuro molto fuori dall'intervallo di tutti gli altri orari sicuri del
    foglio (es. "16:00" dove nessun altro supera le 14:00) e' segnalato, mai cambiato;
    entro l'intervallo, o con pochi riferimenti, nessun segnale."""
    from sirio.models import DayRow

    rows = [DayRow(giorno=g, prog_entrata="11:00", prog_uscita="14:00", eff_entrata="11:00", eff_uscita="14:00")
            for g in (2, 3, 4)]
    rows.append(DayRow(giorno=9, prog_entrata="11:00", prog_uscita="16:00", assenza_alunno=True))
    rows.append(DayRow(giorno=10, prog_entrata="10:00", prog_uscita="13:00"))
    flagged = le._flag_time_outliers(rows)
    assert flagged == [(9, "prog_uscita")]
    assert rows[3].prog_uscita == "16:00" and "prog_uscita" in rows[3].incerti
    assert not rows[4].incerti and not rows[0].incerti
    pochi = [DayRow(giorno=2, prog_entrata="08:00", prog_uscita="10:00"),
             DayRow(giorno=3, prog_entrata="14:00", prog_uscita="17:00")]
    assert le._flag_time_outliers(pochi) == []


def test_two_line_note_written_but_not_read_is_flagged():
    """Nota scritta piccola su due righe (es. un recupero): se nessuna lettura viene
    accettata la nota e' illeggibile, mai vuota in silenzio (prima: note=None, nessun avviso)."""
    data = {"rows": [{"giorno": 11, "prog_entrata": "08:00", "prog_uscita": "11:00", "eff_entrata": "08:00",
                      "eff_uscita": "11:00", "ore_dichiarate": 3.0, "firma": True, "note": "RECUPERO\nDEL 9/2"}]}
    page, truth = _degraded_page(data=data, seed=3)
    junk = {"rows.11.note#1": le.Reading([], "RE", -2.0, 2), "rows.11.note#2": le.Reading([], "9", -2.0, 2),
            "rows.11.note": le.Reading([], "RE 9", -3.0, 3)}
    fake = _SplitNoteRecognizer(truth, junk)
    res = le.LocalEngine(recognizer=fake).extract(page, lambda *_: None)
    assert any(k.startswith("rows.11.note") for k in fake.keys)
    r = res.rows[10]
    assert r.note is None and "note" in r.illeggibili
    assert not _wrong_unflagged(truth, res)


def test_short_note_written_but_not_read_is_flagged():
    """Nota breve di tre segni ("104", larga circa una riga): se la lettura non viene
    accettata e' illeggibile, non vuota in silenzio (prima serviva una larghezza di 1,5
    righe: un "104" vero ne misura 1-1,5)."""
    for seed in (0, 5):
        page, truth = _degraded_page(seed=seed)
        junk = le.Reading([], "RE 9", -3.0, 3)
        fake = FakeRecognizer(truth, {"rows.10.note": junk})
        res = le.LocalEngine(recognizer=fake).extract(page, lambda *_: None)
        assert "rows.10.note" in fake.keys
        r = res.rows[9]
        assert r.note is None and "note" in r.illeggibili, (seed, r)
        assert not _wrong_unflagged(truth, res)
    # un tratto solo (coda di firma) largo una riga resta ignorato
    canvas = np.full((40, 120), 255, np.uint8)
    canvas[20:23, 10:70] = 0
    assert not le._substantial_note(canvas, 50.0, le.NOTA_BREVE_LARGHEZZA)


def test_unread_note_flagged_only_when_really_written():
    """Lettura breve fuori vocabolario: scrittura vera (``note_scritta``) -> illeggibile;
    segno piccolo (coda di firma, tratto) -> ignorato come prima."""
    plan = _plan_with({5: {"note": le.TESTO}, 6: {"note": le.TESTO}})
    plan.note_scritta.add(5)
    readings = {f"rows.{g}.note": le.Reading([("104", -9.0)], "DE", -3.0, 3) for g in (5, 6)}
    res = le.assemble(plan, readings, _grid())
    assert res.rows[4].note is None and "note" in res.rows[4].illeggibili
    assert res.rows[5].note is None and not res.rows[5].illeggibili and not res.rows[5].incerti


def test_substantial_note_shape():
    unit = 50.0
    canvas = np.full((40, 200), 255, np.uint8)
    assert not le._substantial_note(canvas, unit)
    word = canvas.copy()
    for i, x in enumerate(range(10, 170, 40)):
        word[8 + (i % 2) * 4:34, x:x + 6] = 0          # quattro tratti su una larghezza di 3 righe
    assert le._substantial_note(word, unit)
    tail = canvas.copy()
    tail[20:23, 5:150] = 0                               # un solo tratto lungo (coda di una firma)
    assert not le._substantial_note(tail, unit)


def test_absence_row_with_implausible_programmed_exit_is_flagged():
    """Alunno assente, ore riconosciute 1,5 e uscita programmata "14:00" letta "16:00" con
    sicurezza: 1,5 / 5 ore non e' una frazione tonda (di solito la meta') e lo stesso
    giorno della settimana ha 11:00-14:00. Prima: nessuna segnalazione."""
    data = {"rows": [
        {"giorno": 19, "prog_entrata": "11:00", "prog_uscita": "16:00", "ore_dichiarate": 1.5,
         "assenza_alunno": True, "firma": True, "trattino_effettivo": True},
        {"giorno": 5, "note": "GITA"}, {"giorno": 26, "note": "GITA"}]}
    page, truth = _degraded_page(data=data, seed=4)
    res = le.LocalEngine(recognizer=FakeRecognizer(truth)).extract(page, lambda *_: None)
    r = res.rows[18]
    assert (r.prog_entrata, r.prog_uscita, r.ore_dichiarate) == ("11:00", "16:00", 1.5)
    assert "prog_uscita" in r.incerti


def test_absence_hours_ratio_flags_programmed_times():
    plan = _plan_with({g: {"prog_entrata": le.TESTO, "prog_uscita": le.TESTO, "ore_dichiarate": le.TESTO,
                           "assenza_alunno": le.CROCETTA, "firma": le.FIRMA} for g in (3, 4)})
    readings = {"rows.3.prog_entrata": _time("11:00"), "rows.3.prog_uscita": _time("16:00"),
                "rows.3.ore_dichiarate": le.Reading([(1.5, -0.05)], "1,5", -0.06, 3),
                "rows.4.prog_entrata": _time("11:00"), "rows.4.prog_uscita": _time("14:00"),
                "rows.4.ore_dichiarate": le.Reading([(1.5, -0.05)], "1,5", -0.06, 3)}
    res = le.assemble(plan, readings, _grid())
    assert {"prog_entrata", "prog_uscita"} <= set(res.rows[2].incerti)      # 1,5 su 5 ore
    assert not {"prog_entrata", "prog_uscita"} & set(res.rows[3].incerti)    # 1,5 su 3 ore: la meta'


def test_weekly_schedule_with_a_single_sure_week_flags_the_outlier():
    """Una sola altra settimana con lo stesso giorno, letta con sicurezza: basta per
    segnalare l'orario programmato diverso (prima servivano due settimane concordi)."""
    days = (5, 12)
    plan = _plan_with({g: {"prog_entrata": le.TESTO, "prog_uscita": le.TESTO} for g in days})
    readings = {"rows.5.prog_entrata": _time("11:00"), "rows.5.prog_uscita": _time("14:00"),
                "rows.12.prog_entrata": _time("11:00"), "rows.12.prog_uscita": _time("16:00", -0.01)}
    res = le.assemble(plan, readings, _grid())
    assert "prog_uscita" in res.rows[11].incerti
    assert "prog_entrata" not in res.rows[11].incerti and "prog_entrata" not in res.rows[4].incerti


def test_programmed_duration_different_from_same_weekday_is_flagged():
    """Orari diversi fra le settimane ma stessa durata (3 ore); in una settimana la durata
    letta e' 5 ore: l'orario programmato di quel giorno va verificato."""
    sched = {3: ("08:00", "11:00"), 10: ("09:00", "12:00"), 17: ("10:00", "15:00")}
    plan = _plan_with({g: {"prog_entrata": le.TESTO, "prog_uscita": le.TESTO} for g in sched})
    readings = {}
    for g, (a, b) in sched.items():
        readings |= {f"rows.{g}.prog_entrata": _time(a), f"rows.{g}.prog_uscita": _time(b)}
    res = le.assemble(plan, readings, _grid())
    assert {"prog_entrata", "prog_uscita"} <= set(res.rows[16].incerti)
    assert not res.rows[2].incerti and not res.rows[9].incerti


def _free_reading(testo: str, conf: float) -> "le.Reading":
    """Lettura libera con fiducia media per token ``conf`` (free_confidence)."""
    n = 5
    return le.Reading([], testo, (n + 1) * math.log(conf), n)


def test_nome_letto_in_loop_e_illeggibile_e_nomi_incerti_segnalati():
    """Una lettura "impazzita" del nome ("MR.R.R.R.R.R.R") non deve finire nel titolo
    del documento come nome sicuro: e' illeggibile. Un nome letto con fiducia
    media resta da verificare; con fiducia molto alta e' sicuro."""
    assert le._nome_degenere("MR.R.R.R.R.R.R.R.R.R.R.R.R")
    assert le._nome_degenere("ANANANANA")
    assert le._nome_degenere("M A R I O R")
    assert not le._nome_degenere("ROSSI MARIO")
    assert not le._nome_degenere("IC 9 VIA ROMA")
    assert not le._nome_degenere("COOP. SOC. SIRIO")

    loop = le._text_field(_free_reading("MR.R.R.R.R.R.R.R.R", 0.95))
    assert loop.illeggibile and loop.value is None
    medio = le._text_field(_free_reading("ROSSI MARIO", 0.7))
    assert medio.value == "ROSSI MARIO" and medio.incerto
    alto = le._text_field(_free_reading("ROSSI MARIO", 0.95))
    assert alto.value == "ROSSI MARIO" and not alto.incerto
