"""Test del motore OCR locale (TrOCR + analisi dell'inchiostro).

Quasi tutti i test girano senza modello: il riconoscitore TrOCR e' sostituito da
uno finto che "legge" i valori veri del foglio sintetico (``tests/synthetic.py``)
o letture costruite a mano. I test con il modello reale richiedono il modello
gia' scaricato nella cache indicata da ``SIRIO_HF_CACHE`` (nessun download
durante i test); quello di accuratezza richiede anche ``SIRIO_SAMPLE_PDF`` e
``SIRIO_SAMPLE_TRUTH``.
"""

from __future__ import annotations

import errno
import functools
import json
import math
import os
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from sirio.engines import local_engine as le
from sirio.engines.base import EngineError, PageInput
from sirio.models import DAY_FIELDS
from sirio.vision.grid import TableGrid, detect_grid, template_grid
from sirio.vision.preprocess import normalize_page
from tests.synthetic import make_synthetic_sheet

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
    assert lex.value("8:10") is None          # minuti non a quarti d'ora
    assert lex.value("Atlas") is None
    assert lex.is_prefix("8:") and lex.is_prefix("1") and not lex.is_prefix("8:7")
    assert lex.signature("8.00") == lex.signature("8:00") == lex.signature("8:OO")


def test_hours_and_other_lexicons():
    ore = le.hours_lexicon()
    assert ore.value("3") == 3.0
    assert ore.value("1,5") == 1.5 and ore.value("1.5") == 1.5 and ore.value("I,5") == 1.5
    assert ore.value("3h") == 3.0
    assert ore.value("8,5") is None and ore.value("9") is None
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


# --------------------------------------------------------------------------
# Assemblaggio: incerti, illeggibili, valori dedotti
# --------------------------------------------------------------------------

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
    grid = template_grid(1654, 2339)
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
    res = le.assemble(plan, readings, template_grid(1654, 2339))
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
    res = le.assemble(plan, readings, template_grid(1654, 2339))
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
    res = le.assemble(plan, readings, template_grid(1654, 2339))
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
    res = le.assemble(plan, readings, template_grid(1654, 2339))
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
    res = le.assemble(plan, readings, template_grid(1654, 2339))
    assert res.rows[15].prog_entrata == "08:00" and "prog_entrata" not in res.rows[15].incerti
    # un giorno della stessa settimana con un orario diverso: nessuna conferma
    readings["rows.9.prog_entrata"] = _time("09:00")
    res = le.assemble(plan, readings, template_grid(1654, 2339))
    assert "prog_entrata" in res.rows[15].incerti


def test_free_reading_outside_form_rules_does_not_block_confirmation():
    plan = _plan_with({2: FULL_ROW, 3: FULL_ROW})
    readings = {**_row_readings(2), **_row_readings(3)}
    # lettura libera "11.10" (minuti non a quarti d'ora: cifra letta male) -> confermabile
    readings["rows.2.eff_uscita"] = le.Reading([("11:00", -0.2), ("11:30", -2.5)], "11.10", -0.15, 4)
    # lettura libera "11.30" (orario ammesso diverso) -> resta da verificare
    readings["rows.3.eff_uscita"] = le.Reading([("11:00", -0.2), ("11:30", -2.5)], "11.30", -0.15, 4)
    res = le.assemble(plan, readings, template_grid(1654, 2339))
    assert res.rows[1].eff_uscita == "11:00" and "eff_uscita" not in res.rows[1].incerti
    assert res.rows[2].eff_uscita == "11:00" and "eff_uscita" in res.rows[2].incerti


def test_notes_snap_to_recurring_phrases_and_neighbours():
    plan = _plan_with({16: {"note": le.TESTO}, 17: {"note": le.TESTO}, 18: {"note": le.TESTO}})
    readings = {
        "rows.16.note": le.Reading([("PONTE DI CARNEVALE", -1.0)], "Ponte di Carnevale", -0.9, 6),
        "rows.17.note": le.Reading([], "Ponte di Casarevali", -2.5, 7),       # simile a una formula
        "rows.18.note": le.Reading([], "Barrie in Glenlevale", -6.0, 7),      # simile alla nota vicina
    }
    res = le.assemble(plan, readings, template_grid(1654, 2339))
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


def test_extract_rejects_pages_without_table():
    blank = np.full((2339, 1654, 3), 250, np.uint8)
    grid = detect_grid(blank)
    fake = FakeRecognizer({"header": {}, "rows": []})
    res = le.LocalEngine(recognizer=fake).extract(PageInput(blank, grid, "vuoto.png", 1), lambda *_: None)
    assert not res.is_foglio_firma and fake.calls == 0
    assert "non sembra un foglio firma" in (res.ocr_notes or "")


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
# fra CPU diverse).
MIN_ACCURACY = {
    "orari": 0.94,           # misurato 69/72
    "ore": 0.94,             # misurato 17/17
    "assenze": 1.0,
    "firme": 1.0,
    "trattini": 1.0,
    "note": 0.90,            # misurato 31/31
    "intestazione": 0.85,    # misurato 10/10
}
# Orari corretti ma segnalati come incerti (lavoro di verifica inutile per l'utente):
# misurati 16 su 69 (prima della conferma per coerenza erano 21).
MAX_FALSI_INCERTI_ORARI = 18


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
