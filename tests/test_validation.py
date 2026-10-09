"""Test di parsing, controlli di coerenza e totali (sirio/validation.py).

Dati sintetici con nomi di fantasia. Il test sul foglio reale legge la verità
attesa da SIRIO_SAMPLE_TRUTH e viene saltato se la variabile non è impostata.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from sirio import validation as v
from sirio.models import Anomaly, DayRow, Document, Header, empty_rows

# Febbraio 2026: inizia di domenica; giorni feriali 2-6, 9-13, 16-20, 23-27.
FERIALI_FEB_2026 = [g for g in range(1, 29) if g not in {1, 7, 8, 14, 15, 21, 22, 28}]


# --- Fabbriche di dati sintetici ---------------------------------------------


def make_header(**kw) -> Header:
    base = dict(
        anno_scolastico="2025/2026",
        lotto="1",
        municipalita="2",
        ente="Cooperativa Esempio",
        istituto="IC 1 Esempio",
        operatore="ROSSI MARIO",
        alunno="BIANCHI LUCA",
        mese=2,
        anno=2026,
        ore_pei=15,
        firma_coordinatore=True,
        timbro_referente=True,
        totale_mensile_dichiarato=60,
    )
    base.update(kw)
    return Header(**base)


def lavorato(g: int, entrata="08:00", uscita="11:00", ore: float | None = 3, **kw) -> DayRow:
    return DayRow(
        giorno=g,
        prog_entrata=entrata,
        prog_uscita=uscita,
        eff_entrata=entrata,
        eff_uscita=uscita,
        ore_dichiarate=ore,
        firma=True,
        **kw,
    )


def mese_pulito() -> list[DayRow]:
    """Febbraio 2026 regolare: 20 giorni feriali da 3 ore (8:00-11:00), tutti firmati."""
    rows = empty_rows()
    for g in FERIALI_FEB_2026:
        rows[g - 1] = lavorato(g)
    return rows


def replace_row(rows: list[DayRow], g: int, **kw) -> list[DayRow]:
    rows = list(rows)
    rows[g - 1] = rows[g - 1].model_copy(update=kw)
    return rows


def codici(anomalie: list[Anomaly]) -> set[str]:
    return {a.codice for a in anomalie}


def trova(anomalie: list[Anomaly], codice: str, giorno: int | None = "any") -> list[Anomaly]:  # type: ignore[assignment]
    return [a for a in anomalie if a.codice == codice and (giorno == "any" or a.giorno == giorno)]


# --- Parsing ------------------------------------------------------------------


@pytest.mark.parametrize(
    "testo, atteso",
    [
        ("8:00", (8, 0)),
        ("8.00", (8, 0)),
        ("8,00", (8, 0)),
        ("8;00", (8, 0)),
        ("800", (8, 0)),
        ("0800", (8, 0)),
        ("8", (8, 0)),
        ("08:00", (8, 0)),
        ("8:oo", (8, 0)),
        ("8:0o", (8, 0)),
        ("l1:00", (11, 0)),
        ("I1:00", (11, 0)),
        ("11:OO", (11, 0)),
        ("8h", (8, 0)),
        ("ore 8", (8, 0)),
        ("8:00h", (8, 0)),
        (" 8 : 00 ", (8, 0)),
        ("8h30", (8, 30)),
        ("8:30", (8, 30)),
        ("14,30", (14, 30)),
        ("1400", (14, 0)),
        ("0:00", (0, 0)),
        ("23:59", (23, 59)),
        ("8 00", (8, 0)),
        ("8:00:00", (8, 0)),
        ("8:3", (8, 30)),
        ("8:00.", (8, 0)),
        ("alle 9", (9, 0)),
        ("８:００", (8, 0)),  # cifre a larghezza piena
    ],
)
def test_parse_time_valido(testo, atteso):
    assert v.parse_time(testo) == atteso


@pytest.mark.parametrize(
    "testo",
    [
        None,
        "",
        "   ",
        "-",
        "–",
        "24:00",
        "25",
        "8:60",
        "2400",
        "12345",
        "abc",
        "8:00-11:00",
        "8 e 30",
        "x",
        "99:99",
    ],
)
def test_parse_time_non_valido(testo):
    assert v.parse_time(testo) is None


@pytest.mark.parametrize(
    "testo, atteso",
    [
        ("8.00", "08:00"),
        ("l1:00", "11:00"),
        ("1400", "14:00"),
        ("7:5", "07:50"),
        ("", None),
        (None, None),
        ("25:00", None),
    ],
)
def test_normalize_time(testo, atteso):
    assert v.normalize_time(testo) == atteso


@pytest.mark.parametrize(
    "valore, atteso",
    [
        ("1,5", 1.5),
        ("1.5", 1.5),
        ("3", 3.0),
        ("3h", 3.0),
        ("3 ore", 3.0),
        ("3 ORE", 3.0),
        ("2:30", 2.5),
        ("2h30", 2.5),
        ("1 e 30", 1.5),
        ("1 ora e 30", 1.5),
        ("1 e mezza", 1.5),
        ("1½", 1.5),
        ("1 ½", 1.5),
        ("½", 0.5),
        ("mezz'ora", 0.5),
        ("30 min", 0.5),
        ("0,25", 0.25),
        ("0", 0.0),
        ("l,5", 1.5),
        ("24", 24.0),
        (3, 3.0),
        (1.5, 1.5),
        (0, 0.0),
    ],
)
def test_parse_hours_valido(valore, atteso):
    assert v.parse_hours(valore) == pytest.approx(atteso)


@pytest.mark.parametrize(
    "valore", [None, "", "-", "-1", -1, "25", 25, 100.0, "abc", "2:75", float("nan"), float("inf"), True]
)
def test_parse_hours_non_valido(valore):
    assert v.parse_hours(valore) is None


@pytest.mark.parametrize(
    "a, b, atteso",
    [
        ("08:00", "11:00", 3.0),
        ("8.00", "14,00", 6.0),
        ("08:00", "09:30", 1.5),
        ("08:00", "08:20", 1 / 3),
        ("11:00", "08:00", None),
        ("08:00", "08:00", None),
        ("08:00", None, None),
        (None, "11:00", None),
        ("abc", "11:00", None),
    ],
)
def test_hours_between(a, b, atteso):
    risultato = v.hours_between(a, b)
    if atteso is None:
        assert risultato is None
    else:
        assert risultato == pytest.approx(atteso)


@pytest.mark.parametrize(
    "x, atteso",
    [
        (3, "3"),
        (3.0, "3"),
        (1.5, "1,5"),
        (2.25, "2,25"),
        (0.3333, "0,33"),
        (10.1, "10,1"),
        (0, "0"),
        (-0.0, "0"),
        (-1.5, "-1,5"),
        (49.5, "49,5"),
        (None, ""),
        (float("nan"), ""),
    ],
)
def test_format_ore(x, atteso):
    assert v.format_ore(x) == atteso


def test_hours_label_e_percentuale():
    assert v.hours_label(1) == "1 ora"
    assert v.hours_label(3) == "3 ore"
    assert v.hours_label(1.5) == "1,5 ore"
    assert v.hours_label(0) == "0 ore"
    assert v.hours_label(None) == ""
    assert v.format_percentuale(50) == "50%"
    assert v.format_percentuale(100 / 3) == "33,3%"
    assert v.format_intervallo("8:00", "11.00") == "08:00–11:00"
    assert v.format_intervallo("8:00", None) == "08:00–?"


@pytest.mark.parametrize(
    "riga, attese",
    [
        (DayRow(giorno=1, eff_entrata="08:00", eff_uscita="11:00", ore_dichiarate=2.5), 2.5),
        (DayRow(giorno=1, eff_entrata="08:00", eff_uscita="11:00"), 3.0),
        (
            DayRow(
                giorno=1, eff_entrata="08:00", eff_uscita="11:00", ore_dichiarate=3, assenza_operatore=True
            ),
            0.0,
        ),
        (DayRow(giorno=1, ore_dichiarate=1.5, assenza_alunno=True, trattino_effettivo=True), 1.5),
        (DayRow(giorno=1, prog_entrata="08:00", prog_uscita="11:00", trattino_effettivo=True), 0.0),
        (DayRow(giorno=1, eff_entrata="11:00", eff_uscita="08:00"), 0.0),
        (DayRow(giorno=1, eff_entrata="08:00", eff_uscita="11:00", ore_dichiarate=40), 3.0),
        (DayRow(giorno=1), 0.0),
    ],
)
def test_ore_riconosciute(riga, attese):
    assert v.ore_riconosciute(riga) == pytest.approx(attese)


# --- Documento pulito: nessuna anomalia ---------------------------------------


def test_mese_pulito_nessuna_anomalia():
    anomalie, t = v.validate(make_header(), mese_pulito())
    assert anomalie == []
    assert t.stato == "ok"
    assert t.ore_riconosciute == 60
    assert t.ore_dichiarate == 60
    assert t.ore_calcolate == 60
    assert t.ore_programmate == 60
    assert t.giorni_programmati == 20
    assert t.giorni_lavorati == 20
    assert t.totale_mensile_dichiarato == 60
    assert t.differenza_totale == 0
    assert (t.n_errori, t.n_attenzioni, t.n_info) == (0, 0, 0)
    assert [(w.dal, w.al, w.ore) for w in t.settimane] == [
        (1, 1, 0),
        (2, 8, 15),
        (9, 15, 15),
        (16, 22, 15),
        (23, 28, 15),
    ]


# --- Ogni codice: casi che lo attivano ----------------------------------------

Mutazione = Callable[[], tuple[Header, list[DayRow]]]


def _caso(
    header_kw: dict | None = None, righe: Callable[[list[DayRow]], list[DayRow]] | None = None
) -> Mutazione:
    def build() -> tuple[Header, list[DayRow]]:
        rows = mese_pulito()
        if righe is not None:
            rows = righe(rows)
        return make_header(**(header_kw or {})), rows

    return build


CASI_ATTIVAZIONE: list[tuple[str, Mutazione, str, str, int | None]] = [
    (
        "E01 ore diverse",
        _caso(
            righe=lambda r: replace_row(r, 3, eff_uscita="14:00"), header_kw={"totale_mensile_dichiarato": 60}
        ),
        "E01_ORE_NON_COERENTI",
        "errore",
        3,
    ),
    (
        "E01 ore non plausibili",
        _caso(righe=lambda r: replace_row(r, 3, ore_dichiarate=30)),
        "E01_ORE_NON_COERENTI",
        "errore",
        3,
    ),
    (
        "E01 assenza alunno con ore superiori",
        _caso(righe=lambda r: replace_row(r, 3, assenza_alunno=True, ore_dichiarate=4)),
        "E01_ORE_NON_COERENTI",
        "errore",
        3,
    ),
    (
        "E02 totale diverso",
        _caso({"totale_mensile_dichiarato": 61.5}),
        "E02_TOTALE_MENSILE_DIVERSO",
        "errore",
        None,
    ),
    (
        "E03 orario illeggibile come numero",
        _caso(righe=lambda r: replace_row(r, 5, eff_entrata="8:7O")),
        "E03_ORARIO_NON_VALIDO",
        "errore",
        5,
    ),
    (
        "E03 uscita prima dell'entrata",
        _caso(righe=lambda r: replace_row(r, 5, eff_entrata="11:00", eff_uscita="08:00")),
        "E03_ORARIO_NON_VALIDO",
        "errore",
        5,
    ),
    (
        "E03 programmato invertito",
        _caso(righe=lambda r: replace_row(r, 5, prog_entrata="11:00", prog_uscita="11:00")),
        "E03_ORARIO_NON_VALIDO",
        "errore",
        5,
    ),
    (
        "E03 orario incompleto",
        _caso(righe=lambda r: replace_row(r, 5, eff_uscita=None)),
        "E03_ORARIO_NON_VALIDO",
        "errore",
        5,
    ),
    (
        "E04 giorno inesistente",
        _caso(righe=lambda r: replace_row(r, 30, eff_entrata="08:00", eff_uscita="11:00", ore_dichiarate=3)),
        "E04_GIORNO_INESISTENTE",
        "errore",
        30,
    ),
    (
        "E05 orario illeggibile",
        _caso(righe=lambda r: replace_row(r, 6, eff_uscita=None, illeggibili=["eff_uscita"])),
        "E05_CAMPO_ILLEGGIBILE",
        "errore",
        6,
    ),
    (
        "E05 ore illeggibili",
        _caso(righe=lambda r: replace_row(r, 6, ore_dichiarate=None, illeggibili=["ore_dichiarate"])),
        "E05_CAMPO_ILLEGGIBILE",
        "errore",
        6,
    ),
    (
        "E05 nota illeggibile",
        _caso(righe=lambda r: replace_row(r, 6, illeggibili=["note"])),
        "E05_CAMPO_ILLEGGIBILE",
        "attenzione",
        6,
    ),
    (
        "E05 totale illeggibile",
        _caso({"totale_mensile_dichiarato": None, "illeggibili": ["totale_mensile_dichiarato"]}),
        "E05_CAMPO_ILLEGGIBILE",
        "errore",
        None,
    ),
    (
        "E05 istituto illeggibile",
        _caso({"istituto": None, "illeggibili": ["istituto"]}),
        "E05_CAMPO_ILLEGGIBILE",
        "attenzione",
        None,
    ),
    (
        "W01 firma mancante",
        _caso(righe=lambda r: replace_row(r, 9, firma=False)),
        "W01_FIRMA_MANCANTE",
        "attenzione",
        9,
    ),
    (
        "W02 ore senza orario",
        _caso(righe=lambda r: replace_row(r, 9, eff_entrata=None, eff_uscita=None)),
        "W02_ORE_SENZA_ORARIO",
        "attenzione",
        9,
    ),
    (
        "W03 domenica",
        _caso(
            {"totale_mensile_dichiarato": 63},
            righe=lambda r: replace_row(r, 8, **lavorato(8).model_dump(exclude={"giorno"})),
        ),
        "W03_GIORNO_FESTIVO",
        "attenzione",
        8,
    ),
    (
        "W03 sabato",
        _caso(
            {"totale_mensile_dichiarato": 63},
            righe=lambda r: replace_row(r, 7, **lavorato(7).model_dump(exclude={"giorno"})),
        ),
        "W03_GIORNO_FESTIVO",
        "info",
        7,
    ),
    (
        "W04 campo incerto",
        _caso(righe=lambda r: replace_row(r, 11, incerti=["eff_uscita"])),
        "W04_CAMPO_INCERTO",
        "attenzione",
        11,
    ),
    ("W04 intestazione incerta", _caso({"incerti": ["alunno"]}), "W04_CAMPO_INCERTO", "attenzione", None),
    (
        "W05 assenza operatore con ore",
        _caso({"totale_mensile_dichiarato": 60}, righe=lambda r: replace_row(r, 12, assenza_operatore=True)),
        "W05_ASSENZA_OPERATORE_CON_ORE",
        "attenzione",
        12,
    ),
    ("W06 ore PEI superate", _caso({"ore_pei": 12}), "W06_ORE_PEI_SUPERATE", "attenzione", None),
    ("W07 operatore mancante", _caso({"operatore": None}), "W07_INTESTAZIONE_INCOMPLETA", "attenzione", None),
    ("W07 alunno vuoto", _caso({"alunno": "   "}), "W07_INTESTAZIONE_INCOMPLETA", "attenzione", None),
    ("W07 mese mancante", _caso({"mese": None}), "W07_INTESTAZIONE_INCOMPLETA", "attenzione", None),
    ("W07 anno mancante", _caso({"anno": None}), "W07_INTESTAZIONE_INCOMPLETA", "attenzione", None),
    ("W07 mese non valido", _caso({"mese": 13}), "W07_INTESTAZIONE_INCOMPLETA", "attenzione", None),
    (
        "W08 firma coordinatore",
        _caso({"firma_coordinatore": False}),
        "W08_FIRMA_COORDINATORE_MANCANTE",
        "attenzione",
        None,
    ),
    (
        "W09 timbro referente",
        _caso({"timbro_referente": False}),
        "W09_TIMBRO_REFERENTE_MANCANTE",
        "attenzione",
        None,
    ),
    (
        "W10 totale assente",
        _caso({"totale_mensile_dichiarato": None}),
        "W10_TOTALE_MENSILE_ASSENTE",
        "attenzione",
        None,
    ),
    (
        "W11 doppia assenza",
        _caso(
            righe=lambda r: replace_row(
                r,
                13,
                assenza_alunno=True,
                assenza_operatore=True,
                eff_entrata=None,
                eff_uscita=None,
                ore_dichiarate=None,
            )
        ),
        "W11_DOPPIA_ASSENZA",
        "attenzione",
        13,
    ),
    (
        "W12 ore non indicate",
        _caso(righe=lambda r: replace_row(r, 13, ore_dichiarate=None)),
        "W12_ORE_NON_INDICATE",
        "attenzione",
        13,
    ),
    (
        "I01 assenza alunno",
        _caso(
            righe=lambda r: replace_row(
                r,
                4,
                assenza_alunno=True,
                eff_entrata=None,
                eff_uscita=None,
                ore_dichiarate=1.5,
                trattino_effettivo=True,
            )
        ),
        "I01_ASSENZA_ALUNNO",
        "info",
        4,
    ),
    (
        "I02 assenza operatore",
        _caso(
            righe=lambda r: replace_row(
                r,
                10,
                assenza_operatore=True,
                eff_entrata=None,
                eff_uscita=None,
                ore_dichiarate=None,
                note="104",
            )
        ),
        "I02_ASSENZA_OPERATORE",
        "info",
        10,
    ),
    (
        "I03 orario diverso",
        _caso(righe=lambda r: replace_row(r, 18, eff_entrata="08:30", eff_uscita="11:30")),
        "I03_ORARIO_DIVERSO",
        "info",
        18,
    ),
    (
        "I04 senza programmato",
        _caso(righe=lambda r: replace_row(r, 19, prog_entrata=None, prog_uscita=None)),
        "I04_ORE_SENZA_PROGRAMMATO",
        "info",
        19,
    ),
    (
        "I05 non svolto",
        _caso(
            righe=lambda r: replace_row(
                r,
                16,
                eff_entrata=None,
                eff_uscita=None,
                ore_dichiarate=None,
                firma=False,
                trattino_effettivo=True,
                note="PONTE DI CARNEVALE",
            )
        ),
        "I05_NON_SVOLTO",
        "info",
        16,
    ),
]


@pytest.mark.parametrize(
    "nome, build, codice, gravita, giorno", CASI_ATTIVAZIONE, ids=[c[0] for c in CASI_ATTIVAZIONE]
)
def test_codice_attivato(nome, build, codice, gravita, giorno):
    header, rows = build()
    anomalie, totals = v.validate(header, rows)
    trovate = [a for a in anomalie if a.codice == codice]
    assert trovate, f"{codice} non segnalato: {[a.codice for a in anomalie]}"
    assert any(a.giorno == giorno and a.gravita == gravita for a in trovate), trovate
    for a in trovate:
        assert a.messaggio and a.messaggio[0].isupper() and a.messaggio.endswith(".")
        assert codice in v.CODICI
    # coerenza dei contatori
    assert totals.n_errori == sum(a.gravita == "errore" for a in anomalie)
    assert totals.n_attenzioni == sum(a.gravita == "attenzione" for a in anomalie)
    assert totals.n_info == sum(a.gravita == "info" for a in anomalie)


def test_tutti_i_codici_hanno_un_caso_e_una_descrizione():
    coperti = {c[2] for c in CASI_ATTIVAZIONE} | {"E06_NON_FOGLIO_FIRMA"}
    assert coperti == set(v.CODICI)
    for codice, info in v.CODICI.items():
        assert codice[0] in "EWI"
        assert info["gravita"] in ("errore", "attenzione", "info")
        assert info["titolo"] and info["descrizione"]


# --- Casi che NON devono attivare un codice -----------------------------------

CASI_NON_ATTIVAZIONE: list[tuple[str, Mutazione, str]] = [
    (
        "E01 entro la tolleranza",
        _caso(righe=lambda r: replace_row(r, 3, ore_dichiarate=3.005)),
        "E01_ORE_NON_COERENTI",
    ),
    (
        "E01 assenza alunno con ore parziali",
        _caso(
            {"totale_mensile_dichiarato": 58.5},
            righe=lambda r: replace_row(r, 3, assenza_alunno=True, ore_dichiarate=1.5),
        ),
        "E01_ORE_NON_COERENTI",
    ),
    (
        "E01 orari non normalizzati",
        _caso(righe=lambda r: replace_row(r, 3, eff_entrata="8.00", eff_uscita="l1:00")),
        "E01_ORE_NON_COERENTI",
    ),
    ("E02 totale uguale", _caso(), "E02_TOTALE_MENSILE_DIVERSO"),
    (
        "E02 totale entro tolleranza",
        _caso({"totale_mensile_dichiarato": 60.004}),
        "E02_TOTALE_MENSILE_DIVERSO",
    ),
    (
        "E03 orari OCR rumorosi",
        _caso(righe=lambda r: replace_row(r, 5, eff_entrata="8:oo", eff_uscita="11:OO")),
        "E03_ORARIO_NON_VALIDO",
    ),
    (
        "E03 orario illeggibile segnalato da E05",
        _caso(righe=lambda r: replace_row(r, 5, eff_uscita=None, illeggibili=["eff_uscita"])),
        "E03_ORARIO_NON_VALIDO",
    ),
    ("E04 giorno 30 vuoto", _caso(), "E04_GIORNO_INESISTENTE"),
    (
        "E04 senza mese",
        _caso(
            {"mese": None}, righe=lambda r: replace_row(r, 30, **lavorato(30).model_dump(exclude={"giorno"}))
        ),
        "E04_GIORNO_INESISTENTE",
    ),
    (
        "W01 assenza operatore",
        _caso(
            {"totale_mensile_dichiarato": 57},
            righe=lambda r: replace_row(
                r,
                9,
                assenza_operatore=True,
                firma=False,
                eff_entrata=None,
                eff_uscita=None,
                ore_dichiarate=None,
            ),
        ),
        "W01_FIRMA_MANCANTE",
    ),
    (
        "W01 firma incerta",
        _caso(righe=lambda r: replace_row(r, 9, firma=False, incerti=["firma"])),
        "W01_FIRMA_MANCANTE",
    ),
    (
        "W01 firma illeggibile",
        _caso(righe=lambda r: replace_row(r, 9, firma=False, illeggibili=["firma"])),
        "W01_FIRMA_MANCANTE",
    ),
    (
        "W01 non svolto",
        _caso(
            {"totale_mensile_dichiarato": 57},
            righe=lambda r: replace_row(
                r,
                9,
                firma=False,
                eff_entrata=None,
                eff_uscita=None,
                ore_dichiarate=None,
                trattino_effettivo=True,
            ),
        ),
        "W01_FIRMA_MANCANTE",
    ),
    (
        "W02 con assenza alunno",
        _caso(
            {"totale_mensile_dichiarato": 58.5},
            righe=lambda r: replace_row(
                r, 9, eff_entrata=None, eff_uscita=None, ore_dichiarate=1.5, assenza_alunno=True
            ),
        ),
        "W02_ORE_SENZA_ORARIO",
    ),
    (
        "W02 orario illeggibile",
        _caso(
            righe=lambda r: replace_row(
                r, 9, eff_entrata=None, eff_uscita=None, illeggibili=["eff_entrata", "eff_uscita"]
            )
        ),
        "W02_ORE_SENZA_ORARIO",
    ),
    ("W03 weekend vuoto", _caso(), "W03_GIORNO_FESTIVO"),
    (
        "W03 senza mese",
        _caso(
            {"mese": None, "totale_mensile_dichiarato": 63},
            righe=lambda r: replace_row(r, 8, **lavorato(8).model_dump(exclude={"giorno"})),
        ),
        "W03_GIORNO_FESTIVO",
    ),
    (
        "W04 campo incerto e illeggibile",
        _caso(
            righe=lambda r: replace_row(
                r, 11, eff_uscita=None, incerti=["eff_uscita"], illeggibili=["eff_uscita"]
            )
        ),
        "W04_CAMPO_INCERTO",
    ),
    ("W06 pari al PEI", _caso({"ore_pei": 15}), "W06_ORE_PEI_SUPERATE"),
    ("W06 PEI assente", _caso({"ore_pei": None}), "W06_ORE_PEI_SUPERATE"),
    (
        "W07 operatore illeggibile",
        _caso({"operatore": None, "illeggibili": ["operatore"]}),
        "W07_INTESTAZIONE_INCOMPLETA",
    ),
    (
        "W08 firma coordinatore incerta",
        _caso({"firma_coordinatore": False, "incerti": ["firma_coordinatore"]}),
        "W08_FIRMA_COORDINATORE_MANCANTE",
    ),
    (
        "W09 timbro illeggibile",
        _caso({"timbro_referente": False, "illeggibili": ["timbro_referente"]}),
        "W09_TIMBRO_REFERENTE_MANCANTE",
    ),
    (
        "W10 totale illeggibile",
        _caso({"totale_mensile_dichiarato": None, "illeggibili": ["totale_mensile_dichiarato"]}),
        "W10_TOTALE_MENSILE_ASSENTE",
    ),
    (
        "W12 ore illeggibili",
        _caso(righe=lambda r: replace_row(r, 13, ore_dichiarate=None, illeggibili=["ore_dichiarate"])),
        "W12_ORE_NON_INDICATE",
    ),
    (
        "W12 assenza operatore",
        _caso(
            {"totale_mensile_dichiarato": 60},
            righe=lambda r: replace_row(r, 13, ore_dichiarate=None, assenza_operatore=True),
        ),
        "W12_ORE_NON_INDICATE",
    ),
    (
        "I03 orari equivalenti",
        _caso(righe=lambda r: replace_row(r, 18, eff_entrata="8.00", eff_uscita="1100")),
        "I03_ORARIO_DIVERSO",
    ),
    (
        "I05 con assenza alunno",
        _caso(
            {"totale_mensile_dichiarato": 57},
            righe=lambda r: replace_row(
                r, 16, eff_entrata=None, eff_uscita=None, ore_dichiarate=None, assenza_alunno=True
            ),
        ),
        "I05_NON_SVOLTO",
    ),
    ("I05 con orario effettivo", _caso(), "I05_NON_SVOLTO"),
    (
        "I05 orario effettivo illeggibile",
        _caso(
            {"totale_mensile_dichiarato": 57},
            righe=lambda r: replace_row(
                r, 16, eff_entrata=None, eff_uscita=None, ore_dichiarate=None, illeggibili=["eff_entrata"]
            ),
        ),
        "I05_NON_SVOLTO",
    ),
]


@pytest.mark.parametrize(
    "nome, build, codice", CASI_NON_ATTIVAZIONE, ids=[c[0] for c in CASI_NON_ATTIVAZIONE]
)
def test_codice_non_attivato(nome, build, codice):
    header, rows = build()
    anomalie, _ = v.validate(header, rows)
    assert codice not in codici(anomalie), [a.messaggio for a in anomalie if a.codice == codice]


# --- Messaggi -----------------------------------------------------------------


def test_messaggio_e01():
    rows = replace_row(mese_pulito(), 3, eff_uscita="14:00")
    anomalie, _ = v.validate(make_header(), rows)
    (e01,) = trova(anomalie, "E01_ORE_NON_COERENTI")
    assert (
        e01.messaggio == "Giorno 3: dichiarate 3 ore ma l'orario effettivo 08:00–14:00 corrisponde a 6 ore."
    )
    assert (e01.campo, e01.valore_letto, e01.valore_atteso) == ("ore_dichiarate", "3", "6")


def test_messaggio_e02():
    rows = mese_pulito()
    for g in (2, 4, 5):
        rows = replace_row(
            rows,
            g,
            ore_dichiarate=None,
            eff_entrata=None,
            eff_uscita=None,
            prog_entrata=None,
            prog_uscita=None,
            firma=False,
        )
    rows = replace_row(rows, 3, assenza_alunno=True, ore_dichiarate=1.5, eff_entrata=None, eff_uscita=None)
    # 16 giorni da 3 ore = 48 ore, più 1,5 di assenza alunno = 49,5; ne dichiariamo 51
    anomalie, t = v.validate(make_header(totale_mensile_dichiarato=51), rows)
    assert t.ore_dichiarate == 49.5
    (e02,) = trova(anomalie, "E02_TOTALE_MENSILE_DIVERSO")
    assert e02.messaggio == "Totale mensile dichiarato 51 ore, somma dei giorni 49,5 ore (differenza 1,5)."
    assert t.differenza_totale == -1.5

    rows48 = replace_row(
        rows, 3, assenza_alunno=False, ore_dichiarate=None, prog_entrata=None, prog_uscita=None, firma=False
    )
    anomalie, t = v.validate(make_header(totale_mensile_dichiarato=49.5), rows48)
    (e02,) = trova(anomalie, "E02_TOTALE_MENSILE_DIVERSO")
    assert e02.messaggio == "Totale mensile dichiarato 49,5 ore, somma dei giorni 48 ore (differenza 1,5)."
    assert t.differenza_totale == -1.5
    assert (e02.valore_letto, e02.valore_atteso) == ("49,5", "48")


def test_messaggi_assenze_e_non_svolto():
    rows = replace_row(
        mese_pulito(),
        4,
        assenza_alunno=True,
        eff_entrata=None,
        eff_uscita=None,
        ore_dichiarate=1.5,
        trattino_effettivo=True,
    )
    rows = replace_row(
        rows,
        10,
        assenza_operatore=True,
        eff_entrata=None,
        eff_uscita=None,
        ore_dichiarate=None,
        note="104",
        trattino_effettivo=True,
    )
    rows = replace_row(
        rows,
        16,
        eff_entrata=None,
        eff_uscita=None,
        ore_dichiarate=None,
        firma=False,
        note="PONTE DI CARNEVALE",
        trattino_effettivo=True,
    )
    anomalie, t = v.validate(make_header(totale_mensile_dichiarato=52.5), rows)
    assert [a.codice for a in anomalie] == ["I01_ASSENZA_ALUNNO", "I02_ASSENZA_OPERATORE", "I05_NON_SVOLTO"]
    i01, i02, i05 = anomalie
    assert i01.messaggio == "Giorno 4: assenza dell'alunno; riconosciute 1,5 ore su 3 programmate (50%)."
    assert "104" in i02.messaggio and "Legge 104/1992" in i02.messaggio
    assert i02.messaggio.startswith("Giorno 10: assenza dell'operatore")
    assert "prestazione programmata non svolta" in i05.messaggio
    assert "«PONTE DI CARNEVALE»" in i05.messaggio
    assert t.stato == "ok"


def test_messaggio_w03_festivo_con_nome():
    # 6 aprile 2026 = Lunedì dell'Angelo
    header = make_header(mese=4, anno=2026, totale_mensile_dichiarato=3)
    rows = empty_rows()
    rows[5] = lavorato(6)
    anomalie, _ = v.validate(header, rows)
    (w03,) = trova(anomalie, "W03_GIORNO_FESTIVO")
    assert w03.gravita == "attenzione"
    assert "Pasquetta" in w03.messaggio and w03.messaggio.startswith("Giorno 6 (lunedì")


def test_messaggio_w06():
    anomalie, _ = v.validate(make_header(ore_pei=12), mese_pulito())
    w06 = trova(anomalie, "W06_ORE_PEI_SUPERATE")
    assert len(w06) == 4  # le 4 settimane piene da 15 ore
    assert w06[0].messaggio == (
        "Settimana 2 (dal 2 all'8 febbraio): 15 ore, oltre il limite settimanale del PEI (12 ore; eccedenza 3 ore)."
    )


def test_messaggio_e04_e_esclusione_dai_totali():
    rows = replace_row(
        mese_pulito(), 30, **{**lavorato(30).model_dump(exclude={"giorno"}), "incerti": ["eff_uscita"]}
    )
    anomalie, t = v.validate(make_header(), rows)
    (e04,) = trova(anomalie, "E04_GIORNO_INESISTENTE")
    assert "febbraio 2026" in e04.messaggio
    assert trova(anomalie, "E04_GIORNO_INESISTENTE", 30)
    # nessun altro controllo sulla riga inesistente, nessun conteggio
    assert [a.codice for a in anomalie if a.giorno == 30] == ["E04_GIORNO_INESISTENTE"]
    assert t.ore_riconosciute == 60 and t.ore_dichiarate == 60 and t.giorni_lavorati == 20
    assert t.campi_incerti == 0
    assert t.stato == "errori"


def test_w04_un_anomalia_per_campo_e_valore_letto():
    rows = replace_row(mese_pulito(), 11, incerti=["eff_uscita", "ore_dichiarate", "eff_uscita", "firma"])
    header = make_header(incerti=["alunno", "ore_pei"])
    anomalie, t = v.validate(header, rows)
    w04 = trova(anomalie, "W04_CAMPO_INCERTO")
    assert [(a.giorno, a.campo) for a in w04] == [
        (None, "alunno"),
        (None, "ore_pei"),
        (11, "eff_uscita"),
        (11, "ore_dichiarate"),
        (11, "firma"),
    ]
    assert w04[2].valore_letto == "11:00"
    assert w04[3].valore_letto == "3"
    assert w04[4].valore_letto == "presente"
    assert w04[0].valore_letto == "BIANCHI LUCA"
    assert t.campi_incerti == 5
    assert t.stato == "da_verificare"


def test_e05_gravita_per_campo():
    rows = replace_row(
        mese_pulito(),
        6,
        eff_entrata=None,
        prog_uscita=None,
        ore_dichiarate=None,
        note=None,
        firma=False,
        illeggibili=["eff_entrata", "prog_uscita", "ore_dichiarate", "note", "firma", "assenza_alunno"],
        incerti=["note"],
    )
    header = make_header(
        totale_mensile_dichiarato=None, operatore=None, illeggibili=["totale_mensile_dichiarato", "operatore"]
    )
    anomalie, t = v.validate(header, rows)
    gravita = {(a.giorno, a.campo): a.gravita for a in anomalie if a.codice == "E05_CAMPO_ILLEGGIBILE"}
    assert gravita == {
        (None, "operatore"): "attenzione",
        (None, "totale_mensile_dichiarato"): "errore",
        (6, "prog_uscita"): "errore",
        (6, "eff_entrata"): "errore",
        (6, "ore_dichiarate"): "errore",
        (6, "assenza_alunno"): "attenzione",
        (6, "firma"): "attenzione",
        (6, "note"): "attenzione",
    }
    # gli illeggibili non sono anche incerti, né generano W07/W10/W01/E03/W12
    assert not trova(anomalie, "W04_CAMPO_INCERTO")
    assert not (
        {
            "W07_INTESTAZIONE_INCOMPLETA",
            "W10_TOTALE_MENSILE_ASSENTE",
            "W01_FIRMA_MANCANTE",
            "E03_ORARIO_NON_VALIDO",
            "W12_ORE_NON_INDICATE",
        }
        & codici(anomalie)
    )
    assert t.campi_illeggibili == 8
    assert t.campi_incerti == 0


@pytest.mark.parametrize(
    "kw, frammento",
    [
        ({"eff_entrata": "8:7O"}, "entrata effettiva non interpretabile come orario («8:7O»)"),
        (
            {"eff_entrata": "11:00", "eff_uscita": "08:00"},
            "l'uscita effettiva (08:00) non è successiva all'entrata (11:00)",
        ),
        ({"eff_uscita": None}, "orario effettivo incompleto, manca l'uscita (entrata 08:00)"),
        ({"prog_entrata": None}, "orario programmato incompleto, manca l'entrata (uscita 11:00)"),
    ],
)
def test_messaggi_e03(kw, frammento):
    anomalie, _ = v.validate(make_header(), replace_row(mese_pulito(), 5, **kw))
    messaggi = [a.messaggio for a in trova(anomalie, "E03_ORARIO_NON_VALIDO", 5)]
    assert any(frammento in m for m in messaggi), messaggi


def test_w01_conteggio_firme_mancanti():
    rows = mese_pulito()
    for g in (2, 3, 4):
        rows = replace_row(rows, g, firma=False)
    anomalie, t = v.validate(make_header(), rows)
    assert t.firme_mancanti == 3
    assert trova(anomalie, "W01_FIRMA_MANCANTE", 2)[0].messaggio == (
        "Giorno 2: manca la firma dell'operatore (orario 08:00–11:00, 3 ore dichiarate)."
    )


def test_w12_ore_calcolate_e_riconosciute():
    rows = replace_row(mese_pulito(), 13, ore_dichiarate=None, eff_uscita="11:30")
    anomalie, t = v.validate(make_header(), rows)
    (w12,) = trova(anomalie, "W12_ORE_NON_INDICATE")
    assert w12.messaggio == (
        "Giorno 13: ore effettive non indicate sul foglio; riconosciute 3,5 ore calcolate dall'orario 08:00–11:30."
    )
    assert t.ore_riconosciute == 60.5
    assert t.ore_dichiarate == 57
    assert t.ore_calcolate == 60.5
    assert t.differenza_totale == -3
    assert "E02_TOTALE_MENSILE_DIVERSO" in codici(anomalie)


# --- Totali e settimane -------------------------------------------------------


def test_totali_con_assenze():
    rows = replace_row(
        mese_pulito(), 4, assenza_alunno=True, eff_entrata=None, eff_uscita=None, ore_dichiarate=1.5
    )
    rows = replace_row(
        rows, 10, assenza_operatore=True, eff_entrata=None, eff_uscita=None, ore_dichiarate=None
    )
    rows = replace_row(rows, 11, assenza_alunno=True, eff_entrata=None, eff_uscita=None, ore_dichiarate=0)
    anomalie, t = v.validate(make_header(totale_mensile_dichiarato=52.5), rows)
    assert t.assenze_alunno == 2
    assert t.assenze_operatore == 1
    assert t.ore_riconosciute == 52.5  # 17 giorni da 3 ore + 1,5
    assert t.ore_dichiarate == 52.5
    assert t.ore_calcolate == 51
    assert t.ore_programmate == 60
    assert t.giorni_programmati == 20
    assert t.giorni_lavorati == 18
    assert t.stato == "ok"
    i01 = trova(anomalie, "I01_ASSENZA_ALUNNO", 11)[0]
    assert "nessuna ora riconosciuta (3 ore programmate)" in i01.messaggio


def test_settimane_ore_pei_proporzionate():
    _, t = v.validate(make_header(ore_pei=15), mese_pulito())
    assert [(w.settimana, w.dal, w.al, w.ore, w.ore_pei, w.differenza) for w in t.settimane] == [
        (1, 1, 1, 0, 0, 0),
        (2, 2, 8, 15, 15, 0),
        (3, 9, 15, 15, 15, 0),
        (4, 16, 22, 15, 15, 0),
        (5, 23, 28, 15, 15, 0),
    ]


def test_settimane_con_festivo_e_sabato():
    # Aprile 2026: Pasquetta lunedì 6; settimana 6-12 con 4 giorni scolastici.
    rows = empty_rows()
    for g in (7, 8, 9, 10):
        rows[g - 1] = lavorato(g)
    header = make_header(mese=4, anno=2026, ore_pei=15, totale_mensile_dichiarato=12)
    _, t = v.validate(header, rows)
    sett = {w.dal: w for w in t.settimane}
    assert sett[6].ore == 12 and sett[6].ore_pei == 12 and sett[6].differenza == 0
    assert sett[1].al == 5 and sett[1].ore_pei == 9  # mer-ven 1-3 aprile
    # Con attività di sabato la settimana scolastica è di 6 giorni.
    rows[10] = lavorato(11)  # sabato 11 aprile
    _, t = v.validate(header.model_copy(update={"totale_mensile_dichiarato": 15}), rows)
    sett = {w.dal: w for w in t.settimane}
    assert sett[6].ore == 15 and sett[6].ore_pei == 12.5  # 5 giorni su 6
    assert sett[1].ore_pei == 10  # mer-sab 1-4 aprile: 4 giorni su 6


def test_senza_mese_salta_calendario_ma_controlla_il_resto():
    rows = replace_row(mese_pulito(), 3, eff_uscita="14:00")
    rows = replace_row(rows, 30, **lavorato(30).model_dump(exclude={"giorno"}))
    rows = replace_row(rows, 8, **lavorato(8).model_dump(exclude={"giorno"}))
    anomalie, t = v.validate(make_header(mese=None, totale_mensile_dichiarato=66, ore_pei=1), rows)
    c = codici(anomalie)
    assert "E01_ORE_NON_COERENTI" in c
    assert "W07_INTESTAZIONE_INCOMPLETA" in c
    assert not {"E04_GIORNO_INESISTENTE", "W03_GIORNO_FESTIVO", "W06_ORE_PEI_SUPERATE"} & c
    assert t.settimane == []
    assert t.ore_riconosciute == 66  # giorni 8 e 30 conteggiati
    assert t.giorni_lavorati == 22


@pytest.mark.parametrize(
    "header_kw, righe, stato",
    [
        ({}, None, "ok"),
        ({"totale_mensile_dichiarato": 1}, None, "errori"),
        ({"firma_coordinatore": False}, None, "da_verificare"),
        ({}, lambda r: replace_row(r, 3, incerti=["eff_entrata"]), "da_verificare"),
        ({}, lambda r: replace_row(r, 3, illeggibili=["note"]), "da_verificare"),
        ({}, lambda r: replace_row(r, 3, eff_entrata="08:30", eff_uscita="11:30"), "ok"),  # solo info
    ],
)
def test_stato(header_kw, righe, stato):
    header, rows = _caso(header_kw, righe)()
    _, t = v.validate(header, rows)
    assert t.stato == stato


def test_incerti_contati_al_netto_degli_illeggibili():
    rows = replace_row(mese_pulito(), 3, incerti=["eff_entrata", "note"], illeggibili=["note"], note=None)
    header = make_header(incerti=["ente", "lotto"], illeggibili=["lotto"], lotto=None)
    _, t = v.validate(header, rows)
    assert t.campi_incerti == 2
    assert t.campi_illeggibili == 2


# --- Ordinamento, purezza, documento ------------------------------------------


def test_ordinamento_anomalie():
    rows = replace_row(mese_pulito(), 3, eff_uscita="14:00", firma=False)  # E01 + W01
    rows = replace_row(rows, 2, eff_entrata="08:30", eff_uscita="11:30")  # I03
    rows = replace_row(rows, 9, firma=False, incerti=["note"])  # W01 + W04
    header = make_header(firma_coordinatore=False, totale_mensile_dichiarato=1)
    anomalie, _ = v.validate(header, rows)
    chiavi = [
        (a.giorno is not None, a.giorno or 0, {"errore": 0, "attenzione": 1, "info": 2}[a.gravita])
        for a in anomalie
    ]
    assert chiavi == sorted(chiavi)
    assert anomalie[0].giorno is None and anomalie[0].gravita == "errore"
    assert [a.codice for a in anomalie if a.giorno == 3] == [
        "E01_ORE_NON_COERENTI",
        "W01_FIRMA_MANCANTE",
        "I03_ORARIO_DIVERSO",
    ]
    assert [a.codice for a in anomalie if a.giorno == 9] == ["W01_FIRMA_MANCANTE", "W04_CAMPO_INCERTO"]


def test_validate_puro_e_deterministico():
    header = make_header(incerti=["alunno"])
    rows = replace_row(mese_pulito(), 3, eff_uscita="14:00", incerti=["eff_uscita"])
    prima = (header.model_dump(), [r.model_dump() for r in rows])
    a1, t1 = v.validate(header, rows)
    a2, t2 = v.validate(header, list(reversed(rows)))
    assert (header.model_dump(), [r.model_dump() for r in rows]) == prima
    assert [a.model_dump() for a in a1] == [a.model_dump() for a in a2]
    assert t1 == t2


def test_validate_document():
    doc = Document(id="abc123", source_file="esempio.pdf", header=make_header(), rows=mese_pulito())
    risultato = v.validate_document(doc)
    assert risultato is doc
    assert doc.totals.ore_riconosciute == 60 and doc.totals.stato == "ok" and doc.anomalies == []


def test_validate_document_non_foglio_firma():
    doc = Document(id="x", source_file="altro.pdf", is_foglio_firma=False)
    v.validate_document(doc)
    assert [a.codice for a in doc.anomalies] == ["E06_NON_FOGLIO_FIRMA"]
    assert doc.anomalies[0].gravita == "errore"
    assert doc.totals.stato == "errori" and doc.totals.n_errori == 1
    # se l'OCR ha comunque letto dei dati, si validano anche quelli
    doc = Document(
        id="y", source_file="dubbio.pdf", is_foglio_firma=False, header=make_header(), rows=mese_pulito()
    )
    v.validate_document(doc)
    assert doc.anomalies[0].codice == "E06_NON_FOGLIO_FIRMA"
    assert doc.totals.ore_riconosciute == 60


# --- Aiuti per interfaccia ed Excel -------------------------------------------


def test_anomalies_for_day_e_documento():
    rows = replace_row(mese_pulito(), 3, eff_uscita="14:00")
    anomalie, _ = v.validate(make_header(firma_coordinatore=False), rows)
    assert [a.codice for a in v.anomalies_for_day(anomalie, 3)] == [
        "E01_ORE_NON_COERENTI",
        "I03_ORARIO_DIVERSO",
    ]
    assert v.anomalies_for_day(anomalie, 4) == []
    assert [a.codice for a in v.anomalies_for_document(anomalie)] == ["W08_FIRMA_COORDINATORE_MANCANTE"]


@pytest.mark.parametrize(
    "kw, atteso",
    [
        ({}, "OK"),
        ({"eff_uscita": "14:00"}, "Errore: ore non coerenti con l'orario"),
        ({"eff_uscita": None, "illeggibili": ["eff_uscita"]}, "Errore: illeggibile: uscita effettiva"),
        ({"firma": False}, "Da verificare: firma operatore mancante"),
        (
            {"incerti": ["eff_entrata", "ore_dichiarate"]},
            "Da verificare: lettura incerta: entrata effettiva, tot. ore effettive",
        ),
        (
            {"firma": False, "incerti": ["note"]},
            "Da verificare: firma operatore mancante; lettura incerta: note",
        ),
        (
            {"assenza_alunno": True, "eff_entrata": None, "eff_uscita": None, "ore_dichiarate": 1.5},
            "Assenza alunno",
        ),
        (
            {"assenza_operatore": True, "eff_entrata": None, "eff_uscita": None, "ore_dichiarate": None},
            "Assenza operatore",
        ),
        (
            {
                "eff_entrata": None,
                "eff_uscita": None,
                "ore_dichiarate": None,
                "firma": False,
                "trattino_effettivo": True,
            },
            "Non svolto",
        ),
        ({"eff_entrata": "08:30", "eff_uscita": "11:30"}, "OK"),
    ],
)
def test_esito_riga(kw, atteso):
    rows = replace_row(mese_pulito(), 3, **kw)
    anomalie, _ = v.validate(make_header(totale_mensile_dichiarato=None), rows)
    assert v.esito_riga(rows[2], anomalie) == atteso


def test_esito_riga_vuota_e_inesistente():
    rows = replace_row(mese_pulito(), 30, note="CHIUSO")
    anomalie, _ = v.validate(make_header(), rows)
    assert v.esito_riga(rows[0], anomalie) == ""  # domenica vuota
    assert v.esito_riga(rows[29], anomalie) == "Errore: giorno inesistente nel mese"
    assert v.esito_riga(DayRow(giorno=5, note="SCIOPERO"), []) == "Non svolto"


def test_stato_campo():
    riga = DayRow(giorno=3, incerti=["eff_uscita", "note"], illeggibili=["note"])
    assert v.stato_campo(riga, "note") == "illeggibile"
    assert v.stato_campo(riga, "eff_uscita") == "incerto"
    assert v.stato_campo(riga, "eff_entrata") == ""
    assert v.stato_campo(make_header(incerti=["alunno"]), "alunno") == "incerto"


# --- Foglio reale (verità attesa, solo se disponibile) -------------------------


def _verita() -> dict:
    percorso = os.environ.get("SIRIO_SAMPLE_TRUTH")
    if not percorso:
        pytest.skip("SIRIO_SAMPLE_TRUTH non impostata")
    p = Path(percorso)
    if not p.is_file():
        pytest.skip(f"File di verità non trovato: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def test_verita_foglio_reale():
    dati = _verita()
    header = Header(**dati["header"])
    rows = [DayRow(**r) for r in dati["rows"]]
    anomalie, t = v.validate(header, rows)

    assert [a for a in anomalie if a.gravita == "errore"] == []
    assert [a for a in anomalie if a.gravita == "attenzione"] == []
    assert t.stato == "ok"
    assert t.ore_dichiarate == 49.5
    assert t.ore_riconosciute == 49.5
    assert t.totale_mensile_dichiarato == 49.5
    assert t.differenza_totale == 0
    assert t.giorni_lavorati == 17
    assert t.assenze_alunno == 1
    assert t.assenze_operatore == 1
    assert t.firme_mancanti == 0

    assert trova(anomalie, "I01_ASSENZA_ALUNNO", 4)
    i02 = trova(anomalie, "I02_ASSENZA_OPERATORE", 10)
    assert i02 and "104" in i02[0].messaggio
    for g in (16, 17):
        i05 = trova(anomalie, "I05_NON_SVOLTO", g)
        assert i05, g
        nota = rows[g - 1].note
        assert nota and nota in i05[0].messaggio

    assert [(w.dal, w.al, w.ore) for w in t.settimane] == [
        (1, 1, 0),
        (2, 8, 13.5),
        (9, 15, 12),
        (16, 22, 9),
        (23, 28, 15),
    ]

    esiti = {r.giorno: v.esito_riga(r, anomalie) for r in rows}
    assert esiti[4] == "Assenza alunno"
    assert esiti[10] == "Assenza operatore"
    assert esiti[16] == esiti[17] == "Non svolto"
    assert esiti[1] == "" and esiti[2] == "OK"

    doc = Document(id="reale", source_file="foglio.pdf", header=header, rows=rows)
    v.validate_document(doc)
    assert doc.totals == t
