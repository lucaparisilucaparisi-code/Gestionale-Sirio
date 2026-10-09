"""Test del calendario italiano (sirio/calendario.py)."""

from __future__ import annotations

from datetime import date

import pytest

from sirio import calendario as cal


def test_nomi_giorni_e_mesi():
    assert cal.GIORNI == ("lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica")
    assert cal.GIORNI_BREVI == ("lun", "mar", "mer", "gio", "ven", "sab", "dom")
    assert len(cal.MESI) == 12
    assert cal.MESI[0] == "gennaio" and cal.MESI[1] == "febbraio" and cal.MESI[11] == "dicembre"
    assert cal.nome_mese(2) == "febbraio"
    assert cal.etichetta_mese(2, 2026) == "febbraio 2026"
    with pytest.raises(ValueError):
        cal.nome_mese(13)


@pytest.mark.parametrize(
    "anno, attesa",
    [
        (2024, date(2024, 3, 31)),
        (2025, date(2025, 4, 20)),
        (2026, date(2026, 4, 5)),
        (2000, date(2000, 4, 23)),
        (2008, date(2008, 3, 23)),
        (2011, date(2011, 4, 24)),
        (2019, date(2019, 4, 21)),
        (2027, date(2027, 3, 28)),
        (2030, date(2030, 4, 21)),
        (2038, date(2038, 4, 25)),
        (1818, date(1818, 3, 22)),  # data più precoce possibile
        (1943, date(1943, 4, 25)),  # data più tardiva possibile
    ],
)
def test_pasqua(anno, attesa):
    assert cal.pasqua(anno) == attesa
    assert cal.pasqua(anno).weekday() == 6


def test_pasqua_sempre_domenica_tra_22_marzo_e_25_aprile():
    for anno in range(1900, 2201):
        p = cal.pasqua(anno)
        assert p.weekday() == 6
        assert date(anno, 3, 22) <= p <= date(anno, 4, 25)


@pytest.mark.parametrize("anno", [0, 1500, 10000, "2026"])
def test_pasqua_anno_non_valido(anno):
    with pytest.raises(ValueError):
        cal.pasqua(anno)


def test_festivita_2026():
    f = cal.festivita(2026)
    attese = {
        date(2026, 1, 1),
        date(2026, 1, 6),
        date(2026, 4, 5),  # Pasqua
        date(2026, 4, 6),  # Pasquetta
        date(2026, 4, 25),
        date(2026, 5, 1),
        date(2026, 6, 2),
        date(2026, 8, 15),
        date(2026, 9, 19),  # San Gennaro
        date(2026, 10, 4),  # San Francesco (festa nazionale dal 2026)
        date(2026, 11, 1),
        date(2026, 12, 8),
        date(2026, 12, 25),
        date(2026, 12, 26),
    }
    assert set(f) == attese
    assert list(f) == sorted(f)
    assert "Pasquetta" in f[date(2026, 4, 6)]
    assert "San Gennaro" in f[date(2026, 9, 19)]
    assert f[date(2026, 4, 5)] == "Pasqua"


def test_festivita_san_francesco_solo_dal_2026():
    assert date(2025, 10, 4) not in cal.festivita(2025)
    assert date(2027, 10, 4) in cal.festivita(2027)
    assert len(cal.festivita(2025)) == 13


def test_festivita_restituisce_copia():
    f = cal.festivita(2026)
    f.clear()
    assert cal.festivita(2026)


@pytest.mark.parametrize(
    "anno, mese, attesi",
    [
        (2026, 2, 28),
        (2024, 2, 29),
        (2000, 2, 29),
        (2100, 2, 28),
        (2026, 4, 30),
        (2026, 1, 31),
        (2026, 12, 31),
    ],
)
def test_giorni_nel_mese(anno, mese, attesi):
    assert cal.giorni_nel_mese(anno, mese) == attesi


@pytest.mark.parametrize("mese", [0, 13, -1])
def test_giorni_nel_mese_mese_non_valido(mese):
    with pytest.raises(ValueError):
        cal.giorni_nel_mese(2026, mese)


def test_tipo_giorno_febbraio_2026():
    weekend = {1, 7, 8, 14, 15, 21, 22, 28}
    for g in range(1, 32):
        tipo = cal.tipo_giorno(2026, 2, g)
        if g >= 29:
            assert tipo == "inesistente", g
        elif g in weekend:
            assert tipo == ("domenica" if g in {1, 8, 15, 22} else "sabato"), g
        else:
            assert tipo == "feriale", g
    assert cal.nome_festivita(2026, 2, 16) is None  # Carnevale non è festività nazionale


@pytest.mark.parametrize(
    "anno, mese, giorno, tipo",
    [
        (2026, 4, 5, "festivo"),  # Pasqua (domenica): prevale la festività
        (2026, 4, 6, "festivo"),  # Pasquetta
        (2026, 4, 25, "festivo"),  # Liberazione (sabato)
        (2026, 9, 19, "festivo"),  # San Gennaro (sabato)
        (2026, 12, 8, "festivo"),  # Immacolata (martedì)
        (2026, 12, 9, "feriale"),
        (2026, 12, 12, "sabato"),
        (2026, 12, 13, "domenica"),
        (2026, 2, 0, "inesistente"),
        (2026, 2, 30, "inesistente"),
        (2026, 13, 1, "inesistente"),
        (2026, 4, 31, "inesistente"),
    ],
)
def test_tipo_giorno(anno, mese, giorno, tipo):
    assert cal.tipo_giorno(anno, mese, giorno) == tipo


def test_nome_festivita_e_giorno():
    assert cal.nome_festivita(2026, 12, 25) == "Natale"
    assert cal.nome_festivita(2026, 12, 24) is None
    assert cal.nome_festivita(2026, 2, 30) is None
    assert cal.nome_giorno(2026, 2, 2) == "lunedì"
    assert cal.nome_giorno(2026, 2, 1, breve=True) == "dom"
    assert cal.nome_giorno(2026, 2, 30) is None
    assert cal.is_lavorativo(2026, 2, 2)
    assert not cal.is_lavorativo(2026, 2, 1)
    assert not cal.is_lavorativo(2026, 4, 6)


@pytest.mark.parametrize(
    "anno, mese, attese",
    [
        (2026, 2, [(1, 1), (2, 8), (9, 15), (16, 22), (23, 28)]),
        (2026, 3, [(1, 1), (2, 8), (9, 15), (16, 22), (23, 29), (30, 31)]),
        (2026, 6, [(1, 7), (8, 14), (15, 21), (22, 28), (29, 30)]),
        (2021, 2, [(1, 7), (8, 14), (15, 21), (22, 28)]),
        (2026, 1, [(1, 4), (5, 11), (12, 18), (19, 25), (26, 31)]),
    ],
)
def test_settimane_del_mese(anno, mese, attese):
    assert cal.settimane_del_mese(anno, mese) == attese


def test_settimane_proprieta():
    for anno in (2024, 2025, 2026, 2027):
        for mese in range(1, 13):
            sett = cal.settimane_del_mese(anno, mese)
            n = cal.giorni_nel_mese(anno, mese)
            assert sett[0][0] == 1 and sett[-1][1] == n
            for (dal, al), succ in zip(sett, [*sett[1:], None], strict=True):
                assert dal <= al
                assert al - dal <= 6
                if succ is not None:
                    assert succ[0] == al + 1
                    assert date(anno, mese, al).weekday() == 6  # termina di domenica
                    assert date(anno, mese, succ[0]).weekday() == 0
