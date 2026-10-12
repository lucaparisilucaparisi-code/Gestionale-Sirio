"""Test del secondo parere offline (GLM-OCR) sulle celle dubbie.

Quasi tutti i test girano senza modello: interpretazione delle letture, politica
di riga (``decidi_riga``) e lettura a lotti con un lettore finto. Il test con il
modello reale richiede GLM-OCR gia' scaricato nella cache indicata da
``SIRIO_HF_CACHE`` (nessun download durante i test).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sirio.engines import secondo_parere as sp
from sirio.engines.base import EngineError
from sirio.engines.secondo_parere import (
    Parere,
    SecondoParere,
    decidi_riga,
    interpreta,
    plausibile,
)


def P(valore, conf=0.9, stato="valore", testo=""):  # noqa: N802 - abbreviazione dei test
    return Parere(testo=testo or str(valore), valore=valore, stato=stato, confidenza=conf)


ROW = {"prog_entrata": "08:00", "prog_uscita": "11:00", "eff_entrata": "08:00", "eff_uscita": "11:00",
       "ore_dichiarate": 3.0}


# --------------------------------------------------------------------------
# Interpretazione delle letture
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("testo", "atteso"), [
    ("8:00", "08:00"), ("8.00", "08:00"), ("10,30", "10:30"), ("800", "08:00"), ("1100", "11:00"),
    ("13:30", "13:30"), ("8h30", "08:30"), ("l1:00", "11:00"), ("1O:30", "10:30"), ("12:30.", "12:30"),
])
def test_interpreta_orari(testo, atteso):
    assert interpreta("orario", testo) == ("valore", atteso)


@pytest.mark.parametrize("testo", ["111:00", "1h:00", "B. $ \\textcircled{1} $", "11i:00", "12=80", "PONTE"])
def test_interpreta_orari_illeggibili(testo):
    assert interpreta("orario", testo) == ("illeggibile", None)


@pytest.mark.parametrize("testo", ["", "-", "---", "/", "EMPTY", "```markdown\n\n```", "  "])
def test_interpreta_celle_vuote(testo):
    assert interpreta("orario", testo)[0] == "vuota"
    assert interpreta("ore", testo)[0] == "vuota"


@pytest.mark.parametrize(("testo", "atteso"), [
    ("3", 3.0), ("1,5", 1.5), ("2.5", 2.5), ("2h30'", 2.5), ("1h15'", 1.25), ("0", 0.0), ("3h", 3.0),
])
def test_interpreta_ore(testo, atteso):
    stato, v = interpreta("ore", testo)
    assert stato == "valore" and v == pytest.approx(atteso)


def test_interpreta_note_e_numeri():
    assert interpreta("nota", "Scuola chiusa") == ("valore", "SCUOLA CHIUSA")
    assert interpreta("nota", ".")[0] == "vuota"
    assert interpreta("numero", "25") == ("valore", 25.0)
    assert interpreta("ore", "abc")[0] == "illeggibile"


def test_plausibile():
    assert plausibile("orario", "08:00") and plausibile("orario", "18:00")
    assert not plausibile("orario", "19:30") and not plausibile("orario", "06:00")
    assert not plausibile("orario", "08:07") and not plausibile("orario", None)
    assert plausibile("ore", 2.5) and plausibile("ore", 1.25)
    assert not plausibile("ore", 0.0) and not plausibile("ore", 2.2) and not plausibile("ore", 12.0)


# --------------------------------------------------------------------------
# Politica di riga
# --------------------------------------------------------------------------

def test_conferma_con_relazione_esatta_su_celle_sicure():
    es = decidi_riga(ROW, {"eff_uscita"}, {"eff_uscita": P("11:00")})
    assert es.confermati == {"eff_uscita"}
    assert es.segnalati == set()
    assert es.valori["eff_uscita"] == "11:00" and not es.sostituiti


def test_conferma_reciproca_solo_con_il_calcolo_delle_ore():
    # entrata e uscita dubbie ma confermate dal secondo parere, ore lette sicure: 11 - 8 = 3
    seg = {"eff_entrata", "eff_uscita"}
    row = dict(ROW, prog_entrata=None, prog_uscita=None)
    es = decidi_riga(row, seg, {"eff_entrata": P("08:00"), "eff_uscita": P("11:00")})
    assert es.confermati == seg


def test_celle_gemelle_lette_male_allo_stesso_modo_non_si_confermano():
    # "14:00" letto "16:00" da entrambi i modelli nelle due colonne: programmato = effettivo
    # torna, ma fra due celle dubbie non e' una conferma (errore correlato della stessa grafia)
    row = {"prog_entrata": "12:00", "prog_uscita": "16:00", "eff_entrata": "12:00", "eff_uscita": "16:00",
           "ore_dichiarate": 2.0}
    seg = {"prog_uscita", "eff_uscita", "ore_dichiarate"}
    es = decidi_riga(row, seg, {"prog_uscita": P("16:00"), "eff_uscita": P("16:00"), "ore_dichiarate": P(2.0)})
    assert es.segnalati == seg and not es.confermati


def test_nessuna_conferma_se_il_secondo_parere_e_diverso():
    es = decidi_riga(ROW, {"eff_uscita"}, {"eff_uscita": P("14:00")})
    assert "eff_uscita" in es.segnalati and not es.confermati


def test_nessuna_conferma_senza_relazione():
    row = dict(ROW, prog_uscita=None, ore_dichiarate=None)
    es = decidi_riga(row, {"eff_uscita"}, {"eff_uscita": P("11:00")})
    assert es.segnalati == {"eff_uscita"}


def test_nessuna_conferma_se_la_relazione_usa_celle_dubbie_non_confermate():
    # prog_uscita e' segnalato e il secondo parere non lo conferma: non puo' fare da garante
    es = decidi_riga(dict(ROW, ore_dichiarate=None), {"eff_uscita", "prog_uscita"},
                     {"eff_uscita": P("11:00"), "prog_uscita": P(None, stato="illeggibile")})
    assert es.segnalati == {"eff_uscita", "prog_uscita"}


def test_nessuna_conferma_con_poca_fiducia_o_campo_bloccato():
    assert decidi_riga(ROW, {"eff_uscita"}, {"eff_uscita": P("11:00", conf=0.2)}).segnalati == {"eff_uscita"}
    es = decidi_riga(ROW, {"eff_uscita"}, {"eff_uscita": P("11:00")}, bloccati={"eff_uscita"})
    assert es.segnalati == {"eff_uscita"}


def test_nessuna_conferma_fuori_fascia_scolastica():
    row = {"prog_entrata": "11:30", "prog_uscita": "19:30", "eff_entrata": "11:30", "eff_uscita": "19:30",
           "ore_dichiarate": None}
    seg = {"prog_uscita", "eff_uscita"}
    es = decidi_riga(row, seg, {"prog_uscita": P("19:30"), "eff_uscita": P("19:30")})
    assert es.segnalati == seg


def test_note_e_campi_estranei_restano_segnalati():
    es = decidi_riga(ROW, {"note", "firma"}, {"note": P("SCUOLA CHIUSA")})
    assert es.segnalati == {"note", "firma"}


def test_ore_con_assenza_alunno_non_confermate_dagli_orari():
    row = dict(ROW, ore_dichiarate=1.5)
    es = decidi_riga(row, {"ore_dichiarate"}, {"ore_dichiarate": P(1.5)}, assenza_alunno=True)
    assert es.segnalati == {"ore_dichiarate"}


def test_sostituzione_resta_segnalata():
    # il motore legge 16:00, il secondo parere 14:00: solo 14:00 rende la riga coerente
    row = {"prog_entrata": "12:00", "prog_uscita": "14:00", "eff_entrata": "12:00", "eff_uscita": "16:00",
           "ore_dichiarate": 2.0}
    es = decidi_riga(row, {"eff_uscita"}, {"eff_uscita": P("14:00")})
    assert es.sostituiti == {"eff_uscita": ("16:00", "14:00")}
    assert es.valori["eff_uscita"] == "14:00"
    assert "eff_uscita" in es.segnalati and not es.confermati


def test_sostituzione_congiunta_di_piu_celle():
    row = {"prog_entrata": "09:00", "prog_uscita": "13:30", "eff_entrata": "09:00", "eff_uscita": "13:30",
           "ore_dichiarate": 2.0}
    seg = {"prog_uscita", "eff_uscita", "ore_dichiarate"}
    es = decidi_riga(row, seg, {"prog_uscita": P("13:00"), "eff_uscita": P("13:00"), "ore_dichiarate": P(4.0)})
    assert es.valori["prog_uscita"] == "13:00" and es.valori["eff_uscita"] == "13:00"
    assert es.valori["ore_dichiarate"] == 4.0
    assert es.segnalati == seg


def test_a_parita_vince_il_motore():
    row = dict(ROW, prog_uscita=None, ore_dichiarate=None)
    es = decidi_riga(row, {"eff_uscita"}, {"eff_uscita": P("12:00")})
    assert not es.sostituiti and es.valori["eff_uscita"] == "11:00"


def test_sostituzione_ignora_valori_implausibili():
    row = {"prog_entrata": "11:30", "prog_uscita": "12:00", "eff_entrata": "11:30", "eff_uscita": "12:30",
           "ore_dichiarate": None}
    es = decidi_riga(row, {"prog_uscita"}, {"prog_uscita": P("19:30")})
    assert not es.sostituiti


def test_cella_illeggibile_riceve_un_valore_ma_resta_segnalata():
    row = dict(ROW, prog_entrata=None)
    es = decidi_riga(row, {"prog_entrata"}, {"prog_entrata": P("08:00")})
    assert es.valori["prog_entrata"] == "08:00" and es.segnalati == {"prog_entrata"}


# --------------------------------------------------------------------------
# Disponibilita', cache e lettura a lotti (lettore finto)
# --------------------------------------------------------------------------

def test_dipendenze_mancanti(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "dependencies_missing", lambda: "manca torch")
    s = SecondoParere(cache_dir=tmp_path)
    assert s.is_available() == (False, "manca torch")
    with pytest.raises(EngineError, match="manca torch"):
        s.load()


def test_ram_insufficiente(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "dependencies_missing", lambda: None)
    monkeypatch.setattr(sp, "ram_totale_gb", lambda: 8.0)
    ok, msg = SecondoParere(cache_dir=tmp_path).is_available()
    assert not ok and "12 GB" in msg and "8 GB" in msg
    with pytest.raises(EngineError):
        SecondoParere(cache_dir=tmp_path).load()
    assert SecondoParere(cache_dir=tmp_path, min_ram_gb=4).is_available()[0]


def test_ram_totale_gb():
    v = sp.ram_totale_gb()
    assert v is None or v > 0.1


def test_disponibile_con_download_al_primo_uso(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "dependencies_missing", lambda: None)
    monkeypatch.setattr(sp, "ram_totale_gb", lambda: 16.0)
    ok, msg = SecondoParere(cache_dir=tmp_path).is_available()
    assert ok and "verrà scaricato" in msg and "GB" in msg


def test_modello_in_cache_riconosciuto(tmp_path):
    s = SecondoParere(cache_dir=tmp_path)
    assert not s.is_downloaded()
    snap = tmp_path / "models--zai-org--GLM-OCR" / "snapshots" / sp.MODEL_REVISION
    snap.mkdir(parents=True)
    for name in sp._REQUIRED_FILES:
        (snap / name).write_text("{}")
    assert not s.is_downloaded()           # mancano i pesi
    (snap / "model.safetensors").write_bytes(b"x")
    assert s.is_downloaded() and sp.model_path(tmp_path) == snap


class _FakeTorch:
    def __init__(self):
        self.threads = 4
        self.calls = []

    def get_num_threads(self):
        return self.threads

    def set_num_threads(self, n):
        self.calls.append(n)
        self.threads = n


class _FakeReader:
    """Restituisce come lettura il valore scritto nel primo pixel (indice in ``texts``)."""

    def __init__(self, texts):
        self.texts = texts
        self.torch = _FakeTorch()
        self.batches = []

    def read(self, crops, max_tokens):
        self.batches.append((len(crops), max_tokens))
        return [(self.texts[int(np.asarray(c).flat[0])], 0.8) for c in crops]


def test_read_many_lotti_ordine_e_interpretazione():
    texts = ["8:00", "1h:00", "2,5", "", "Scuola chiusa"]
    kinds = ["orario", "orario", "ore", "orario", "nota"]
    s = SecondoParere(cache_dir=Path("."), batch_size=2, threads=2)
    s._reader = _FakeReader(texts)
    reqs = [(np.full((20, 40, 3), i, np.uint8), k) for i, k in enumerate(kinds)]
    out = s.read_many(reqs)
    assert [p.stato for p in out] == ["valore", "illeggibile", "valore", "vuota", "valore"]
    assert out[0].valore == "08:00" and out[2].valore == 2.5 and out[4].valore == "SCUOLA CHIUSA"
    assert all(p.confidenza == pytest.approx(0.8) for p in out)
    assert all(n <= 2 for n, _ in s._reader.batches)
    assert s._reader.torch.calls == [2, 4]       # thread ripristinati
    assert s.read(reqs[0][0], "orario") == ("08:00", pytest.approx(0.8))
    assert s.read(reqs[3][0], "orario") == (None, pytest.approx(0.8))
    with pytest.raises(ValueError):
        s.read_many([(reqs[0][0], "firma")])


def test_to_pil_grigi_bgr_e_ritagli_minuscoli():
    pytest.importorskip("PIL")
    im = sp._to_pil(np.zeros((5, 9), np.uint8))
    assert im.mode == "RGB" and min(im.size) >= 16
    bgr = np.zeros((20, 20, 3), np.uint8)
    bgr[..., 0] = 255                                  # blu in BGR
    assert sp._to_pil(bgr).getpixel((0, 0)) == (0, 0, 255)


# --------------------------------------------------------------------------
# Modello reale (solo se gia' in cache)
# --------------------------------------------------------------------------

def _real_cache() -> Path | None:
    raw = os.environ.get("SIRIO_HF_CACHE")
    if not raw or sp.dependencies_missing() or sp.model_path(Path(raw)) is None:
        return None
    return Path(raw)


@pytest.mark.skipif(_real_cache() is None, reason="modello GLM-OCR non presente nella cache SIRIO_HF_CACHE")
def test_lettura_reale_cella_sintetica():
    cv2 = pytest.importorskip("cv2")
    crops = []
    for text in ("8:00", "11:30", "2,5"):
        img = np.full((70, 200, 3), 255, np.uint8)
        cv2.putText(img, text, (14, 52), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (90, 40, 20), 3, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (199, 69), (0, 0, 0), 2)     # bordi della cella
        crops.append(img)
    s = SecondoParere(cache_dir=_real_cache())
    out = s.read_many([(crops[0], "orario"), (crops[1], "orario"), (crops[2], "ore")])
    assert out[0].valore == "08:00" and out[1].valore == "11:30"
    assert out[2].valore == pytest.approx(2.5)
    assert all(0 < p.confidenza <= 1 for p in out)
