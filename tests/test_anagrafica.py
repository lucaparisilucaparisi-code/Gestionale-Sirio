"""Test dell'anagrafica appresa (sirio/anagrafica.py): apprendimento solo da dati
affidabili, riconoscimento dei nomi letti male, associazioni e persistenza.

Solo nomi fittizi: le corruzioni imitano quelle reali del motore offline
(lettere confuse come O/0, I/1, M/H, lettere perse o scambiate)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from sirio import anagrafica as an
from sirio.models import Document, ExtractionResult, Header


@pytest.fixture()
def reg(tmp_path: Path) -> an.Anagrafica:
    return an.Anagrafica(tmp_path / "anagrafica.json")


_ids = iter(range(10_000))


def make_doc(verified: bool = True, edited: tuple[str, ...] = (), doc_id: str | None = None, **header) -> Document:
    values = {"operatore": "ROSSI MARIO", "alunno": "BIANCHI LUCA", "istituto": "IC 1 VERDI",
              "ente": "COOPERATIVA SOCIALE ESEMPIO", "lotto": "1", "municipalita": "3", "ore_pei": 15.0}
    values.update(header)
    return Document(id=doc_id or f"doc{next(_ids)}", source_file="foglio.pdf", user_verified=verified,
                    user_edited=list(edited), header=Header(**values))


def confirm(reg: an.Anagrafica, **header) -> Document:
    doc = make_doc(**header)
    reg.impara_documento(doc)
    return doc


def counts(reg: an.Anagrafica, campo: str) -> dict[str, int]:
    return {v["valore"]: v["conteggio"] for v in reg.esporta()["campi"][campo]}


# --------------------------------------------------------------------------
# Normalizzazione e somiglianza
# --------------------------------------------------------------------------

def test_normalization_and_key() -> None:
    assert an.normalizza("  D'Angelo   Lucà ") == "D ANGELO LUCA"
    assert an.normalizza("IC.9 Cuoco-Schipa") == "IC 9 CUOCO SCHIPA"
    assert an.chiave("Mario Rossi") == an.chiave("ROSSI  MARIO") == "MARIO ROSSI"
    assert an.pulisci("  Rossi   Mario ") == "Rossi Mario" and an.pulisci("   ") is None
    assert not an.valido("1") and not an.valido("--") and an.valido("Ro")


def test_weighted_distance() -> None:
    assert an.distanza("ROSSI", "ROSSI") == 0
    assert an.distanza("ROSSI", "R0SSI") == an.COSTO_SIMILE           # O/0 si confondono
    assert an.distanza("ROSSI", "RXSSI") == 1
    assert an.distanza("TOMBERLI", "TOMBRELI") == 1                    # scambio di lettere adiacenti
    assert an.distanza("", "ABC") == 3


def test_similarity_of_ocr_like_corruptions() -> None:
    assert an.confronta("MARIO ROSSI", "ROSSI MARIO").valore == 1.0   # ordine dei nomi scambiato
    assert an.confronta("R0SSI MAR1O", "ROSSI MARIO").valore >= 0.80
    assert an.confronta("ESPOSIT0 ANA", "ESPOSITO ANNA").valore >= 0.80
    assert an.confronta("VERDI GIUSEPPE", "ROSSI MARIO").valore < 0.4
    # stesso cognome, persona diversa: la somiglianza globale e' alta ma una parola non si ritrova
    other = an.confronta("ESPOSITO MARIA", "ESPOSITO ANNA")
    assert other.valore > 0.75 and other.copertura <= 0.5
    # nomi che differiscono solo per la vocale finale e istituti con numeri diversi: sospetti
    assert an.confronta("ROSSI MARIA", "ROSSI MARIO").sospetta
    assert an.confronta("IC 10 VERDI", "IC 9 VERDI").sospetta
    assert not an.confronta("R0SSI MAR1O", "ROSSI MARIO").sospetta


# --------------------------------------------------------------------------
# Apprendimento: solo dati affidabili
# --------------------------------------------------------------------------

def test_raw_ocr_is_never_learned(reg: an.Anagrafica) -> None:
    assert reg.impara_documento(make_doc(verified=False)) is False
    assert len(reg) == 0 and reg.esporta()["associazioni"] == []
    assert not reg.path.exists()


def test_user_edit_learns_only_the_edited_field(reg: an.Anagrafica) -> None:
    doc = make_doc(verified=False, edited=("header.operatore", "rows.3.eff_uscita"), operatore="ROSSI MARIO",
                   alunno="BLANCHJ LUKA")
    assert reg.impara_documento(doc) is True
    assert counts(reg, "operatore") == {"ROSSI MARIO": 1}
    assert counts(reg, "alunno") == {} and counts(reg, "ente") == {}
    assert reg.esporta()["associazioni"] == []           # niente associazioni senza conferma
    # stesso documento salvato di nuovo: idempotente
    assert reg.impara_documento(doc) is False
    assert counts(reg, "operatore") == {"ROSSI MARIO": 1}


def test_confirmed_document_learns_header_and_association(reg: an.Anagrafica) -> None:
    doc = confirm(reg)
    data = reg.esporta()
    assert {c: [v["valore"] for v in data["campi"][c]] for c in an.CAMPI} == {
        "operatore": ["ROSSI MARIO"], "alunno": ["BIANCHI LUCA"], "istituto": ["IC 1 VERDI"],
        "ente": ["COOPERATIVA SOCIALE ESEMPIO"]}
    assert data["campi"]["operatore"][0]["conteggio"] == 1 and data["campi"]["operatore"][0]["ultimo_uso"]
    (assoc,) = data["associazioni"]
    assert assoc == {**assoc, "operatore": "ROSSI MARIO", "alunno": "BIANCHI LUCA", "istituto": "IC 1 VERDI",
                     "ente": "COOPERATIVA SOCIALE ESEMPIO", "lotto": "1", "municipalita": "3", "ore_pei": 15.0,
                     "conteggio": 1}
    assert reg.impara_documento(doc) is False            # ricalcolo idempotente
    confirm(reg)                                          # un altro mese dello stesso operatore
    assert counts(reg, "operatore") == {"ROSSI MARIO": 2}
    assert reg.esporta()["associazioni"][0]["conteggio"] == 2
    # un operatore puo' seguire piu' alunni
    confirm(reg, alunno="VERDI ANNA", ore_pei=10.0)
    assert len(reg.esporta()["associazioni"]) == 2


def test_correction_replaces_value_learned_from_same_document(reg: an.Anagrafica) -> None:
    doc = make_doc(operatore="ROSSI MARIOO")              # errore di battitura confermato
    reg.impara_documento(doc)
    doc.header.operatore = "ROSSI MARIO"                  # poi corretto
    assert reg.impara_documento(doc) is True
    assert counts(reg, "operatore") == {"ROSSI MARIO": 1}
    assert [a["operatore"] for a in reg.esporta()["associazioni"]] == ["ROSSI MARIO"]
    # conferma tolta: resta solo cio' che l'utente ha scritto a mano
    doc.user_verified = False
    doc.user_edited = ["header.alunno"]
    reg.impara_documento(doc)
    assert counts(reg, "operatore") == {} and counts(reg, "alunno") == {"BIANCHI LUCA": 1}
    assert reg.esporta()["associazioni"] == []


def test_detached_documents_keep_what_was_learned(reg: an.Anagrafica) -> None:
    doc = confirm(reg)
    reg.scollega_documento(doc.id)                        # documento eliminato o riletto
    assert counts(reg, "operatore") == {"ROSSI MARIO": 1}
    doc.user_verified = False                             # la nuova lettura non ritira nulla
    assert reg.impara_documento(doc) is False
    assert counts(reg, "operatore") == {"ROSSI MARIO": 1}
    doc.user_verified = True                              # riconfermato: un'altra conferma
    reg.impara_documento(doc)
    assert counts(reg, "operatore") == {"ROSSI MARIO": 2}
    reg.scollega_tutti()
    assert counts(reg, "operatore") == {"ROSSI MARIO": 2}


def test_documents_that_are_not_timesheets_are_ignored(reg: an.Anagrafica) -> None:
    doc = make_doc()
    doc.is_foglio_firma = False
    assert reg.impara_documento(doc) is False and len(reg) == 0


def test_spelling_of_the_latest_confirmation_wins(reg: an.Anagrafica) -> None:
    confirm(reg, operatore="Rossi Mario")
    confirm(reg, operatore="ROSSI  MARIO")
    assert counts(reg, "operatore") == {"ROSSI MARIO": 2}


# --------------------------------------------------------------------------
# Riconoscimento e soglie
# --------------------------------------------------------------------------

def correct(reg: an.Anagrafica, **header) -> tuple[Header, dict[str, an.Riconoscimento], list[str]]:
    h = Header(**header)
    decisions = reg.correggi(h)
    return h, {d.campo: d for d in decisions}, an.note(decisions)


def test_empty_registry_changes_nothing(reg: an.Anagrafica) -> None:
    h, dec, notes = correct(reg, operatore="R0SSI MAR1O", incerti=["operatore"])
    assert h.operatore == "R0SSI MAR1O" and h.incerti == ["operatore"] and dec == {} and notes == []


def test_misread_name_replaced_and_original_reading_noted(reg: an.Anagrafica) -> None:
    confirm(reg)
    confirm(reg, operatore="ESPOSITO ANNA", alunno="FERRARA LUCIA", istituto="IC 2 BLU")
    h, dec, notes = correct(reg, operatore="R0SSI MAR1O")
    assert h.operatore == "ROSSI MARIO"
    assert dec["operatore"].esito == "sostituito"
    assert "operatore" in h.incerti                       # 85%: sotto la soglia di certezza, da verificare
    assert notes == ["Operatore riconosciuto dall'anagrafica (somiglianza 85%, da verificare): "
                     "letto «R0SSI MAR1O», proposto «ROSSI MARIO»."]
    h, dec, notes = correct(reg, operatore="ESPOSIT0 ANA")
    assert h.operatore == "ESPOSITO ANNA" and "operatore" in h.incerti
    # una sola lettera confusa: sostituito e non piu' incerto
    h, dec, notes = correct(reg, operatore="ROSSI MARI0", incerti=["operatore"])
    assert h.operatore == "ROSSI MARIO" and "operatore" not in h.incerti
    assert notes == ["Operatore riconosciuto dall'anagrafica (somiglianza 92%): "
                     "letto «ROSSI MARI0», usato «ROSSI MARIO»."]


def test_exact_and_swapped_readings(reg: an.Anagrafica) -> None:
    confirm(reg)
    h, dec, notes = correct(reg, operatore="Mario Rossi", incerti=["operatore"])
    assert h.operatore == "ROSSI MARIO" and h.incerti == [] and dec["operatore"].esito == "identico"
    assert notes == ["Operatore uniformato all'anagrafica: letto «Mario Rossi», usato «ROSSI MARIO»."]
    h, dec, notes = correct(reg, operatore="ROSSI MARIO", incerti=["operatore"])
    assert h.incerti == [] and notes == ["Operatore «ROSSI MARIO» presente nell'anagrafica: lettura confermata."]
    h, dec, notes = correct(reg, operatore="ROSSI MARIO")
    assert notes == []


def test_ambiguous_candidates_keep_the_reading(reg: an.Anagrafica) -> None:
    confirm(reg, operatore="ROSSI MARIO")
    confirm(reg, operatore="RUSSO MARIO", alunno="VERDI ANNA")
    h, dec, notes = correct(reg, operatore="R0SSU MARIO")
    assert h.operatore == "R0SSU MARIO" and "operatore" in h.incerti
    assert dec["operatore"].esito == "ambiguo"
    assert "«ROSSI MARIO»" in notes[0] and "«RUSSO MARIO»" in notes[0] and "più voci simili" in notes[0]


def test_different_people_are_not_merged(reg: an.Anagrafica) -> None:
    confirm(reg, operatore="ESPOSITO ANNA")
    confirm(reg, operatore="ROSSI MARIO", alunno="VERDI ANNA")
    # stesso cognome, nome diverso (una parola non si ritrova)
    for reading in ("ESPOSITO MARIA", "ESPOSITO ANNALISA"):
        h, dec, _ = correct(reg, operatore=reading)
        assert h.operatore == reading and dec["operatore"].esito == "simile", reading
        assert "operatore" in h.incerti
    # solo la vocale finale diversa: probabilmente un'altra persona
    h, dec, notes = correct(reg, operatore="ROSSI MARIA")
    assert h.operatore == "ROSSI MARIA" and dec["operatore"].esito == "simile"
    assert "possibile corrispondenza" in notes[0]
    # nome nuovo, lontano da tutti: nessuna modifica e nessuna nota
    h, dec, notes = correct(reg, operatore="FERRARA CHIARA")
    assert h.operatore == "FERRARA CHIARA" and h.incerti == [] and dec["operatore"].esito == "nuovo"
    assert notes == []


def test_school_numbers_must_match(reg: an.Anagrafica) -> None:
    confirm(reg, istituto="IC 9 VERDI")
    h, dec, _ = correct(reg, istituto="IC 10 VERDI")
    assert h.istituto == "IC 10 VERDI" and dec["istituto"].esito == "simile"
    h, dec, _ = correct(reg, istituto="IC.9 VERBI")
    assert h.istituto == "IC 9 VERDI" and dec["istituto"].esito == "sostituito"


def test_operator_association_is_a_prior(reg: an.Anagrafica) -> None:
    confirm(reg)
    confirm(reg, operatore="ESPOSITO ANNA", alunno="FERRARA LUCIA", istituto="IC 2 BLU",
            ente="ASSOCIAZIONE ARCOBALENO")
    garbled = {"alunno": "BLAMEHI IVCA", "ente": "CODPERATN NORMALE ESEHPU"}
    # senza operatore riconosciuto le letture sono troppo rovinate per sostituirle
    h, dec, _ = correct(reg, operatore="FERRARA CHIARA", **garbled)
    assert h.alunno == "BLAMEHI IVCA" and h.ente == "CODPERATN NORMALE ESEHPU"
    # con l'operatore riconosciuto valgono le sue associazioni
    h, dec, notes = correct(reg, operatore="R0SSI MAR1O", **garbled)
    assert (h.operatore, h.alunno, h.ente) == ("ROSSI MARIO", "BIANCHI LUCA", "COOPERATIVA SOCIALE ESEMPIO")
    assert dec["alunno"].associato and dec["ente"].associato
    assert {"operatore", "alunno", "ente"} <= set(h.incerti)
    assert any("Alunno riconosciuto dall'anagrafica" in n and "associazione nota" in n for n in notes)


def test_operator_recognized_through_the_student(reg: an.Anagrafica) -> None:
    confirm(reg, operatore="ROSSI MARIO", alunno="BIANCHI LUCA")
    confirm(reg, operatore="RUSSO MARIO", alunno="VERDI ANNA")
    h, dec, _ = correct(reg, operatore="R0SSU MARIO", alunno="BIANCHI LUCA")
    assert h.operatore == "ROSSI MARIO" and dec["operatore"].associato and "operatore" in h.incerti


def test_illegible_fields_filled_only_from_unique_associations(reg: an.Anagrafica) -> None:
    confirm(reg)
    h, dec, notes = correct(reg, operatore="ROSSI MARIO", alunno=None, lotto=None,
                            illeggibili=["alunno", "lotto"])
    assert (h.alunno, h.lotto) == ("BIANCHI LUCA", "1")
    assert h.illeggibili == [] and {"alunno", "lotto"} <= set(h.incerti)
    assert dec["alunno"].esito == "dedotto"
    assert "Alunno non leggibile: proposto «BIANCHI LUCA» dall'anagrafica" in notes[0]
    h, dec, notes = correct(reg, operatore="ROSSI MARIO", alunno="BIANCHI LUCA", ore_pei=None,
                            illeggibili=["ore_pei"])
    assert h.ore_pei == 15.0 and "ore_pei" in h.incerti
    assert notes == ["Ore da PEI non leggibile: proposto «15» dall'anagrafica (unico valore associato), "
                     "da verificare."]
    # campo vuoto (non illeggibile): nessuna deduzione
    h, _, _ = correct(reg, operatore="ROSSI MARIO", alunno=None)
    assert h.alunno is None and h.incerti == []
    # due alunni noti per l'operatore: l'alunno illeggibile non si puo' dedurre
    confirm(reg, alunno="VERDI ANNA")
    h, dec, _ = correct(reg, operatore="ROSSI MARIO", alunno=None, illeggibili=["alunno"])
    assert h.alunno is None and h.illeggibili == ["alunno"] and "alunno" not in dec
    # operatore illeggibile ma alunno riconosciuto con un solo operatore noto
    h, dec, _ = correct(reg, operatore=None, alunno="VERDI ANNA", illeggibili=["operatore"])
    assert h.operatore == "ROSSI MARIO" and "operatore" in h.incerti and dec["operatore"].esito == "dedotto"
    # operatore non riconosciuto: nessuna deduzione
    h, dec, _ = correct(reg, operatore="FERRARA CHIARA", alunno=None, illeggibili=["alunno"])
    assert h.alunno is None and dec == {"operatore": dec["operatore"]}


def test_apply_to_result_appends_notes(reg: an.Anagrafica) -> None:
    reg.aggiungi("operatore", "ROSSI MARIO")
    result = ExtractionResult(header=Header(operatore="R0SSI MAR1O"), ocr_notes="Nota del motore.")
    decisions = an.applica_al_risultato(result, reg)
    assert result.header.operatore == "ROSSI MARIO" and decisions[0].esito == "sostituito"
    assert result.ocr_notes.startswith("Nota del motore. Operatore riconosciuto dall'anagrafica")
    other = ExtractionResult(is_foglio_firma=False, header=Header(operatore="R0SSI MAR1O"))
    assert an.applica_al_risultato(other, reg) == [] and other.header.operatore == "R0SSI MAR1O"


def test_matching_is_fast_enough(reg: an.Anagrafica) -> None:
    surnames = ["ESPOSITO", "RUSSO", "ROMANO", "COPPOLA", "SANTORO", "AMATO", "GRECO", "CIOFFI", "GALLO", "IZZO"]
    names = ["ANNA", "LUCA", "SARA", "CIRO", "ROSA", "PAOLO", "ELENA", "GENNARO", "TERESA", "VINCENZO", "CHIARA"]
    for i, s in enumerate(surnames):
        for j, n in enumerate(names):
            confirm(reg, operatore=f"{s} {n}", alunno=f"{names[-1 - j]} {surnames[-1 - i]} {n}",
                    istituto=f"IC {i + j} SCUOLA")
    started = time.perf_counter()
    h, dec, _ = correct(reg, operatore="ESP0SITO CHIAPA", alunno="ANMA IZO CHIAPA", istituto="IC 1O SCUDLA",
                        ente="COOPERATIVA SOCIALE ESEMPLO")
    assert time.perf_counter() - started < 2.0
    assert (h.operatore, h.alunno) == ("ESPOSITO CHIARA", "ANNA IZZO CHIARA")
    # "CHIRA" potrebbe essere CHIARA o CIRO: candidati troppo vicini, lettura invariata
    h, dec, _ = correct(reg, operatore="ESP0SITO CHIRA")
    assert h.operatore == "ESP0SITO CHIRA" and dec["operatore"].esito == "ambiguo"


# --------------------------------------------------------------------------
# Modifiche manuali e persistenza
# --------------------------------------------------------------------------

def test_manual_add_and_remove(reg: an.Anagrafica) -> None:
    voce = reg.aggiungi("operatore", "  Rossi   Mario ")
    assert voce["valore"] == "Rossi Mario" and voce["manuale"] is True and voce["conteggio"] == 0
    assert reg.valori("operatore") == ["Rossi Mario"]
    with pytest.raises(ValueError, match="sconosciuto"):
        reg.aggiungi("mese", "ROSSI")
    with pytest.raises(ValueError, match="almeno due lettere"):
        reg.aggiungi("alunno", " 1 ")
    with pytest.raises(ValueError, match="troppo lungo"):
        reg.aggiungi("ente", "A" * 300)
    # le voci manuali restano anche quando un documento ritira il suo contributo
    doc = confirm(reg, operatore="ROSSI MARIO")
    doc.user_verified = False
    reg.impara_documento(doc)
    assert reg.valori("operatore") == ["ROSSI MARIO"] and reg.valori("alunno") == []
    # rimozione di una voce errata: anche dalle associazioni
    confirm(reg, alunno="BIANCHI LUCA")
    confirm(reg, alunno="VERDI ANNA")
    assert reg.rimuovi("alunno", "verdi anna") is True
    assert reg.valori("alunno") == ["BIANCHI LUCA"]
    assert [a["alunno"] for a in reg.esporta()["associazioni"]] == ["BIANCHI LUCA"]
    assert reg.rimuovi("istituto", "IC 1 VERDI") is True
    assert reg.esporta()["associazioni"][0]["istituto"] is None
    assert reg.rimuovi("alunno", "NESSUNO") is False
    with pytest.raises(ValueError):
        reg.rimuovi("lotto", "1")


def test_persistence_across_restarts(tmp_path: Path) -> None:
    path = tmp_path / "dati" / "anagrafica.json"
    reg = an.Anagrafica(path)
    doc = confirm(reg)
    reg.aggiungi("ente", "ASSOCIAZIONE ARCOBALENO")
    assert path.is_file() and not list(path.parent.glob("*.tmp"))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["versione"] == an.VERSIONE and doc.id in data["documenti"]
    again = an.Anagrafica(path)                            # riavvio
    assert again.esporta() == reg.esporta()
    # il contributo del documento e' ancora collegato: una correzione lo sostituisce
    doc.header.operatore = "ROSSI MARIA"
    again.impara_documento(doc)
    assert again.valori("operatore") == ["ROSSI MARIA"]
    h = Header(operatore="ROSSI MARIA")
    again.correggi(h)
    assert h.incerti == []


def test_unreadable_file_is_set_aside(tmp_path: Path) -> None:
    path = tmp_path / "anagrafica.json"
    path.write_text("{ non e' json", encoding="utf-8")
    reg = an.Anagrafica(path)
    assert len(reg) == 0
    assert list(tmp_path.glob("anagrafica.json.illeggibile-*"))
    reg.aggiungi("operatore", "ROSSI MARIO")
    assert an.Anagrafica(path).valori("operatore") == ["ROSSI MARIO"]


def test_concurrent_learning_is_consistent(reg: an.Anagrafica) -> None:
    docs = [make_doc(alunno=f"ALUNNO {chr(65 + i)}{chr(65 + i)}") for i in range(20)]
    threads = [threading.Thread(target=reg.impara_documento, args=(d,)) for d in docs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert counts(reg, "operatore") == {"ROSSI MARIO": 20}
    assert len(reg.esporta()["associazioni"]) == 20
    assert an.Anagrafica(reg.path).esporta() == reg.esporta()


def test_shared_instance_per_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIRIO_DATA_DIR", str(tmp_path / "a"))
    first = an.anagrafica_predefinita()
    assert first is an.anagrafica_predefinita()
    assert first.path == (tmp_path / "a" / "anagrafica.json").resolve()
    monkeypatch.setenv("SIRIO_DATA_DIR", str(tmp_path / "b"))
    assert an.anagrafica_predefinita() is not first
