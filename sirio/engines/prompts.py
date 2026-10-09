"""Prompt e schemi JSON (output strutturato) per il motore Claude Vision.

Due richieste:

1. **Lettura completa** (``SYSTEM_PROMPT`` + ``first_pass_text`` + ``SCHEMA``):
   pagina intera, intestazione e due ritagli della tabella; restituisce
   intestazione, 31 righe, pie' di pagina, confidenza e note.
2. **Verifica incrociata** (stesso ``SYSTEM_PROMPT`` + ``verification_text`` +
   ``VERIFY_SCHEMA``): strisce ingrandite delle sole righe dubbie, da rileggere
   in modo indipendente con lo stesso schema di riga.

Gli schemi rispettano i vincoli dell'output strutturato: ogni oggetto ha
``additionalProperties: false`` e tutte le proprieta' in ``required``; nessuna
parola chiave non supportata (minimum, maxLength, pattern, minItems...).
L'API ammette al massimo 16 proprieta' con tipi unione (``anyOf`` o
``"type": [..., "null"]``) per richiesta: per questo i valori testuali assenti
sono la stringa vuota ``""`` invece di ``null`` (nessun tipo unione).
``claude_engine`` converte ``""`` in ``None``.

Il prompt di sistema e' fisso (nessun dato variabile) per sfruttare la cache
dei prompt tra un documento e l'altro.
"""

from __future__ import annotations

from typing import Any

from sirio.models import DAY_FIELDS

# Campi segnalabili come incerti/illeggibili nelle righe (= DAY_FIELDS).
ROW_FLAG_FIELDS: tuple[str, ...] = DAY_FIELDS

# Campi segnalabili nell'intestazione: come HEADER_FIELDS ma con "mese_anno"
# (un unico valore scritto "MM/AAAA") al posto di "mese" e "anno".
HEADER_FLAG_FIELDS: tuple[str, ...] = (
    "anno_scolastico",
    "lotto",
    "municipalita",
    "ente",
    "istituto",
    "operatore",
    "alunno",
    "mese_anno",
    "ore_pei",
    "sostituzione",
    "data_compilazione",
    "firma_coordinatore",
    "timbro_referente",
    "totale_mensile_dichiarato",
)

# Parole chiave JSON Schema non supportate dall'output strutturato (usate nei test).
UNSUPPORTED_SCHEMA_KEYWORDS: frozenset[str] = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "patternProperties",
        "dependentRequired",
        "if",
        "then",
        "else",
        "not",
        "$ref",
        "$defs",
        "definitions",
    }
)


# ==========================================================================
# Schemi
# ==========================================================================

def _obj(properties: dict[str, Any], description: str | None = None) -> dict[str, Any]:
    """Oggetto rigoroso: nessuna proprieta' aggiuntiva, tutte obbligatorie."""
    out: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    if description:
        out["description"] = description
    return out


def _text(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _flag(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}


def _flags(fields: tuple[str, ...], description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string", "enum": list(fields)},
        "description": description,
    }


_TIME = (
    "Orario come scritto, normalizzato HH:MM (24 ore); "
    "stringa vuota se la cella è vuota, contiene un trattino o è illeggibile."
)

ROW_SCHEMA: dict[str, Any] = _obj(
    {
        "giorno": {"type": "integer", "description": "Numero del giorno stampato nella prima colonna (1-31)."},
        "prog_entrata": _text("Orario programmato - Entrata. " + _TIME),
        "prog_uscita": _text("Orario programmato - Uscita. " + _TIME),
        "eff_entrata": _text("Orario effettivo - Entrata. " + _TIME),
        "eff_uscita": _text("Orario effettivo - Uscita. " + _TIME),
        "ore_dichiarate": _text(
            "Colonna «Tot. Ore effettive» trascritta come scritta (es. \"3\", \"1,5\"); stringa vuota se vuota, "
            "trattino o illeggibile. Mai calcolata dagli orari."
        ),
        "assenza_alunno": _flag("Crocetta (o segno) nella colonna «Assenza Alunno»."),
        "assenza_operatore": _flag("Crocetta (o segno) nella colonna «Assenza Operat.»."),
        "firma": _flag("Firma o sigla a mano presente nella colonna «Firma Operatore»."),
        "note": _text("Testo della colonna «Note» in maiuscolo, come scritto; stringa vuota se vuota."),
        "trattino_effettivo": _flag("Trattino «-» scritto in almeno una cella dell'orario effettivo."),
        "incerti": _flags(ROW_FLAG_FIELDS, "Campi letti e trascritti ma con lettura non sicura."),
        "illeggibili": _flags(ROW_FLAG_FIELDS, "Campi con scrittura presente ma non leggibile (valore vuoto)."),
    },
    "Una riga della tabella giornaliera.",
)

HEADER_SCHEMA: dict[str, Any] = _obj(
    {
        "anno_scolastico": _text("Anno scolastico del titolo, formato AAAA/AAAA (es. \"2025/2026\")."),
        "lotto": _text("Valore scritto dopo «LOTTO» (es. \"1\")."),
        "municipalita": _text("Valore scritto dopo «Municipalità» (es. \"2\")."),
        "ente": _text("Valore scritto dopo «Ente:» in maiuscolo."),
        "istituto": _text("Valore scritto dopo «Istituto scolastico:» in maiuscolo."),
        "operatore": _text("«Nome e Cognome Operatore» in maiuscolo, nell'ordine in cui è scritto."),
        "alunno": _text("«Nome e Cognome Alunno» in maiuscolo, nell'ordine in cui è scritto."),
        "mese_anno": _text("«MESE/ANNO DI RIF.» come scritto, formato MM/AAAA (es. \"02/2026\")."),
        "ore_pei": _text("«Ore da PEI» come scritto (ore settimanali, es. \"15\" o \"12,5\")."),
        "sostituzione": {
            "type": "string",
            "enum": ["SI", "NO", ""],
            "description": "Casella «Sostituzione» barrata: \"SI\" o \"NO\"; stringa vuota se nessuna è segnata.",
        },
        "data_compilazione": _text(
            "Data «Napoli, __/__/____» in fondo alla pagina, formato GG/MM/AAAA; stringa vuota se non compilata."
        ),
        "firma_coordinatore": _flag("Firma o sigla nello spazio «Firma Coordinatore dell'Ente»."),
        "timbro_referente": _flag("Timbro e/o firma nello spazio «Timbro e firma Referente Scolastico»."),
        "totale_mensile_dichiarato": _text(
            "Valore scritto nella riga «Totale ore effettive mensili» (es. \"49,5\"); stringa vuota se non scritto."
        ),
        "incerti": _flags(HEADER_FLAG_FIELDS, "Campi letti e trascritti ma con lettura non sicura."),
        "illeggibili": _flags(HEADER_FLAG_FIELDS, "Campi con scrittura presente ma non leggibile (valore vuoto)."),
    },
    "Intestazione e piè di pagina del foglio firma.",
)

SCHEMA: dict[str, Any] = _obj(
    {
        "is_foglio_firma": _flag(
            "true se la pagina è il foglio firma dell'Assistenza Specialistica del Comune di Napoli."
        ),
        "header": HEADER_SCHEMA,
        "rows": {
            "type": "array",
            "items": ROW_SCHEMA,
            "description": "Esattamente 31 righe, giorni da 1 a 31 in ordine, anche se vuote.",
        },
        "confidence": {"type": "number", "description": "Confidenza complessiva della trascrizione, da 0 a 1."},
        "ocr_notes": _text(
            "Brevi osservazioni in italiano per il revisore (correzioni, cancellature, cifre sovrascritte, "
            "timbri sopra le celle); stringa vuota se non c'è nulla da segnalare."
        ),
    },
    "Trascrizione completa del foglio firma.",
)

VERIFY_SCHEMA: dict[str, Any] = _obj(
    {
        "rows": {
            "type": "array",
            "items": ROW_SCHEMA,
            "description": "Una riga per ciascun giorno richiesto, nell'ordine indicato.",
        },
        "totale_mensile_dichiarato": _text(
            "Valore della riga «Totale ore effettive mensili» se ne è richiesta la rilettura, altrimenti stringa vuota."
        ),
        "totale_incerto": _flag("true se la lettura del totale mensile non è sicura (false se non richiesta)."),
        "totale_illeggibile": _flag(
            "true se il totale mensile è scritto ma non leggibile (false se non richiesta)."
        ),
        "ocr_notes": _text("Brevi osservazioni in italiano sulle righe rilette; stringa vuota se nulla da segnalare."),
    },
    "Rilettura mirata di alcune righe del foglio firma.",
)


# ==========================================================================
# Prompt
# ==========================================================================

SYSTEM_PROMPT = """\
Sei un sistema di trascrizione di precisione. Leggi i "fogli firma" compilati a mano \
dell'Assistenza Specialistica all'integrazione scolastica del Comune di Napoli e ne \
trascrivi fedelmente il contenuto nel formato JSON richiesto. I dati servono alla \
rendicontazione ufficiale delle ore di servizio: un valore sbagliato riportato come \
sicuro è molto più grave di un valore segnalato come dubbio.

# IL MODULO

Pagina A4 verticale. Dall'alto verso il basso:

1. A sinistra lo stemma e la scritta "COMUNE DI NAPOLI". Al centro il titolo stampato \
"Assistenza Specialistica all'integrazione scolastica destinato agli alunni disabili \
frequentanti le scuole del Comune di Napoli. Anno Scolastico 2025/2026" \
-> anno_scolastico (formato "AAAA/AAAA").
2. Righe d'intestazione compilate a mano:
   - "LOTTO __ - Municipalità __" -> lotto, municipalita (solo il valore scritto, es. "1", "2");
   - "Ente: ____" -> ente;
   - "Istituto scolastico: ____" -> istituto;
   - "Nome e Cognome Operatore: ____" -> operatore;
   - "Nome e Cognome Alunno: ____" -> alunno.
   A destra:
   - "MESE/ANNO DI RIF.: __/____" -> mese_anno (es. "02/2026");
   - "Ore da PEI: __" -> ore_pei (ore settimanali, es. "15");
   - "Sostituzione: [SI] [NO]" -> sostituzione: "SI" o "NO" solo se la casella corrispondente \
è barrata, cerchiata o crocettata; "" se nessuna casella è segnata (le parole SI e NO \
stampate nelle caselle non contano).
3. La tabella giornaliera, con 10 colonne e intestazione su due righe:
   | Giorno | Orario programmato: Entrata | Uscita | Orario effettivo: Entrata | Uscita | \
Tot. Ore effettive | Assenza Alunno | Assenza Operat. | Firma Operatore | Note |
   Seguono 31 righe, una per ogni giorno da 1 a 31 (il numero del giorno è stampato \
nella prima colonna). Corrispondenza con i campi:
   - Orario programmato Entrata / Uscita -> prog_entrata, prog_uscita;
   - Orario effettivo Entrata / Uscita -> eff_entrata, eff_uscita;
   - Tot. Ore effettive -> ore_dichiarate;
   - Assenza Alunno -> assenza_alunno (true se c'è una crocetta "X" o un segno simile);
   - Assenza Operat. -> assenza_operatore (idem);
   - Firma Operatore -> firma (true se c'è una firma o una sigla a mano; il nome non va letto);
   - Note -> note (testo libero).
4. Sotto il giorno 31 la riga "Totale ore effettive mensili": il valore scritto nella \
colonna "Tot. Ore effettive" -> totale_mensile_dichiarato; nella stessa riga, a destra, \
"Firma Coordinatore dell'Ente" -> firma_coordinatore (true se nello spazio accanto c'è \
una firma o sigla).
5. In fondo: "Napoli, __/__/____" -> data_compilazione (formato "GG/MM/AAAA"; "" se non \
compilata) e "Timbro e firma Referente Scolastico" -> timbro_referente (true se nello \
spazio c'è un timbro e/o una firma). Timbri e firme possono debordare sopra le ultime \
righe della tabella o nei margini: non confonderli con dati delle righe.

# CONVENZIONI DI COMPILAZIONE

- Orari: restituiscili nel formato 24 ore "HH:MM" con zero iniziale ("8:00", "8.00", \
"8,00" e "8" -> "08:00"; "11:00" -> "11:00"). I turni tipici sono di mattina (es. \
08:00-11:00, 11:00-14:00).
- Un trattino "-" negli orari effettivi significa prestazione non svolta: eff_entrata ed \
eff_uscita "" e trattino_effettivo true. Un trattino in qualsiasi altra colonna = cella \
vuota ("").
- Ore: si usa la virgola decimale ("1,5" = un'ora e mezza). Trascrivi ore_dichiarate e \
ore_pei come stringa, così come sono scritte ("3", "1,5").
- Crocetta "X" nelle colonne di assenza -> true; cella vuota -> false.
- Con l'assenza dell'alunno l'operatore può vedersi riconosciute ore parziali (es. "1,5" \
su un orario programmato di 3 ore) e gli orari effettivi possono essere trattini: è normale.
- Note: es. "104" (permesso Legge 104), "PONTE DI CARNEVALE", "MALATTIA", "SCIOPERO", \
"FESTIVITÀ". Trascrivile in MAIUSCOLO, parola per parola, senza correggere né sciogliere \
abbreviazioni. Se la stessa nota è scritta su più righe riportala in ciascuna. Se una \
nota prosegue oltre il bordo della cella, trascrivila per intero. Se è usato un segno di \
ripetizione (virgolette, "idem"), riporta il testo ripetuto e segnala il campo come incerto.
- Sabati, domeniche e festivi sono di norma righe vuote. Una riga vuota ha tutti i campi \
vuoti ("") o false e nessuna segnalazione.
- Nomi e testi dell'intestazione: in MAIUSCOLO, nell'ordine e con l'ortografia con cui sono \
scritti, senza aggiungere né togliere parole.

# GRAFIA AMBIGUA

- Cifre facili da confondere: 1/4/7, 0/6/8/9, 3/5/8, 2/7; in particolare "11:00" e \
"14:00" (un 4 aperto in alto somiglia a un 1). Il segno ":" può essere scritto come \
".", "," o essere appena accennato.
- Nel dubbio scegli la lettura coerente con il resto della riga e del foglio: la \
differenza tra uscita ed entrata effettive di solito corrisponde a "Tot. Ore effettive"; \
gli orari effettivi di solito coincidono con quelli programmati della stessa riga; le \
altre righe mostrano come lo scrivente traccia ogni cifra. Quando la scelta dipende dal \
contesto e non dal tratto, segnala il campo come incerto.
- Se però un valore è scritto chiaramente ed è incoerente (ore che non corrispondono \
agli orari, orario effettivo diverso dal programmato, totale mensile diverso dalla somma), \
trascrivilo ESATTAMENTE COME SCRITTO. Non correggere, non ricalcolare e non completare \
mai i dati: le incoerenze vengono segnalate da un controllo automatico successivo.
- Non ricavare mai un valore assente: se "Tot. Ore effettive" è vuota lascia "" anche se \
gli orari ci sono; se il totale mensile non è scritto restituisci ""
- Valori corretti o sovrascritti: riporta il valore finale valido (non quello cancellato); \
se la correzione rende la lettura dubbia segnala il campo come incerto e descrivi \
brevemente la correzione in ocr_notes.
- Leggi ogni riga allineandoti al numero del giorno stampato a sinistra: la scrittura a \
mano può sconfinare nella riga sopra o sotto.

# SEGNALAZIONI: INCERTI E ILLEGGIBILI

Per l'intestazione e per ogni riga compila due elenchi di nomi di campo:
- "incerti": campi letti e trascritti, ma la cui lettura non è sicura (tratto ambiguo, \
cifre sovrascritte, segno parziale, scelta guidata dal contesto). Il valore va riportato.
- "illeggibili": campi in cui c'è qualcosa di scritto che non riesci a leggere in modo \
attendibile. Il valore DEVE essere la stringa vuota "". Non usarlo per le celle vuote.
Un campo non può comparire in entrambi gli elenchi. Non segnalare i valori scritti in \
modo chiaro e le celle vuote. Nell'intestazione "mese_anno" indica il campo "MESE/ANNO DI RIF.".

# RISPOSTA

- is_foglio_firma: false se la pagina non è questo modulo (pagina bianca, altro \
documento, retro): in tal caso intestazione con valori vuoti ("") e false e righe vuote.
- rows: esattamente 31 oggetti, giorni da 1 a 31 in ordine, uno per ogni riga stampata, \
anche se vuota.
- confidence: numero da 0 a 1 che esprime quanto sei sicuro della correttezza dell'intera \
trascrizione.
- ocr_notes: poche frasi in italiano utili a chi revisiona (correzioni, cancellature, \
scritte fuori dalle celle, timbri che coprono dati, righe difficili); "" se non c'è \
nulla da segnalare.
"""


def first_pass_text(image_labels: list[str], grid_detected: bool) -> str:
    """Testo della richiesta di lettura completa (dopo le immagini)."""
    lines = ["Trascrivi il foglio firma mostrato nelle immagini seguenti, tutte tratte dalla stessa pagina:"]
    for i, label in enumerate(image_labels, start=1):
        lines.append(f"- Immagine {i}: {label}")
    lines.append(
        "Usa i ritagli ad alta risoluzione per leggere la scrittura e la pagina intera per orientarti. "
        "Il giorno 16 compare in entrambi i ritagli della tabella: è la stessa riga."
    )
    if not grid_detected:
        lines.append(
            "Attenzione: i ritagli sono approssimativi (la tabella non è stata individuata con precisione); "
            "fai riferimento ai numeri dei giorni stampati nella prima colonna."
        )
    lines.append("Restituisci intestazione, 31 righe, piè di pagina, segnalazioni, confidenza e note.")
    return "\n".join(lines)


def verification_text(days: list[int], image_labels: list[str], include_total: bool) -> str:
    """Testo della richiesta di verifica incrociata (dopo le immagini)."""
    elenco = ", ".join(str(d) for d in days)
    lines = [
        "VERIFICA MIRATA. Le immagini seguenti sono ritagli ingranditi di un foglio firma:",
    ]
    for i, label in enumerate(image_labels, start=1):
        lines.append(f"- Immagine {i} = {label}")
    lines += [
        "",
        "Ogni striscia mostra una riga della tabella; ai bordi possono comparire parti delle righe "
        "vicine: leggi solo la riga del giorno indicato, riconoscibile dal numero stampato nella prima "
        "colonna. L'intestazione serve solo a riconoscere le colonne.",
        "Rileggi con la massima attenzione ogni cella di queste righe, in modo indipendente e con le "
        "stesse convenzioni (orari HH:MM, trattino negli orari effettivi = \"\" con trattino_effettivo "
        "true, ore come scritte con la virgola, crocette, firma, note in maiuscolo) e le stesse regole "
        "per incerti e illeggibili. Trascrivi i valori come sono scritti, senza correggerli.",
        f"In \"rows\" restituisci un oggetto per ciascuno di questi giorni, in quest'ordine: {elenco}.",
    ]
    if include_total:
        lines.append(
            "Rileggi anche il valore scritto nella riga \"Totale ore effettive mensili\" "
            "(totale_mensile_dichiarato, con totale_incerto e totale_illeggibile)."
        )
    else:
        lines.append(
            "Non è richiesta la rilettura del totale mensile: totale_mensile_dichiarato \"\", "
            "totale_incerto false, totale_illeggibile false."
        )
    return "\n".join(lines)


__all__ = [
    "HEADER_FLAG_FIELDS",
    "HEADER_SCHEMA",
    "ROW_FLAG_FIELDS",
    "ROW_SCHEMA",
    "SCHEMA",
    "SYSTEM_PROMPT",
    "UNSUPPORTED_SCHEMA_KEYWORDS",
    "VERIFY_SCHEMA",
    "first_pass_text",
    "verification_text",
]
