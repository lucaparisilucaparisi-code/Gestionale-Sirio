# Sirio OCR — Architettura e contratti tra i moduli

Documento tecnico di riferimento. Ogni modulo DEVE rispettare le firme e i
formati qui descritti: i componenti sono sviluppati in parallelo e si
integrano solo tramite questi contratti. Il modello dati è in
`sirio/models.py` (non modificarne i campi esistenti; aggiunte solo se
indispensabili e retro-compatibili).

## 1. Obiettivo

Applicazione desktop "a un clic" che:

1. importa in blocco PDF (anche multipagina) e immagini (JPG/PNG/TIFF) di
   **fogli firma** dell'*Assistenza Specialistica all'integrazione scolastica
   — Comune di Napoli* (modello con intestazione Lotto, Municipalità, Ente,
   Istituto, Operatore, Alunno, Mese/Anno, Ore da PEI, Sostituzione e tabella
   giornaliera 1–31);
2. legge la scrittura a mano con il miglior OCR disponibile
   (**Claude Vision**, cloud) o con un **motore locale offline** (OpenCV + TrOCR);
3. valida i dati (coerenza orari/ore, totali, firme, festività, ore PEI…),
   evidenziando campi **incerti** e **illeggibili**;
4. mostra i dati in una **griglia stile Excel modificabile** accanto alla
   scansione, per la revisione;
5. genera un **Excel** professionale e preciso con riepilogo, dettaglio
   giornaliero, una scheda per ogni foglio firma, anomalie e totali.

Lingua dell'interfaccia, dei messaggi e dell'Excel: **italiano**.

## 2. Struttura del repository

```
Avvia Sirio OCR.bat            avvio Windows (doppio clic)
Avvia Sirio OCR.command        avvio macOS
avvia-sirio-ocr.sh             avvio Linux
launcher/                      script di bootstrap (uv, Python, dipendenze, collegamento desktop)
pyproject.toml                 dipendenze (uv); extra "offline" = torch+transformers; extra "dev" = test
sirio/
  __init__.py                  __version__ = "1.0.0"
  __main__.py                  python -m sirio  -> sirio.app.main()
  app.py                       avvio server + finestra applicazione + ciclo di vita
  config.py                    percorsi, impostazioni, chiave API
  models.py                    modello dati (contratto)
  pdf_io.py                    lettura PDF/immagini -> pagine (numpy BGR)
  calendario.py                giorni della settimana, festività italiane
  validation.py                parsing orari/ore, controlli, totali
  excel_export.py              generazione del file .xlsx
  store.py                     archivio documenti su disco
  processing.py                coda di elaborazione (thread)
  server.py                    API FastAPI + file statici dell'interfaccia
  vision/
    preprocess.py              raddrizzamento (deskew), normalizzazione pagina
    grid.py                    individuazione della tabella e delle celle
    ink.py                     analisi inchiostro (crocette, firme, celle vuote)
  engines/
    base.py                    interfaccia comune dei motori OCR
    prompts.py                 prompt e schema JSON per Claude
    claude_engine.py           motore Claude Vision
    local_engine.py            motore locale (TrOCR + OpenCV)
  web/
    index.html  styles.css  app.js  assets/(logo.svg, sirio.ico, favicon.svg)
tests/                         pytest (nessun dato personale nel repository)
docs/                          documentazione
```

Il PDF d'esempio reale contiene dati personali (minori): **non va mai
aggiunto al repository**. I test che lo usano leggono il percorso dalla
variabile d'ambiente `SIRIO_SAMPLE_PDF` e vengono saltati se assente; la
verità attesa è in `SIRIO_SAMPLE_TRUTH` (JSON). Per i test nel repository si
usa un foglio firma **sintetico** generato da `tests/synthetic.py`.

## 3. Il modulo (layout del foglio firma)

Pagina A4 verticale. Dall'alto:

* stemma e "COMUNE DI NAPOLI" a sinistra; titolo "Assistenza Specialistica
  all'integrazione scolastica destinato agli alunni disabili frequentanti le
  scuole del Comune di Napoli. Anno Scolastico 2025/2026";
* righe d'intestazione: `LOTTO __ - Municipalità __`, `Ente: ____`,
  `Istituto scolastico: ____`, `Nome e Cognome Operatore: ____`,
  `Nome e Cognome Alunno: ____`; a destra `MESE/ANNO DI RIF.: mm/aaaa`,
  `Ore da PEI: __`, `Sostituzione: [SI] [NO]`;
* tabella con 10 colonne:
  `Giorno | Orario programmato Entrata | Uscita | Orario effettivo Entrata | Uscita | Tot. Ore effettive | Assenza Alunno | Assenza Operat. | Firma Operatore | Note`
  intestazione su due righe, poi 31 righe (giorni 1–31), poi la riga
  `Totale ore effettive mensili | <valore> | Firma Coordinatore dell'Ente`;
* in fondo: `Napoli, __/__/____` e `Timbro e firma Referente Scolastico`.

Convenzioni di compilazione osservate: orari tipo `8:00`, `11:00`, `14:00`;
"-" negli orari effettivi = prestazione non svolta; crocetta (X) nelle colonne
assenza; ore con virgola decimale (`1,5`); note libere (es. `104` = permesso
L.104, `PONTE DI CARNEVALE`); weekend e festivi vuoti. Con **assenza alunno**
l'operatore può vedersi riconosciute ore parziali (es. 1,5 su 3).

## 4. Contratti dei moduli

### 4.1 `sirio/config.py`

```python
APP_NAME = "Sirio OCR"
def data_dir() -> Path          # platformdirs.user_data_dir("SirioOCR", appauthor=False); override env SIRIO_DATA_DIR; crea la cartella
def documents_dir() -> Path     # data_dir()/"documenti"
def logs_dir() -> Path          # data_dir()/"log"
def export_dir() -> Path        # platformdirs.user_documents_dir()/"Sirio OCR"/"Export"; override env SIRIO_EXPORT_DIR

class Settings(BaseModel):
    engine: Literal["claude", "locale"] = "claude"
    claude_model: str = "claude-opus-5-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    verifica_incrociata: bool = True     # seconda lettura mirata delle righe dubbie (solo Claude)
    concorrenza: int = 3                 # documenti elaborati in parallelo (Claude); il locale usa 1
    local_model: str = "microsoft/trocr-base-handwritten"
    export_fogli_per_documento: bool = True
    export_giorni_vuoti: bool = False
    tema: Literal["auto", "chiaro", "scuro"] = "auto"

def load_settings() -> Settings     # da data_dir()/"impostazioni.json"; valori mancanti -> default
def save_settings(s: Settings) -> None   # scrittura atomica
def get_api_key() -> str | None     # 1) env ANTHROPIC_API_KEY 2) keyring("SirioOCR","anthropic_api_key") 3) file data_dir()/".chiave" (permessi 600)
def set_api_key(key: str | None) -> None   # None/"" = cancella; prova keyring, altrimenti file
def api_key_source() -> Literal["env", "keyring", "file"] | None
def api_key_hint() -> str | None    # es. "sk-ant-…a1b2" (mai la chiave intera)

CLAUDE_MODELS = [  # mostrati nelle impostazioni; prezzi in USD per milione di token
  {"id": "claude-opus-5-5",   "label": "Claude Opus 5.5 — massima precisione (consigliato)", "input": 4.00, "output": 20.00},
  {"id": "claude-sonnet-5-5", "label": "Claude Sonnet 5.5 — bilanciato",                     "input": 2.00, "output": 10.00},
  {"id": "claude-haiku-5-5",  "label": "Claude Haiku 5.5 — economico",                       "input": 0.10, "output": 0.50},
]
```

### 4.2 `sirio/pdf_io.py`

```python
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
def count_pages(data: bytes, filename: str) -> int
def iter_pages(data: bytes, filename: str, dpi: int = 300) -> Iterator[tuple[int, np.ndarray]]
    # (indice pagina 0-based, immagine BGR uint8). PDF via PyMuPDF (import pymupdf).
    # Per PDF scansionati: renderizza a `dpi` (300). TIFF multipagina supportati via Pillow.
def decode_image(data: bytes) -> np.ndarray          # BGR, gestisce EXIF orientation
def encode_jpeg(img: np.ndarray, quality: int = 90, max_side: int | None = None) -> bytes
def encode_png(img: np.ndarray, max_side: int | None = None) -> bytes
def resize_max(img, max_side: int | None = None, max_pixels: int | None = None) -> np.ndarray
```

### 4.3 `sirio/vision/`

```python
# preprocess.py
def normalize_page(img: np.ndarray) -> np.ndarray
    # raddrizza (deskew entro ±7°, stimato dalle linee orizzontali della tabella),
    # porta in verticale (ruota di 90/180° se il modulo è capovolto o orizzontale: usa
    # la posizione della tabella e la direzione delle righe), NON ritaglia.
def deskew(img) -> tuple[np.ndarray, float]   # (immagine, angolo in gradi)

# grid.py
COLUMNS = ("giorno", "prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita",
           "ore_dichiarate", "assenza_alunno", "assenza_operatore", "firma", "note")

@dataclass
class TableGrid:
    width: int                 # dimensioni dell'immagine analizzata
    height: int
    col_x: list[int]           # 11 ascisse: bordi delle 10 colonne (col_x[0] = bordo sinistro tabella)
    row_y: list[int]           # 32 ordinate: row_y[i] = bordo superiore del giorno i+1, row_y[31] = bordo inferiore del giorno 31
    header_top: int            # bordo superiore dell'intestazione della tabella ("Giorno | Orario programmato…")
    total_row: tuple[int, int] | None   # (y0, y1) della riga "Totale ore effettive mensili"
    detected: bool             # True = da linee reali; False = modello proporzionale di ripiego
    score: float               # qualità 0..1
    def cell(self, giorno: int, colonna: str, pad: int = 0) -> tuple[int, int, int, int]  # (x0, y0, x1, y1)
    def row_box(self, giorno: int, pad: int = 0) -> tuple[int, int, int, int]
    def rows_box(self, g_from: int, g_to: int, include_table_header: bool = False, pad: int = 0) -> tuple[int, int, int, int]
    def table_box(self) -> tuple[int, int, int, int]       # da header_top al fondo della riga totale
    def header_region(self) -> tuple[int, int, int, int]   # parte alta della pagina (sopra header_top)
    def footer_region(self) -> tuple[int, int, int, int]   # sotto il giorno 31 fino al fondo pagina
    def total_value_box(self) -> tuple[int, int, int, int] | None  # cella del valore "Totale ore effettive mensili"
    def to_dict(self) -> dict ; @classmethod from_dict(d) -> TableGrid
    def scaled(self, sx: float, sy: float) -> "TableGrid"

def detect_grid(img: np.ndarray) -> TableGrid     # mai eccezioni: in caso di insuccesso modello proporzionale

# ink.py
def crop(img, box, pad: int = 0) -> np.ndarray
def ink_ratio(img, box, margin: float = 0.12, remove_lines: bool = True) -> float  # frazione di pixel d'inchiostro
def is_blank(img, box) -> bool
def has_cross(img, box) -> bool          # crocetta nelle colonne assenza
def has_signature(img, box) -> bool      # firma presente
def is_dash(img, box) -> bool            # trattino isolato "-"
```

### 4.4 `sirio/engines/`

```python
# base.py
@dataclass
class PageInput:
    image: np.ndarray          # BGR normalizzata (normalize_page)
    grid: TableGrid
    source_file: str
    page: int                  # 1-based

ProgressFn = Callable[[float, str], None]     # (0..1, messaggio italiano)

class EngineError(Exception):  # messaggio già comprensibile per l'utente (italiano)
    ...

class OCREngine(Protocol):
    name: str                                  # "claude" | "locale"
    def is_available(self) -> tuple[bool, str]  # (pronto?, spiegazione in italiano)
    def extract(self, page: PageInput, progress: ProgressFn) -> ExtractionResult

def get_engine(settings: Settings) -> OCREngine   # costruisce il motore scelto (import pigri)
```

**Claude** (`claude_engine.py`, SDK `anthropic>=1.12`):

* modello da `settings.claude_model` (default `claude-opus-5-5`), effort da
  `settings.claude_effort` (default `high`) in `output_config={"effort": ..., "format": {"type": "json_schema", "schema": ...}}`;
  thinking adattivo (omettere `thinking`); **mai** `temperature`, `budget_tokens`, prefill o `tool_choice` forzato;
* streaming: `client.beta.messages.stream(..., max_tokens=32000)` + `get_final_message()`;
  per Opus 5.5 e Sonnet 5.5 aggiungere `betas=["server-side-fallback-2026-07-01"], fallbacks="default"`
  (NON per Haiku 5.5); controllare `stop_reason` (`refusal`, `max_tokens`) prima di leggere il contenuto;
* immagini JPEG base64 entro i limiti: lato lungo ≤ 2576 px **e** ≤ 3,75 megapixel per immagine;
  inviare: pagina intera + 2 ritagli ad alta risoluzione della tabella (giorni 1–16 con intestazione
  tabella, giorni 16–31 con la riga del totale), ricavati da `TableGrid`;
* risposta JSON con schema rigoroso (tutti gli oggetti `additionalProperties: false`, tutti i campi in
  `required`, valori assenti = `null`); per ogni campo: valore normalizzato + elenchi `incerti`
  e `illeggibili`;
* **verifica incrociata** (se `settings.verifica_incrociata`): dopo la prima lettura si esegue
  `validation.validate(...)`; per le righe con anomalie di coerenza, campi incerti o illeggibili si
  invia un secondo messaggio con le sole strisce di riga ingrandite (2×) chiedendo una rilettura
  mirata; si fondono i risultati;
* errori SDK → `EngineError` con messaggio italiano (chiave non valida, credito esaurito, rete, limiti);
  `max_retries=4`;
* `Usage`: token di input/output sommati su tutte le chiamate, costo stimato con i prezzi di `CLAUDE_MODELS`.

**Locale** (`local_engine.py`): rileva la griglia, usa `ink.py` per crocette/firme/trattini/celle
vuote e TrOCR (`transformers.VisionEncoderDecoderModel` + `TrOCRProcessor`, modello da
`settings.local_model`, solo CPU) per orari, ore, note e campi d'intestazione; post-elaborazione con
vincoli (formato HH:MM, minuti 00/15/30/45, 6:00–20:00, coerenza programmato/effettivo/ore); campi con
bassa confidenza → `incerti`, campi con inchiostro ma lettura impossibile → `illeggibili`.
Download del modello alla prima esecuzione con avanzamento tramite `progress`.
`is_available()` → False con messaggio se torch/transformers non sono installati.

### 4.5 `sirio/calendario.py`

```python
GIORNI = ("lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica")
GIORNI_BREVI = ("lun", "mar", "mer", "gio", "ven", "sab", "dom")
MESI = ("gennaio", ..., "dicembre")
def pasqua(anno: int) -> date
def festivita(anno: int) -> dict[date, str]     # festività nazionali + S. Gennaro (19/9, patrono di Napoli)
def giorni_nel_mese(anno: int, mese: int) -> int
def tipo_giorno(anno: int, mese: int, giorno: int) -> Literal["feriale", "sabato", "domenica", "festivo", "inesistente"]
def nome_festivita(anno, mese, giorno) -> str | None
def settimane_del_mese(anno: int, mese: int) -> list[tuple[int, int]]   # [(dal, al), ...] lun–dom
```

### 4.6 `sirio/validation.py`

```python
TOLLERANZA = 0.01
def parse_time(s: str | None) -> tuple[int, int] | None     # "8:00","8.00","8,00","800","8","08:00","8h","ore 8" -> (8,0); invalido -> None
def normalize_time(s: str | None) -> str | None             # -> "08:00" | None
def parse_hours(s: str | float | None) -> float | None      # "1,5","1.5","3","3h","2:30"(=2.5),"1 e 30" -> float
def hours_between(a: str | None, b: str | None) -> float | None   # uscita - entrata in ore; None se mancante/invalido/negativo
def ore_riconosciute(row: DayRow) -> float                  # dichiarate se presenti, altrimenti calcolate (0 se assenza operatore)
def validate(header: Header, rows: list[DayRow]) -> tuple[list[Anomaly], Totals]
def validate_document(doc: Document) -> Document            # imposta doc.anomalies e doc.totals e restituisce doc
```

Codici anomalia (gravità tra parentesi). Messaggi sempre in italiano, concreti, con giorno e valori:

| Codice | Gravità | Condizione |
|---|---|---|
| `E01_ORE_NON_COERENTI` | errore | ore dichiarate ≠ (uscita − entrata effettive) |
| `E02_TOTALE_MENSILE_DIVERSO` | errore | somma ore giornaliere ≠ totale mensile dichiarato |
| `E03_ORARIO_NON_VALIDO` | errore | orario non interpretabile, oppure uscita ≤ entrata |
| `E04_GIORNO_INESISTENTE` | errore | dati in un giorno che non esiste nel mese (es. 30 febbraio) |
| `E05_CAMPO_ILLEGGIBILE` | errore se il campo incide sulle ore (orari, ore, totale), altrimenti attenzione | campo scritto ma illeggibile |
| `E06_NON_FOGLIO_FIRMA` | errore | la pagina non è un foglio firma riconosciuto |
| `W01_FIRMA_MANCANTE` | attenzione | ore/orari effettivi presenti ma nessuna firma operatore |
| `W02_ORE_SENZA_ORARIO` | attenzione | ore dichiarate > 0 senza orari effettivi e senza assenza alunno |
| `W03_GIORNO_FESTIVO` | attenzione (domenica/festivo), info (sabato) | prestazione in giorno non lavorativo |
| `W04_CAMPO_INCERTO` | attenzione | lettura OCR incerta |
| `W05_ASSENZA_OPERATORE_CON_ORE` | attenzione | assenza operatore ma ore/orari effettivi presenti |
| `W06_ORE_PEI_SUPERATE` | attenzione | ore della settimana > ore da PEI |
| `W07_INTESTAZIONE_INCOMPLETA` | attenzione | manca operatore, alunno, mese o anno |
| `W08_FIRMA_COORDINATORE_MANCANTE` | attenzione | — |
| `W09_TIMBRO_REFERENTE_MANCANTE` | attenzione | — |
| `W10_TOTALE_MENSILE_ASSENTE` | attenzione | totale mensile non scritto |
| `W11_DOPPIA_ASSENZA` | attenzione | crocette in entrambe le colonne assenza |
| `W12_ORE_NON_INDICATE` | attenzione | orari effettivi presenti ma colonna "Tot. ore" vuota (ore ricavate dagli orari) |
| `I01_ASSENZA_ALUNNO` | info | assenza alunno (con ore riconosciute e % sul programmato) |
| `I02_ASSENZA_OPERATORE` | info | assenza operatore (con eventuale nota, es. 104) |
| `I03_ORARIO_DIVERSO` | info | orario effettivo diverso dal programmato |
| `I04_ORE_SENZA_PROGRAMMATO` | info | orario effettivo senza orario programmato |
| `I05_NON_SVOLTO` | info | prestazione programmata ma non svolta (senza ore né assenze), con l'eventuale nota |

`Totals.stato`: `errori` se n_errori > 0; altrimenti `da_verificare` se n_attenzioni > 0 o campi
incerti/illeggibili > 0; altrimenti `ok`. Le settimane vanno da lunedì a domenica, troncate al mese.
`WeekTotal.ore_pei` sono le ore PEI **attese** nella settimana, proporzionate ai giorni scolastici
che la settimana contiene nel mese (W06 confronta invece con le ore PEI settimanali piene). Con
assenza alunno sono ammesse ore dichiarate inferiori all'orario effettivo. Dal 2026 il 4 ottobre
(San Francesco) è di nuovo festa nazionale (L. 151/2025). Helper per Excel e interfaccia:
`CODICI` (legenda), `esito_riga`, `stato_campo` ("illeggibile" | "incerto" | ""), `etichetta_campo`.

### 4.7 `sirio/excel_export.py`

```python
class ExportOptions(BaseModel):
    fogli_per_documento: bool = True
    giorni_vuoti: bool = False         # includere nel dettaglio anche i giorni senza dati
    titolo: str | None = None
def default_filename(docs: list[Document]) -> str          # "Rendicontazione_2026-02_20261009-1530.xlsx"
def export_workbook(docs: list[Document], path: Path, options: ExportOptions | None = None) -> Path
```

Libreria: **XlsxWriter**. Orari come veri valori orari Excel (`hh:mm`), date come date
(`gg/mm/aaaa`), ore come numeri (`0.00`), formule per totali e differenze (il file resta "vivo").
Fogli:

1. **Riepilogo** — titolo, data di generazione, indicatori (fogli, ore totali, anomalie…), tabella
   Excel (una riga per foglio firma) con collegamenti ipertestuali alla scheda del documento, riga
   totali, formattazione condizionale dello stato (verde OK / ambra da verificare / rosso errori).
2. **Dettaglio giornaliero** — una riga per ogni giorno con dati di ogni documento: operatore,
   alunno, istituto, data, giorno della settimana, orari programmati/effettivi, ore programmate,
   calcolate, dichiarate, differenza, assenze, firma, note, esito, anomalie. Filtri e riquadri bloccati.
3. **Una scheda per foglio firma** (nome ≤ 31 caratteri, univoco, es. `TOMBERLI A. 02-2026`):
   replica fedele del modulo — intestazione, tabella di 31 giorni con data e giorno della settimana,
   weekend/festivi in grigio, celle **illeggibili** (sfondo rosso chiaro, testo "ILLEGGIBILE") e
   **incerte** (sfondo ambra) evidenziate con commento della cella, celle corrette a mano in blu
   con commento "Valore OCR originale: …", colonna "Ore calcolate" con formula, totali con formule,
   confronto con il totale dichiarato, riepilogo settimanale vs ore PEI, firme/timbro, elenco anomalie.
4. **Anomalie** — elenco completo (documento, giorno, gravità, codice, descrizione, valore letto/atteso).
5. **Totali per operatore** — ore e giorni per operatore/alunno/istituto (con formule SOMMA.SE ove utile).
6. **Legenda e note** — significato dei colori, regole di calcolo, motore OCR e modello usati.

Impostazione di stampa: A4 orizzontale, adatta alla larghezza, righe d'intestazione ripetute.

### 4.8 `sirio/store.py`

```python
class DocumentStore:
    def __init__(self, root: Path)                       # root = config.documents_dir()
    revision: int                                        # incrementata a ogni modifica (per il polling)
    def add_file(self, filename: str, data: bytes) -> list[Document]
        # una Document per pagina; salva <root>/<id>/page.jpg (pagina normalizzata, JPEG q92, ≤ 3600 px),
        # thumb.jpg (≤ 480 px), grid.json (TableGrid.to_dict), doc.json; status "in_coda".
        # Dedup: se esiste già un documento con lo stesso sha256+pagina lo restituisce senza duplicarlo.
    def list(self) -> list[Document]                     # ordinati per created_at, source_file, source_page
    def get(self, doc_id: str) -> Document | None
    def save(self, doc: Document) -> None                # scrittura atomica (tmp + os.replace), aggiorna updated_at
    def delete(self, doc_id: str) -> bool
    def clear(self) -> None
    def page_path(self, doc_id) -> Path ; def thumb_path(self, doc_id) -> Path
    def load_page(self, doc_id) -> np.ndarray ; def load_grid(self, doc_id) -> TableGrid | None
```

Thread-safe (`threading.RLock`). Id documento: `uuid4().hex[:12]`.

### 4.9 `sirio/processing.py`

```python
class Processor:
    def __init__(self, store: DocumentStore, settings_provider: Callable[[], Settings],
                 engine_factory: Callable[[Settings], OCREngine] = get_engine)
    def start(self) -> None          # avvia i worker; rimette in coda i documenti rimasti "in_lavorazione"/"in_coda"
    def stop(self) -> None
    def enqueue(self, doc_id: str) -> None
    def kick(self) -> None           # riprova i documenti in attesa (es. dopo aver salvato la chiave API)
    def status(self) -> dict         # {"in_coda": n, "in_lavorazione": n, "completati": n, "errori": n, "motore_pronto": bool, "messaggio": str}
```

Ciclo per documento: `in_lavorazione` → carica pagina e griglia → `engine.extract` (avanzamento
→ `doc.progress`, `doc.status_message`, salvataggio con frequenza limitata) →
`doc.apply_extraction` → `validate_document` → `completato` (o `scartato` se non è un foglio firma,
`errore` con messaggio in caso di eccezione). Se il motore non è disponibile, i documenti restano
`in_coda` con `status_message` esplicativo finché `kick()`.
Concorrenza: `settings.concorrenza` per Claude, 1 per il motore locale.

### 4.10 `sirio/server.py` — API HTTP (FastAPI)

`create_app(store, processor, token: str, on_shutdown: Callable | None = None) -> FastAPI`.
Ascolto solo su `127.0.0.1`. Ogni richiesta `/api/*` deve avere l'header `X-Sirio-Token: <token>`
(403 altrimenti) e un header `Host` locale (`127.0.0.1:<porta>` o `localhost:<porta>`). La pagina
`index.html` è servita con il token inserito in `<meta name="sirio-token" content="...">`
(sostituzione del segnaposto `__SIRIO_TOKEN__`). File statici da `sirio/web/` su `/static/...`.
Risposte JSON; errori `{"detail": "messaggio italiano"}`.

| Metodo e percorso | Corpo / parametri | Risposta |
|---|---|---|
| `GET /` | — | index.html con token |
| `GET /api/health` | — | `{"ok": true, "version": "1.0.0"}` (senza token) |
| `GET /api/state?since=<rev>` | — | se `since == revision`: `{"revision": n, "unchanged": true}`; altrimenti `{"revision", "documents": [DocSummary], "queue": Processor.status(), "kpi": {...}}` |
| `POST /api/upload` | multipart `files` (più file) | `{"documents": [DocSummary], "errors": [{"file", "message"}]}` |
| `GET /api/documents/{id}` | — | `Document` completo + `"grid": TableGrid.to_dict()` + `"image_url"` |
| `PUT /api/documents/{id}` | `{"header"?: Header, "rows"?: [DayRow], "user_verified"?: bool}` | `Document` aggiornato (rivalidato) |
| `POST /api/documents/{id}/reprocess` | — | `DocSummary` |
| `DELETE /api/documents/{id}` | — | `{"ok": true}` |
| `DELETE /api/documents` | — | cancella tutto `{"ok": true}` |
| `GET /api/documents/{id}/image` | `?max=<px>` | JPEG pagina |
| `GET /api/documents/{id}/thumb` | — | JPEG miniatura |
| `GET /api/documents/{id}/crop` | `?x0&y0&x1&y1&scale=2` | JPEG ritaglio (per il dettaglio cella) |
| `GET /api/settings` | — | `{"settings": Settings, "api_key_set": bool, "api_key_hint", "api_key_source", "engines": {"claude": {"available", "message"}, "locale": {"available", "message"}}, "models": CLAUDE_MODELS, "data_dir", "export_dir", "version"}` |
| `PUT /api/settings` | Settings parziali + `"api_key"?: str` (`""` = cancella) | come GET |
| `POST /api/settings/test` | — | `{"ok": bool, "message": str}` (verifica la chiave con una richiesta minima) |
| `POST /api/export` | `{"ids"?: [str], "options"?: ExportOptions}` (default: tutti i completati) | `{"filename", "path", "url": "/api/exports/<filename>"}` |
| `GET /api/exports` | — | `[{"filename", "path", "size", "created_at", "url"}]` |
| `GET /api/exports/{filename}` | — | download `.xlsx` (solo file nella cartella export) |
| `POST /api/open` | `{"target": "export_dir" \| "file", "filename"?: str}` | apre cartella/file con il programma predefinito del sistema |
| `POST /api/heartbeat` | — | `{"ok": true}` |
| `POST /api/bye` | — | `{"ok": true}` (la finestra si sta chiudendo) |

`DocSummary` = `{"id", "display_name", "source_file", "source_page", "page_count", "status",
"progress", "status_message", "error", "is_foglio_firma", "engine", "model", "user_verified",
"created_at", "updated_at", "header": {operatore, alunno, istituto, ente, mese, anno, ore_pei},
"totals": {ore_riconosciute, ore_dichiarate, ore_calcolate, totale_mensile_dichiarato,
differenza_totale, giorni_lavorati, n_errori, n_attenzioni, n_info, campi_incerti,
campi_illeggibili, stato}, "cost_usd", "thumb_url"}`.

`kpi` = `{"documenti", "completati", "ore_totali", "errori", "attenzioni", "illeggibili", "da_verificare", "verificati", "costo_usd"}`.

Modifica (`PUT`): i campi cambiati rispetto al valore precedente vengono aggiunti a
`user_edited` (chiavi `header.<campo>` / `rows.<giorno>.<campo>`), il valore OCR originale viene
memorizzato in `ocr_originali` (solo la prima volta) e il campo viene tolto da `incerti`/`illeggibili`.

### 4.11 `sirio/app.py` — avvio e ciclo di vita

`python -m sirio [--porta N] [--senza-finestra] [--cartella-dati DIR]`:

1. istanza singola: file `data_dir()/istanza.json` `{port, pid, token}`; se un'istanza risponde a
   `/api/health`, apre solo una nuova finestra verso di essa ed esce;
2. avvia uvicorn su `127.0.0.1:<porta libera>` in un thread; token casuale (`secrets.token_urlsafe`);
3. apre la finestra applicazione: Microsoft Edge / Google Chrome / Chromium in modalità app
   (`--app=URL --user-data-dir=<data_dir>/finestra --window-size=1480,940 --no-first-run
   --no-default-browser-check`), attende la chiusura del processo; se non trovato, browser
   predefinito (`webbrowser.open`);
4. chiusura: quando il processo della finestra termina (e non arrivano heartbeat per 10 s), oppure
   dopo `/api/bye` senza nuovi heartbeat per 8 s, oppure senza heartbeat per 15 minuti → arresto
   ordinato (processor.stop, server.should_exit);
5. log su `logs_dir()/sirio.log` (rotazione), eccezioni non gestite registrate.

### 4.12 Interfaccia (sirio/web)

Stile premium e professionale, HTML/CSS/JS senza dipendenze esterne (funziona offline), tema chiaro e
scuro. Viste:

* **Dashboard / Importa**: indicatori (fogli, ore totali, da verificare, illeggibili, errori), grande
  area di trascinamento PDF/immagini (anche cartelle intere e selezione multipla), coda con barre di
  avanzamento per documento.
* **Documenti**: elenco filtrabile/ordinabile (stato, operatore, mese) con miniature e badge.
* **Revisione documento** (vista principale): a sinistra la scansione con zoom/trascinamento e
  evidenziazione della riga/cella selezionata (usando `grid`); a destra intestazione modificabile e
  **griglia stile Excel** dei 31 giorni, modificabile cella per cella (tastiera: frecce, Invio, Tab,
  Esc, Canc), celle **illeggibili** in rosso con icona, **incerte** in ambra, modificate in blu,
  weekend/festivi in grigio; pannello anomalie cliccabile (porta alla cella); salva automatico,
  "Conferma documento" (user_verified), navigazione documento precedente/successivo.
* **Anteprima Excel**: visualizzazione a schede dei fogli che verranno generati (Riepilogo,
  Dettaglio giornaliero, Anomalie) in griglia, con modifica diretta dei valori del dettaglio
  (scrive sul documento con `PUT`), poi "Genera Excel" → apri file / apri cartella / esportazioni precedenti.
* **Impostazioni**: motore OCR (Claude consigliato / Locale offline), chiave API (campo password,
  verifica), modello, effort, verifica incrociata, concorrenza, tema, cartelle dati/export, nota privacy.

## 5. Avvio "a un clic"

* Windows: doppio clic su `Avvia Sirio OCR.bat` → `launcher/avvia.ps1`.
* macOS: `Avvia Sirio OCR.command`; Linux: `avvia-sirio-ocr.sh` → `launcher/avvia.sh`.

Il bootstrap scarica **uv** (Astral) nella cartella `.runtime/` del programma, che installa Python
3.12 gestito e tutte le dipendenze (`uv sync --extra offline`, PyTorch solo CPU) in
`.runtime/venv`; ai successivi avvii verifica solo che tutto sia aggiornato (pochi secondi). Ripiego:
Python di sistema ≥ 3.10 + `venv` + `pip`. Al primo avvio su Windows crea il collegamento "Sirio OCR"
sul desktop con icona. Nessun privilegio di amministratore richiesto.
