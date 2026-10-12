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
2. legge la scrittura a mano con il **motore locale offline** (OpenCV + TrOCR,
   **predefinito**: gratuito, nessuna chiave API, i documenti non lasciano mai il
   computer) oppure, in alternativa facoltativa, con **Claude Vision** (cloud, a pagamento,
   precisione superiore);
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
  fsutil.py                    sostituzione atomica dei file con nuovi tentativi (Windows)
  models.py                    modello dati (contratto)
  pdf_io.py                    lettura PDF/immagini -> pagine (numpy BGR)
  calendario.py                giorni della settimana, festività italiane
  validation.py                parsing orari/ore, controlli, totali
  excel_export.py              generazione del file .xlsx
  store.py                     archivio documenti su disco
  anagrafica.py                anagrafica appresa (nomi confermati) e correzione dei nomi letti
  processing.py                coda di elaborazione (thread)
  server.py                    API FastAPI + file statici dell'interfaccia
  vision/
    preprocess.py              raddrizzamento (deskew), normalizzazione pagina
    grid.py                    individuazione della tabella e delle celle
    ink.py                     analisi inchiostro (crocette, firme, celle vuote, inchiostro proprio delle celle)
  engines/
    base.py                    interfaccia comune dei motori OCR
    prompts.py                 prompt e schema JSON per Claude
    claude_engine.py           motore Claude Vision
    local_engine.py            motore locale (TrOCR + OpenCV)
    strokes.py                 barrature e barrette "/" (tratti a mano che non sono valori)
    secondo_parere.py          secondo parere offline facoltativo (GLM-OCR) sulle celle dubbie
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
    engine: Literal["claude", "locale"] = "locale"   # predefinito: offline gratuito
    claude_model: str = "claude-opus-5-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    verifica_incrociata: bool = True     # seconda lettura mirata delle righe dubbie (solo Claude)
    concorrenza: int = 3                 # documenti elaborati in parallelo (Claude); il locale usa 1
    local_model: str = "microsoft/trocr-base-handwritten"
    secondo_parere: bool = False         # rilettura GLM-OCR delle celle dubbie del motore locale (vedi 4.4)
    export_fogli_per_documento: bool = True
    export_giorni_vuoti: bool = False
    tema: Literal["auto", "chiaro", "scuro"] = "auto"

def load_settings() -> Settings     # da data_dir()/"impostazioni.json"; valori mancanti -> default
def save_settings(s: Settings) -> None   # scrittura atomica (fsutil.replace_with_retry)
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
    # Per PDF scansionati: renderizza a `dpi` (300), al massimo quanto un A4 a `dpi` con margine
    # (~14 Mpx): le scansioni impaginate a 72 dpi (pagine "di 65 cm") non diventano immagini da
    # 40 Mpx. TIFF multipagina supportati via Pillow.
def decode_image(data: bytes) -> np.ndarray          # BGR, gestisce EXIF orientation
def encode_jpeg(img: np.ndarray, quality: int = 90, max_side: int | None = None) -> bytes
def encode_png(img: np.ndarray, max_side: int | None = None) -> bytes
def resize_max(img, max_side: int | None = None, max_pixels: int | None = None) -> np.ndarray
```

### 4.3 `sirio/vision/`

```python
# preprocess.py
def normalize_page(img: np.ndarray) -> np.ndarray
    # 1) verso (0/90/180/270) da tratti rettilinei anche inclinati o curvi e dalla disposizione
    #    delle colonne del modulo ("Giorno" stretta a sinistra, Firma/Note larghe a destra), in tre
    #    fasce; si gira solo con un indizio forte; 2) inclinazione (±7°) e prospettiva (foto di
    #    sbieco); 3) foglio incurvato/piegato: linee della tabella seguite come curve e pagina
    #    rimappata a maglia (solo oltre 0,15 righe di curvatura, tenuta solo se la griglia non
    #    peggiora); 4) se la griglia non è affidabile ma quella della pagina girata di 180° sì, la
    #    pagina era capovolta; 5) se la griglia resta inaffidabile la geometria è rimisurata su
    #    una copia a contrasto aumentato; 6) se resta inaffidabile e si era corretta la prospettiva
    #    (su un foglio incurvato il "bordo" trovato può essere una piega) si riprova senza: la
    #    maglia segue le linee reali e corregge anche la prospettiva. Si tiene sempre la versione
    #    con la griglia migliore. NON ritaglia; una scansione già dritta esce identica.
def normalize_page_info(img) -> tuple[np.ndarray, dict]
    # come normalize_page + correzioni applicate: {"rotazione", "inclinazione", "prospettiva",
    # "maglia", "contrasto", "capovolta_dalla_tabella"} (salvate dallo store in preparazione.json)
def correzioni_testo(info: dict | None) -> list[str]   # frasi per l'utente ("era capovolta", ...)
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
    score: float               # qualità 0..1 (limitata dall'allineamento)
    alignment: float | None    # autoverifica (check_alignment): None = non misurata (griglie vecchie)
    row_quality: list[float] | None   # 31 valori 0..1, uno per riga-giorno
    reliable: bool             # proprietà: detected e alignment >= GRID_ALIGNMENT_MIN (0,7)
    def row_reliable(self, giorno: int) -> bool   # reliable e row_quality >= ROW_QUALITY_MIN (0,5)
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
def check_alignment(img, grid) -> dict           # autoverifica senza OCR (vedi sotto)

# ink.py
def crop(img, box, pad: int = 0) -> np.ndarray
def ink_ratio(img, box, margin: float = 0.12, remove_lines: bool = True) -> float  # frazione di pixel d'inchiostro
def is_blank(img, box) -> bool
def has_cross(img, box) -> bool          # crocetta nelle colonne assenza
def has_signature(img, box) -> bool      # firma presente
def is_dash(img, box) -> bool            # trattino isolato "-"

class PageInk:                           # analisi dell'inchiostro della tabella, una volta per pagina (~1 s)
    def __init__(self, img, grid, creases=None)   # creases: risultato di find_creases (None = cercate qui)
    strokes: list[Stroke]                # barrature "diagonale" (righe non usate) e linee "orizzontale"
    creases: list[Crease]                # pieghe del foglio, tolte dall'inchiostro come le barrature
    struck_days: set[int] ; line_days: set[int]
    def own_ink(self, giorno, colonna) -> CellInkResult
        # .crop/.mask (solo la scrittura propria: linee, barrature, pieghe e tratti delle celle
        # vicine sbiancati), .ink_ratio, .empty, .struck, .row_struck, .line_across ("-" tirato a
        # mano), .slash ("/" isolata), .stroke_angle, .overflow_removed, .strike_removed, .crease

def find_creases(img, grid) -> tuple[list[Crease], np.ndarray | None]
    # pieghe del foglio: linee sottili quasi verticali (anche chiare, leggermente curve) continue
    # per almeno 8 righe e lontane dai bordi di colonna; maschera dei pixel da togliere (i tratti
    # di scrittura che le attraversano restano). Crease: y0, y1, path (x per ogni y),
    # half_width, days, level (grigio tipico); .crosses(box)
def ink_under_crease(img, box, creases, unit, mask=None) -> bool
    # scrittura piu' scura della piega rimasta sotto di essa (sparirebbe togliendo la piega)
```

**Autoverifica della griglia** (`check_alignment`, applicata da `detect_grid`): una griglia
"rilevata" può essere sfalsata di una riga (righe periodiche, intestazione o totali scambiati per
una riga-giorno) o giusta solo in parte (foglio incurvato). Si misura, senza OCR: quanta parte di
ogni bordo previsto cade su una linea reale; se i numeri stampati nella colonna "Giorno" hanno una
cifra nelle righe 1–9 e due nelle 10–31 (letti dopo una leggera sfocatura: nelle fotocopie la
colonna è retinata scura e il retino non va preso per inchiostro); se la riga dei totali ha la
cella unita. Uno sfalsamento di righe intere viene corretto solo se la griglia spostata poggia
sulle linee almeno quanto l'originale e la sua prima e ultima riga sono righe vere (un foglio
incurvato con le righe sbagliate solo in una parte resta "non affidabile"); il resto produce
`alignment` e `row_quality`. Una tabella con
quasi tutte le righe trovate non viene scartata solo perché una o due linee verticali sono deboli
(foto con lo sfondo attorno al foglio): decide l'autoverifica. **Contratto per i
motori:** con `grid.reliable` False ogni valore della pagina va segnalato; con
`grid.row_reliable(g)` False ogni valore della riga g.

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
  inviare: pagina intera + intestazione + 2 ritagli ad alta risoluzione della tabella (giorni 1–16 con intestazione
  tabella, giorni 16–31 con la riga del totale), ricavati da `TableGrid`;
* risposta JSON con schema rigoroso (tutti gli oggetti `additionalProperties: false`, tutti i campi in
  `required`; i testi assenti sono la stringa vuota `""` nello schema, convertita in `None` — l'API
  ammette al massimo 16 proprietà annullabili per richiesta); per ogni campo: valore normalizzato + elenchi `incerti`
  e `illeggibili`;
* **verifica incrociata** (se `settings.verifica_incrociata`): dopo la prima lettura si esegue
  `validation.validate(...)`; per le righe con anomalie di coerenza, campi incerti o illeggibili si
  invia un secondo messaggio con le sole strisce di riga ingrandite (2×) chiedendo una rilettura
  mirata; si fondono i risultati;
* errori SDK → `EngineError` con messaggio italiano (chiave non valida, credito esaurito, rete, limiti);
  `max_retries=4`;
* `Usage`: token di input/output sommati su tutte le chiamate, costo stimato con i prezzi di `CLAUDE_MODELS`.

**Locale** (`local_engine.py`): usa `ink.py` per crocette/firme/trattini/celle vuote e TrOCR
(`transformers.VisionEncoderDecoderModel` + `TrOCRProcessor`, modello da `settings.local_model`,
solo CPU) con decodifica vincolata ai valori ammessi (lessici) più una lettura libera per orari,
ore, note e campi d'intestazione. Regola guida: **un valore sbagliato non deve mai apparire
sicuro**; nel dubbio la cella va in `incerti` (o `illeggibili` se scritta ma non interpretabile).
Download del modello alla prima esecuzione con avanzamento tramite `progress`.
`is_available()` → False con messaggio se torch/transformers non sono installati.

1. *Analisi della pagina* (`plan_page`, senza OCR). I segni a mano che non sono valori sono
   cercati da **due rilevatori indipendenti** riuniti in `PageMarks`: `strokes.find_strikes`
   (tratti lunghi, sottili e rettilinei, cancellati dall'immagine prima dell'analisi delle celle)
   e `ink.PageInk` (righe barrate per intero, linee orizzontali, barrette "/", inchiostro proprio
   di ogni cella). Un segno visto da uno solo dei due basta a **segnalare** la cella; solo quelli
   del primo (o di entrambi) tolgono un valore senza segnalarlo. Regole: riga barrata (diagonale
   su più celle) → nessun valore, l'inchiostro rimasto (crocette, firme, testo) è segnalato;
   linea tirata sugli orari effettivi → `trattino_effettivo`; "/" isolata in una riga senza altri
   orari → nulla (altrove letta e segnalata); code di firme nella colonna Note tolte; nota senza
   inchiostro proprio → vuota, salvo che l'inchiostro attribuito alle celle vicine sia una
   scrittura consistente (es. "FERIE" staccata di poco dalla firma, presa per la sua
   continuazione): allora si legge e si segnala (motivo `attribuzione`). Le linee stampate non
   allineate alla griglia si cancellano solo lungo la linea, e una "fascia di scrittura" (cifre di
   celle vicine appoggiate sulla linea della riga, ricucite fra loro: spessa e alta insieme) non è
   mai una linea né una barratura. Crocette e firme di inchiostro chiarissimo (meno di 60 livelli
   sulla carta: ombre, aloni del retro del foglio) sono celle vuote; fra 60 e 110 sono segni da
   verificare (le crocette e le firme vere misurate sono tutte oltre 130). Una cella di orari con
   **due fasce scritte una sopra l'altra** ("9:00" sopra "13:00") si legge riga per riga: entrata
   = la prima, uscita = l'ultima, sempre segnalata (motivo `due_orari`). Se il ritaglio pulito non trova scrittura ma `PageInk` sì, si legge
   il ritaglio di `PageInk` e il valore resta segnalato.
   **Pieghe del foglio** (`ink.find_creases`, foto di fogli piegati): la linea lunga quasi
   verticale che attraversa molte righe non è un segno in nessuna colonna; viene tolta come le
   barrature (anche dalla ricerca delle barrature) e non genera crocette, firme o testo; la
   scrittura che taglia (o che resta sotto di essa, più scura della piega) è letta e segnalata
   (motivo `piega`), una crocetta che attraversa diventa da verificare.
   **Timbri e firme del piè di pagina** (`foreign_ink`): l'inchiostro unito a una struttura che
   scende in buona parte sotto la tabella (timbro della scuola, firme del referente o del
   coordinatore appoggiate sull'angolo della tabella) non è una firma, una crocetta o una nota
   della riga: se è la maggior parte dell'inchiostro della cella la firma diventa `TIMBRO`
   (falsa, segnalata) o, se tolto resta una firma, firma da verificare; una quota minore rende
   comunque il segno da verificare. Nella colonna Note si legge (segnalato) solo ciò che resta
   tolta la struttura, se è scrittura vera.
2. *Lettura* TrOCR a lotti, con bianco sopra/sotto i ritagli di orari e ore.
3. *Assemblaggio* (`assemble`): formati reali degli orari ("8:30", "8,30", "830", "8h30", "8")
   a passi di 5 minuti (orari sfalsati "8:10 - 13:10"; i minuti non a quarti d'ora partono
   sfavoriti e diventano normali solo se la pagina li usa: almeno 3 letture sicure),
   ore a quarti ("2,5", "2h30'", "1:15", "0"), fascia scolastica plausibile (fuori 7–18 sempre
   segnalato), scambi di cifre tipici di TrOCR (1/4/7, 2/9, 0/6) solo come alternative segnalate,
   riconciliazione della riga (ore = uscita − entrata; uguaglianza programmato/effettivo usata solo
   per gli spareggi) che **segnala invece di correggere**; conferma per coerenza dei soli valori
   poco leggibili preferiti dalla lettura e imposti da celle sicure; note sicure solo se composte
   di parole del vocabolario scolastico generico e senza cifre. Regole di prudenza (ognuna rende
   la cella "da verificare", nessuna cambia un valore): probabilità del valore < 0,7 fra i valori
   ammessi; un'altra lettura ammessa ≥ 0,25; la coerenza della riga rafforza la fiducia solo se
   una relazione *esatta* impone il valore (una durata solo plausibile non basta); programmato ≠
   effettivo con il valore gemello fra le letture della cella, anche quando le due coppie tornano
   entrambe con le ore, e quando la cella gemella è stata riportata a un valore diverso dalla sua
   lettura; orario programmato diverso da quello (uguale) degli stessi giorni della settimana
   (bastano due settimane concordi, o una sola letta con sicurezza); durata programmata diversa
   di almeno un'ora da quella (uguale) degli stessi giorni della settimana; con l'alunno assente,
   ore riconosciute che non sono una frazione "tonda" (¼, ⅓, ½, ⅔, ¾, intera) della durata
   programmata (es. 1,5 su 11:00–16:00) → orario programmato e ore da verificare; un
   solo orario programmato (o effettivo) leggibile; crocetta di assenza dell'operatore in una riga
   firmata con l'orario effettivo; firma o crocetta in una riga senza alcun valore sicuro (segno
   isolato: timbro, firma del piè di pagina, macchia, barra "/" di una riga annullata); minuti di
   un orario diversi da quelli delle celle di altre righe con la scrittura più simile (stessa
   grafia, es. "13:ↄↄ" letto con sicurezza "13:30" dove gli altri "ↄↄ" sono letti ":00": la riga
   resta coerente perché anche l'entrata si sposta, quindi la coerenza non basta a confermare);
   nota fatta di sole parole vuote ("IL", "DI LA") mai sicura; nota breve del vocabolario ("104")
   proposta da verificare invece di essere scartata; nota scritta per davvero (due righe, o
   almeno tre segni larghi e alti come una parola breve, es. "104") di cui nessuna lettura è accettata →
   `illeggibili`, mai vuota in silenzio (lo stesso per ogni campo: inchiostro proprio presente e
   nessun valore → segnalato). Firma del coordinatore: contano solo i segni per lo più dentro la
   sua riga; i tratti che vengono da fuori (firma del referente che sale nella riga, timbro che
   la copre) la rendono da verificare. Intestazione: mese/anno sicuri
   quando il nome del mese è letto (lettura vincolata e libera concordi) e il calendario dei
   giorni compilati esclude ogni altro anno non futuro; totale mensile segnalato se diverso dalla
   somma delle ore lette (salvo ore illeggibili che spieghino la differenza).
4. *Griglia*: con `grid.reliable` False tutti i valori (anche l'intestazione) sono segnalati e il
   risultato porta l'avviso `W14_GRIGLIA_INCERTA` ("Griglia incerta: verificare tutti i valori
   della pagina"); con righe non allineate, avviso per quei giorni. `ExtractionResult.avvisi`
   contiene questi avvisi (vedi 4.6).
5. *Secondo parere* facoltativo (`settings.secondo_parere`, predefinito **disattivato**):
   `secondo_parere.SecondoParere` (GLM-OCR, MIT, ~2,65 GB scaricati al primo uso nella stessa
   cartella dei modelli, ~4,5 GB di RAM; non si attiva sotto 12 GB di RAM totale) rilegge le sole
   celle dubbie di orari/ore (cella grezza + 0,2 righe di margine). Politica `decidi_riga`: una
   segnalazione viene tolta solo se i due modelli leggono lo stesso valore *e* una relazione
   esatta della riga lo conferma con celle sicure; per le altre celle il valore può essere
   sostituito ma resta segnalato. Mai sulle righe non allineate, sulle celle barrate, con "/" o
   fuori fascia. Se il modello non è disponibile (RAM, download, memoria) la lettura prosegue
   senza e le note lo spiegano. Misure sulle pagine reali di sviluppo nel README.

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
MAX_ORE_GIORNO = 24.0 ; MAX_ORE_PEI = 60.0 ; MAX_ORE_MESE = 744.0
MAX_ORE_CAMPO = {"ore_dichiarate": 24, "ore_pei": 60, "totale_mensile_dichiarato": 744}
def parse_time(s: str | None) -> tuple[int, int] | None     # "8:00","8.00","8,00","800","8","08:00","8h","ore 8" -> (8,0); invalido -> None
def normalize_time(s: str | None) -> str | None             # -> "08:00" | None
def parse_hours(s: str | float | None, *, massimo: float = 24.0) -> float | None
    # "1,5","1.5","3","3h","2:30"(=2.5),"1 e 30" -> float; None se < 0 o > massimo.
    # massimo = 24 per le ore di un giorno; per l'intestazione MAX_ORE_CAMPO[campo]
    # (60 ore PEI settimanali, 744 totale mensile): server (PUT) ed Excel usano lo stesso limite
def hours_between(a: str | None, b: str | None) -> float | None   # uscita - entrata in ore; None se mancante/invalido/negativo
def ore_riconosciute(row: DayRow) -> float                  # dichiarate se presenti, altrimenti calcolate (0 se assenza operatore)
def validate(header: Header, rows: list[DayRow]) -> tuple[list[Anomaly], Totals]
def validate_document(doc: Document) -> Document            # imposta doc.anomalies e doc.totals e restituisce doc
def gruppi_possibili_duplicati(docs) -> list[list[Document]]   # ≥ 2 fogli con stesso operatore, alunno, mese e anno
def segnala_possibili_duplicati(docs) -> None               # aggiunge W13 a ogni documento dei gruppi (idempotente)
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
| `W14_GRIGLIA_INCERTA` | attenzione | avviso della lettura: la tabella della scansione non è allineata con sicurezza alle righe del modulo (pagina intera o giorni indicati): i valori sono tutti da verificare |
| `W13_POSSIBILE_DUPLICATO` | attenzione | un altro foglio firma ha lo stesso operatore, alunno, mese e anno (es. stessa pagina scansionata due volte in file diversi: le ore verrebbero conteggiate due volte); controllo **tra documenti**, vedi sotto |
| `I01_ASSENZA_ALUNNO` | info | assenza alunno (con ore riconosciute e % sul programmato) |
| `I02_ASSENZA_OPERATORE` | info | assenza operatore (con eventuale nota, es. 104) |
| `I03_ORARIO_DIVERSO` | info | orario effettivo diverso dal programmato |
| `I04_ORE_SENZA_PROGRAMMATO` | info | orario effettivo senza orario programmato |
| `I05_NON_SVOLTO` | info | prestazione programmata ma non svolta (senza ore né assenze), con l'eventuale nota |
| `I06_PAGINA_RADDRIZZATA` | info | avviso della lettura: pagina capovolta, ruotata, storta, fotografata di sbieco o incurvata, raddrizzata prima della lettura |

**Avvisi della lettura** (`W14`, `I06`): non si ricavano da intestazione e righe, quindi il motore
(o `processing`) li mette in `ExtractionResult.avvisi`, copiati in `Document.avvisi`;
`validate_document` li aggiunge alle anomalie a ogni validazione (anche dopo le correzioni
dell'utente). Costruttori: `avviso_griglia_incerta(giorni=None)`, `avviso_pagina_raddrizzata(frasi)`.

`Totals.stato`: `errori` se n_errori > 0; altrimenti `da_verificare` se n_attenzioni > 0 o campi
incerti/illeggibili > 0; altrimenti `ok`. Le settimane vanno da lunedì a domenica, troncate al mese.
`WeekTotal.ore_pei` sono le ore PEI **attese** nella settimana, proporzionate ai giorni scolastici
che la settimana contiene nel mese (W06 confronta invece con le ore PEI settimanali piene). Con
assenza alunno sono ammesse ore dichiarate inferiori all'orario effettivo. Dal 2026 il 4 ottobre
(San Francesco) è di nuovo festa nazionale (L. 151/2025). Helper per Excel e interfaccia:
`CODICI` (legenda), `esito_riga`, `stato_campo` ("illeggibile" | "incerto" | ""), `etichetta_campo`.

**W13 (possibile foglio duplicato)** non dipende da un solo foglio: i nomi sono confrontati senza
maiuscole, accenti, punteggiatura e ordine delle parole; i fogli senza operatore, alunno o periodo
valido non sono confrontati. Non viene mai salvata in `doc.json`: l'Excel la calcola sui documenti
esportati e il server (4.10) la calcola al volo sui documenti correnti, così sparisce da sola
quando uno dei due fogli viene eliminato o corretto.

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
3. **Una scheda per foglio firma** (nome ≤ 31 caratteri, univoco, es. `ROSSI M. 02-2026`):
   replica fedele del modulo — intestazione, tabella di 31 giorni con data e giorno della settimana,
   weekend/festivi in grigio, celle **illeggibili** (sfondo rosso chiaro, testo "ILLEGGIBILE") e
   **incerte** (sfondo ambra) evidenziate con commento della cella, celle corrette a mano in blu
   con commento "Valore OCR originale: …", colonna "Ore calcolate" con formula, totali con formule,
   confronto con il totale dichiarato, riepilogo settimanale vs ore PEI, firme/timbro, elenco anomalie.
4. **Anomalie** — elenco completo (documento, giorno, gravità, codice, descrizione, valore letto/atteso).
5. **Totali per operatore** — ore e giorni per operatore/alunno/istituto (con formule SOMMA.SE ove utile).
6. **Legenda e note** — significato dei colori, regole di calcolo, motore OCR e modello usati.

Impostazione di stampa: A4 orizzontale, adatta alla larghezza, righe d'intestazione ripetute.
Il file viene scritto accanto alla destinazione e poi sostituito atomicamente
(`fsutil.replace_with_retry`): se il blocco persiste (file aperto in Excel) `PermissionError` con
messaggio italiano «… il file è aperto in un altro programma …».

### 4.8 `sirio/store.py`

```python
class DocumentStore:
    def __init__(self, root: Path)                       # root = config.documents_dir()
    revision: int                                        # incrementata a ogni modifica (per il polling)
    def add_file(self, filename: str, data: bytes,
                 on_document: Callable[[Document], None] | None = None) -> list[Document]
        # una Document per pagina; salva <root>/<id>/page.jpg (pagina normalizzata con
        # normalize_page_info e poi ridimensionata, JPEG q92, ≤ 3600 px), thumb.jpg (≤ 480 px),
        # grid.json (TableGrid.to_dict, cercata sulla stessa pagina salvata: coordinate valide sia per
        # il motore sia per i riquadri dell'interfaccia), preparazione.json (correzioni applicate),
        # doc.json; status "in_coda".
        # Dedup: se esiste già un documento con lo stesso sha256+pagina lo restituisce senza duplicarlo.
        # on_document: chiamata per ogni pagina nuova appena salvata (il server la mette in coda).
    def pending_imports(self) -> list[str]               # file la cui importazione è rimasta in sospeso
    def resume_pending_imports(self, on_document=None) -> list[Document]   # completa le importazioni interrotte
    def list(self) -> list[Document]                     # ordinati per created_at, source_file, source_page
    def get(self, doc_id: str) -> Document | None
    def save(self, doc: Document) -> None                # scrittura atomica (tmp + os.replace), aggiorna updated_at
    def delete(self, doc_id: str) -> bool
    def clear(self) -> None
    def page_path(self, doc_id) -> Path ; def thumb_path(self, doc_id) -> Path
    def load_page(self, doc_id) -> np.ndarray ; def load_grid(self, doc_id) -> TableGrid | None
    def load_preparation(self, doc_id) -> dict | None    # preparazione.json (None se assente)
```

Thread-safe (`threading.RLock`). Id documento: `uuid4().hex[:12]`. Scritture atomiche
(tmp + `fsutil.replace_with_retry`, che ritenta se un antivirus blocca per un istante la destinazione).

**Ripresa delle importazioni interrotte.** Le pagine di un file vengono importate in una sola
richiesta: prima del ciclo sulle pagine `add_file` salva una copia del file in
`<root>/.import-<sha256>.bin` e il nome in `.import-<sha256>.json`, eliminate al termine (anche se
l'importazione non riesce: l'errore è già segnalato all'utente). Se il programma si chiude a metà
(es. pagina 2 di 4), i due file restano: all'avvio `app.py` chiama `resume_pending_imports`, che
ripete `add_file` sulla copia salvata (le pagine già presenti sono riconosciute da sha256 e numero
di pagina, quelle mancanti vengono aggiunte e messe in coda con la stessa callback del caricamento).
Una copia scritta a metà (senza `.json`) viene eliminata al caricamento dell'archivio; una copia
danneggiata (sha256 diverso) viene scartata. Ogni ripresa incrementa `tentativi` nel `.json`
prima di iniziare: dopo 3 riprese non completate (es. un file che fa chiudere il programma)
l'importazione viene abbandonata, così un file problematico non blocca ogni avvio.

### 4.9 `sirio/processing.py`

```python
class Processor:
    def __init__(self, store: DocumentStore, settings_provider: Callable[[], Settings],
                 engine_factory: Callable[[Settings], OCREngine] = get_engine,
                 anagrafica: Anagrafica | None = None)      # None = anagrafica_predefinita()
    anagrafica: Anagrafica                                   # proprietà: quella usata (condivisa col server)
    def start(self) -> None          # avvia i worker; rimette in coda i documenti rimasti "in_lavorazione"/"in_coda"
    def stop(self) -> None
    def enqueue(self, doc_id: str) -> None
    def kick(self) -> None           # riprova i documenti in attesa (es. dopo aver salvato la chiave API)
    def status(self) -> dict         # {"in_coda": n, "in_lavorazione": n, "completati": n, "errori": n, "motore_pronto": bool, "messaggio": str}
```

Ciclo per documento: `in_lavorazione` → carica pagina e griglia → `engine.extract` (avanzamento
→ `doc.progress`, `doc.status_message`, salvataggio con frequenza limitata) → **anagrafica**
(`anagrafica.applica_al_risultato`: nomi letti male ricondotti a quelli già confermati, per
qualunque motore; un suo errore viene registrato e non blocca la lettura) →
correzioni della pagina (`store.load_preparation` → avviso `I06` e nota "Pagina raddrizzata
automaticamente: …", per qualunque motore) → `doc.apply_extraction` → `validate_document` →
`completato` (messaggio "Lettura completata: griglia incerta, verificare i valori" se c'è un avviso
`W14`; `scartato` se non è un foglio firma,
`errore` con messaggio in caso di eccezione). Dopo una rilettura il contributo del documento
all'anagrafica viene scollegato (`scollega_documento`): quanto appreso resta. Se il motore non è disponibile, i documenti restano
`in_coda` con `status_message` esplicativo finché `kick()`.
Concorrenza: `settings.concorrenza` per Claude, 1 per il motore locale.

### 4.10 `sirio/server.py` — API HTTP (FastAPI)

`create_app(store, processor, token: str, on_shutdown: Callable | None = None, ..., anagrafica: Anagrafica | None = None) -> FastAPI`
(anagrafica predefinita: `processor.anagrafica`).
Ascolto solo su `127.0.0.1`. Ogni richiesta `/api/*` deve avere l'header `X-Sirio-Token: <token>`
(403 altrimenti) e un header `Host` locale (`127.0.0.1:<porta>` o `localhost:<porta>`).

**Il token è un segreto dell'avvio.** L'indirizzo 127.0.0.1 è condiviso da tutti gli utenti dello
stesso computer (PC condivisi, Desktop remoto/RDS), quindi la pagina principale non consegna il
token a chiunque la chieda: `index.html` è servita con il token inserito in
`<meta name="sirio-token" content="...">` (sostituzione del segnaposto `__SIRIO_TOKEN__`) e con il
cookie `sirio_token` (HttpOnly, SameSite=Strict) **solo** se l'indirizzo contiene `?k=<token>`
(confronto a tempo costante) — è l'indirizzo con cui `app.py` apre la finestra — oppure se la
richiesta ha già il cookie valido (ricaricamento della pagina). Altrimenti risponde 403 con una
piccola pagina in italiano, senza token: «Per motivi di sicurezza apri Sirio OCR dal collegamento
sul desktop o con «Avvia Sirio OCR».». Appena caricata, la pagina toglie `k` dalla barra degli
indirizzi (`history.replaceState`), così non resta nella cronologia né nei preferiti
(`Referrer-Policy: no-referrer` su tutte le risposte). I file statici da `sirio/web/` su
`/static/...` restano pubblici (non contengono segreti).
Per le sole letture (GET/HEAD: immagini, download) e per `heartbeat`/`bye` vale anche il cookie.
Risposte JSON; errori `{"detail": "messaggio italiano"}`.

| Metodo e percorso | Corpo / parametri | Risposta |
|---|---|---|
| `GET /` | `?k=<token>` (o cookie `sirio_token`) | index.html con token; senza: 403, pagina «apri Sirio OCR dal collegamento» |
| `GET /api/health` | — | `{"ok": true, "version": "1.0.0"}` (senza token) |
| `GET /api/state?since=<rev>` | — | se `since == revision`: `{"revision": n, "unchanged": true}`; altrimenti `{"revision", "documents": [DocSummary], "queue": Processor.status(), "kpi": {...}}` |
| `POST /api/upload` | multipart `files` (più file) | `{"documents": [DocSummary], "errors": [{"file", "message"}]}` |
| `GET /api/documents/{id}` | — | `Document` completo (con l'eventuale W13) + `"grid": TableGrid.to_dict()` + `"image_url"` |
| `PUT /api/documents/{id}` | `{"header"?: Header, "rows"?: [DayRow], "user_verified"?: bool}` | `Document` aggiornato (rivalidato, con l'eventuale W13) |
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
| `GET /api/anagrafica` | — | `{"campi": {"operatore"\|"alunno"\|"istituto"\|"ente": [{"valore", "conteggio", "ultimo_uso", "manuale"}]}, "associazioni": [{"operatore", "alunno", "istituto", "ente", "lotto", "municipalita", "ore_pei", "conteggio", "ultimo_uso"}]}` |
| `POST /api/anagrafica` | `{"campo", "valore"}` (aggiunta manuale) | come GET; 422 se campo sconosciuto o valore non valido |
| `DELETE /api/anagrafica/{campo}/{valore}` | valore codificato nell'URL (può contenere `/`; maiuscole, accenti e ordine dei nomi indifferenti) | come GET; 404 se la voce non esiste |
| `POST /api/heartbeat` | — | `{"ok": true}` |
| `POST /api/bye` | — | `{"ok": true}` (la finestra si sta chiudendo) |

`DocSummary` = `{"id", "display_name", "source_file", "source_page", "page_count", "status",
"progress", "status_message", "error", "is_foglio_firma", "engine", "model", "user_verified",
"created_at", "updated_at", "header": {operatore, alunno, istituto, ente, mese, anno, ore_pei},
"totals": {ore_riconosciute, ore_dichiarate, ore_calcolate, totale_mensile_dichiarato,
differenza_totale, giorni_lavorati, n_errori, n_attenzioni, n_info, campi_incerti,
campi_illeggibili, stato}, "cost_usd", "thumb_url"}`.

`kpi` = `{"documenti", "completati", "ore_totali", "errori", "attenzioni", "illeggibili", "incerti",
"da_verificare", "verificati", "possibili_duplicati", "documenti_con_errori", "in_coda", "in_lavorazione",
"errori_elaborazione", "scartati", "costo_usd"}` (`possibili_duplicati` = fogli con la W13).

**Possibili duplicati (W13).** `/api/state`, `GET /api/documents`, `GET`/`PUT /api/documents/{id}` e
`/api/preview` calcolano al volo `validation.segnala_possibili_duplicati` sui fogli firma completati
(per l'anteprima: sui documenti dell'anteprima, gli stessi che finirebbero nell'Excel): i documenti
coinvolti mostrano l'attenzione nell'elenco delle anomalie, contatori aggiornati e stato «da
verificare» già prima dell'esportazione. La W13 non viene salvata in `doc.json`.

Modifica (`PUT`): i campi cambiati rispetto al valore precedente vengono aggiunti a
`user_edited` (chiavi `header.<campo>` / `rows.<giorno>.<campo>`), il valore OCR originale viene
memorizzato in `ocr_originali` (solo la prima volta) e il campo viene tolto da `incerti`/`illeggibili`.
Ore ammesse: 0–24 per `ore_dichiarate`, 0–60 per `ore_pei`, 0–744 per `totale_mensile_dichiarato`
(`validation.MAX_ORE_CAMPO`; fuori intervallo → 422 «… indicare un numero di ore tra 0 e N»).
Dopo ogni `PUT` riuscito il documento viene passato a `Anagrafica.impara_documento` (vedi 4.13: se
confermato si apprende l'intera intestazione, altrimenti solo i campi anagrafici corretti a mano).
L'eliminazione di documenti (`DELETE /api/documents[/{id}]`) scollega i loro contributi senza
cancellare quanto appreso.

### 4.11 `sirio/app.py` — avvio e ciclo di vita

`python -m sirio [--porta N] [--senza-finestra] [--cartella-dati DIR]`

`python -m sirio --scarica-modello [MODELLO] [--cartella-dati DIR]`: scarica soltanto il modello del
motore offline (predefinito quello di `settings.local_model`) in `local_engine.default_cache_dir()`
con una riga d'avanzamento aggiornata sul posto (percentuale e MB) e termina, senza server né
finestra né lucchetto d'istanza. Uscita 0 se il modello è disponibile (anche "Modello già presente"),
1 con il messaggio d'errore in italiano su stderr. Registro solo su file. Usato dai programmi di avvio.

Avvio normale:

1. istanza singola: lucchetto `data_dir()/istanza.lock` e file `data_dir()/istanza.json`
   `{port, pid, token, version, url, started_at}` (scrittura atomica, permessi 600); se
   un'istanza risponde a `/api/health`, apre solo una nuova finestra verso di essa (con il
   segreto `?k=` letto da `istanza.json`) ed esce;
2. avvia uvicorn su `127.0.0.1:<porta libera>` in un thread; token casuale (`secrets.token_urlsafe`);
   avvia la coda di elaborazione e, in un thread a parte, `store.resume_pending_imports` (pagine
   mancanti delle importazioni interrotte, messe in coda con `processor.enqueue`: la finestra non
   attende la conversione delle pagine);
3. apre la finestra applicazione all'indirizzo `http://127.0.0.1:<porta>/?k=<token>` (segreto
   dell'avvio, vedi 4.10; mai scritto nel registro): Microsoft Edge / Google Chrome / Chromium in
   modalità app (`--app=URL --user-data-dir=<data_dir>/finestra --window-size=1480,940
   --no-first-run --no-default-browser-check`), attende la chiusura del processo; se non trovato,
   browser predefinito (`webbrowser.open`, stesso indirizzo); con `--senza-finestra` l'indirizzo
   completo è stampato sulla console;
4. chiusura: quando il processo della finestra termina e non arrivano heartbeat per 75 s
   (`WINDOW_GRACE`), oppure dopo `/api/bye` senza nuovi heartbeat per 75 s (`BYE_GRACE`), oppure
   senza heartbeat per 15 minuti → arresto ordinato (processor.stop, server.should_exit). Le attese
   superano l'intervallo tra due heartbeat anche di una finestra ridotta a icona (il browser ne
   rallenta i timer fino a circa uno al minuto): un'altra finestra ancora aperta, o il passaggio
   della finestra a un processo del browser già attivo, non provocano l'arresto; dopo una
   sospensione del sistema i tempi ripartono da zero;
5. log su `logs_dir()/sirio.log` (rotazione), eccezioni non gestite registrate.

### 4.12 Interfaccia (sirio/web)

Stile premium e professionale, HTML/CSS/JS senza dipendenze esterne (funziona offline), tema chiaro e
scuro. Viste:

* **Dashboard / Importa**: indicatori (fogli, ore totali, da verificare con gli eventuali possibili
  duplicati, illeggibili, errori), grande
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
* **Impostazioni**: motore OCR — **Locale offline (predefinito, gratuito, privacy totale)** oppure
  Claude Vision (facoltativo, a pagamento, precisione superiore); la sezione chiave API/modello/effort/
  verifica incrociata è visibile solo se si sceglie Claude; concorrenza, tema, cartelle dati/export,
  nota privacy. Nessun banner "configura la chiave API" quando il motore è quello locale.

### 4.13 `sirio/anagrafica.py` — anagrafica appresa

Operatori, alunni, istituti ed enti ricorrono ogni mese: i nomi letti male (soprattutto dal motore
offline, es. «R0SSI MAR1O» per «ROSSI MARIO») vengono ricondotti ai nomi già **confermati dall'utente**.
File `data_dir()/"anagrafica.json"` (scrittura atomica, `threading.RLock`, istanza condivisa per
cartella dati con `anagrafica_predefinita()`; un file illeggibile viene messo da parte come
`anagrafica.json.illeggibile-<data>` e si riparte da zero).

```python
CAMPI = ("operatore", "alunno", "istituto", "ente")
CAMPI_ASSOCIAZIONE = ("alunno", "istituto", "ente", "lotto", "municipalita", "ore_pei")
class Anagrafica:
    def __init__(self, path: Path)
    def impara_documento(self, doc: Document) -> bool        # idempotente; True se cambiata
    def scollega_documento(self, doc_id: str) -> None        # documento eliminato/riletto: quanto appreso resta
    def scollega_tutti(self) -> None
    def correggi(self, header: Header) -> list[Riconoscimento]   # modifica header (valori, incerti, illeggibili)
    def aggiungi(self, campo: str, valore: str) -> dict      # ValueError (italiano) se non valido
    def rimuovi(self, campo: str, valore: str) -> bool       # anche dalle associazioni
    def valori(self, campo: str) -> list[str]                # per uso decrescente
    def esporta(self) -> dict                                # contenuto di GET /api/anagrafica
def applica_al_risultato(result: ExtractionResult, anagrafica: Anagrafica) -> list[Riconoscimento]
    # correggi(result.header) + frasi aggiunte a result.ocr_notes
def note(decisioni) -> list[str]
def normalizza(testo) -> str ; def chiave(testo) -> str ; def confronta(letto, noto) -> Somiglianza
def anagrafica_predefinita() -> Anagrafica
```

**Apprendimento solo da dati affidabili** (mai dalla lettura OCR grezza): documento confermato
(`user_verified`) → i quattro campi anagrafici e l'associazione operatore → {alunno, istituto, ente,
lotto, municipalità, ore PEI} (chiave operatore+alunno: un operatore può seguire più alunni; vale
l'ultimo valore confermato); documento non confermato → solo i campi anagrafici corretti a mano
(`header.<campo>` in `user_edited`); aggiunta manuale. Il contributo di ogni documento è
memorizzato (`documenti`): il ricalcolo a ogni salvataggio è idempotente, una correzione
successiva sostituisce il valore appreso prima dallo stesso documento (gli errori di battitura
corretti non restano), togliere la conferma ritira ciò che non è stato scritto a mano. Il
`conteggio` di una voce è il numero di documenti che la confermano; le voci manuali restano anche
con conteggio 0.

**Somiglianza** (`confronta`): testi normalizzati (maiuscole, senza accenti/punteggiatura, spazi
compattati, ordine dei nomi «NOME COGNOME»/«COGNOME NOME» indifferente — rotazioni delle parole);
`valore` = media fra `difflib.SequenceMatcher.ratio` e una distanza di modifica pesata (scambi di
lettere adiacenti, confusioni tipiche come O/0, I/1, M/H, U/N a costo 0,5); `copertura` = parola
peggio ritrovata (ricerca approssimata di ogni parola di ≥ 3 lettere nell'altro testo, in entrambe
le direzioni); `sospetta` = differiscono solo per vocali finali (MARIO/MARIA) o numeri diversi
(«IC 9»/«IC 10»).

**Regole di decisione** (`SOGLIE`, calibrate su letture reali del motore locale e ~1.700 letture
simulate: ~72% dei nomi letti male ricondotti alla voce giusta, nessuna sostituzione errata, < 1%
di persone nuove sostituite — e comunque segnalate incerte):

| Campo | sostituzione (punteggio) | copertura (con associazione) | suggerimento |
|---|---|---|---|
| operatore, alunno | 0,72 | 0,60 (0,40) | 0,70 |
| istituto | 0,62 | 0,35 (0,25) | 0,55 |
| ente | 0,70 | 0,55 (0,30) | 0,55 |

* lettura identica a una voce (a meno di maiuscole/accenti/ordine): si usa la voce, il campo non è
  più incerto ("lettura confermata" nelle note se il motore lo aveva segnalato);
* punteggio = somiglianza + bonus d'associazione (0,10 se la voce è associata all'operatore — o
  all'alunno — riconosciuto, 0,15 se alla coppia operatore+alunno); se ≥ soglia, copertura
  sufficiente, distacco ≥ `MARGINE` (0,08) dal secondo candidato e non `sospetta` → **sostituita**;
  resta incerta se somiglianza < `CERTEZZA` (0,90), copertura < 0,80 o numeri diversi. Nota:
  «Operatore riconosciuto dall'anagrafica (somiglianza 92%): letto «…», usato «…».» (la lettura
  originale sta in `ocr_notes`, non in `ocr_originali`, che resta riservato alle correzioni
  dell'utente);
* candidati troppo vicini (**ambiguo**) o un solo candidato sopra la soglia di suggerimento ma non
  sostituibile (**simile**): lettura invariata, campo incerto, candidati nelle note;
* sotto la soglia di suggerimento: nome nuovo, nessuna modifica;
* ordine: operatore, alunno (con l'associazione dell'operatore), di nuovo l'operatore con
  l'associazione dell'alunno se non era stato riconosciuto, istituto ed ente;
* campi **illeggibili** (valore `None` e in `illeggibili`) proposti solo da un'associazione
  univoca dell'operatore/alunno riconosciuti (es. un solo alunno noto per quell'operatore; anche
  operatore, istituto, ente, lotto, municipalità, ore PEI): il campo passa da `illeggibili` a
  `incerti`.

## 5. Avvio "a un clic"

* Windows: doppio clic su `Avvia Sirio OCR.bat` → `launcher/avvia.ps1`.
* macOS: `Avvia Sirio OCR.command`; Linux: `avvia-sirio-ocr.sh` → `launcher/avvia.sh`.

Il bootstrap scarica **uv** (Astral) nella cartella dei componenti — su Windows
`%LOCALAPPDATA%\SirioOCR\runtime` (locale, non sincronizzata con OneDrive; `.runtime\` del programma
solo se `LOCALAPPDATA` manca; la vecchia `.runtime\` delle versioni precedenti viene eliminata se non
in uso), su macOS e Linux `.runtime/` nella cartella del programma — che installa Python 3.12
gestito e tutte le dipendenze (`uv sync --extra offline`, PyTorch solo CPU) in `<componenti>/venv`;
subito dopo un'installazione riuscita (salvo `SIRIO_SENZA_OFFLINE=1`) esegue
`<python> -m sirio --scarica-modello` per scaricare il modello di riconoscimento (circa 1,3 GB, con
l'avanzamento in console): un errore non blocca l'avvio, il modello verrà scaricato al primo
utilizzo. Ai successivi avvii verifica solo che tutto sia aggiornato (pochi secondi). Ripiego:
Python di sistema ≥ 3.10 + `venv` + `pip`. Al primo avvio su Windows crea il collegamento "Sirio OCR"
sul desktop con icona. Nessun privilegio di amministratore richiesto.
