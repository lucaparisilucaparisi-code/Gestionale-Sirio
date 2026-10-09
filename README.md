# Sirio OCR

**Lettura automatica dei fogli firma dell'Assistenza Specialistica e rendicontazione Excel.**

Sirio OCR legge i fogli firma compilati a mano (modello *Assistenza Specialistica
all'integrazione scolastica — Comune di Napoli*), anche centinaia di PDF alla
volta, controlla che i dati siano coerenti, mostra in una griglia stile Excel
tutto ciò che ha letto — evidenziando i dati **illeggibili** o **incerti** — e
genera un file Excel preciso e pronto per la rendicontazione.

---

## Avvio in un clic

1. Scarica il programma (pulsante **Code → Download ZIP** su GitHub) ed estrai la
   cartella dove preferisci, ad esempio in *Documenti*.
2. Fai doppio clic su:
   - **Windows:** `Avvia Sirio OCR.bat`
   - **macOS:** `Avvia Sirio OCR.command`
   - **Linux:** `avvia-sirio-ocr.sh`

Al **primo avvio** il programma scarica da solo tutto ciò che serve (Python 3.12 e
le librerie, circa 1 GB compreso il motore offline): servono una connessione a
Internet e qualche minuto. Non servono privilegi di amministratore e non viene
modificato nulla fuori dalla cartella del programma. Su Windows viene creato
anche il collegamento **Sirio OCR** sul desktop.

Dagli avvii successivi l'applicazione si apre in pochi secondi, in una finestra
dedicata (Microsoft Edge o Google Chrome in modalità applicazione).

> **Windows:** se compare l'avviso "Windows ha protetto il PC", scegli
> *Ulteriori informazioni → Esegui comunque* (succede con i file scaricati da Internet).
> **macOS:** la prima volta apri il file con *clic destro → Apri*.

## Configurazione (una sola volta)

Il motore di lettura consigliato è **Claude Vision** di Anthropic, oggi il sistema
più accurato nella lettura della scrittura a mano su moduli.

1. Crea una chiave API su [console.anthropic.com](https://console.anthropic.com)
   (*Settings → API Keys → Create Key*) e ricarica un piccolo credito.
2. In Sirio OCR apri **Impostazioni**, incolla la chiave e premi **Verifica chiave**.
   La chiave viene conservata nel portachiavi protetto del sistema operativo.

In alternativa è disponibile il **motore locale offline** (OpenCV + TrOCR): i
documenti non lasciano mai il computer e non ci sono costi, ma la precisione sulla
scrittura a mano è inferiore e la revisione manuale è più impegnativa.

## Come si usa

1. **Importa** — trascina nella finestra i PDF (anche multipagina), le immagini
   (JPG, PNG, TIFF) o intere cartelle. Ogni pagina diventa un foglio firma da elaborare.
2. **Lettura automatica** — i fogli vengono letti in parallelo; per ognuno si vede
   l'avanzamento. Con la *verifica incrociata* (attiva in modo predefinito) le righe
   dubbie vengono rilette una seconda volta con immagini ingrandite.
3. **Revisione** — per ogni foglio la scansione è affiancata a una griglia stile
   Excel con i 31 giorni, modificabile cella per cella:
   - 🟥 **illeggibile**: scritto ma non leggibile, va inserito a mano guardando la scansione;
   - 🟨 **incerto**: letto, ma da controllare;
   - 🟦 **corretto a mano**: il valore originale letto resta tracciato;
   - selezionando una cella, la scansione evidenzia il punto corrispondente.
   Il tasto **F8** porta al prossimo campo da verificare. Ogni modifica viene salvata
   e ricontrollata subito.
4. **Anteprima Excel** — mostra le schede che verranno generate (Riepilogo,
   Dettaglio giornaliero, Anomalie), anch'esse modificabili, poi **Genera Excel**.

Il file viene salvato in *Documenti/Sirio OCR/Export*.

## L'Excel generato

| Foglio | Contenuto |
|---|---|
| **Riepilogo** | Indicatori del periodo e una riga per ogni foglio firma: operatore, alunno, istituto, mese, ore programmate / dichiarate / calcolate, totale mensile, differenze, assenze, firme mancanti, stato (OK / Da verificare / Errori), collegamento alla scheda. |
| **Dettaglio giornaliero** | Ogni giornata di ogni foglio: data e giorno della settimana, orari programmati ed effettivi, ore, assenze, firma, note, esito dei controlli. Con filtri. |
| **Una scheda per foglio firma** | Replica fedele del modulo cartaceo con i 31 giorni, weekend e festivi evidenziati, celle illeggibili / incerte / corrette colorate e commentate, formule per i totali, riepilogo settimanale rispetto alle ore da PEI, anomalie. |
| **Anomalie** | Tutti i controlli non superati, con giorno, gravità e valori letti/attesi. |
| **Totali per operatore** | Ore e giornate per operatore, alunno e istituto. |
| **Legenda e note** | Significato dei colori, regole di calcolo, motore OCR utilizzato. |

Orari, date e ore sono veri valori Excel (non testo) e i totali sono formule: il
file resta "vivo" e si aggiorna se si correggono i dati.

## Controlli automatici

Per ogni foglio vengono verificati, tra gli altri:

- coerenza tra orario effettivo e ore dichiarate di ogni giorno;
- somma dei giorni rispetto al **totale ore effettive mensili** dichiarato;
- orari non validi, uscita prima dell'entrata, giorni inesistenti nel mese;
- firme dell'operatore mancanti, firma del coordinatore e timbro del referente;
- prestazioni in domenica o festivi (calendario italiano e S. Gennaro);
- ore settimanali oltre le **ore da PEI**;
- assenze dell'alunno (con ore riconosciute) e dell'operatore (es. permesso *104*);
- prestazioni programmate ma non svolte (es. *Ponte di Carnevale*);
- intestazione incompleta, campi incerti o illeggibili.

## Costi indicativi (motore Claude)

Il costo dipende dal modello scelto e da quanto è fitta la scrittura; l'applicazione
mostra il costo effettivo di ogni foglio. Stime indicative per foglio firma:

| Modello | Precisione | Costo stimato per foglio |
|---|---|---|
| Claude Opus 5.5 (consigliato) | massima | ~ 0,15 – 0,35 $ |
| Claude Sonnet 5.5 | molto alta | ~ 0,08 – 0,18 $ |
| Claude Haiku 5.5 | buona | meno di 0,01 $ |

## Privacy

I fogli firma contengono dati personali di minori con disabilità (categorie
particolari di dati, art. 9 GDPR).

- Con il **motore Claude** le immagini dei fogli vengono inviate all'API di
  Anthropic per la lettura. Anthropic non usa i dati inviati tramite API per
  addestrare i modelli; valutate comunque con il vostro DPO la nomina a
  responsabile del trattamento e le condizioni contrattuali applicabili.
- Con il **motore locale** tutto resta sul computer.
- Documenti, immagini ed export sono salvati solo in locale (cartella dati
  dell'utente e *Documenti/Sirio OCR*). Dalla schermata Documenti si possono
  eliminare in qualsiasi momento.

## Risoluzione dei problemi

- **Il primo avvio si interrompe:** controlla la connessione e riprova; i dettagli
  sono in `.runtime/avvio.log` nella cartella del programma.
- **Reinstallare da zero:** chiudi il programma, elimina la cartella `.runtime`
  e riavvia.
- **Non installare il motore offline** (risparmia ~1 GB): imposta la variabile
  d'ambiente `SIRIO_SENZA_OFFLINE=1` prima del primo avvio.
- **Registro dell'applicazione:** `sirio.log` nella cartella dati
  (Windows: `%LOCALAPPDATA%\SirioOCR\log`).

## Per sviluppatori

```bash
uv sync --extra offline --extra dev     # ambiente di sviluppo
uv run python -m sirio                  # avvia l'applicazione
uv run pytest                           # test
```

L'architettura, i contratti tra i moduli e le API interne sono descritti in
[`docs/ARCHITETTURA.md`](docs/ARCHITETTURA.md). I test che usano un foglio firma
reale leggono il percorso da `SIRIO_SAMPLE_PDF` (e la verità attesa da
`SIRIO_SAMPLE_TRUTH`): i documenti reali non devono mai essere aggiunti al repository.
