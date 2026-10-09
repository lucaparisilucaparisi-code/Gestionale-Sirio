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

## Motore di lettura

Sirio OCR funziona **subito, senza configurazione**: il motore predefinito è il
**motore locale offline** (OpenCV + TrOCR), gratuito, senza chiave API e senza
abbonamenti. I fogli firma **non lasciano mai il computer**. Al primo utilizzo
viene scaricato una sola volta il modello di riconoscimento della scrittura a mano;
da quel momento il programma funziona anche senza Internet.

La scrittura a mano è difficile da leggere per qualunque sistema automatico: per
questo ogni valore viene controllato (coerenza tra orari e ore, totali, firme…) e
i campi **incerti** o **illeggibili** vengono evidenziati per una rapida verifica
nella griglia di revisione, prima di generare l'Excel.

**Facoltativo — Claude Vision (a pagamento).** Per una precisione superiore sulla
scrittura difficile, in *Impostazioni* si può scegliere il motore cloud Claude di
Anthropic: serve una chiave API da [console.anthropic.com](https://console.anthropic.com)
(*Settings → API Keys*) con un piccolo credito. In questo caso le immagini dei
fogli vengono inviate ad Anthropic (vedi *Privacy*).

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

## Costi

Con il motore locale offline (predefinito) il programma è **completamente gratuito**.

Solo se si sceglie il motore facoltativo Claude, il costo dipende dal modello scelto e da quanto è fitta la scrittura; l'applicazione
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
  Anthropic per la lettura. Per impostazione predefinita Anthropic non usa i dati inviati tramite API per
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
- **Installazione senza motore offline** (solo per chi usa esclusivamente Claude,
  risparmia ~1 GB): imposta la variabile d'ambiente `SIRIO_SENZA_OFFLINE=1` prima
  del primo avvio.
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
