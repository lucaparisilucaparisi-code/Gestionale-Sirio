# Sirio OCR

**Lettura automatica dei fogli firma dell'Assistenza Specialistica e rendicontazione Excel.**

Sirio OCR legge i fogli firma compilati a mano (modello *Assistenza Specialistica
all'integrazione scolastica — Comune di Napoli*), anche centinaia di PDF alla
volta, controlla che i dati siano coerenti, mostra in una griglia stile Excel
tutto ciò che ha letto — evidenziando i dati **illeggibili** o **incerti** — e
genera un file Excel preciso e pronto per la rendicontazione.

---

## Avvio in un clic

1. Scarica il programma (pulsante **Code → Download ZIP** su GitHub) ed estrai
   tutti i file (*clic destro sul file ZIP → Estrai tutto*) in una cartella **non
   sincronizzata** con OneDrive o iCloud, ad esempio `C:\Users\<nome>\SirioOCR`
   su Windows.
2. Fai doppio clic su:
   - **Windows:** `Avvia Sirio OCR.bat`
   - **macOS:** `Avvia Sirio OCR.command`
   - **Linux:** `avvia-sirio-ocr.sh`

Al **primo avvio** il programma scarica da solo tutto ciò che serve: Python 3.12,
le librerie e il modello di riconoscimento della scrittura a mano del motore
offline, **circa 1,7 GB da scaricare**; servono **circa 4 GB di spazio libero**, una
connessione a Internet e qualche minuto. Non servono privilegi di amministratore.
I componenti vengono installati in `%LOCALAPPDATA%\SirioOCR\runtime` su Windows
(cartella locale, non sincronizzata con OneDrive) e nella cartella `.runtime` del
programma su macOS e Linux. Su Windows viene creato anche il collegamento
**Sirio OCR** sul desktop (aggiornato automaticamente se in seguito si avvia una
copia del programma estratta in un'altra cartella).

Dagli avvii successivi l'applicazione si apre in pochi secondi, in una finestra
dedicata (Microsoft Edge o Google Chrome in modalità applicazione).

> **Windows:** se compare l'avviso "Windows ha protetto il PC", scegli
> *Ulteriori informazioni → Esegui comunque* (succede con i file scaricati da Internet).
> **macOS:** se macOS blocca il file perché scaricato da Internet, apri
> *Impostazioni di Sistema → Privacy e sicurezza* e scegli *Apri comunque*
> (fino a macOS 14 basta anche *clic destro sul file → Apri*).

## Motore di lettura

Sirio OCR funziona **subito, senza configurazione**: il motore predefinito è il
**motore locale offline** (OpenCV + TrOCR), gratuito, senza chiave API e senza
abbonamenti. I fogli firma **non lasciano mai il computer**. Il modello di
riconoscimento della scrittura a mano (circa 1,3 GB) viene scaricato una sola volta
durante il primo avvio; da quel momento la lettura funziona anche senza Internet.

Il motore offline richiede Windows a 64 bit, Linux a 64 bit oppure un Mac con
processore Apple (M1 o successivi) e macOS 14 o successivo. Sugli altri computer
(ad esempio i Mac con processore Intel) il programma si installa comunque e si può
usare il motore Claude. Su Windows serve anche il componente *Microsoft Visual C++
Redistributable*, già presente sulla maggior parte dei PC: se manca, il programma di
avvio lo segnala e indica dove scaricarlo.

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
- Con il **motore locale** la lettura avviene interamente sul computer.
- Documenti e immagini importati sono salvati nella cartella dati dell'utente
  (Windows: `%LOCALAPPDATA%\SirioOCR`, non sincronizzata con OneDrive). Dalla
  schermata Documenti si possono eliminare in qualsiasi momento.
- Gli Excel generati sono salvati in *Documenti\Sirio OCR\Export*. **Attenzione:**
  se la cartella *Documenti* è sincronizzata con OneDrive (backup delle cartelle di
  Windows, proposto durante la configurazione di Windows 11; il percorso indicato in
  *Impostazioni* contiene «OneDrive») o con iCloud, gli Excel, che contengono dati
  personali dei minori, vengono caricati anche nel cloud. In questo caso escludete
  *Documenti* dal backup, oppure fate impostare dal referente informatico la
  variabile d'ambiente `SIRIO_EXPORT_DIR` con una cartella locale non sincronizzata
  (ad esempio `C:\SirioOCR\Export`).

## Risoluzione dei problemi

- **Il primo avvio si interrompe:** controlla la connessione e riprova; i dettagli
  (compresi i messaggi dell'installazione) sono in `avvio.log`, nella cartella dei
  componenti: `%LOCALAPPDATA%\SirioOCR\runtime\avvio.log` su Windows,
  `.runtime/avvio.log` nella cartella del programma su macOS e Linux.
- **Antivirus o proxy aziendale:** il programma di avvio usa i certificati di
  sistema, quindi funziona anche con gli antivirus che analizzano le connessioni
  HTTPS e con i proxy aziendali già configurati sul computer.
- **Reinstallare da zero:** chiudi il programma, elimina la cartella dei componenti
  (`%LOCALAPPDATA%\SirioOCR\runtime` su Windows, `.runtime` nella cartella del
  programma su macOS e Linux) e riavvia.
- **Installazione senza motore offline** (solo per chi usa esclusivamente Claude,
  risparmia circa 1,5 GB di download e più spazio su disco): imposta la variabile
  d'ambiente `SIRIO_SENZA_OFFLINE=1` prima del primo avvio.
- **Registro dell'applicazione:** `sirio.log` nella cartella dati
  (Windows: `%LOCALAPPDATA%\SirioOCR\log`).
- **Disinstallare:** chiudi il programma ed elimina la cartella del programma, la
  cartella dei dati (documenti importati, modello del motore offline e, su Windows,
  componenti): `%LOCALAPPDATA%\SirioOCR` su Windows,
  `~/Library/Application Support/SirioOCR` su macOS, `~/.local/share/SirioOCR` su
  Linux. Su Windows elimina anche il collegamento **Sirio OCR** sul desktop. Gli
  Excel generati restano in *Documenti/Sirio OCR/Export*.

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
