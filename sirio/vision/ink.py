"""Analisi dell'inchiostro nelle celle del foglio firma.

Funzioni per capire, senza OCR, se una cella e' vuota, contiene una crocetta
(colonne di assenza), una firma o un trattino isolato ("-").

Principi
--------
* L'inchiostro e' cio' che e' sensibilmente piu' scuro della carta circostante
  (soglia relativa allo sfondo locale: funziona con scansioni chiare o scure).
* Le linee della griglia che cadono nel riquadro (griglia imprecisa di qualche
  pixel, linee ondulate) vengono rimosse prima di misurare.
* Firme e scritte debordano spesso nelle righe vicine: ogni tratto (componente
  connessa) viene attribuito alla cella che ne contiene la parte maggiore, cosi'
  le code delle firme della riga sopra o sotto non generano falsi positivi.

Tutti i riquadri sono ``(x0, y0, x1, y1)`` in pixel dell'immagine (BGR o grigia).

Inchiostro "proprio" di una cella (API per il motore)
----------------------------------------------------
Per leggere le celle di una pagina si costruisce una volta ``PageInk`` e poi si
chiede ogni cella (circa 1-2 s per pagina a 300 dpi, poi ~2 ms per cella)::

    page = ink.PageInk(img, grid)              # img BGR normalizzata, grid = TableGrid
    res = page.own_ink(giorno, "eff_entrata")  # oppure ink.cell_own_ink(img, grid, g, col, page=page)
    res.crop             # grigio uint8 della cella, solo scrittura propria (righe, barrature,
                         # tratti delle celle vicine sbiancati): da dare all'OCR
    res.mask             # bool, inchiostro proprio (stessa forma di crop)
    res.ink_ratio        # inchiostro proprio / area interna; res.empty = sotto BLANK_RATIO
    res.struck           # una barratura diagonale lunga attraversa la cella
    res.row_struck       # la riga del giorno e' barrata (riga non usata): nessun valore
    res.line_across      # linea orizzontale a penna attraverso la cella (= "-", non svolta)
    res.slash            # la cella contiene solo un "/" (giornata non lavorata, non un valore;
                         # nella colonna ore un "1" inclinato puo' somigliargli: res.stroke_angle)
    res.overflow_removed # frazione d'inchiostro tolta perche' di celle vicine (> 0.3: segnalare)
    res.strike_removed   # frazione tolta perche' parte di barrature/linee
    page.strokes         # list[Stroke]: kind "diagonale"/"orizzontale", points, days, columns
    page.struck_days     # giorni con righe barrate; page.line_days: giorni con linee orizzontali
    page.creases         # list[Crease]: pieghe del foglio (tolte come le barrature; res.crease)

Cosa riconosce:

* barrature diagonali che attraversano piu' righe e almeno un bordo di colonna
  (righe 29-31 cancellate), anche a matita chiara o tratteggiate (pezzi
  rettilinei sottili concatenati + trasformata di Hough con verifica che
  attorno al tratto non ci sia scrittura);
* linee orizzontali tirate a mano attraverso le celle di una riga (orario
  effettivo "non svolto"), nella fascia centrale della riga; le linee doppie o
  le ombre del modulo, perfettamente parallele alla griglia, sono ignorate;
* pieghe del foglio (``find_creases``): lunghe linee sottili quasi verticali,
  anche chiare e leggermente curve, continue per almeno 8 righe e lontane dai
  bordi di colonna stampati (l'ombra della piega di un foglio piegato e
  fotografato): tolte dall'inchiostro proprio come le barrature, salvo i tratti
  di scrittura che le attraversano; ``ink_under_crease`` dice se sotto la piega
  c'e' scrittura piu' scura della piega (una cifra che togliendo la piega
  sparirebbe);
* tratti debordanti: un tratto appartiene alla cella che ne contiene la parte
  maggiore (ricucito solo attraverso le linee orizzontali); le code sottili
  (virgole, aste) della riga sopra/sotto che lasciano nella cella solo un
  pezzetto addossato al bordo vengono tolte; nella colonna "Note" vengono tolte
  le continuazioni in corsivo delle firme che attraversano la linea di confine
  (se la firma e' nettamente piu' grande del pezzo che deborda).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from sirio.vision.preprocess import to_gray

Box = tuple[int, int, int, int]

# Soglie calibrate sul modulo reale (scansione a 300 dpi) e sui fogli sintetici.
BLANK_RATIO = 0.0035        # sotto questa frazione d'inchiostro la cella e' vuota
SIGNATURE_RATIO = 0.018     # inchiostro minimo per una firma
SIGNATURE_WIDTH = 0.22      # estensione orizzontale minima della firma (frazione della cella)
CROSS_SPAN = 0.16           # estensione minima (larghezza e altezza) di una crocetta
DASH_MAX_HEIGHT = 0.22      # altezza massima di un trattino (frazione della cella)
DASH_MIN_ASPECT = 1.8       # rapporto larghezza/altezza minimo di un trattino
SHARED_STROKE_RATIO = 0.045  # inchiostro minimo nella cella per un tratto condiviso con altre righe
SHARED_STROKE_EXTENT = 0.4   # ... e altezza minima occupata nella cella (frazione)


@dataclass
class CellInk:
    """Misure dell'inchiostro attribuito a una cella."""

    ratio: float = 0.0          # pixel d'inchiostro della cella / area della cella
    pixels: int = 0
    width_frac: float = 0.0     # larghezza dell'inchiostro / larghezza della cella
    height_frac: float = 0.0    # altezza dell'inchiostro / altezza della cella
    n_components: int = 0       # tratti significativi attribuiti alla cella
    largest_frac: float = 0.0   # tratto piu' grande / totale inchiostro della cella
    cx: float = 0.5             # baricentro orizzontale (0 = bordo sinistro, 1 = destro)
    cy: float = 0.5             # baricentro verticale (0 = bordo superiore, 1 = inferiore)
    bbox: Box | None = None     # riquadro dell'inchiostro in coordinate dell'immagine
    foreign: float = 0.0        # inchiostro presente nella cella ma attribuito ad altre celle
    cell_w: int = 0             # dimensioni della cella analizzata (pixel)
    cell_h: int = 0

    @property
    def empty(self) -> bool:
        return self.pixels == 0

    @property
    def ink_w(self) -> float:
        """Larghezza dell'inchiostro in pixel."""
        return self.width_frac * self.cell_w

    @property
    def ink_h(self) -> float:
        """Altezza dell'inchiostro in pixel."""
        return self.height_frac * self.cell_h


# --------------------------------------------------------------------------
# Utilita'
# --------------------------------------------------------------------------

def _norm_box(img: np.ndarray, box: Box, pad: int = 0) -> Box:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0 = max(0, min(w, x0 - pad))
    y0 = max(0, min(h, y0 - pad))
    x1 = max(0, min(w, x1 + pad))
    y1 = max(0, min(h, y1 + pad))
    return x0, y0, x1, y1


def crop(img: np.ndarray, box: Box, pad: int = 0) -> np.ndarray:
    """Ritaglio (copia) del riquadro, allargato di ``pad`` pixel e limitato all'immagine."""
    x0, y0, x1, y1 = _norm_box(img, box, pad)
    if x1 <= x0 or y1 <= y0:
        shape = (0, 0) + img.shape[2:]
        return np.zeros(shape, dtype=img.dtype)
    return img[y0:y1, x0:x1].copy()


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Inchiostro (bool) rispetto al livello della carta nella zona."""
    bg = float(np.percentile(gray, 90))
    if bg < 80:                       # zona quasi tutta scura (bordo di scansione, timbro pieno)
        bg = max(bg, float(np.percentile(gray, 99)))
    thr = max(35.0, bg - max(45.0, 0.2 * bg))
    return gray < thr


def _remove_grid_lines(ink: np.ndarray, line_len_x: int, line_len_y: int, grow: float = 2.0,
                       rows: list[tuple[int, int]] | None = None,
                       cols: list[tuple[int, int]] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Toglie le linee lunghe orizzontali/verticali.

    ``rows``/``cols`` (facoltativi) limitano le linee orizzontali/verticali alle
    fasce in cui possono trovarsi i bordi della griglia: un tratto di scrittura
    lungo e dritto in mezzo alla cella non viene scambiato per una linea.
    Restituisce (inchiostro, linee orizzontali, linee verticali)."""
    m = ink.astype(np.uint8) * 255
    hl = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, line_len_x), 1)))
    vl = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, line_len_y))))
    if rows is not None:
        keep = np.zeros(hl.shape[0], dtype=bool)
        for a, b in rows:
            keep[max(0, a):max(0, b)] = True
        hl[~keep, :] = 0
    if cols is not None:
        keep = np.zeros(vl.shape[1], dtype=bool)
        for a, b in cols:
            keep[max(0, a):max(0, b)] = True
        vl[:, ~keep] = 0
    # allarga le linee trovate di qualche pixel nel verso dello spessore: i bordi
    # delle linee ondulate o sfrangiate non devono restare come "inchiostro"
    k = max(2, int(round(grow)))
    hl = cv2.dilate(hl, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2 * k + 1)))
    vl = cv2.dilate(vl, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 3)))
    h_lines, v_lines = hl > 0, vl > 0
    return ink & ~(h_lines | v_lines), h_lines, v_lines


def _line_kernel(extent: int, bh: int) -> int:
    """Lunghezza minima di un tratto rettilineo per essere considerato linea della
    griglia: proporzionale alla zona analizzata, entro 1,4-2 altezze di riga."""
    return int(max(1.4 * bh, min(0.45 * extent, 2.0 * bh)))


def _drop_line_remnants(ink: np.ndarray, h_lines: np.ndarray, v_lines: np.ndarray, cell: Box,
                        bh: int) -> np.ndarray:
    """Elimina i frammenti di linea rimasti lungo i bordi della cella: bordi
    sfrangiati di linee ondulate (sottilissimi) e spezzoni di linee interrotte
    (allineati con il resto della linea gia' rimossa)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    if n <= 1:
        return ink
    fx0, fy0, fx1, fy1 = cell
    H, W = ink.shape
    sliver = max(3.0, 0.05 * bh)      # spessore massimo di un bordo sfrangiato
    piece = max(6.0, 0.16 * bh)       # spessore massimo di uno spezzone di linea
    edge = 0.18 * bh
    reach = int(3 * bh)
    lines = h_lines | v_lines
    drop = np.zeros(n, dtype=bool)
    for k in range(1, n):
        x, y, w, h, _ = (int(v) for v in stats[k])
        yc, xc = y + h / 2.0, x + w / 2.0
        near_h = min(abs(yc - fy0), abs(yc - fy1)) <= edge
        near_v = min(abs(xc - fx0), abs(xc - fx1)) <= edge
        # frammento minuscolo attaccato a una linea (es. negli incroci)
        if (max(w, h) <= piece and (near_h or near_v)
                and lines[max(0, y - 2):min(H, y + h + 2), max(0, x - 2):min(W, x + w + 2)].any()):
            drop[k] = True
            continue
        if h <= piece and w >= h and near_h:
            band = h_lines[max(0, y - 1):min(H, y + h + 1)]
            left = band[:, max(0, x - reach):max(0, x - 1)].any()
            right = band[:, min(W, x + w + 1):min(W, x + w + reach)].any()
            if (h <= sliver and w >= 3 * h) or left or right:
                drop[k] = True
        elif w <= piece and h >= max(w, 0.6 * bh) and near_v:
            band = v_lines[:, max(0, x - 1):min(W, x + w + 1)]
            up = band[max(0, y - reach):max(0, y - 1)].any()
            down = band[min(H, y + h + 1):min(H, y + h + reach)].any()
            if (w <= sliver and h >= 3 * w) or up or down:
                drop[k] = True
    if not drop.any():
        return ink
    return ink & ~drop[labels]


# --------------------------------------------------------------------------
# Analisi di una cella
# --------------------------------------------------------------------------

def analyze_cell(img: np.ndarray, box: Box, margin: float = 0.04) -> CellInk:
    """Misura l'inchiostro che appartiene alla cella ``box``.

    I tratti che debordano dalle celle vicine (parte maggiore fuori dal
    riquadro) sono esclusi; ``margin`` (frazione del lato) esclude i bordi.
    """
    x0, y0, x1, y1 = _norm_box(img, box)
    bw, bh = x1 - x0, y1 - y0
    empty = CellInk(cell_w=max(0, bw), cell_h=max(0, bh))
    if bw < 4 or bh < 4:
        return empty
    H, W = img.shape[:2]
    # Zona estesa: righe sopra e sotto (per attribuire i tratti che debordano)
    # e un po' di contesto a sinistra e a destra.
    ex0 = max(0, int(x0 - 0.3 * bh))
    ex1 = min(W, int(x1 + 0.3 * bh))
    ey0 = max(0, int(y0 - 0.8 * bh))
    ey1 = min(H, int(y1 + 0.8 * bh))
    gray = to_gray(img[ey0:ey1, ex0:ex1])
    ink = _ink_mask(gray)
    ew, eh = ex1 - ex0, ey1 - ey0
    fx0, fy0, fx1, fy1 = x0 - ex0, y0 - ey0, x1 - ex0, y1 - ey0
    # le linee della griglia possono stare solo attorno ai bordi della cella
    tol = int(round(0.3 * bh)) + 1
    ink, h_lines, v_lines = _remove_grid_lines(
        ink, _line_kernel(ew, bh), _line_kernel(eh, bh),
        rows=[(fy0 - tol, fy0 + tol), (fy1 - tol, fy1 + tol)],
        cols=[(fx0 - tol, fx0 + tol), (fx1 - tol, fx1 + tol)],
    )
    lines = h_lines | v_lines
    ink = _drop_line_remnants(ink, h_lines, v_lines, (fx0, fy0, fx1, fy1), bh)

    # Connettivita': i pixel di linea vicini ai tratti vengono rimessi solo per
    # ricucire i tratti che attraversano una linea (es. l'asta di una firma che
    # sale nella riga sopra), non per contarli come inchiostro.
    lt = max(3, int(round(0.15 * bh)))
    ink_u8 = ink.astype(np.uint8)
    near = cv2.dilate(ink_u8, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * lt + 1, 2 * lt + 1))) > 0
    conn = (ink | (lines & near)).astype(np.uint8)
    n, labels = cv2.connectedComponents(conn, connectivity=8)
    if n <= 1:
        return empty

    lab = np.where(ink, labels, 0)
    total = np.bincount(lab.ravel(), minlength=n)
    # riquadro della cella (al netto del margine) in coordinate della zona estesa
    mx = int(round(margin * bw))
    my = int(round(margin * bh))
    cx0, cy0 = fx0 + mx, fy0 + my
    cx1, cy1 = fx1 - mx, fy1 - my
    if cx1 <= cx0 or cy1 <= cy0:
        return empty
    inner = lab[cy0:cy1, cx0:cx1]
    inside = np.bincount(inner.ravel(), minlength=n)
    # pixel nella cella vera (senza margine): per l'attribuzione
    cell_lab = lab[fy0:fy1, fx0:fx1]
    in_cell = np.bincount(cell_lab.ravel(), minlength=n)

    min_area = max(6.0, (0.07 * bh) ** 2)
    owned = np.zeros(n, dtype=bool)
    for k in range(1, n):
        if total[k] < min_area or inside[k] == 0:
            continue
        share = in_cell[k] / float(total[k])
        if share >= 0.5:
            owned[k] = True
        elif share >= 0.15 and in_cell[k] >= SHARED_STROKE_RATIO * bw * bh:
            # Firme (o scritte) di righe consecutive che si toccano formano un
            # solo tratto, ma ognuna occupa buona parte dell'altezza della
            # propria cella; le code che debordano dalle righe vicine restano
            # invece addossate al bordo superiore o inferiore.
            ys_k = np.nonzero((cell_lab == k).any(axis=1))[0]
            if ys_k.size and (ys_k[-1] - ys_k[0] + 1) >= SHARED_STROKE_EXTENT * bh:
                owned[k] = True
    area = float((cx1 - cx0) * (cy1 - cy0))
    foreign = float(inside[1:][~owned[1:]].sum()) / area if area > 0 else 0.0
    empty.foreign = foreign
    if not owned.any():
        return empty
    # le misure di forma usano solo i tratti significativi dentro la cella
    # (un pixel isolato non deve allargare il riquadro dell'inchiostro)
    major = owned & (inside >= min_area)
    if not major.any():
        major = owned
    sel = major[inner]
    pixels = int(owned[inner].sum())
    if pixels == 0 or not sel.any():
        return empty
    ys, xs = np.nonzero(sel)
    comp_sizes = inside[owned]
    bx0, bx1 = int(xs.min()), int(xs.max()) + 1
    by0, by1 = int(ys.min()), int(ys.max()) + 1
    return CellInk(
        ratio=pixels / area,
        pixels=pixels,
        width_frac=(bx1 - bx0) / float(bw),
        height_frac=(by1 - by0) / float(bh),
        n_components=int((comp_sizes >= min_area).sum()),
        largest_frac=float(comp_sizes.max()) / pixels,
        cx=(float(xs.mean()) + mx) / bw,
        cy=(float(ys.mean()) + my) / bh,
        bbox=(x0 + mx + bx0, y0 + my + by0, x0 + mx + bx1, y0 + my + by1),
        foreign=foreign,
        cell_w=bw,
        cell_h=bh,
    )


def ink_ratio(img: np.ndarray, box: Box, margin: float = 0.12, remove_lines: bool = True) -> float:
    """Frazione di pixel d'inchiostro nel riquadro, escluso un margine interno
    (``margin`` = frazione di larghezza/altezza su ogni lato) ed escluse le
    linee della griglia. Misura grezza: non attribuisce i tratti alle celle."""
    x0, y0, x1, y1 = _norm_box(img, box)
    bw, bh = x1 - x0, y1 - y0
    if bw < 2 or bh < 2:
        return 0.0
    mx = int(round(max(0.0, margin) * bw))
    my = int(round(max(0.0, margin) * bh))
    if remove_lines:
        H, W = img.shape[:2]
        ex0, ex1 = max(0, x0 - bh // 2), min(W, x1 + bh // 2)
        ey0, ey1 = max(0, y0 - bh // 2), min(H, y1 + bh // 2)
        ink = _ink_mask(to_gray(img[ey0:ey1, ex0:ex1]))
        ink, _, _ = _remove_grid_lines(ink, _line_kernel(ex1 - ex0, bh), _line_kernel(ey1 - ey0, bh))
        sub = ink[y0 - ey0 + my:y1 - ey0 - my, x0 - ex0 + mx:x1 - ex0 - mx]
    else:
        sub = _ink_mask(to_gray(img[y0:y1, x0:x1]))[my:bh - my, mx:bw - mx]
    if sub.size == 0:
        return 0.0
    return float(sub.mean())


# --------------------------------------------------------------------------
# Classificazione
# --------------------------------------------------------------------------

def _is_speck(info: CellInk) -> bool:
    """Puntino isolato (polvere, punto di penna): piccolo e non allungato."""
    w_px, h_px = info.ink_w, info.ink_h
    return max(w_px, h_px) < 0.22 * info.cell_h and w_px < DASH_MIN_ASPECT * max(h_px, 1.0)


def is_blank(img: np.ndarray, box: Box) -> bool:
    """True se nella cella non c'e' inchiostro proprio (puntini, residui di
    linee e code di firme delle righe vicine non contano)."""
    info = analyze_cell(img, box)
    return info.ratio < BLANK_RATIO or _is_speck(info)


def has_cross(img: np.ndarray, box: Box) -> bool:
    """True se nella cella (colonne di assenza) c'e' una crocetta."""
    info = analyze_cell(img, box, margin=0.06)
    if info.ratio < BLANK_RATIO or _is_speck(info):
        return False
    if info.width_frac < CROSS_SPAN or info.height_frac < CROSS_SPAN:
        return False
    # un trattino orizzontale o una barretta verticale non sono crocette
    aspect = info.ink_w / max(1e-6, info.ink_h)
    return 0.25 <= aspect <= 4.5


def has_signature(img: np.ndarray, box: Box) -> bool:
    """True se nella cella "Firma operatore" c'e' una firma."""
    info = analyze_cell(img, box)
    return info.ratio >= SIGNATURE_RATIO and info.width_frac >= SIGNATURE_WIDTH and info.height_frac >= 0.2


def is_dash(img: np.ndarray, box: Box) -> bool:
    """True se la cella contiene soltanto un trattino ("-")."""
    info = analyze_cell(img, box)
    if info.ratio < BLANK_RATIO or info.pixels == 0:
        return False
    if info.height_frac > DASH_MAX_HEIGHT or info.ink_w < DASH_MIN_ASPECT * info.ink_h:
        return False
    if info.width_frac < 0.05 or info.width_frac > 0.7:
        return False
    return info.n_components <= 2 and info.largest_frac >= 0.75


# --------------------------------------------------------------------------
# Inchiostro "proprio" di una cella: barrature, linee attraverso, debordamenti
# --------------------------------------------------------------------------

STRIKE_MAX_WIDTH = 0.16      # spessore massimo di un tratto di barratura (frazione della riga)
STRIKE_MIN_PIECE = 0.6       # lunghezza minima di un pezzo rettilineo (righe)
STRIKE_MIN_ROWS = 1.2        # estensione verticale minima di una barratura diagonale (righe)
LINE_MIN_COLS = 1.3          # estensione minima di una linea orizzontale (larghezze di colonna dati)


@dataclass
class Stroke:
    """Tratto a penna lungo e quasi rettilineo che attraversa piu' celle.

    ``kind``: ``"diagonale"`` (barratura di righe non usate, attraversa piu'
    righe) oppure ``"orizzontale"`` (linea tirata attraverso le celle di una
    riga: prestazione non svolta). ``points``: polilinea (x, y) in pixel
    dell'immagine, ordinata da sinistra a destra. ``days``: giorni le cui
    righe sono attraversate; ``columns``: colonne attraversate."""

    kind: str
    points: np.ndarray
    angle: float                      # gradi, 0 = orizzontale
    days: tuple[int, ...]
    columns: tuple[str, ...]


@dataclass
class CellInkResult:
    """Inchiostro di una cella dopo la pulizia (``cell_own_ink``).

    * ``crop``: ritaglio in scala di grigi (``uint8``, inchiostro scuro su
      bianco) del riquadro ``box``, con linee della griglia, barrature e
      tratti delle celle vicine sbiancati: pronto per l'OCR;
    * ``mask``: inchiostro proprio (bool, stesse dimensioni di ``crop``);
    * ``ink_ratio``: inchiostro proprio / area interna della cella;
    * ``struck``: la cella e' attraversata da una barratura diagonale che
      copre piu' righe (riga non usata: il contenuto non e' un valore);
    * ``line_across``: una linea orizzontale a penna attraversa la cella
      (prestazione non svolta, equivale al trattino "-");
    * ``overflow_removed``: frazione dell'inchiostro presente nella cella
      tolta perche' appartiene a celle vicine (firme lunghe, aste delle righe
      sopra/sotto, continuazioni in corsivo): se > 0 conviene segnalare;
    * ``strike_removed``: frazione tolta perche' parte di barrature/linee;
    * ``slash``: l'inchiostro proprio e' un solo tratto rettilineo obliquo
      ("/" o "\\", inclinato fra 15 e 70 gradi): nelle colonne degli orari
      e' il segno di giornata non lavorata, non un valore (attenzione: nella
      colonna delle ore un "1" inclinato gli somiglia; ``stroke_angle`` aiuta);
    * ``stroke_angle``: inclinazione in gradi (0 = orizzontale, 90 = verticale)
      del tratto principale, ``None`` se la cella e' vuota;
    * ``row_struck``: la riga del giorno e' attraversata da una barratura
      diagonale (anche se il tratto non passa per questa cella);
    * ``crease``: la cella e' attraversata da una piega del foglio (``Crease``),
      tolta dall'inchiostro proprio come le barrature;
    * ``box``: riquadro della cella (x0, y0, x1, y1) nell'immagine.
    """

    crop: np.ndarray
    mask: np.ndarray
    ink_ratio: float
    struck: bool
    line_across: bool
    overflow_removed: float
    strike_removed: float
    box: Box
    slash: bool = False
    stroke_angle: float | None = None
    row_struck: bool = False
    crease: bool = False

    @property
    def empty(self) -> bool:
        return self.ink_ratio < BLANK_RATIO


class PageInk:
    """Analisi dell'inchiostro della tabella di una pagina, calcolata una volta
    e riusata per tutte le celle (vedi ``cell_own_ink``).

    Costruzione: ``PageInk(img, grid)`` (``img`` BGR o grigia, ``grid`` la
    ``TableGrid`` della stessa immagine). Attributi: ``strokes`` (elenco di
    ``Stroke``), ``strike_mask`` (bool, coordinate della zona ``origin``).
    """

    def __init__(self, img: np.ndarray, grid,  # noqa: ANN001 - TableGrid (import circolare evitato)
                 creases: tuple[list[Crease], np.ndarray | None] | None = None) -> None:
        gray = _darkness_gray(img)
        H, W = gray.shape[:2]
        if (grid.width, grid.height) != (W, H):        # griglia di un'immagine ridimensionata
            grid = grid.scaled(W / float(grid.width), H / float(grid.height))
            grid.width, grid.height = W, H
        self.grid = grid
        rh = float(grid.row_height)
        self.rh = rh
        x0 = max(0, int(grid.col_x[0] - 0.5 * rh))
        x1 = min(W, int(grid.col_x[-1] + 0.5 * rh))
        y0 = max(0, int(grid.row_y[0] - 1.5 * rh))
        y1 = min(H, int(grid.row_y[-1] + 1.5 * rh))
        self.origin = (x0, y0)
        region = gray[y0:y1, x0:x1]
        self.region = region
        self.paper = _paper_level(region)
        ink, faint = _local_ink(region, rh, faint=True)
        tol = int(round(0.22 * rh)) + 1
        rows = [(int(y - y0) - tol, int(y - y0) + tol) for y in list(grid.row_y) + [grid.header_top]
                + ([grid.total_row[1]] if grid.total_row else [])]
        cols = [(int(x - x0) - tol, int(x - x0) + tol) for x in grid.col_x]
        lk = _line_kernel(min(region.shape[1], int(4 * rh)), int(rh))
        core, h_lines, v_lines = _remove_grid_lines(ink, lk, lk, rows=rows, cols=cols)
        self.ink = ink
        self.h_lines = h_lines
        self.v_lines = v_lines
        self.lines = h_lines | v_lines
        faint_core, _, _ = _remove_grid_lines(faint, lk, lk, rows=rows, cols=cols)
        # pieghe del foglio (linee lunghe quasi verticali): non sono scrittura ne' barrature
        if creases is None:
            creases = find_creases(img, grid)
        self.creases = list(creases[0])
        crease_mask = np.zeros_like(faint_core)
        if creases[1] is not None and creases[1].shape[:2] == (H, W):
            crease_mask = creases[1][y0:y1, x0:x1].copy()
        faint_core &= ~crease_mask
        self.strokes, strike = _find_strokes(faint_core, grid, rh, (x0, y0))
        self.strike_mask = strike | crease_mask
        core = core & ~self.strike_mask
        self.core = core
        # pixel delle linee orizzontali che ricuciono un tratto che le attraversa:
        # inchiostro sopra e sotto entro 0,16 righe (la fascia della linea tolta
        # e' spessa qualche pixel: il raggio deve superarla)
        r = max(3, int(round(0.16 * rh)))
        c8 = core.astype(np.float32)
        up = cv2.filter2D(c8, -1, np.vstack([np.ones((r, 1)), np.zeros((r + 1, 1))]).astype(np.float32)) > 0
        down = cv2.filter2D(c8, -1, np.vstack([np.zeros((r + 1, 1)), np.ones((r, 1))]).astype(np.float32)) > 0
        self.bridge = h_lines & ~v_lines & up & down
        # tratti che attraversano una linea verticale (inchiostro a sinistra e a
        # destra): non ricuciti nelle componenti (la scrittura di celle vicine
        # appoggiata alla stessa linea si fonderebbe), usati solo per
        # riconoscere le continuazioni in corsivo dalla cella di sinistra
        rv = max(r, int(round(0.25 * rh)))
        left = cv2.filter2D(c8, -1, np.hstack([np.ones((1, rv)), np.zeros((1, rv + 1))]).astype(np.float32)) > 0
        right = cv2.filter2D(c8, -1, np.hstack([np.zeros((1, rv + 1)), np.ones((1, rv))]).astype(np.float32)) > 0
        self.vcross = v_lines & left & right
        self.reach = rv

    @property
    def struck_days(self) -> set[int]:
        """Giorni le cui righe sono attraversate da una barratura diagonale
        (righe non usate cancellate con un tratto lungo)."""
        return {d for st in self.strokes if st.kind == "diagonale" for d in st.days}

    @property
    def line_days(self) -> set[int]:
        """Giorni attraversati da una linea orizzontale a penna."""
        return {d for st in self.strokes if st.kind == "orizzontale" for d in st.days}

    # ------------------------------------------------------------------ celle
    def _local_box(self, box: Box) -> Box:
        ox, oy = self.origin
        h, w = self.region.shape[:2]
        x0, y0, x1, y1 = box
        return (max(0, x0 - ox), max(0, y0 - oy), min(w, x1 - ox), min(h, y1 - oy))

    def cell_strike(self, giorno: int, colonna: str | int) -> tuple[bool, bool]:
        """(barrata da una diagonale, attraversata da una linea orizzontale)."""
        g = self.grid
        c = g.col_index(colonna)
        x0, y0, x1, y1 = g.cell(giorno, c)
        cw, ch = x1 - x0, y1 - y0
        struck = across = False
        for st in self.strokes:
            pts = st.points
            if st.kind == "diagonale":
                ix0, ix1 = x0 + 0.1 * cw, x1 - 0.1 * cw
                iy0, iy1 = y0 + 0.1 * ch, y1 - 0.1 * ch
                xs = np.linspace(ix0, ix1, 16)
                ys = np.interp(xs, pts[:, 0], pts[:, 1], left=np.nan, right=np.nan)
                if np.any((ys >= iy0) & (ys <= iy1)):
                    struck = True
            else:
                xs = np.linspace(x0, x1, 24)
                ys = np.interp(xs, pts[:, 0], pts[:, 1], left=np.nan, right=np.nan)
                inside = (ys >= y0 + 0.15 * ch) & (ys <= y1 - 0.15 * ch)
                if inside.mean() >= 0.5:
                    across = True
        return struck, across

    def own_ink(self, giorno: int, colonna: str | int, pad: int = 0, remove_lines: bool = True) -> CellInkResult:
        g = self.grid
        c = g.col_index(colonna)
        box = g.cell(giorno, c)
        bx0, by0, bx1, by1 = self._local_box(box)
        h, w = self.region.shape[:2]
        px0, py0 = max(0, bx0 - pad), max(0, by0 - pad)
        px1, py1 = min(w, bx1 + pad), min(h, by1 + pad)
        cw, ch = bx1 - bx0, by1 - by0
        struck, across = self.cell_strike(giorno, c)
        crease = any(cr.crosses(box) for cr in self.creases)
        empty_crop = np.full((max(0, py1 - py0), max(0, px1 - px0)), 255, np.uint8)
        if cw < 3 or ch < 3:
            return CellInkResult(empty_crop, empty_crop > 0, 0.0, struck, across, 0.0, 0.0, box,
                                 row_struck=int(giorno) in self.struck_days, crease=crease)
        # finestra: le celle vicine per intero (a sinistra, a destra, sopra, sotto)
        ox, oy = self.origin
        wl = (g.col_x[c] - g.col_x[c - 1]) if c > 0 else int(0.5 * self.rh)
        wr = (g.col_x[c + 2] - g.col_x[c + 1]) if c + 2 < len(g.col_x) else int(0.5 * self.rh)
        wx0, wx1 = max(0, bx0 - wl), min(w, bx1 + wr)
        wy0, wy1 = max(0, int(by0 - 1.0 * self.rh)), min(h, int(by1 + 1.0 * self.rh))
        core = self.core[wy0:wy1, wx0:wx1]
        conn = (core | self.bridge[wy0:wy1, wx0:wx1]).astype(np.uint8)
        n, lab_all = cv2.connectedComponents(conn, connectivity=8)
        lab = np.where(core, lab_all, 0)
        total = np.bincount(lab.ravel(), minlength=n)
        fx0, fy0, fx1, fy1 = bx0 - wx0, by0 - wy0, bx1 - wx0, by1 - wy0
        cell_lab = lab[fy0:fy1, fx0:fx1]
        in_cell = np.bincount(cell_lab.ravel(), minlength=n)
        mx, my = int(round(0.04 * cw)), int(round(0.04 * ch))
        inner = lab[fy0 + my:fy1 - my, fx0 + mx:fx1 - mx]
        inside = np.bincount(inner.ravel(), minlength=n)
        min_area = max(4.0, (0.035 * self.rh) ** 2)
        own = np.zeros(n, dtype=bool)
        for k in np.nonzero(inside)[0]:
            if k == 0 or total[k] < min_area:
                continue
            share = in_cell[k] / float(total[k])
            if share >= 0.5:
                own[k] = True
            elif share >= 0.15 and in_cell[k] >= SHARED_STROKE_RATIO * cw * ch:
                ys_k = np.nonzero((cell_lab == k).any(axis=1))[0]
                if ys_k.size and (ys_k[-1] - ys_k[0] + 1) >= SHARED_STROKE_EXTENT * ch:
                    own[k] = True
        # code di cifre/virgole della riga sopra (o aste della riga sotto) che
        # attraversano il bordo: nella cella ne resta solo un pezzetto addossato
        # al bordo, anche se e' la parte piu' lunga del tratto
        for k in np.nonzero(own)[0]:
            if total[k] <= in_cell[k]:
                continue
            part = cell_lab == k
            ys_k = np.nonzero(part.any(axis=1))[0]
            xs_k = np.nonzero(part.any(axis=0))[0]
            if ys_k.size == 0 or (xs_k[-1] - xs_k[0] + 1) > 0.25 * ch:
                continue                        # non e' una coda sottile
            ext = (ys_k[-1] - ys_k[0] + 1) / float(ch)
            comp_rows = np.nonzero((lab == k).any(axis=1))[0]
            from_above = comp_rows[0] < fy0 and ys_k[-1] < 0.45 * ch
            from_below = comp_rows[-1] >= fy1 and ys_k[0] > 0.55 * ch
            if ext <= 0.4 and (from_above or from_below):
                own[k] = False
        # continuazioni in corsivo dalla colonna delle firme (firma lunga che
        # finisce nelle "Note"): un tratto proprio che tocca il bordo sinistro
        # proprio dove un tratto della cella di sinistra lo attraversa, se la
        # parte di sinistra e' nettamente piu' grande (la firma), e i pezzi
        # attaccati in corsivo (entro 0,3 righe, alla stessa altezza)
        # solo a destra della colonna delle firme: le cifre di celle vicine che
        # toccano la stessa linea non sono "continuazioni"
        if c > 1 and _COLUMN_NAMES[c - 1] == "firma" and own.any():
            lx = g.col_x[c] - ox - wx0
            r = self.reach
            band = (max(0, lx - 2 * r), min(lab.shape[1], lx + 2 * r + 1))
            cross_rows = np.nonzero(self.vcross[wy0:wy1, wx0:wx1][:, band[0]:band[1]].any(axis=1))[0]
            cross_rows = cross_rows[(cross_rows >= fy0) & (cross_rows < fy1)]
            if cross_rows.size:
                # il pezzo che prosegue puo' staccarsi dalla linea (penna sollevata):
                # si guarda fino a 0,4 righe a destra, all'altezza dell'attraversamento
                ys_c = np.arange(max(0, cross_rows.min() - r), min(lab.shape[0], cross_rows.max() + r + 1))
                rb = lab[ys_c, lx + 3:min(lab.shape[1], lx + r + int(0.4 * self.rh) + 1)]
                lb = lab[ys_c, max(0, lx - 3 * r):max(0, lx - 3)]
                right_k = [int(k) for k in np.unique(rb) if k and own[k]]
                # tratti della cella di sinistra (in maggioranza dentro di essa)
                left_cell = lab[fy0:fy1, max(0, fx0 - (g.col_x[c] - g.col_x[c - 1])):fx0]
                in_left = np.bincount(left_cell.ravel(), minlength=n)
                left_k = [int(k) for k in np.unique(lb)
                          if k and not own[k] and in_left[k] >= 0.5 * total[k]]
                # massa della scrittura della cella di sinistra (la firma intera,
                # anche se fatta di piu' tratti)
                size_left = float(sum(total[k] for k in np.nonzero(in_left)[0]
                                      if k and in_left[k] >= 0.5 * total[k])) if left_k else 0.0
                chain = set(right_k)
                if chain:
                    cmask = np.isin(cell_lab, list(chain))
                    reach = max(2, int(round(0.3 * self.rh)))
                    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * reach + 1, 2 * (reach // 2) + 1))
                    for _ in range(6):
                        grown = cv2.dilate(cmask.astype(np.uint8), kern) > 0
                        ys_c = np.nonzero(cmask.any(axis=1))[0]
                        new = []
                        for k in np.unique(cell_lab[grown]):
                            k = int(k)
                            if not k or k in chain or not own[k]:
                                continue
                            ys_k = np.nonzero((cell_lab == k).any(axis=1))[0]
                            if ys_k.size and ys_c.size and ys_k[0] <= ys_c[-1] and ys_c[0] <= ys_k[-1]:
                                new.append(k)
                        if not new:
                            break
                        chain.update(new)
                        cmask |= np.isin(cell_lab, new)
                    size_right = float(sum(in_cell[k] for k in chain))
                    if size_left >= 1.5 * size_right:
                        for k in chain:
                            own[k] = False
        own_win = own[lab]
        own_mask = own_win[py0 - wy0:py1 - wy0, px0 - wx0:px1 - wx0]
        cell_own = own_win[fy0:fy1, fx0:fx1]
        ink_cell = self.ink[by0:by1, bx0:bx1] & ~self.lines[by0:by1, bx0:bx1]
        ink_in_cell = int(ink_cell.sum())
        strike_in_cell = int((self.strike_mask[by0:by1, bx0:bx1] & ink_cell).sum())
        own_px = int(cell_own.sum())
        removed_foreign = max(0, ink_in_cell - own_px - strike_in_cell)
        denom = max(1, ink_in_cell)
        area = float(max(1, (cw - 2 * mx) * (ch - 2 * my)))
        own_inner = int(own_win[fy0 + my:fy1 - my, fx0 + mx:fx1 - mx].sum())
        # ritaglio pulito: tutto cio' che non e' inchiostro proprio diventa carta
        crop = self.region[py0:py1, px0:px1].copy()
        keep = own_mask
        if not remove_lines:
            keep = keep | self.lines[py0:py1, px0:px1]
        keep_soft = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        crop[~keep_soft] = int(round(self.paper))
        slash, angle = _slash_shape(cell_own, ch)
        return CellInkResult(
            crop=crop,
            mask=own_mask,
            ink_ratio=own_inner / area,
            struck=struck,
            line_across=across,
            overflow_removed=removed_foreign / denom if ink_in_cell else 0.0,
            strike_removed=strike_in_cell / denom if ink_in_cell else 0.0,
            box=box,
            slash=slash,
            stroke_angle=angle,
            row_struck=int(giorno) in self.struck_days,
            crease=crease,
        )


def _slash_shape(mask: np.ndarray, ch: int) -> tuple[bool, float | None]:
    """(un solo tratto obliquo rettilineo?, inclinazione del tratto principale)."""
    if not mask.any():
        return False, None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return False, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    ys, xs = np.nonzero(labels == k)
    if ys.size < 5:
        return False, None
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts -= pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(pts.T))
    d = evecs[:, 1]
    angle = float(abs(np.degrees(np.arctan2(d[1], d[0]))))
    angle = 180.0 - angle if angle > 90 else angle
    proj = pts @ d
    length = float(proj.max() - proj.min())
    width = float(ys.size) / max(length, 1.0)
    across = np.abs(pts @ evecs[:, 0])
    dominant = areas[k - 1] >= 0.85 * areas.sum()
    straight = float(np.percentile(across, 95)) <= max(3.0, 0.08 * ch)
    slash = bool(dominant and straight and length >= 0.4 * ch and width <= 0.14 * ch and 15 <= angle <= 70)
    return slash, round(angle, 1)


def _darkness_gray(img: np.ndarray) -> np.ndarray:
    """Scala di grigi per l'inchiostro: il canale piu' scuro di ogni pixel, cosi'
    la biro blu chiara (poco scura in luminanza) resta ben distinta dalla carta."""
    if img.ndim == 3 and img.shape[2] >= 3:
        return np.ascontiguousarray(img[:, :, :3].min(axis=2))
    return to_gray(img)


def _paper_level(gray: np.ndarray) -> float:
    step = max(1, int(np.sqrt(gray.size / 200_000.0)))
    return float(np.percentile(gray[::step, ::step], 90))


def _local_ink(gray: np.ndarray, rh: float, faint: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Inchiostro rispetto alla carta circostante: regge ombre e fondini grigi. Con ``faint`` restituisce anche una seconda maschera
    piu' sensibile (soglia dimezzata) per i tratti chiari (matita, penna scarica)."""
    # finestra di ~0,6 righe: piu' larga di un tratto di penna, piu' stretta dei
    # fondini grigi del modulo (colonna "Giorno"), che restano "carta"
    k = max(9, int(0.6 * rh) | 1)
    bg = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))   # carta locale (massimo)
    bg = cv2.blur(bg, (k, k)).astype(np.float32)
    diff = bg - gray.astype(np.float32)
    # contrasto tipico della scrittura nella pagina: soglia relativa, minimo 18 livelli
    strong = np.percentile(diff, 99.5)
    thr = max(18.0, min(45.0, 0.3 * float(strong)))
    if faint:
        return diff > thr, diff > max(14.0, 0.5 * thr)
    return diff > thr


def _pieces(core: np.ndarray, rh: float) -> list[dict]:
    """Pezzi di tratto sottili e rettilinei (componenti connesse allungate)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    out = []
    min_len = STRIKE_MIN_PIECE * rh
    for k in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[k])
        if max(w, h) < min_len or area < min_len:
            continue
        ys, xs = np.nonzero(labels[y:y + h, x:x + w] == k)
        pts = np.column_stack([xs + x, ys + y]).astype(np.float64)
        mean = pts.mean(axis=0)
        cov = np.cov((pts - mean).T)
        evals, evecs = np.linalg.eigh(cov)
        d = evecs[:, 1]
        proj = (pts - mean) @ d
        length = float(proj.max() - proj.min())
        across = (pts - mean) @ evecs[:, 0]
        width = float(area) / max(length, 1.0)
        if length < min_len or width > max(3.0, STRIKE_MAX_WIDTH * rh):
            continue
        # quasi rettilineo: scarto trasversale entro 0,12 righe (archi ampi ammessi)
        if float(np.percentile(np.abs(across), 95)) > max(4.0, 0.2 * rh):
            continue
        if d[0] < 0:
            d = -d
        p0 = mean + d * proj.min()
        p1 = mean + d * proj.max()
        out.append({"k": k, "p0": p0, "p1": p1, "dir": d, "len": length,
                    "angle": float(np.degrees(np.arctan2(d[1], d[0])))})
    return out


def _hough_pieces(closed: np.ndarray, rh: float) -> list[dict]:
    """Segmenti rettilinei lunghi anche se tratteggiati (tratti molto chiari):
    trasformata di Hough probabilistica, poi verifica che l'inchiostro stia
    sulla retta (copertura) e non attorno (non e' una zona di scrittura)."""
    h, w = closed.shape
    m8 = closed.astype(np.uint8)
    segs = cv2.HoughLinesP(m8, 1, np.pi / 180, threshold=max(10, int(0.35 * rh)),
                           minLineLength=int(1.2 * rh), maxLineGap=int(0.45 * rh))
    out: list[dict] = []
    if segs is None:
        return out
    near = cv2.dilate(m8, np.ones((3, 3), np.uint8)) > 0
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4)[:400]:
        p0 = np.array([x1, y1], dtype=np.float64)
        p1 = np.array([x2, y2], dtype=np.float64)
        length = float(np.linalg.norm(p1 - p0))
        if length < 1.2 * rh:
            continue
        d = (p1 - p0) / length
        nrm = np.array([-d[1], d[0]])
        t = np.arange(0.0, length, 2.0)
        pts = p0[None, :] + t[:, None] * d[None, :]
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        on = float(near[ys, xs].mean())
        if on < 0.55:
            continue
        # attorno alla retta (fra 0,12 e 0,3 righe per parte) non c'e' scrittura
        offs = [f * rh for f in (0.12, 0.18, 0.24, 0.3)]
        off_pts = np.concatenate([pts + o * nrm for o in offs] + [pts - o * nrm for o in offs])
        ox = np.clip(np.round(off_pts[:, 0]).astype(int), 0, w - 1)
        oy = np.clip(np.round(off_pts[:, 1]).astype(int), 0, h - 1)
        if float(closed[oy, ox].mean()) > 0.05:
            continue
        if d[0] < 0:
            d, p0, p1 = -d, p1, p0
        out.append({"k": None, "p0": p0, "p1": p1, "dir": d, "len": length,
                    "angle": float(np.degrees(np.arctan2(d[1], d[0])))})
    return out


def _find_strokes(core: np.ndarray, grid, rh: float, origin: tuple[int, int]) -> tuple[list[Stroke], np.ndarray]:  # noqa: ANN001
    """Barrature diagonali e linee orizzontali a penna nella zona della tabella.

    I pezzi rettilinei (interrotti dalle linee della griglia tolte) vengono
    concatenati se allineati: stessa direzione (entro 15 gradi) e buco fra le
    estremita' piccolo (entro 0,8 righe) nella stessa direzione."""
    ox, oy = origin
    # i tratti chiari (matita, biro scarica) escono puntinati: una chiusura
    # piccola li ricompone (la scrittura vicina si ingrossa e viene scartata)
    r = max(2, int(round(0.05 * rh)))
    closed = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    pieces = _pieces(closed, rh) + _hough_pieces(closed, rh)
    mask = np.zeros(core.shape, dtype=bool)
    if not pieces:
        return [], mask
    m = len(pieces)
    parent = list(range(m))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(m):
        for j in range(i + 1, m):
            a, b = pieces[i], pieces[j]
            dang = abs(((a["angle"] - b["angle"]) + 90) % 180 - 90)
            if dang > 15:
                continue
            # estremi piu' vicini
            best = None
            for pa in (a["p0"], a["p1"]):
                for pb in (b["p0"], b["p1"]):
                    dd = float(np.linalg.norm(pa - pb))
                    if best is None or dd < best[0]:
                        best = (dd, pa, pb)
            dist, pa, pb = best
            if dist > 1.2 * rh:
                continue
            if dist > 0.25 * rh:
                v = (pb - pa) / dist
                cosang = min(abs(float(v @ a["dir"])), abs(float(v @ b["dir"])))
                if cosang < np.cos(np.radians(30)):
                    continue
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(m):
        groups.setdefault(find(i), []).append(i)
    col_w = float(np.median(np.diff(grid.col_x[1:6])))
    strokes: list[Stroke] = []
    labels_needed: list[int] = []
    segs_needed: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in groups.values():
        pts = np.array([p for i in idx for p in (pieces[i]["p0"], pieces[i]["p1"])])
        xs, ys = pts[:, 0], pts[:, 1]
        ext_x = float(xs.max() - xs.min())
        ext_y = float(ys.max() - ys.min())
        total_len = sum(pieces[i]["len"] for i in idx)
        angle = float(np.degrees(np.arctan2(ext_y, max(ext_x, 1e-6))))
        order = np.argsort(xs)
        poly = np.column_stack([xs[order] + ox, ys[order] + oy])
        kind = None
        if ext_x >= LINE_MIN_COLS * col_w and ext_y <= 0.6 * rh and total_len >= 0.8 * ext_x:
            kind = "orizzontale"
        elif ext_y >= STRIKE_MIN_ROWS * rh and ext_x >= 0.8 * col_w and 8 <= angle <= 82 \
                and total_len >= 0.75 * float(np.hypot(ext_x, ext_y)):
            kind = "diagonale"
        if kind is None:
            continue
        if kind == "orizzontale":
            # una linea della griglia rimasta (non e' un tratto a penna): sta sul
            # bordo di riga, non all'interno
            yc = float(np.median(poly[:, 1]))
            if min(abs(yc - float(y)) for y in grid.row_y) < 0.27 * rh:
                continue
            if not grid.row_y[0] <= yc <= grid.row_y[-1]:
                continue
            # perfettamente parallela alla griglia e dritta come una linea
            # stampata: e' un'ombra/linea doppia del modulo, non un tratto a mano
            fit = np.polyfit(poly[:, 0], poly[:, 1], 1)
            resid = float(np.std(poly[:, 1] - np.polyval(fit, poly[:, 0])))
            if abs(float(np.degrees(np.arctan(fit[0])))) < 0.3 and resid < 0.012 * rh:
                continue
        days = tuple(d for d in range(1, 32)
                     if np.any((poly[:, 1] >= grid.row_y[d - 1]) & (poly[:, 1] <= grid.row_y[d]))
                     or (poly[:, 1].min() <= grid.row_y[d - 1] and poly[:, 1].max() >= grid.row_y[d]))
        if not days or (kind == "diagonale" and len(days) < 2):
            continue
        cols = tuple(name for ci, name in enumerate(_COLUMN_NAMES)
                     if poly[:, 0].min() <= grid.col_x[ci + 1] and poly[:, 0].max() >= grid.col_x[ci])
        if kind == "diagonale":
            # una barratura attraversa almeno un bordo di colonna (le firme
            # inclinate di righe consecutive restano nella loro colonna)
            gx0, gx1 = poly[:, 0].min(), poly[:, 0].max()
            if not any(gx0 + 0.1 * col_w < x < gx1 - 0.1 * col_w for x in grid.col_x[1:-1]):
                continue
        strokes.append(Stroke(kind=kind, points=poly, angle=angle, days=days, columns=cols))
        labels_needed.extend(pieces[i]["k"] for i in idx if pieces[i]["k"] is not None)
        segs_needed.extend((pieces[i]["p0"], pieces[i]["p1"]) for i in idx)
    strokes = _merge_horizontal(strokes, rh, grid)
    if strokes:
        _, labels = cv2.connectedComponents(closed.astype(np.uint8), connectivity=8)
        mask = np.isin(labels, [k for k in labels_needed]) if labels_needed else np.zeros(core.shape, bool)
        lines = np.zeros(core.shape, np.uint8)
        th = max(3, int(round(0.12 * rh)))
        for a, b in segs_needed:
            cv2.line(lines, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1, th)
        # anche i tratti deboli lungo l'intera polilinea (buchi fra i pezzi)
        for st in strokes:
            cv2.polylines(lines, [(st.points - np.array([ox, oy])).astype(np.int32).reshape(-1, 1, 2)], False, 1, th)
        mask = (mask | (lines > 0)) & core
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return strokes, mask


def _merge_horizontal(strokes: list[Stroke], rh: float, grid) -> list[Stroke]:  # noqa: ANN001
    """Unisce le linee orizzontali della stessa riga allineate (stessa altezza
    entro 0,2 righe, buco fino a 2,5 righe): un tratto debole a meta' strada
    spezza la linea senza cambiarne il significato."""
    hor = sorted([s for s in strokes if s.kind == "orizzontale"], key=lambda s: s.points[:, 0].min())
    other = [s for s in strokes if s.kind != "orizzontale"]
    out: list[Stroke] = []
    for st in hor:
        if out:
            last = out[-1]
            gap = st.points[:, 0].min() - last.points[:, 0].max()
            ya = float(np.interp(last.points[:, 0].max(), last.points[:, 0], last.points[:, 1]))
            yb = float(np.interp(st.points[:, 0].min(), st.points[:, 0], st.points[:, 1]))
            if gap <= 2.5 * rh and abs(ya - yb) <= 0.2 * rh and set(last.days) & set(st.days):
                pts = np.vstack([last.points, st.points])
                pts = pts[np.argsort(pts[:, 0])]
                cols = tuple(n for ci, n in enumerate(_COLUMN_NAMES)
                             if pts[:, 0].min() <= grid.col_x[ci + 1] and pts[:, 0].max() >= grid.col_x[ci])
                ang = float(np.degrees(np.arctan2(np.ptp(pts[:, 1]), max(np.ptp(pts[:, 0]), 1e-6))))
                out[-1] = Stroke("orizzontale", pts, ang, tuple(sorted(set(last.days) | set(st.days))), cols)
                continue
        out.append(st)
    return other + out


_COLUMN_NAMES = ("giorno", "prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita", "ore_dichiarate",
                 "assenza_alunno", "assenza_operatore", "firma", "note")


# --------------------------------------------------------------------------
# Pieghe del foglio: lunghe linee quasi verticali che non sono la griglia
# --------------------------------------------------------------------------

CREASE_MIN_ROWS = 8.0        # estensione verticale minima di una piega (righe): la scrittura di righe
                             # consecutive allineata in colonna forma al massimo 5-6 righe continue
CREASE_MAX_WIDTH = 0.25      # spessore mediano massimo (righe), ombra compresa
CREASE_COVERAGE = 0.75       # frazione dell'estensione in cui la linea e' presente (quasi continua)
CREASE_GAP = 1.0             # interruzione massima (righe) ricucita: una piega poco visibile in qualche
                             # punto resta una; la scrittura dei giorni feriali si ferma al fine settimana
CREASE_MAX_SLOPE = 0.35      # inclinazione massima rispetto alla verticale (dx/dy, circa 20 gradi)
CREASE_MAX_DEV = 0.15        # scarto massimo (righe, 90 percentile) dalla curva adattata
CREASE_PRINTED = 0.35        # piu' vicina di cosi' (righe) a un bordo di colonna: e' la linea stampata


@dataclass
class Crease:
    """Linea lunga, sottile e continua quasi verticale (piega del foglio fotografato,
    ombra della piega, graffio della scansione) che attraversa piu' righe della
    tabella senza essere una linea stampata della griglia. Non e' scrittura: va
    tolta dall'analisi di crocette, firme e testo in tutte le colonne.

    ``path``: ascissa (pixel dell'immagine) per ogni ordinata da ``y0`` a ``y1``
    (esclusa); ``half_width``: semispessore in pixel; ``days``: giorni attraversati."""

    y0: int
    y1: int
    path: np.ndarray
    half_width: float
    days: tuple[int, ...]
    level: float = 0.0          # grigio tipico (mediano) della piega: la scrittura sotto e' piu' scura

    def x_at(self, y: float) -> float:
        i = int(round(min(max(y, self.y0), self.y1 - 1))) - self.y0
        return float(self.path[i])

    def crosses(self, box: Box, inner: float = 0.08) -> bool:
        """True se la linea passa dentro il riquadro (al netto di un margine ``inner``
        della larghezza) per almeno meta' della sua altezza."""
        x0, y0, x1, y1 = box
        ya, yb = max(y0, self.y0), min(y1, self.y1)
        if yb - ya < 0.5 * (y1 - y0):
            return False
        xs = self.path[ya - self.y0:yb - self.y0]
        m = inner * (x1 - x0)
        return bool(((xs >= x0 + m) & (xs <= x1 - m)).mean() >= 0.5)


def _crossing_rows(ink: np.ndarray, path: np.ndarray, half: float, ext: int) -> np.ndarray:
    """Righe (bool, una per ordinata) in cui l'inchiostro attorno alla curva ``path``
    continua per piu' di ``ext`` pixel oltre la fascia larga ``2 * half``: un tratto
    di scrittura attraversa la piega. Allargate di qualche pixel in verticale (spessore
    del tratto)."""
    h, w = ink.shape
    out = np.zeros(h, dtype=bool)
    for i in range(h):
        a = int(np.floor(path[i] - half))
        b = int(np.ceil(path[i] + half))
        row = ink[i]
        la, rb = max(0, a - ext), min(w, b + ext + 1)
        if a - 1 >= 0 and la < a and row[la:a].all():
            out[i] = True
        elif b + 1 <= w and b + 1 < rb and row[b + 1:rb].all():
            out[i] = True
    if out.any():
        out = np.convolve(out.astype(np.float32), np.ones(5, np.float32), mode="same") > 0
    return out


def find_creases(img: np.ndarray, grid) -> tuple[list[Crease], np.ndarray | None]:  # noqa: ANN001
    """Pieghe del foglio nella tabella dei giorni (vedi ``Crease``) e maschera dei
    loro pixel (bool, dimensioni dell'immagine; None se non ce ne sono).

    Un foglio piegato in verticale e fotografato mostra l'ombra della piega: una
    linea scura sottile, leggermente curva, che attraversa tutte le righe dentro
    una colonna. Le crocette e le firme non si estendono per piu' righe con una
    linea continua: si cercano tratti quasi verticali (anche chiari) lunghi almeno
    ``CREASE_MIN_ROWS`` righe, sottili, presenti in modo continuo e lontani dai
    bordi di colonna stampati."""
    gray = _darkness_gray(img)
    H, W = gray.shape[:2]
    if (grid.width, grid.height) != (W, H):
        grid = grid.scaled(W / float(grid.width), H / float(grid.height))
    rh = float(grid.row_height)
    if rh <= 6:
        return [], None
    x0 = max(0, int(grid.col_x[1] - 0.2 * rh))
    x1 = min(W, int(grid.col_x[-1] + 0.2 * rh))
    y0 = max(0, int(grid.row_y[0] - 0.1 * rh))
    y1 = min(H, int(grid.row_y[-1] + 0.1 * rh))
    if x1 - x0 < 10 or y1 - y0 < CREASE_MIN_ROWS * rh:
        return [], None
    region = gray[y0:y1, x0:x1]
    _strong, faint = _local_ink(region, rh, faint=True)
    m = faint.copy()
    # i bordi di colonna stampati (e cio' che vi si confonde) non sono pieghe
    band = max(2, int(round(0.12 * rh)))
    for x in grid.col_x:
        xl = int(round(x)) - x0
        m[:, max(0, xl - band):max(0, xl + band + 1)] = False
    # tratti quasi verticali: allargati di poco in orizzontale (linee inclinate o
    # curve restano continue), poi apertura verticale (toglie scrittura e linee orizzontali)
    d = max(1, int(round(0.04 * rh)))
    k = max(7, int(round(0.35 * rh)))
    wide = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (2 * d + 1, 1)))
    vert = cv2.morphologyEx(wide, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    gap = max(3, int(round(CREASE_GAP * rh)))
    linked = cv2.morphologyEx(vert, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(linked, connectivity=8)
    creases: list[Crease] = []
    mask = np.zeros((H, W), dtype=bool)
    for i in range(1, n):
        bx, by, bw, bh, _a = (int(v) for v in stats[i])
        if bh < CREASE_MIN_ROWS * rh:
            continue
        sub = (lab[by:by + bh, bx:bx + bw] == i) & (vert[by:by + bh, bx:bx + bw] > 0)
        rows_on = sub.any(axis=1)
        if rows_on.mean() < CREASE_COVERAGE:
            continue
        ys = np.nonzero(rows_on)[0]
        first = np.argmax(sub[ys], axis=1)
        last = sub.shape[1] - 1 - np.argmax(sub[ys][:, ::-1], axis=1)
        widths = (last - first + 1).astype(np.float64) - 2 * d
        centers = (first + last) / 2.0
        # righe in cui c'e' solo la linea (sottile): almeno meta'; le altre sono
        # attraversate da scrittura unita alla linea e non servono a misurarla
        clean = widths <= CREASE_MAX_WIDTH * rh
        if clean.sum() < 0.5 * ys.size or clean.sum() < CREASE_MIN_ROWS * rh * 0.5:
            continue
        # curva regolare (piega leggermente incurvata), quasi verticale
        try:
            coef = np.polyfit(ys[clean].astype(np.float64), centers[clean], 2)
        except (np.linalg.LinAlgError, ValueError):
            continue
        resid = np.abs(centers[clean] - np.polyval(coef, ys[clean]))
        if float(np.percentile(resid, 90)) > CREASE_MAX_DEV * rh:
            continue
        yy = np.arange(bh, dtype=np.float64)
        slope = np.abs(2 * coef[0] * yy + coef[1])
        if float(slope.max()) > CREASE_MAX_SLOPE:
            continue
        half = 0.5 * float(np.percentile(widths[clean], 75)) + 1.5
        # presente lungo la curva (non solo nella componente, che puo' includere scrittura)
        fit = np.polyval(coef, yy)
        cols_sub = np.arange(bw, dtype=np.float64)[None, :]
        near_fit = np.abs(cols_sub - fit[:, None]) <= half + d + 1
        if float((sub & near_fit).any(axis=1).mean()) < CREASE_COVERAGE:
            continue
        path = fit + bx + x0
        # vicina e parallela a un bordo di colonna: e' la linea stampata (o vi si confonde)
        near = [abs(float(np.median(path)) - float(x)) for x in grid.col_x]
        if min(near) < CREASE_PRINTED * rh:
            continue
        gy0 = by + y0
        days = tuple(g for g in range(1, 32)
                     if min(gy0 + bh, grid.row_y[g]) - max(gy0, grid.row_y[g - 1]) >= 0.5 * rh)
        if not days:
            continue
        cross = _crossing_rows(faint[by:by + bh], path - x0, half, max(2, int(round(0.06 * rh))))
        px = np.clip(np.round(path).astype(int), 0, W - 1)
        along = gray[np.arange(gy0, gy0 + bh), px]
        level = float(np.median(along[~cross])) if (~cross).any() else float(np.median(along))
        creases.append(Crease(y0=gy0, y1=gy0 + bh, path=path, half_width=half, days=days, level=level))
        # pixel d'inchiostro (anche chiaro) lungo la curva, salvo dove un tratto di
        # scrittura la attraversa (inchiostro che continua oltre la fascia della piega
        # da una parte o dall'altra): li' si lascia tutto, per non tagliare le cifre
        lo, hi = max(x0, int(path.min() - half - 2)), min(x1, int(path.max() + half + 3))
        cols = np.arange(lo, hi, dtype=np.float64)[None, :]
        sel = np.abs(cols - path[:, None]) <= half
        band_ink = faint[by:by + bh, lo - x0:hi - x0]
        sel &= cv2.dilate(band_ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        sel[cross] = False
        mask[gy0:gy0 + bh, lo:hi] |= sel
    if not creases:
        return [], None
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return creases, mask


CREASE_INK_DARKER = 40.0     # scrittura sotto la piega: piu' scura della piega di almeno tanti livelli


def ink_under_crease(img: np.ndarray, box: Box, creases: list[Crease], unit: float,
                     mask: np.ndarray | None = None) -> bool:
    """True se dentro la fascia di una piega che attraversa il riquadro c'e' inchiostro
    nettamente piu' scuro della piega stessa: scrittura (una cifra stretta, un "1")
    finita sotto la piega, che togliendo la piega sparirebbe. Con ``mask`` (pixel
    tolti con la piega, ``find_creases``) contano solo i pixel tolti: i tratti che
    attraversano la piega restano nell'immagine e non sono "nascosti"."""
    hits = [c for c in creases if c.crosses(box)]
    if not hits:
        return False
    gray = _darkness_gray(img)
    H, W = gray.shape[:2]
    x0, y0, x1, y1 = _norm_box(gray, box)
    my = int(round(0.08 * (y1 - y0)))
    need = max(8.0, (0.1 * unit) ** 2)
    for c in hits:
        ya, yb = max(y0 + my, c.y0), min(y1 - my, c.y1)
        if yb <= ya:
            continue
        xs = c.path[ya - c.y0:yb - c.y0]
        cols = np.arange(x0, x1)[None, :]
        band = np.abs(cols - xs[:, None]) <= c.half_width + 1
        dark = gray[ya:yb, x0:x1] < c.level - CREASE_INK_DARKER
        if mask is not None and mask.shape[:2] == (H, W):
            dark &= mask[ya:yb, x0:x1]
        if float((band & dark).sum()) >= need:
            return True
    return False


def find_strokes(img: np.ndarray, grid) -> list[Stroke]:  # noqa: ANN001
    """Barrature diagonali e linee orizzontali a penna della tabella (vedi ``Stroke``)."""
    return PageInk(img, grid).strokes


def cell_own_ink(img: np.ndarray, grid, giorno: int, colonna: str | int,  # noqa: ANN001
                 page: PageInk | None = None, pad: int = 0, remove_lines: bool = True) -> CellInkResult:
    """Inchiostro proprio della cella (giorno, colonna) e indicatori di
    barratura/linea/debordamento (vedi ``CellInkResult``). Per analizzare piu'
    celle della stessa pagina passare ``page=PageInk(img, grid)``, calcolato
    una sola volta."""
    if page is None:
        page = PageInk(img, grid)
    return page.own_ink(giorno, colonna, pad=pad, remove_lines=remove_lines)


__all__ = [
    "CellInk",
    "CellInkResult",
    "Crease",
    "PageInk",
    "Stroke",
    "cell_own_ink",
    "find_creases",
    "find_strokes",
    "ink_under_crease",
    "analyze_cell",
    "crop",
    "has_cross",
    "has_signature",
    "ink_ratio",
    "is_blank",
    "is_dash",
]
