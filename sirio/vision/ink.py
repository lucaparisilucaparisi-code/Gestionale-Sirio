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


__all__ = [
    "CellInk",
    "analyze_cell",
    "crop",
    "has_cross",
    "has_signature",
    "ink_ratio",
    "is_blank",
    "is_dash",
]
