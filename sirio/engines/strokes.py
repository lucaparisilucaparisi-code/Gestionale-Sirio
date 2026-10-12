"""Tratti a mano che non sono scrittura: barrature e barrette.

Sui fogli firma reali compaiono segni che l'OCR non deve "leggere":

* **barrature**: righe non usate annullate con lunghe linee diagonali che
  attraversano piu' celle e piu' righe; linee orizzontali tirate a mano sugli
  orari effettivi per dire "prestazione non svolta";
* **barrette**: una "/" (o una "\\") isolata nella prima cella di una riga
  non lavorata.

Un tratto e' una *barratura* se e' lungo (almeno ``STRIKE_MIN_LEN`` altezze di
riga), sottile (spessore medio = area / lunghezza) e quasi rettilineo (scarto
da una parabola adattata lungo l'asse principale; le foto di fogli incurvati
piegano leggermente le linee). Le linee stampate della griglia vengono tolte
prima (vicino alle posizioni note di righe e colonne); quelle che attraversano
quasi tutta la tabella (griglia non allineata, foglio piegato) vengono
cancellate come linee ma non contano come barrature.

Funzioni pure su immagini in scala di grigi; nessuna dipendenza dal modello.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from sirio.vision.grid import TableGrid

STRIKE_MIN_LEN = 2.6        # lunghezza minima di una barratura (altezze di riga)
STRIKE_MAX_THICK = 0.13     # spessore medio massimo (altezze di riga)
STRIKE_MAX_DEV = 0.16       # scarto massimo (90 percentile) dalla linea adattata (altezze di riga)
PRINTED_SPAN = 0.8          # frazione della tabella oltre la quale un tratto e' una linea stampata
HORIZONTAL_DEG = 20.0       # inclinazione massima di una linea "orizzontale" (trattino lungo)
HORIZONTAL_MAX_ROWS = 1.0   # ... e altezza massima (altezze di riga): resta dentro una riga
SLASH_MIN_DEG = 18.0        # inclinazione di una barretta "/" rispetto all'orizzontale
SLASH_MAX_DEG = 65.0        # oltre e' piu' probabilmente un "1"
CELL_MIN_RUN = 0.35         # tratto di barratura minimo dentro una cella (altezze di riga)
WEAK_INK_MIN = 28.0         # soglia d'inchiostro (bassa) per i tratti leggeri delle barrature
WEAK_INK_REL = 0.12
LINE_GROW = 0.06            # allargamento delle linee stampate tolte (altezze di riga)
SPECK_AREA = 0.002          # area (altezze di riga al quadrato) sotto cui un pezzo e' un puntino
LINK_RADIUS = 0.1           # raggio (altezze di riga) per ricucire i frammenti vicini
PIECE_MIN_LEN = 0.5         # pezzo rettilineo minimo di una barratura spezzata (altezze di riga)
JOIN_MAX_DEG = 12.0         # pezzi della stessa linea: direzioni simili...
JOIN_MAX_GAP = 1.5          # ... estremi vicini (altezze di riga)...
JOIN_MAX_OFFSET = 0.15      # ... e allineati (scarto laterale, altezze di riga)


@dataclass
class Stroke:
    """Una barratura trovata nella tabella (coordinate dell'immagine)."""

    x0: int
    y0: int
    x1: int
    y1: int
    angle: float                # gradi rispetto all'orizzontale (0..90)
    length: float               # pixel
    thickness: float            # pixel

    horizontal: bool = False    # linea lungo una sola riga (prestazione non svolta)


@dataclass
class StrikeInfo:
    """Barrature della tabella e celle che attraversano."""

    mask: np.ndarray | None = None                  # pixel da cancellare (bool, dimensioni dell'immagine)
    strokes: list[Stroke] = field(default_factory=list)
    # (giorno, indice colonna) -> (lunghezza del tratto nella cella in altezze di riga, orizzontale?)
    cells: dict[tuple[int, int], tuple[float, bool]] = field(default_factory=dict)

    def struck(self, giorno: int, col: int) -> bool:
        return (giorno, col) in self.cells

    def horizontal(self, giorno: int, col: int) -> bool:
        c = self.cells.get((giorno, col))
        return bool(c and c[1])


def paper_background(gray: np.ndarray, unit: float) -> np.ndarray:
    """Livello della carta punto per punto (massimo locale sfumato): regge le
    ombre e la luce non uniforme delle foto."""
    k = max(5, int(round(0.7 * unit)) | 1)
    bg = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return cv2.blur(bg, (k, k))


def ink_mask(gray: np.ndarray, unit: float) -> np.ndarray:
    """Inchiostro rispetto alla carta locale (stessa soglia dei ritagli)."""
    bg = paper_background(gray, unit).astype(np.float32)
    diff = bg - gray.astype(np.float32)
    return diff > np.maximum(45.0, 0.2 * bg)


def _printed_lines(ink: np.ndarray, unit: float, rows: list[float], cols: list[float]) -> np.ndarray:
    """Linee stampate: tratti rettilinei vicini alle righe/colonne note della griglia."""
    m = ink.astype(np.uint8)
    h, w = m.shape
    band = max(3, int(round(0.22 * unit)))
    klen = max(9, int(round(0.8 * unit)))
    hl = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1)))
    vl = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen)))
    keep = np.zeros(h, dtype=bool)
    for y in rows:
        keep[max(0, int(y) - band):max(0, int(y) + band + 1)] = True
    hl[~keep, :] = 0
    keep = np.zeros(w, dtype=bool)
    for x in cols:
        keep[max(0, int(x) - band):max(0, int(x) + band + 1)] = True
    vl[:, ~keep] = 0
    t = max(1, int(round(LINE_GROW * unit)))
    hl = cv2.dilate(hl, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2 * t + 1)))
    vl = cv2.dilate(vl, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * t + 1, 3)))
    return (hl | vl) > 0


def stroke_shape(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    """Forma di un tratto dai suoi pixel: (lunghezza, spessore medio, scarto dalla
    curva adattata al 90 percentile, inclinazione in gradi 0..90)."""
    n = xs.size
    if n < 3:
        return 0.0, 0.0, 0.0, 0.0
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    minor = np.array([-major[1], major[0]])
    along = (pts - mean) @ major
    perp = (pts - mean) @ minor
    length = float(along.max() - along.min()) + 1.0
    thickness = n / length
    # scarto da una parabola lungo l'asse principale (linee leggermente curve nelle foto)
    try:
        coef = np.polyfit(along, perp, 2) if length > 4 else np.array([0.0, 0.0, float(perp.mean())])
        resid = perp - np.polyval(coef, along)
    except (np.linalg.LinAlgError, ValueError):
        resid = perp
    dev = float(np.percentile(np.abs(resid), 90))
    angle = math.degrees(math.atan2(abs(major[1]), abs(major[0])))
    return length, thickness, dev, angle


def find_strikes(gray: np.ndarray, grid: TableGrid, ignore: np.ndarray | None = None) -> StrikeInfo:
    """Barrature a mano nella parte giornaliera della tabella (giorni 1-31,
    colonne dagli orari alle note).

    ``ignore`` (facoltativo, bool, dimensioni dell'immagine): pixel da non
    considerare (es. le pieghe del foglio, ``sirio.vision.ink.find_creases``: una
    piega quasi verticale non e' una barratura).

    Le barrature sono spesso tracciate con poca pressione e risultano spezzate:
    si usa una soglia d'inchiostro piu' bassa di quella dei ritagli e i pezzi
    vicini vengono raggruppati (dilatazione) prima di misurarne la forma sui
    soli pixel d'inchiostro."""
    info = StrikeInfo()
    H, W = gray.shape[:2]
    unit = float(grid.row_height)
    if unit <= 4:
        return info
    x0 = max(0, int(grid.col_x[1] + 0.1 * unit))      # la colonna dei giorni e' spesso grigia
    x1 = min(W, int(grid.col_x[-1] + 0.2 * unit))
    y0 = max(0, int(grid.row_y[0] + 0.1 * unit))       # sotto l'intestazione (spesso grigia)
    y1 = min(H, int(grid.row_y[-1] + 0.3 * unit))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return info
    region = gray[y0:y1, x0:x1]
    bg = paper_background(region, unit).astype(np.float32)
    weak = (bg - region.astype(np.float32)) > np.maximum(WEAK_INK_MIN, WEAK_INK_REL * bg)
    if ignore is not None and ignore.shape[:2] == (H, W):
        weak &= ~ignore[y0:y1, x0:x1]
    row_y = [y - y0 for y in grid.row_y]
    lines = _printed_lines(weak, unit, row_y, [x - x0 for x in grid.col_x])
    rest = weak & ~lines
    # puntini (grana della carta, retino grigio delle colonne stampate) non si ricuciono
    n0, lab0, stats0, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8), connectivity=8)
    small = stats0[:, cv2.CC_STAT_AREA] < max(6.0, SPECK_AREA * unit * unit)
    small[0] = False
    rest &= ~small[lab0]
    r = max(2, int(round(LINK_RADIUS * unit)))
    link = cv2.dilate(rest.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(link, connectivity=8)
    # 1) pezzi rettilinei e sottili (i tratti di scrittura, curvi o spessi, sono esclusi)
    pieces: list[dict] = []
    for k in range(1, n):
        bx, by, bw, bh, _area = (int(v) for v in stats[k])
        if max(bw, bh) < PIECE_MIN_LEN * unit:
            continue
        sub = (lab[by:by + bh, bx:bx + bw] == k) & rest[by:by + bh, bx:bx + bw]
        ys, xs = np.nonzero(sub)
        if xs.size < 8:
            continue
        xs = xs + bx
        ys = ys + by
        length, thick, dev, _angle = stroke_shape(xs, ys)
        if length < PIECE_MIN_LEN * unit or thick > STRIKE_MAX_THICK * unit or dev > STRIKE_MAX_DEV * unit:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        c = pts.mean(axis=0)
        evals, evecs = np.linalg.eigh(np.cov((pts - c).T))
        d = evecs[:, int(np.argmax(evals))]
        t = (pts - c) @ d
        pieces.append({"xs": xs, "ys": ys, "c": c, "d": d, "a": c + d * t.min(), "b": c + d * t.max()})
    # 2) pezzi allineati della stessa linea (spezzata dalla poca pressione o dalle linee della griglia)
    parent = list(range(len(pieces)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, p in enumerate(pieces):
        for j in range(i + 1, len(pieces)):
            q = pieces[j]
            if abs(float(p["d"] @ q["d"])) < math.cos(math.radians(JOIN_MAX_DEG)):
                continue
            gap = min(float(np.linalg.norm(e - f)) for e in (p["a"], p["b"]) for f in (q["a"], q["b"]))
            if gap > JOIN_MAX_GAP * unit:
                continue
            off = q["c"] - p["c"]
            lateral = abs(float(off @ np.array([-p["d"][1], p["d"][0]])))
            if lateral > JOIN_MAX_OFFSET * unit + 0.05 * abs(float(off @ p["d"])):
                continue
            parent[root(j)] = root(i)
    groups: dict[int, list[dict]] = {}
    for i, p in enumerate(pieces):
        groups.setdefault(root(i), []).append(p)
    mask = np.zeros_like(rest)
    table_w = float(grid.col_x[-1] - grid.col_x[1])
    table_h = float(grid.row_y[-1] - grid.row_y[0])
    min_len = STRIKE_MIN_LEN * unit
    gy = np.asarray(grid.row_y)
    gx = np.asarray(grid.col_x)
    for members in groups.values():
        xs = np.concatenate([m["xs"] for m in members])
        ys = np.concatenate([m["ys"] for m in members])
        length, thick, dev, angle = stroke_shape(xs, ys)
        # spessore sui soli tratti (i vuoti fra i pezzi non contano)
        thick = xs.size / max(1.0, sum(float(np.linalg.norm(m["b"] - m["a"])) + 1.0 for m in members))
        if length < min_len or thick > STRIKE_MAX_THICK * unit or dev > STRIKE_MAX_DEV * unit:
            continue
        bx, by = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max()) - bx + 1, int(ys.max()) - by + 1
        mask[ys, xs] = True
        ym = float(np.median(ys))
        printed = ((angle <= 10 and bw >= PRINTED_SPAN * table_w) or (angle >= 80 and bh >= PRINTED_SPAN * table_h)
                   or (angle <= 5 and min(abs(ym - y) for y in row_y) <= 0.25 * unit))
        if printed:
            # linea stampata non allineata con la griglia: si cancella ma non e' una barratura
            continue
        horiz = angle <= HORIZONTAL_DEG and bh <= HORIZONTAL_MAX_ROWS * unit
        info.strokes.append(Stroke(bx + x0, by + y0, bx + bw + x0, by + bh + y0, angle, length, thick, horiz))
        # celle attraversate: lunghezza del tratto dentro ogni cella
        rows = np.searchsorted(gy, ys + y0, side="right")      # 1..31 = giorno
        cols = np.searchsorted(gx, xs + x0, side="right") - 1  # 0..9 = colonna
        ok = (rows >= 1) & (rows <= 31) & (cols >= 0) & (cols <= 9)
        if not ok.any():
            continue
        # "orizzontale" = linea tirata lungo una sola riga (una diagonale poco inclinata
        # che scende per piu' righe e' invece una barratura di righe non usate)
        keys, counts = np.unique(rows[ok] * 16 + cols[ok], return_counts=True)
        for key, cnt in zip(keys.tolist(), counts.tolist()):
            run = cnt / max(1.0, thick) / unit
            if run < CELL_MIN_RUN:
                continue
            g, c = divmod(int(key), 16)
            prev = info.cells.get((g, c))
            if prev is None or run > prev[0]:
                info.cells[(g, c)] = (run, horiz if prev is None else (horiz and prev[1]))
    if mask.any():
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        full = np.zeros((H, W), dtype=bool)
        full[y0:y1, x0:x1] = mask
        info.mask = full
    return info


def erase(gray: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Copia dell'immagine con i pixel di ``mask`` portati al colore della carta."""
    if mask is None or not mask.any():
        return gray
    out = gray.copy()
    out[mask] = 255
    return out


def is_slash(crop: np.ndarray, unit: float) -> bool:
    """True se il ritaglio pulito (inchiostro scuro su bianco) contiene solo una
    barretta obliqua ("/" o "\\"): un tratto unico, sottile, rettilineo, inclinato
    fra ``SLASH_MIN_DEG`` e ``SLASH_MAX_DEG`` gradi (un "1" e' quasi verticale)."""
    if crop is None or crop.size == 0:
        return False
    m = (crop < 160).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return False
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    total = float(areas.sum())
    min_piece = max(4.0, (0.05 * unit) ** 2)
    if (areas >= min_piece).sum() > 1 or areas.max() < 0.9 * total:
        return False
    ys, xs = np.nonzero(lab == k)
    length, thick, dev, angle = stroke_shape(xs, ys)
    if not 0.35 * unit <= length <= 2.5 * unit:
        return False
    if thick > 0.16 * unit or dev > 0.07 * unit + 1.5:
        return False
    return SLASH_MIN_DEG <= angle <= SLASH_MAX_DEG


__all__ = ["StrikeInfo", "Stroke", "erase", "find_strikes", "ink_mask", "is_slash", "paper_background",
           "stroke_shape"]
