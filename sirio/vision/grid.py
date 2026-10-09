"""Individuazione della tabella del foglio firma e delle sue celle.

La tabella ha 10 colonne (``COLUMNS``), un'intestazione su due righe, 31 righe
giornaliere di altezza uniforme e la riga "Totale ore effettive mensili".
``detect_grid`` ricava dalle linee reali della scansione le 11 ascisse dei
bordi di colonna e le 32 ordinate dei bordi di riga; se la tabella non e'
riconoscibile restituisce un modello proporzionale (``detected=False``)
misurato sul modulo reale. Non solleva mai eccezioni.

Tutte le coordinate sono in pixel dell'immagine analizzata (di norma la pagina
normalizzata da ``preprocess.normalize_page``); i riquadri sono tuple
``(x0, y0, x1, y1)`` con ``x1``/``y1`` esclusi, gia' limitati all'immagine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from sirio.vision.preprocess import binarize, resize_to_width, to_bgr, to_gray

log = logging.getLogger(__name__)

COLUMNS: tuple[str, ...] = (
    "giorno",
    "prog_entrata",
    "prog_uscita",
    "eff_entrata",
    "eff_uscita",
    "ore_dichiarate",
    "assenza_alunno",
    "assenza_operatore",
    "firma",
    "note",
)
_COL_INDEX = {name: i for i, name in enumerate(COLUMNS)}

N_DAYS = 31
N_COLS = len(COLUMNS)

# --- Modello proporzionale (misurato sul modulo reale, A4 verticale) --------
# Bordi delle colonne in frazione della larghezza della tabella.
TEMPLATE_COL_FRACTIONS: tuple[float, ...] = (
    0.0, 0.0556, 0.1311, 0.2063, 0.2809, 0.3561, 0.4282, 0.4915, 0.5551, 0.7550, 1.0,
)
# Posizioni in frazione della pagina.
TEMPLATE_TABLE_X = (0.0390, 0.8867)        # bordo sinistro / destro della tabella
TEMPLATE_HEADER_TOP = 0.1584               # bordo superiore dell'intestazione della tabella
TEMPLATE_FIRST_ROW = 0.1818                # bordo superiore del giorno 1
TEMPLATE_LAST_ROW = 0.9307                 # bordo inferiore del giorno 31
TEMPLATE_TOTAL_BOTTOM = 0.9522             # bordo inferiore della riga dei totali
# Altezza dell'intestazione e della riga dei totali in righe-giorno.
HEADER_ROWS = 0.96
TOTAL_ROWS = 0.88

_WORK_WIDTH = 1700


Box = tuple[int, int, int, int]


@dataclass
class TableGrid:
    """Geometria della tabella giornaliera di un foglio firma."""

    width: int                 # dimensioni dell'immagine analizzata
    height: int
    col_x: list[int]           # 11 ascisse: bordi delle 10 colonne
    row_y: list[int]           # 32 ordinate: row_y[i] = bordo superiore del giorno i+1
    header_top: int            # bordo superiore dell'intestazione della tabella
    total_row: tuple[int, int] | None  # (y0, y1) della riga "Totale ore effettive mensili"
    detected: bool             # True = da linee reali; False = modello proporzionale
    score: float               # qualita' 0..1
    details: dict = field(default_factory=dict, compare=False, repr=False)

    # ---------------------------------------------------------------- utilita'
    @staticmethod
    def col_index(colonna: str | int) -> int:
        """Indice 0..9 di una colonna (nome di ``COLUMNS`` o indice)."""
        if isinstance(colonna, (int, np.integer)):
            idx = int(colonna)
            if 0 <= idx < N_COLS:
                return idx
            raise ValueError(f"Indice di colonna non valido: {colonna}")
        try:
            return _COL_INDEX[str(colonna)]
        except KeyError:
            raise ValueError(f"Colonna sconosciuta: {colonna!r}") from None

    @staticmethod
    def _check_day(giorno: int) -> int:
        g = int(giorno)
        if not 1 <= g <= N_DAYS:
            raise ValueError(f"Giorno non valido: {giorno} (ammessi 1-31)")
        return g

    def _clip(self, x0: float, y0: float, x1: float, y1: float, pad: int = 0) -> Box:
        xa = int(round(min(x0, x1))) - pad
        xb = int(round(max(x0, x1))) + pad
        ya = int(round(min(y0, y1))) - pad
        yb = int(round(max(y0, y1))) + pad
        xa = max(0, min(xa, self.width - 1))
        ya = max(0, min(ya, self.height - 1))
        xb = max(xa + 1, min(xb, self.width))
        yb = max(ya + 1, min(yb, self.height))
        return xa, ya, xb, yb

    @property
    def row_height(self) -> float:
        """Altezza media di una riga-giorno."""
        return (self.row_y[-1] - self.row_y[0]) / float(N_DAYS)

    # ------------------------------------------------------------- riquadri
    def cell(self, giorno: int, colonna: str | int, pad: int = 0) -> Box:
        """Riquadro di una cella. ``pad`` > 0 allarga (contesto), < 0 restringe."""
        g = self._check_day(giorno)
        c = self.col_index(colonna)
        return self._clip(self.col_x[c], self.row_y[g - 1], self.col_x[c + 1], self.row_y[g], pad)

    def row_box(self, giorno: int, pad: int = 0) -> Box:
        """Riquadro dell'intera riga di un giorno (tutte le colonne)."""
        g = self._check_day(giorno)
        return self._clip(self.col_x[0], self.row_y[g - 1], self.col_x[-1], self.row_y[g], pad)

    def rows_box(self, g_from: int, g_to: int, include_table_header: bool = False, pad: int = 0) -> Box:
        """Riquadro delle righe dal giorno ``g_from`` al giorno ``g_to`` inclusi,
        eventualmente con l'intestazione della tabella."""
        a, b = sorted((self._check_day(g_from), self._check_day(g_to)))
        y0 = self.header_top if include_table_header else self.row_y[a - 1]
        return self._clip(self.col_x[0], y0, self.col_x[-1], self.row_y[b], pad)

    def table_box(self) -> Box:
        """Intera tabella: dall'intestazione al fondo della riga dei totali."""
        bottom = self.total_row[1] if self.total_row else self.row_y[-1]
        return self._clip(self.col_x[0], self.header_top, self.col_x[-1], bottom)

    def header_region(self) -> Box:
        """Parte alta della pagina (sopra l'intestazione della tabella)."""
        return self._clip(0, 0, self.width, max(1, self.header_top))

    def footer_region(self) -> Box:
        """Dal fondo del giorno 31 al fondo della pagina (totali, firme, timbro)."""
        return self._clip(0, self.row_y[-1], self.width, self.height)

    def total_value_box(self) -> Box | None:
        """Cella del valore "Totale ore effettive mensili" (colonna delle ore)."""
        if not self.total_row:
            return None
        c = _COL_INDEX["ore_dichiarate"]
        return self._clip(self.col_x[c], self.total_row[0], self.col_x[c + 1], self.total_row[1])

    def coordinator_signature_box(self) -> Box | None:
        """Spazio per la firma del coordinatore dell'ente (riga dei totali, colonna "Note")."""
        if not self.total_row:
            return None
        return self._clip(self.col_x[9], self.total_row[0], self.col_x[10], self.total_row[1])

    def referent_stamp_box(self) -> Box | None:
        """Spazio per timbro e firma del referente scolastico (sotto la riga dei totali)."""
        if not self.total_row:
            return None
        th = self.total_row[1] - self.total_row[0]
        return self._clip(self.col_x[9], self.total_row[1], self.col_x[10], self.total_row[1] + 1.1 * th)

    # ----------------------------------------------------------- conversione
    def to_dict(self) -> dict:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "col_x": [int(v) for v in self.col_x],
            "row_y": [int(v) for v in self.row_y],
            "header_top": int(self.header_top),
            "total_row": [int(self.total_row[0]), int(self.total_row[1])] if self.total_row else None,
            "detected": bool(self.detected),
            "score": round(float(self.score), 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TableGrid":
        try:
            col_x = [int(v) for v in d["col_x"]]
            row_y = [int(v) for v in d["row_y"]]
            total = d.get("total_row")
            grid = cls(
                width=int(d["width"]),
                height=int(d["height"]),
                col_x=col_x,
                row_y=row_y,
                header_top=int(d["header_top"]),
                total_row=(int(total[0]), int(total[1])) if total else None,
                detected=bool(d.get("detected", False)),
                score=float(d.get("score", 0.0)),
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"Descrizione della griglia non valida: {exc}") from exc
        if len(grid.col_x) != N_COLS + 1 or len(grid.row_y) != N_DAYS + 1:
            raise ValueError("Descrizione della griglia non valida: numero di colonne o righe errato.")
        if grid.width <= 0 or grid.height <= 0:
            raise ValueError("Descrizione della griglia non valida: dimensioni non positive.")
        return grid

    def scaled(self, sx: float, sy: float) -> "TableGrid":
        """Stessa griglia per un'immagine ridimensionata di (sx, sy)."""
        return TableGrid(
            width=max(1, int(round(self.width * sx))),
            height=max(1, int(round(self.height * sy))),
            col_x=[int(round(v * sx)) for v in self.col_x],
            row_y=[int(round(v * sy)) for v in self.row_y],
            header_top=int(round(self.header_top * sy)),
            total_row=(int(round(self.total_row[0] * sy)), int(round(self.total_row[1] * sy)))
            if self.total_row else None,
            detected=self.detected,
            score=self.score,
        )


# ==========================================================================
# Modello proporzionale di ripiego
# ==========================================================================

def template_grid(width: int, height: int, table_x: tuple[float, float] | None = None,
                  col_x: list[float] | None = None, score: float = 0.05) -> TableGrid:
    """Griglia proporzionale del modulo standard per una pagina ``width`` x ``height``.
    ``table_x``/``col_x`` permettono di usare bordi della tabella gia' noti."""
    w, h = float(width), float(height)
    if col_x is None:
        x0, x1 = table_x if table_x else (TEMPLATE_TABLE_X[0] * w, TEMPLATE_TABLE_X[1] * w)
        col_x = [x0 + f * (x1 - x0) for f in TEMPLATE_COL_FRACTIONS]
    y0 = TEMPLATE_FIRST_ROW * h
    y1 = TEMPLATE_LAST_ROW * h
    rows = [y0 + i * (y1 - y0) / N_DAYS for i in range(N_DAYS + 1)]
    return TableGrid(
        width=int(width),
        height=int(height),
        col_x=[int(round(v)) for v in col_x],
        row_y=[int(round(v)) for v in rows],
        header_top=int(round(TEMPLATE_HEADER_TOP * h)),
        total_row=(int(round(y1)), int(round(min(h - 1, TEMPLATE_TOTAL_BOTTOM * h)))),
        detected=False,
        score=float(score),
    )


# ==========================================================================
# Rilevamento
# ==========================================================================

@dataclass
class _Line:
    pos: float                 # coordinata del centro della linea
    lo: int                    # estremi della zona occupata (asse perpendicolare)
    hi: int
    start: int                 # estensione lungo la linea
    end: int
    strength: float            # pixel coperti lungo la linea
    cover: np.ndarray | None = None   # maschera booleana di copertura lungo la linea


def _longest_run(mask: np.ndarray, max_gap: int) -> tuple[int, int]:
    """Estremi del tratto piu' lungo di ``mask`` tollerando buchi fino a ``max_gap``."""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return 0, 0
    best = (int(idx[0]), int(idx[0]))
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > max_gap:
            if prev - start > best[1] - best[0]:
                best = (start, prev)
            start = i
        prev = i
    if prev - start > best[1] - best[0]:
        best = (start, prev)
    return best


def _cluster_lines(mask: np.ndarray, axis: int, min_count: float, merge: int, pad: int,
                   max_gap: int) -> list[_Line]:
    """Linee (orizzontali se axis=0, verticali se axis=1) dalla maschera morfologica."""
    m = mask > 0
    if axis == 1:
        m = m.T
    prof = m.sum(axis=1).astype(np.float64)
    if prof.size >= 3:
        prof = np.convolve(prof, np.ones(3) / 3.0, mode="same")
    idx = np.nonzero(prof >= min_count)[0]
    lines: list[_Line] = []
    if idx.size == 0:
        return lines
    groups: list[list[int]] = [[int(idx[0])]]
    for i in idx[1:]:
        i = int(i)
        if i - groups[-1][-1] <= merge:
            groups[-1].append(i)
        else:
            groups.append([i])
    n = m.shape[0]
    for gr in groups:
        lo, hi = gr[0], gr[-1]
        weights = prof[lo:hi + 1]
        pos = float(np.dot(np.arange(lo, hi + 1), weights) / max(weights.sum(), 1e-9))
        band = m[max(0, lo - pad):min(n, hi + pad + 1)]
        cover = band.any(axis=0)
        start, end = _longest_run(cover, max_gap)
        lines.append(_Line(pos=pos, lo=lo, hi=hi, start=start, end=end,
                           strength=float(cover.sum()), cover=cover))
    return lines


def _coverage(line: _Line, a: float, b: float) -> float:
    """Frazione dell'intervallo [a, b) coperta dalla linea."""
    if line.cover is None:
        return 0.0
    ia = max(0, int(round(a)))
    ib = min(line.cover.size, int(round(b)))
    if ib <= ia:
        return 0.0
    return float(line.cover[ia:ib].mean())


def _fit_columns(vlines: list[_Line], x_left: float, x_right: float, w: int) -> tuple[list[float], int] | None:
    """Sceglie gli 11 bordi di colonna confrontando le linee verticali con le
    proporzioni del modulo. Restituisce (ascisse, linee effettivamente trovate)."""
    if not vlines:
        return None
    xs = np.array([ln.pos for ln in vlines])
    st = np.array([ln.strength for ln in vlines])
    st_n = st / max(st.max(), 1e-9)
    fr = np.array(TEMPLATE_COL_FRACTIONS)
    min_gap = float(np.min(np.diff(fr)))

    def evaluate(a: float, b: float) -> tuple[float, list[float], int]:
        tw = b - a
        pos: list[float] = []
        found = 0
        score = 0.0
        for j, f in enumerate(fr):
            pred = a + f * tw
            tol = 0.32 * min_gap * tw
            d = np.abs(xs - pred)
            k = int(np.argmin(d))
            if d[k] <= tol:
                pos.append(float(xs[k]))
                found += 1
                score += st_n[k] * (1.0 - 0.5 * d[k] / tol)
            else:
                pos.append(float(pred))
                score -= 0.35
        return score, pos, found

    best: tuple[float, list[float], int] | None = None
    # Candidati per i bordi: linee verticali vicine agli estremi delle righe lunghe.
    lefts = [ln.pos for ln in vlines if abs(ln.pos - x_left) <= 0.06 * w] or [x_left]
    rights = [ln.pos for ln in vlines if abs(ln.pos - x_right) <= 0.06 * w] or [x_right]
    for a in lefts:
        for b in rights:
            if b - a < 0.4 * w:
                continue
            res = evaluate(a, b)
            if best is None or res[0] > best[0]:
                best = res
    if best is None:
        return None
    _, pos, found = best
    # Ordine strettamente crescente (difesa contro abbinamenti incrociati).
    for j in range(1, len(pos)):
        if pos[j] <= pos[j - 1]:
            return None
    return pos, found


class _BandStats:
    """Presenza delle linee verticali interne (colonne 1..9) riga per riga, per
    misurare quanto una fascia orizzontale "somiglia" a una riga-giorno."""

    def __init__(self, vmask: np.ndarray, col_x: list[float], tol: int):
        h, w = vmask.shape
        self.h = h
        self.cum: list[np.ndarray] = []
        for x in col_x[1:-1]:
            xa = max(0, int(round(x)) - tol)
            xb = min(w, int(round(x)) + tol + 1)
            on = (vmask[:, xa:xb] > 0).any(axis=1).astype(np.float64)
            self.cum.append(np.concatenate([[0.0], np.cumsum(on)]))

    def fills(self, ya: float, yb: float) -> np.ndarray:
        span = yb - ya
        a = int(round(ya + 0.2 * span))
        b = int(round(yb - 0.2 * span))
        a = max(0, min(a, self.h - 1))
        b = max(a + 1, min(b, self.h))
        return np.array([(c[b] - c[a]) / (b - a) for c in self.cum])

    def body_like(self, ya: float, yb: float) -> float:
        f = np.sort(self.fills(ya, yb))
        return float(f[:3].mean())


def _estimate_period(ys: np.ndarray, h: int) -> float | None:
    if ys.size < 4:
        return None
    d = np.diff(np.sort(ys))
    d = d[(d >= 0.012 * h) & (d <= 0.045 * h)]
    if d.size < 3:
        return None
    counts = np.array([np.sum(np.abs(d - v) <= 0.07 * v) for v in d])
    v0 = float(d[int(np.argmax(counts))])
    close = d[np.abs(d - v0) <= 0.07 * v0]
    return float(np.median(close))


def _header_like(partial: list[_Line], ya: float, yb: float, col_x: list[float]) -> float:
    """1 se nella fascia c'e' la linea intermedia dell'intestazione (sotto
    "Orario programmato/effettivo": colonne 1-4 soltanto)."""
    span = yb - ya
    best = 0.0
    for ln in partial:
        if not (ya + 0.25 * span <= ln.pos <= yb - 0.25 * span):
            continue
        inside = _coverage(ln, col_x[1], col_x[5])
        outside = _coverage(ln, col_x[6], col_x[10])
        if inside >= 0.5 and outside <= 0.35:
            best = max(best, min(1.0, inside) * (1.0 - outside))
    return best


def _fit_rows(strong: list[_Line], partial: list[_Line], bands: _BandStats, col_x: list[float],
              h: int) -> dict | None:
    """Trova le 32 linee delle righe-giorno (progressione quasi regolare)."""
    ys = np.array(sorted(ln.pos for ln in strong))
    period = _estimate_period(ys, h)
    if period is None:
        return None
    cands: set[int] = set()
    q = max(1.0, 0.1 * period)
    for y in ys:
        for k in range(N_DAYS + 1):
            y0 = y - k * period
            if y0 < -0.5 * period or y0 + N_DAYS * period > h + 0.5 * period:
                continue
            cands.add(int(round(y0 / q)))
    best: dict | None = None
    idx = np.arange(N_DAYS + 1, dtype=np.float64)

    def match(pred: np.ndarray, tol: float) -> np.ndarray:
        pos = np.searchsorted(ys, pred)
        out = np.full(pred.shape, np.nan)
        for i, (p, k) in enumerate(zip(pred, pos)):
            best_d = tol
            for kk in (k - 1, k):
                if 0 <= kk < ys.size and abs(ys[kk] - p) <= best_d:
                    best_d = abs(ys[kk] - p)
                    out[i] = ys[kk]
        return out

    for c in cands:
        y0 = c * q
        pred = y0 + idx * period
        m = match(pred, 0.3 * period)
        ok = ~np.isnan(m)
        if ok.sum() < 12:
            continue
        b, a = np.polyfit(idx[ok], m[ok], 1)
        if not 0.85 * period <= b <= 1.15 * period:
            continue
        pred = a + b * idx
        m = match(pred, 0.25 * b)
        ok = ~np.isnan(m)
        n_ok = int(ok.sum())
        if n_ok < 12:
            continue
        rows = pred.copy()
        rows[ok] = m[ok]
        # interpolazione locale delle linee mancanti
        if n_ok < len(rows):
            rows[~ok] = np.interp(idx[~ok], idx[ok], m[ok], left=np.nan, right=np.nan)
            nan = np.isnan(rows)
            rows[nan] = (a + b * idx)[nan]
        diffs = np.diff(rows)
        if np.any(diffs <= 0.5 * b):
            continue
        body = float(np.mean([bands.body_like(rows[i], rows[i + 1]) for i in range(N_DAYS)]))
        # fascia sopra il giorno 1 (intestazione) e sotto il giorno 31 (totali)
        above = [ln for ln in strong if 0.6 * b <= rows[0] - ln.pos <= 1.8 * b]
        hdr_top = max((ln.pos for ln in above), default=rows[0] - HEADER_ROWS * b)
        below = [ln for ln in strong + partial if 0.55 * b <= ln.pos - rows[-1] <= 1.6 * b]
        tot_bottom = min((ln.pos for ln in below), default=rows[-1] + TOTAL_ROWS * b)
        like_above = bands.body_like(hdr_top, rows[0]) if hdr_top > 0 else 0.0
        like_below = bands.body_like(rows[-1], tot_bottom) if tot_bottom < h else 0.0
        hdr = _header_like(partial, hdr_top, rows[0], col_x)
        ref = max(body, 1e-6)
        score = n_ok / (N_DAYS + 1.0) + body + 0.4 * hdr
        if like_above > 0.8 * ref and hdr < 0.5:
            score -= 0.6 * like_above / ref        # sopra c'e' ancora una riga-giorno
        if like_below > 0.8 * ref:
            score -= 0.6 * like_below / ref        # sotto c'e' ancora una riga-giorno
        if best is None or score > best["score"]:
            best = {
                "score": score, "rows": rows, "matched": n_ok, "body": body, "period": b,
                "header_top": hdr_top, "header_found": bool(above), "header_like": hdr,
                "total_bottom": tot_bottom, "total_found": bool(below),
            }
    return best


def _detect(gray: np.ndarray) -> TableGrid | None:
    H, W = gray.shape[:2]
    small, scale = resize_to_width(gray, _WORK_WIDTH)
    h, w = small.shape[:2]
    if h < 200 or w < 150:
        return None
    bw = binarize(small, c=12)

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 40), 1))
    hmask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    hmask = cv2.morphologyEx(hmask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 150), 1)))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 55)))
    vmask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    vmask = cv2.morphologyEx(vmask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, h // 200))))

    merge = max(3, h // 350)
    hlines = _cluster_lines(hmask, 0, 0.06 * w, merge=merge, pad=2, max_gap=max(4, w // 50))
    if len(hlines) < 10:
        return None
    max_st = max(ln.strength for ln in hlines)
    if max_st < 0.35 * w:
        return None
    longest = [ln for ln in hlines if ln.strength >= 0.7 * max_st]
    x_left = float(np.median([ln.start for ln in longest]))
    x_right = float(np.median([ln.end for ln in longest]))
    y_top = min(ln.pos for ln in longest)
    y_bot = max(ln.pos for ln in longest)
    if y_bot - y_top < 0.2 * h:
        return None

    # Linee verticali misurate nel corpo della tabella.
    ya, yb = int(y_top), int(y_bot) + 1
    vband = np.zeros_like(vmask)
    vband[ya:yb] = vmask[ya:yb]
    xa = max(0, int(x_left - 0.04 * w))
    xb = min(w, int(x_right + 0.04 * w) + 1)
    vband[:, :xa] = 0
    vband[:, xb:] = 0
    vlines = _cluster_lines(vband, 1, 0.2 * (yb - ya), merge=max(3, w // 300), pad=2,
                            max_gap=max(4, h // 60))
    cols = _fit_columns(vlines, x_left, x_right, w)
    if cols is None:
        return None
    col_x, col_found = cols
    tw = col_x[-1] - col_x[0]

    # Righe: linee che attraversano la tabella.
    for ln in hlines:
        ln.strength = _coverage(ln, col_x[0], col_x[-1])
    strong = [ln for ln in hlines if ln.strength >= 0.5]
    partial = [ln for ln in hlines if 0.12 <= ln.strength < 0.5]
    tol = max(3, int(round(0.004 * w)))
    bands = _BandStats(vmask, col_x, tol)
    fit = _fit_rows(strong, partial, bands, col_x, h)
    if fit is None:
        return None
    rows = fit["rows"]
    diffs = np.diff(rows)
    regularity = float(1.0 - min(1.0, 4.0 * float(np.std(diffs) / max(np.mean(diffs), 1e-9))))
    m_rows = fit["matched"] / float(N_DAYS + 1)
    m_cols = col_found / float(N_COLS + 1)
    body = max(0.0, min(1.0, fit["body"]))
    score = 0.35 * m_rows + 0.25 * m_cols + 0.25 * body + 0.15 * regularity
    detected = m_rows >= 0.6 and m_cols >= 0.6 and body >= 0.35 and tw >= 0.4 * w
    if not detected:
        log.info("Tabella riconosciuta solo in parte (righe %.0f%%, colonne %.0f%%)",
                 100 * m_rows, 100 * m_cols)
        fallback = template_grid(W, H, col_x=[x / scale for x in col_x] if m_cols >= 0.6 else None,
                                 score=min(0.3, 0.3 * score))
        return fallback
    total_bottom = float(fit["total_bottom"])
    total = (int(round(rows[-1])), int(round(total_bottom))) if total_bottom < h else None
    grid = TableGrid(
        width=w,
        height=h,
        col_x=[int(round(x)) for x in col_x],
        row_y=[int(round(y)) for y in rows],
        header_top=int(round(max(0.0, fit["header_top"]))),
        total_row=total,
        detected=True,
        score=float(max(0.0, min(1.0, score))),
    )
    if scale != 1.0:
        grid = grid.scaled(W / float(w), H / float(h))
        grid.width, grid.height = W, H
    grid.details = {
        "righe_trovate": fit["matched"],
        "colonne_trovate": col_found,
        "periodo": fit["period"] / scale,
        "intestazione_trovata": fit["header_found"],
        "totale_trovato": fit["total_found"],
    }
    return grid


def detect_grid(img: np.ndarray) -> TableGrid:
    """Individua la tabella giornaliera. Non solleva mai eccezioni: se la tabella
    non e' riconoscibile restituisce il modello proporzionale (``detected=False``)."""
    try:
        gray = to_gray(img)
    except Exception:  # noqa: BLE001
        log.warning("Immagine non valida per il rilevamento della griglia")
        h, w = (img.shape[:2] if isinstance(img, np.ndarray) and img.ndim >= 2 else (3508, 2480))
        return template_grid(max(1, int(w)), max(1, int(h)), score=0.0)
    H, W = gray.shape[:2]
    try:
        grid = _detect(gray)
    except Exception:  # noqa: BLE001
        log.exception("Rilevamento della griglia non riuscito")
        grid = None
    if grid is None:
        return template_grid(W, H, score=0.0)
    return grid


# ==========================================================================
# Diagnostica
# ==========================================================================

def draw_grid(img: np.ndarray, grid: TableGrid) -> np.ndarray:
    """Copia BGR dell'immagine con la griglia disegnata (per il debug)."""
    out = to_bgr(img).copy()
    h, w = out.shape[:2]
    sx = w / float(grid.width) if grid.width else 1.0
    sy = h / float(grid.height) if grid.height else 1.0
    g = grid if (abs(sx - 1) < 1e-6 and abs(sy - 1) < 1e-6) else grid.scaled(sx, sy)
    t = max(1, int(round(min(w, h) / 900)))
    col_color = (255, 120, 0) if g.detected else (0, 140, 255)
    row_color = (0, 170, 0) if g.detected else (0, 140, 255)
    y_top, y_bot = g.header_top, (g.total_row[1] if g.total_row else g.row_y[-1])
    for x in g.col_x:
        cv2.line(out, (x, y_top), (x, y_bot), col_color, t, cv2.LINE_AA)
    for y in g.row_y:
        cv2.line(out, (g.col_x[0], y), (g.col_x[-1], y), row_color, t, cv2.LINE_AA)
    cv2.line(out, (g.col_x[0], g.header_top), (g.col_x[-1], g.header_top), (0, 0, 230), t + 1, cv2.LINE_AA)
    if g.total_row:
        cv2.rectangle(out, (g.col_x[0], g.total_row[0]), (g.col_x[-1], g.total_row[1]), (200, 0, 200), t,
                      cv2.LINE_AA)
        tv = g.total_value_box()
        if tv:
            cv2.rectangle(out, tv[:2], tv[2:], (200, 0, 200), t + 1, cv2.LINE_AA)
    fs = max(0.4, min(w, h) / 2400.0)
    for d in range(1, N_DAYS + 1):
        y = int((g.row_y[d - 1] + g.row_y[d]) / 2)
        cv2.putText(out, str(d), (max(0, g.col_x[0] - int(60 * fs)), y + int(8 * fs)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 230), max(1, t), cv2.LINE_AA)
    label = f"{'rilevata' if g.detected else 'modello'}  score={g.score:.2f}"
    cv2.putText(out, label, (10, int(40 * fs) + 10), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.2, (0, 0, 230),
                max(1, t), cv2.LINE_AA)
    return out


__all__ = [
    "COLUMNS",
    "TableGrid",
    "detect_grid",
    "draw_grid",
    "template_grid",
]
