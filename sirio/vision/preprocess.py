"""Raddrizzamento (deskew) e normalizzazione delle pagine scansionate.

Convenzioni
-----------
* Le immagini sono array numpy ``uint8`` BGR (oppure in scala di grigi).
* L'angolo di inclinazione e' espresso in gradi con la convenzione di OpenCV:
  positivo = contenuto ruotato in senso antiorario. ``deskew`` ruota
  l'immagine dell'angolo opposto per riportare orizzontali le righe della
  tabella e restituisce l'angolo stimato.
* Le rotazioni di 90/180/270 gradi sono esatte (``cv2.rotate``) e il formato
  della pagina non viene mai ritagliato.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

MAX_SKEW_DEG = 7.0          # inclinazione massima corretta (gradi)
MIN_SKEW_DEG = 0.05         # sotto questa soglia l'immagine non viene ruotata

_COARSE_WIDTH = 800         # lato orizzontale per la stima grossolana
_FINE_WIDTH = 1600          # lato orizzontale per la stima fine
_COARSE_STEP = 0.2
_FINE_RANGE = 0.3
_FINE_STEP = 0.02


# --------------------------------------------------------------------------
# Utilita' comuni (usate anche da grid.py e ink.py)
# --------------------------------------------------------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    """Converte un'immagine BGR/BGRA/grigia in scala di grigi ``uint8``."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Immagine vuota o non valida.")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Formato d'immagine non supportato: forma {img.shape}.")


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Converte un'immagine grigia/BGRA in BGR ``uint8`` (copia se necessario)."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0 or img.ndim not in (2, 3):
        raise ValueError("Immagine vuota o non valida.")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def binarize(gray: np.ndarray, c: int = 15) -> np.ndarray:
    """Soglia adattiva: inchiostro = 255, sfondo = 0.

    La finestra e' proporzionale alla larghezza dell'immagine, cosi' il
    risultato non dipende dalla risoluzione di scansione e le aree scure
    uniformi (bordi neri dello scanner) non diventano "inchiostro".
    """
    h, w = gray.shape[:2]
    block = max(15, int(round(max(h, w) / 70.0)) | 1)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, c
    )


def resize_to_width(img: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """Ridimensiona (solo riduzione) alla larghezza indicata. Restituisce (img, scala)."""
    h, w = img.shape[:2]
    if w <= width:
        return img, 1.0
    scale = width / float(w)
    out = cv2.resize(img, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return out, scale


def rotate_right_angle(img: np.ndarray, degrees_cw: int) -> np.ndarray:
    """Rotazione esatta di 0/90/180/270 gradi in senso orario."""
    k = int(degrees_cw) % 360
    if k == 0:
        return img
    if k == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if k == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if k == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("Sono ammesse solo rotazioni multiple di 90 gradi.")


def rotate_small_angle(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Ruota di ``angle_deg`` (antiorario positivo) attorno al centro, stesso formato,
    riempiendo gli angoli scoperti di bianco."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    border = 255 if img.ndim == 2 else (255,) * img.shape[2]
    return cv2.warpAffine(
        img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border
    )


# --------------------------------------------------------------------------
# Stima dell'inclinazione
# --------------------------------------------------------------------------

def _profile_scores(ys: np.ndarray, xs: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Nitidezza (somma dei quadrati) del profilo di proiezione orizzontale per
    ogni angolo candidato. Un contenuto ruotato di +a (antiorario) ha righe
    y + x*tan(a) = costante."""
    if ys.size == 0:
        return np.zeros(len(angles))
    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)
    span = float(xs_f.max()) if xs_f.size else 0.0
    scores = np.empty(len(angles))
    for i, a in enumerate(angles):
        t = math.tan(math.radians(float(a)))
        proj = ys_f + xs_f * t
        offset = span * abs(t) + 1.0
        idx = np.round(proj + offset).astype(np.int64)
        hist = np.bincount(idx).astype(np.float64)
        scores[i] = float(np.dot(hist, hist))
    return scores


def _subsample(ys: np.ndarray, xs: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    if ys.size <= limit:
        return ys, xs
    step = int(math.ceil(ys.size / limit))
    return ys[::step], xs[::step]


def _coarse_skew(bw_small: np.ndarray) -> tuple[float, float]:
    """Ricerca grossolana su +-MAX_SKEW_DEG. Restituisce (angolo, guadagno rispetto a 0 gradi)."""
    ys, xs = np.nonzero(bw_small)
    if ys.size < 200:
        return 0.0, 0.0
    ys, xs = _subsample(ys, xs, 250_000)
    angles = np.arange(-MAX_SKEW_DEG, MAX_SKEW_DEG + 1e-9, _COARSE_STEP)
    scores = _profile_scores(ys, xs, angles)
    best = int(np.argmax(scores))
    zero = scores[int(np.argmin(np.abs(angles)))]
    gain = float(scores[best] / zero) if zero > 0 else 0.0
    # interpolazione parabolica attorno al massimo
    a = float(angles[best])
    if 0 < best < len(angles) - 1:
        s0, s1, s2 = scores[best - 1], scores[best], scores[best + 1]
        den = s0 - 2 * s1 + s2
        if den < 0:
            a += 0.5 * _COARSE_STEP * float(s0 - s2) / float(den)
    return a, gain


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = values[order]
    cw = np.cumsum(weights[order])
    return float(v[int(np.searchsorted(cw, cw[-1] / 2.0))])


def _fine_skew(gray: np.ndarray, coarse: float) -> float | None:
    """Raffina l'angolo usando solo le linee orizzontali lunghe (righe della
    tabella, sottolineature dell'intestazione): firme e scrittura non incidono.

    Ogni linea lunga viene interpolata con una retta; si prende la mediana
    (pesata sulla lunghezza) delle inclinazioni, robusta alle scansioni con
    il foglio leggermente incurvato (righe alte e basse inclinate in modo diverso).
    """
    small, _ = resize_to_width(gray, _FINE_WIDTH)
    if abs(coarse) >= MIN_SKEW_DEG:
        small = rotate_small_angle(small, -coarse)
    bw = binarize(small)
    h, w = bw.shape
    klen = max(15, w // 30)
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    angles: list[float] = []
    weights: list[float] = []
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        if cw < 0.15 * w or ch > 0.04 * h:
            continue
        ys, xs = np.nonzero(labels[y:y + ch, x:x + cw] == i)
        if xs.size < cw:
            continue
        slope = float(np.polyfit(xs.astype(np.float64), ys.astype(np.float64), 1)[0])
        angles.append(-math.degrees(math.atan(slope)))
        weights.append(float(cw))
    if len(angles) >= 3:
        residual = _weighted_median(np.asarray(angles), np.asarray(weights))
        if abs(residual) <= 1.0:
            return coarse + residual
    # Ripiego: nitidezza del profilo delle sole linee orizzontali.
    ys, xs = np.nonzero(lines)
    if ys.size < w * 0.5:          # meno di mezza riga di tabella: nessuna stima affidabile
        return None
    ys, xs = _subsample(ys, xs, 400_000)
    grid = np.arange(-_FINE_RANGE, _FINE_RANGE + 1e-9, _FINE_STEP)
    scores = _profile_scores(ys, xs, grid)
    best = int(np.argmax(scores))
    a = float(grid[best])
    if 0 < best < len(grid) - 1:
        s0, s1, s2 = scores[best - 1], scores[best], scores[best + 1]
        den = s0 - 2 * s1 + s2
        if den < 0:
            a += 0.5 * _FINE_STEP * float(s0 - s2) / float(den)
    return coarse + a


def estimate_skew(img: np.ndarray) -> float:
    """Stima l'inclinazione della pagina (gradi, antiorario positivo) entro +-7 gradi.
    Restituisce 0.0 se non ci sono linee sufficienti per una stima affidabile."""
    gray = to_gray(img)
    small, _ = resize_to_width(gray, _COARSE_WIDTH)
    bw = binarize(small)
    coarse, gain = _coarse_skew(bw)
    if gain < 1.0 + 1e-6:
        coarse = 0.0
    fine = _fine_skew(gray, coarse)
    angle = coarse if fine is None else fine
    if fine is None and gain < 1.02:
        return 0.0
    return float(max(-MAX_SKEW_DEG, min(MAX_SKEW_DEG, angle)))


def deskew(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Raddrizza la pagina. Restituisce (immagine raddrizzata, angolo stimato in gradi).

    L'immagine mantiene il formato originale; gli angoli scoperti sono bianchi.
    Se l'inclinazione e' trascurabile l'immagine viene restituita invariata.
    """
    try:
        angle = estimate_skew(img)
    except Exception:  # noqa: BLE001 - il raddrizzamento non deve mai bloccare l'elaborazione
        log.exception("Stima dell'inclinazione non riuscita")
        return img, 0.0
    if abs(angle) < MIN_SKEW_DEG:
        return img, angle
    return rotate_small_angle(img, -angle), angle


# --------------------------------------------------------------------------
# Orientamento (90/180/270 gradi)
# --------------------------------------------------------------------------

def _peaks(profile: np.ndarray, threshold: float, merge: int) -> list[int]:
    """Centri delle zone del profilo sopra soglia (zone vicine unite)."""
    idx = np.nonzero(profile >= threshold)[0]
    if idx.size == 0:
        return []
    out: list[int] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > merge:
            seg = profile[start:prev + 1]
            out.append(start + int(np.argmax(seg)))
            start = i
        prev = i
    seg = profile[start:prev + 1]
    out.append(start + int(np.argmax(seg)))
    return out


def _long_lines(bw: np.ndarray, axis: int, min_cover: float) -> list[int]:
    """Posizioni delle linee lunghe orizzontali (axis=0) o verticali (axis=1)."""
    h, w = bw.shape
    if axis == 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 40), 1))
        m = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
        prof = (m > 0).sum(axis=1).astype(np.float64)
        prof = np.convolve(prof, np.ones(3), mode="same")
        return _peaks(prof, min_cover * w, merge=max(2, h // 300))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(9, h // 40)))
    m = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    prof = (m > 0).sum(axis=0).astype(np.float64)
    prof = np.convolve(prof, np.ones(3), mode="same")
    return _peaks(prof, min_cover * h, merge=max(2, w // 300))


def _table_chain(ys: list[int]) -> tuple[int, int, int] | None:
    """Sequenza piu' lunga di linee quasi equidistanti (le righe della tabella).
    Restituisce (inizio, fine, numero di linee)."""
    if len(ys) < 6:
        return None
    ys = sorted(ys)
    diffs = np.diff(ys)
    med = float(np.median(diffs))
    if med <= 0:
        return None
    best: tuple[int, int, int] | None = None
    start = 0
    for i in range(1, len(ys) + 1):
        if i == len(ys) or ys[i] - ys[i - 1] > 1.7 * med:
            n = i - start
            if best is None or n > best[2]:
                best = (ys[start], ys[i - 1], n)
            start = i
    return best


def _upright_votes(bw: np.ndarray) -> tuple[int, int]:
    """Indizi di verso su un'immagine binaria con le righe della tabella
    orizzontali: (margini, colonne), ciascuno +1 = diritto, -1 = capovolto, 0 = incerto."""
    h, w = bw.shape
    rows = _long_lines(bw, 0, 0.4)
    chain = _table_chain(rows)
    if chain is None or chain[2] < 12:
        return 0, 0
    t0, t1, _ = chain
    margin_vote = col_vote = 0
    top_margin = t0 / h
    bottom_margin = (h - t1) / h
    # Il modulo ha l'intestazione (testo) sopra la tabella e poco spazio sotto.
    if top_margin > bottom_margin + 0.04:
        margin_vote = 1
    elif bottom_margin > top_margin + 0.04:
        margin_vote = -1
    # Colonna "Giorno" stretta a sinistra, colonne "Firma"/"Note" larghe a destra.
    band = bw[t0:t1 + 1]
    if band.shape[0] > 20:
        cols = _long_lines(band, 1, 0.4)
        if len(cols) >= 5:
            cols = sorted(cols)
            gaps = np.diff(cols)
            gaps = gaps[gaps > w * 0.01]
            if gaps.size >= 4:
                left = float(gaps[0])
                right = float(gaps[-1])
                if left < 0.6 * right:
                    col_vote = 1
                elif right < 0.6 * left:
                    col_vote = -1
    return margin_vote, col_vote


def _resize_long(gray: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    long_side = max(h, w)
    target = int(_COARSE_WIDTH * 1.414)
    if long_side <= target:
        return gray, 1.0
    s = target / float(long_side)
    return cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA), s


def detect_orientation(img: np.ndarray) -> int:
    """Rotazione (gradi in senso orario: 0, 90, 180, 270) che porta il modulo in
    verticale e nel verso di lettura. Restituisce 0 se gli indizi non sono chiari."""
    gray = to_gray(img)
    small, _ = _resize_long(gray)
    bw = binarize(small)

    # Direzione delle righe della tabella: il modulo ha ~34 linee orizzontali
    # lunghe (righe dei giorni) e ~11 verticali (colonne).
    ang_h, _ = _coarse_skew(bw)
    bw_h = rotate_small_angle(bw, -ang_h) if abs(ang_h) >= MIN_SKEW_DEG else bw
    n_h = len(_long_lines(_fix_binary(bw_h), 0, 0.35))
    n_v = 0
    if n_h < 20:   # con le righe gia' orizzontali l'analisi verticale e' superflua
        bw_t = np.ascontiguousarray(bw.T)
        ang_v, _ = _coarse_skew(bw_t)
        bw_v = rotate_small_angle(bw_t, -ang_v) if abs(ang_v) >= MIN_SKEW_DEG else bw_t
        n_v = len(_long_lines(_fix_binary(bw_v), 0, 0.35))

    if n_v >= 15 and n_v >= 1.6 * max(n_h, 1):
        # Righe verticali: ruota di 90 gradi e decide il verso.
        cand = rotate_right_angle(bw, 90)
        ang, _ = _coarse_skew(cand)
        if abs(ang) >= MIN_SKEW_DEG:
            cand = _fix_binary(rotate_small_angle(cand, -ang))
        # il verso va scelto comunque: basta un indizio non contraddetto
        margin_vote, col_vote = _upright_votes(cand)
        return 270 if margin_vote + col_vote < 0 else 90

    if n_h < 12:
        return 0
    # Capovolgimento solo con l'indizio piu' affidabile (colonne) non smentito
    # dai margini: una pagina diritta non deve mai essere girata per errore
    # (es. scansione con l'intestazione tagliata).
    margin_vote, col_vote = _upright_votes(_fix_binary(bw_h))
    return 180 if col_vote < 0 and margin_vote <= 0 else 0


def _fix_binary(bw: np.ndarray) -> np.ndarray:
    """Dopo una rotazione interpolata riporta l'immagine a 0/255."""
    return np.where(bw > 127, 255, 0).astype(np.uint8)


# --------------------------------------------------------------------------
# Prospettiva (foto scattate di sbieco)
# --------------------------------------------------------------------------

PERSPECTIVE_MIN_DEV = 0.005   # scarto minimo dei vertici (frazione del lato lungo della tabella)
_PERSP_WIDTH = 1400           # lato orizzontale per la ricerca dei bordi della tabella


@dataclass
class _FitLine:
    """Retta ``b = m * a + q`` (orizzontale: a = x, b = y; verticale: a = y, b = x)."""

    m: float
    q: float
    lo: int                   # estensione lungo la linea
    hi: int

    def at(self, a: float) -> float:
        return self.m * a + self.q


def _fit_line(along: np.ndarray, across: np.ndarray) -> tuple[float, float] | None:
    """Retta robusta per i pixel di una linea: centro trasversale per ogni
    posizione, scartando le posizioni toccate da altri tratti (firme che la
    attraversano: spessore anomalo) e poi i punti lontani dalla prima stima."""
    a0 = int(along.min())
    idx = (along - a0).astype(np.intp)
    counts = np.bincount(idx)
    sums = np.bincount(idx, weights=across.astype(np.float64))
    pos = np.nonzero(counts)[0]
    thick = counts[pos]
    keep = thick <= 2 * float(np.median(thick)) + 1
    pos, mid = pos[keep], sums[pos][keep] / thick[keep]
    if pos.size < 10:
        return None
    x = (pos + a0).astype(np.float64)
    m, q = np.polyfit(x, mid, 1)
    res = np.abs(mid - (m * x + q))
    keep = res <= max(1.5, 2.5 * float(np.median(res)))
    if int(keep.sum()) >= 10:
        m, q = np.polyfit(x[keep], mid[keep], 1)
    return float(m), float(q)


def _line_components(mask: np.ndarray, horizontal: bool, min_len: float) -> list[_FitLine]:
    """Linee lunghe della maschera morfologica, ciascuna interpolata con una retta."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: list[_FitLine] = []
    for i in range(1, n):
        x, y, cw, ch, _area = (int(v) for v in stats[i])
        length = cw if horizontal else ch
        if length < min_len:
            continue
        ys, xs = np.nonzero(labels[y:y + ch, x:x + cw] == i)
        fit = _fit_line(xs + x, ys + y) if horizontal else _fit_line(ys + y, xs + x)
        if fit is None:
            continue
        lo, hi = (x, x + cw) if horizontal else (y, y + ch)
        out.append(_FitLine(fit[0], fit[1], lo, hi))
    return out


def _intersect(hor: _FitLine, ver: _FitLine) -> tuple[float, float]:
    """Punto d'incontro di una retta orizzontale (y = m x + q) e di una verticale (x = m y + q)."""
    den = 1.0 - ver.m * hor.m
    x = (ver.m * hor.q + ver.q) / den
    return x, hor.at(x)


def table_quad(img: np.ndarray) -> np.ndarray | None:
    """Vertici (alto-sinistra, alto-destra, basso-destra, basso-sinistra) dei
    bordi esterni della tabella, in pixel di ``img``; ``None`` se la tabella non
    si individua con sicurezza.

    I bordi sono le rette interpolate sulle linee lunghe della scansione: le due
    verticali esterne e la prima e l'ultima linea orizzontale che le congiungono
    (di norma il bordo superiore dell'intestazione e il fondo della riga dei
    totali). Le linee possono essere un po' inclinate (prospettiva); il bordo del
    foglio sullo sfondo di una foto non viene scambiato per la tabella perche' le
    righe della tabella non lo raggiungono."""
    gray = to_gray(img)
    small, scale = resize_to_width(gray, _PERSP_WIDTH)
    h, w = small.shape[:2]
    if h < 200 or w < 150:
        return None
    bw = binarize(small)
    # una leggera dilatazione trasversale tiene unite le linee non perfettamente orizzontali/verticali
    hmask = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    hmask = cv2.morphologyEx(hmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 30), 1)))
    hmask = cv2.morphologyEx(hmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 100), 1)))
    vmask = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    vmask = cv2.morphologyEx(vmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 40))))
    vmask = cv2.morphologyEx(vmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, h // 140))))

    hlines = _line_components(hmask, True, 0.35 * w)
    if len(hlines) < 10:
        return None
    # estremi delle righe della tabella: le righe-giorno sono la maggioranza
    x_left = float(np.median([ln.lo for ln in hlines]))
    x_right = float(np.median([ln.hi for ln in hlines]))
    if x_right - x_left < 0.35 * w:
        return None
    mid_x = 0.5 * (x_left + x_right)
    ys = sorted(ln.at(mid_x) for ln in hlines)
    vlines = _line_components(vmask, False, 0.25 * (ys[-1] - ys[0]))
    tol = 0.025 * w

    def border(x_target: float) -> _FitLine | None:
        cands = []
        for ln in vlines:
            mid_y = 0.5 * (ln.lo + ln.hi)
            d = abs(ln.at(mid_y) - x_target)
            if d <= tol:
                cands.append((d, ln))
        return min(cands, key=lambda c: c[0])[1] if cands else None

    left, right = border(x_left), border(x_right)
    if left is None or right is None:
        return None
    y_lo = max(left.lo, right.lo) - tol
    y_hi = min(left.hi, right.hi) + tol
    rows = []
    for ln in hlines:
        y_l = ln.at(left.at(ln.at(x_left)))
        y_r = ln.at(right.at(ln.at(x_right)))
        if not (y_lo <= y_l <= y_hi and y_lo <= y_r <= y_hi):
            continue
        # la riga deve congiungere i due bordi (non il bordo del foglio, piu' esterno)
        if abs(ln.lo - left.at(y_l)) <= tol and abs(ln.hi - right.at(y_r)) <= tol:
            rows.append(ln)
    if len(rows) < 10:
        return None
    rows.sort(key=lambda ln: ln.at(mid_x))
    top, bottom = rows[0], rows[-1]
    quad = np.array([_intersect(top, left), _intersect(top, right),
                     _intersect(bottom, right), _intersect(bottom, left)], dtype=np.float64)
    if not np.all(np.isfinite(quad)):
        return None
    tw = 0.5 * (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3]))
    th = 0.5 * (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1]))
    if tw < 0.35 * w or th < 0.3 * h:
        return None
    # solo prospettive moderate: lati opposti simili, angoli vicini a 90 gradi
    for a, b in ((np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])),
                 (np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))):
        if min(a, b) < 0.75 * max(a, b):
            return None
    for i in range(4):
        u = quad[(i + 1) % 4] - quad[i]
        v = quad[(i - 1) % 4] - quad[i]
        cos = float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9))
        if abs(cos) > math.sin(math.radians(20)):
            return None
    return quad / scale


def _rectify(quad: np.ndarray, min_dev: float) -> np.ndarray | None:
    """Omografia dal quadrilatero ``quad`` al rettangolo con i lati orizzontali e
    verticali (stesso centro, lati medi); ``None`` se i vertici si discostano
    dal rettangolo meno di ``min_dev`` (frazione del lato lungo)."""
    tw = 0.5 * (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3]))
    th = 0.5 * (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1]))
    cx, cy = quad.mean(axis=0)
    rect = np.array([[cx - tw / 2, cy - th / 2], [cx + tw / 2, cy - th / 2],
                     [cx + tw / 2, cy + th / 2], [cx - tw / 2, cy + th / 2]], dtype=np.float64)
    dev = float(np.linalg.norm(quad - rect, axis=1).max()) / max(tw, th)
    if dev < min_dev:
        return None
    log.debug("Prospettiva della tabella: scarto dei vertici %.1f%%", 100 * dev)
    return cv2.getPerspectiveTransform(quad.astype(np.float32), rect.astype(np.float32))


def perspective_matrix(img: np.ndarray, min_dev: float = PERSPECTIVE_MIN_DEV) -> np.ndarray | None:
    """Omografia 3x3 che porta i bordi della tabella su un rettangolo con i lati
    orizzontali e verticali (stesso centro, lati medi del quadrilatero). ``None``
    se la tabella non si individua o se e' gia' rettangolare: i vertici si
    discostano meno di ``min_dev`` (frazione del lato lungo) dal rettangolo."""
    quad = table_quad(img)
    return None if quad is None else _rectify(quad, min_dev)


def _warp(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Applica l'omografia mantenendo il formato. Le zone scoperte ripetono il
    bordo dell'immagine: un riempimento bianco accanto allo sfondo scuro di una
    foto formerebbe una lunga "linea" scambiabile per una riga della tabella."""
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def normalize_page(img: np.ndarray) -> np.ndarray:
    """Porta la pagina in verticale e nel verso giusto, la raddrizza e, se la
    tabella e' deformata dalla prospettiva (foto scattata di sbieco), la riporta
    rettangolare.

    Restituisce sempre un'immagine BGR ``uint8``. Non ritaglia: il formato resta
    quello della pagina (scambiato se ruotata di 90 gradi). In caso di dubbio
    l'immagine non viene ruotata ne' deformata. Solleva ``ValueError`` solo per
    immagini non valide.
    """
    img = to_bgr(img)
    try:
        rot = detect_orientation(img)
    except Exception:  # noqa: BLE001
        log.exception("Rilevamento dell'orientamento non riuscito")
        rot = 0
    if rot:
        log.info("Pagina ruotata di %d gradi", rot)
        img = rotate_right_angle(img, rot)
    out, angle = deskew(img)
    if abs(angle) >= MIN_SKEW_DEG:
        log.debug("Inclinazione corretta: %.2f gradi", angle)
    try:
        matrix = None
        quad = table_quad(out)
        if quad is not None:
            matrix = _rectify(quad, PERSPECTIVE_MIN_DEV)
            if matrix is not None and out is not img:
                # rotazione e prospettiva in un'unica interpolazione
                h, w = img.shape[:2]
                rot_m = np.vstack([cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0), [0.0, 0.0, 1.0]])
                matrix = matrix @ rot_m
        elif out is not img:
            # una prospettiva forte puo' ingannare la stima dell'inclinazione:
            # si riprova sulla pagina non ruotata
            quad = table_quad(img)
            if quad is not None:
                matrix = _rectify(quad, PERSPECTIVE_MIN_DEV)
    except Exception:  # noqa: BLE001 - la correzione non deve mai bloccare l'elaborazione
        log.exception("Correzione della prospettiva non riuscita")
        matrix = None
    if matrix is not None:
        log.info("Prospettiva della pagina corretta")
        out = _warp(img, matrix)
    return np.ascontiguousarray(out)
