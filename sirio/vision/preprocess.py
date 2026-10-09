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


def normalize_page(img: np.ndarray) -> np.ndarray:
    """Porta la pagina in verticale e nel verso giusto, poi la raddrizza.

    Restituisce sempre un'immagine BGR ``uint8``. Non ritaglia: il formato resta
    quello della pagina (scambiato se ruotata di 90 gradi). In caso di dubbio
    l'immagine non viene ruotata. Solleva ``ValueError`` solo per immagini non valide.
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
    return np.ascontiguousarray(out)
