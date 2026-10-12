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

Catena di ``normalize_page``
----------------------------
1. ``detect_orientation``: verso della pagina (0/90/180/270) da tratti
   rettilinei anche inclinati o curvi (foto, fogli incurvati), proporzioni
   delle colonne del modulo (colonna "Giorno" stretta a sinistra), margini e
   testo sopra/sotto la tabella. Si gira solo con un indizio forte.
2. ``deskew`` (+-7 gradi) e prospettiva (``table_quad``: foto di sbieco).
3. ``dewarp``: le linee della tabella seguite come curve (``measure_mesh``);
   se si scostano dalla forma dritta piu' di ``DEWARP_MIN_DEV`` righe
   (foglio incurvato o piegato) la pagina viene rimappata a maglia
   (``cv2.remap``) con rotazione e prospettiva in un'unica interpolazione, e
   il risultato e' tenuto solo se la griglia rilevata non peggiora.
4. ``_confirm_upright``: se la griglia non e' affidabile ma quella della
   pagina girata di 180 gradi si', la pagina era capovolta.
Una scansione gia' dritta esce identica (nessun ricampionamento).
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
    # Moduli stampati chiari (fotocopie sbiadite, stampa a colori, toner
    # scarico): la soglia si adatta al contrasto dell'inchiostro piu' scuro
    # della pagina, cosi' le linee chiare non spariscono. Sulle scansioni
    # normali il limite non interviene.
    step = max(1, int(math.sqrt(h * w / 250_000.0)))
    sample = gray[::step, ::step]
    paper = float(np.percentile(sample, 90))
    ink = float(np.percentile(sample, 0.5))
    c = int(max(4, min(c, round(0.2 * (paper - ink)))))
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


def _resize_long(gray: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    long_side = max(h, w)
    target = int(_COARSE_WIDTH * 1.414)
    if long_side <= target:
        return gray, 1.0
    s = target / float(long_side)
    return cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA), s


# Proporzioni delle colonne del modulo (bordi in frazione della larghezza
# della tabella; copia di ``grid.TEMPLATE_COL_FRACTIONS``, che importa questo modulo).
_FORM_COL_FRACTIONS = (0.0, 0.0548, 0.1306, 0.2059, 0.2807, 0.3561, 0.4280, 0.4914, 0.5553, 0.7555, 1.0)
_ORIENT_ANGLES = (0.0, -7.0, 7.0, -14.0, 14.0)
_COL_TOL = 0.018              # tolleranza sulle posizioni delle colonne (frazione della tabella)


def _oriented_open(bw: np.ndarray, length: int, vertical: bool) -> np.ndarray:
    """Tratti rettilinei lunghi almeno ``length`` quasi orizzontali (o verticali),
    anche inclinati fino a ~15 gradi o un po' curvi."""
    out = np.zeros_like(bw)
    for a in _ORIENT_ANGLES:
        out |= cv2.morphologyEx(bw, cv2.MORPH_OPEN, _oriented_kernel(length, a + (90.0 if vertical else 0.0)))
    return out


def _skew_of_mask(hm: np.ndarray, max_deg: float = 15.0) -> float:
    """Inclinazione (gradi, antiorario positivo) delle linee orizzontali di una maschera."""
    ys, xs = np.nonzero(hm)
    if ys.size < 200:
        return 0.0
    ys, xs = _subsample(ys, xs, 150_000)
    angles = np.arange(-max_deg, max_deg + 1e-9, 0.5)
    sc = _profile_scores(ys, xs, angles)
    best = int(np.argmax(sc))
    a = float(angles[best])
    if 0 < best < len(angles) - 1:
        s0, s1, s2 = sc[best - 1], sc[best], sc[best + 1]
        den = s0 - 2 * s1 + s2
        if den < 0:
            a += 0.25 * float(s0 - s2) / float(den)
    return a


def _column_match(vm: np.ndarray, y0: int, y1: int) -> tuple[float, float]:
    """Quanto le linee verticali della tabella (fra ``y0`` e ``y1``) somigliano
    alle colonne del modulo nel verso giusto e in quello speculare (colonna
    "Giorno" stretta a sinistra, "Firma" e "Note" larghe a destra). Restituisce
    (accordo diritto, accordo capovolto), in numero di linee ritrovate."""
    h, w = vm.shape
    if y1 - y0 < 60:
        return 0.0, 0.0
    # tre fasce separate: con la prospettiva o il foglio incurvato le colonne
    # non sono parallele e un unico profilo le confonderebbe
    up = down = 0.0
    for i in range(3):
        a = y0 + (y1 - y0) * i // 3
        b = y0 + (y1 - y0) * (i + 1) // 3
        u, d = _column_match_band(vm[a:b])
        up += u
        down += d
    return up / 3.0, down / 3.0


def _column_match_band(band: np.ndarray) -> tuple[float, float]:
    h, w = band.shape
    # allargamento orizzontale: una colonna un po' inclinata (prospettiva)
    # conta comunque in ogni riga della fascia
    k = max(3, int(0.03 * h)) | 1
    wide = cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    prof = (wide > 0).sum(axis=0).astype(np.float64)
    if prof.max() <= 0:
        return 0.0, 0.0
    cand = _peaks(prof, 0.3 * h, merge=max(2, w // 120))
    if len(cand) < 5:
        return 0.0, 0.0
    # soglia relativa alle linee tipiche, non alla piu' marcata (bordo del foglio)
    ref = float(np.median(sorted(prof[cand])[-12:]))
    pk = [p for p in cand if prof[p] >= 0.45 * ref]
    if len(pk) < 5:
        return 0.0, 0.0
    xs = np.array(pk, dtype=np.float64)
    st = np.minimum(1.0, prof[pk] / ref)
    fr = np.array(_FORM_COL_FRACTIONS)

    def best(fracs: np.ndarray) -> float:
        out = 0.0
        for i in range(xs.size):
            for j in range(i + 1, xs.size):
                a, b = xs[i], xs[j]
                if b - a < 0.4 * w:
                    continue
                pred = a + fracs * (b - a)
                d = np.abs(xs[None, :] - pred[:, None])
                near = d.argmin(axis=1)
                ok = d[np.arange(fracs.size), near] <= _COL_TOL * (b - a)
                out = max(out, float((ok * np.minimum(1.0, 0.5 + st[near])).sum()))
        return out

    return best(fr), best(1.0 - fr[::-1])


def _upright_cues(bw: np.ndarray) -> tuple[float, int, int]:
    """Indizi di verso su un'immagine binaria con le righe della tabella quasi
    orizzontali: (colonne, margini, testo). ``colonne`` e' la differenza fra
    le linee verticali ritrovate nelle proporzioni del modulo diritto e in
    quelle del modulo capovolto (positiva = diritto, 11 = tutte); margini e
    testo valgono +1 = diritto, -1 = capovolto, 0 = incerto. Funziona anche
    con il foglio incurvato, inclinato fino a ~15 gradi o fotografato con lo
    sfondo attorno."""
    h, w = bw.shape
    L = max(15, max(h, w) // 40)
    vm = _oriented_open(bw, L, vertical=True)
    # inclinazione stimata sulle linee verticali (il profilo delle colonne e'
    # cio' che conta; con il foglio incurvato le orizzontali sono ad arco)
    ang = -_skew_of_mask(np.ascontiguousarray(vm.T))
    if abs(ang) >= 0.5:
        bw = _fix_binary(rotate_small_angle(bw, -ang))
        vm = _oriented_open(bw, L, vertical=True)
    hm = _oriented_open(bw, L, vertical=False)
    # estensione verticale della tabella: dove stanno le linee verticali
    vrow = (vm > 0).sum(axis=1).astype(np.float64)
    if vrow.sum() < 5 * h:
        return 0, 0, 0
    cum = np.cumsum(vrow) / vrow.sum()
    y0 = int(np.searchsorted(cum, 0.03))
    y1 = int(np.searchsorted(cum, 0.97))
    if y1 - y0 < 0.3 * h:
        return 0, 0, 0
    s_up, s_down = _column_match(vm, y0, y1)
    col = (s_up - s_down) if max(s_up, s_down) >= 5 else 0.0
    top, bottom = y0 / h, (h - y1) / h
    margin = 1 if top > bottom + 0.04 else (-1 if bottom > top + 0.04 else 0)
    # testo stampato e scritto sopra la tabella (intestazione) e sotto (pie' di pagina)
    text = bw.copy()
    text[hm > 0] = 0
    text[vm > 0] = 0
    band = int(0.1 * h)
    above = float(text[max(0, y0 - band):y0].mean()) if y0 > 0 else 0.0
    below = float(text[y1:min(h, y1 + band)].mean()) if y1 < h else 0.0
    txt = 1 if above > 1.8 * below + 1e-3 else (-1 if below > 1.8 * above + 1e-3 else 0)
    return col, margin, txt


def detect_orientation(img: np.ndarray) -> int:
    """Rotazione (gradi in senso orario: 0, 90, 180, 270) che porta il modulo in
    verticale e nel verso di lettura. Restituisce 0 se gli indizi non sono chiari.

    1. Direzione delle righe: la tabella ha ~34 linee orizzontali lunghe e ~11
       verticali; si confronta la quantita' di tratti rettilinei lunghi
       orizzontali e verticali (anche inclinati o curvi: foto, fogli incurvati).
    2. Verso (0/180 o 90/270), sulla pagina con le righe orizzontali:
       colonne (la colonna stretta "Giorno" a sinistra, "Firma" e "Note" larghe
       a destra), margini (intestazione sopra la tabella, poco spazio sotto) e
       testo (molto sopra la tabella, poco sotto). Una pagina e' capovolta
       solo con un indizio forte non smentito dagli altri: una pagina diritta
       non deve mai essere girata per errore."""
    gray = to_gray(img)
    small, _ = _resize_long(gray)
    bw = binarize(small)
    h, w = bw.shape
    L = max(15, max(h, w) // 40)
    h_mass = float((_oriented_open(bw, L, vertical=False) > 0).sum())
    v_mass = float((_oriented_open(bw, L, vertical=True) > 0).sum())
    if max(h_mass, v_mass) < 3.0 * max(h, w):
        return 0                                    # nessuna tabella riconoscibile
    if v_mass > 1.3 * h_mass:
        cand = np.ascontiguousarray(rotate_right_angle(bw, 90))
        col, margin, txt = _upright_cues(cand)
        vote = _verso(col, margin, txt)
        return 270 if vote < 0 else 90
    if h_mass < 1.3 * v_mass:
        return 0
    col, margin, txt = _upright_cues(bw)
    return 180 if _verso(col, margin, txt) < 0 and _flip_is_safe(col, margin, txt) else 0


ORIENT_COL_STRONG = 3.0     # differenza di colonne ritrovate che da sola decide il verso


def _verso(col: float, margin: int, txt: int) -> float:
    """Voto complessivo per il verso (positivo = diritto)."""
    c = 0.0 if abs(col) < 1.5 else col
    return 2.0 * c + margin + 0.5 * txt


def _flip_is_safe(col: float, margin: int, txt: int) -> bool:
    """Capovolgere una pagina richiede un indizio forte: le colonne nettamente
    speculari, oppure colonne speculari (anche poco) non smentite dai
    margini, oppure - senza indizio dalle colonne - margini e testo concordi."""
    if col <= -ORIENT_COL_STRONG:
        return True
    if col <= -1.5:
        return margin <= 0
    if abs(col) < 1.5:
        return margin < 0 and txt < 0
    return False


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


# --------------------------------------------------------------------------
# Fogli incurvati o piegati (foto): raddrizzamento a maglia ("dewarp")
# --------------------------------------------------------------------------
#
# Le linee della tabella vengono seguite come CURVE: l'immagine e' divisa in
# strisce strette (verticali per le linee orizzontali, orizzontali per le
# verticali); in ogni striscia i picchi del profilo della maschera delle linee
# sono le posizioni locali delle linee, collegati da una striscia all'altra con
# una previsione che usa la pendenza della linea e lo spostamento "di consenso"
# delle linee vicine (le linee del modulo sono quasi parallele: una piega o una
# curvatura le sposta tutte insieme). Ogni linea seguita fornisce campioni dello
# spostamento verticale (orizzontali) o orizzontale (verticali) fra la pagina
# fotografata e quella ideale, in cui la linea e' dritta alla sua posizione
# mediana. I due campi di spostamento, interpolati fra le linee ed estesi
# costanti oltre la tabella, definiscono una mappa a maglia applicata con
# ``cv2.remap`` in un'unica interpolazione (insieme a rotazione e prospettiva).

DEWARP_MIN_DEV = 0.15         # curvatura minima (frazione dell'altezza di riga) per applicare il dewarp
DEWARP_RETRY_DEV = 0.04       # soglia del secondo tentativo, se la griglia non e' affidabile
_DEWARP_WIDTH = 1600          # larghezza di lavoro per il tracciamento delle linee
_MAP_STEP = 4                 # passo (pixel di lavoro) della mappa a maglia prima dell'ingrandimento


@dataclass
class _Traced:
    """Linee seguite su una maschera: ``pos[i, k]`` posizione trasversale della
    linea ``i`` nella striscia ``k`` (NaN = non misurata), ``centers[k]``
    coordinata della striscia lungo le linee."""

    pos: np.ndarray
    centers: np.ndarray
    spacing: float

    @property
    def n(self) -> int:
        return int(self.pos.shape[0])


def _oriented_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Elemento strutturante: segmento lungo ``length`` inclinato di ``angle_deg``
    rispetto all'orizzontale."""
    r = length // 2
    k = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
    dx = r * math.cos(math.radians(angle_deg))
    dy = r * math.sin(math.radians(angle_deg))
    cv2.line(k, (int(round(r - dx)), int(round(r - dy))), (int(round(r + dx)), int(round(r + dy))), 1, 1)
    ys, xs = np.nonzero(k)
    return k[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


_TRACE_ANGLES = (0.0, -8.0, 8.0, -16.0, 16.0, -24.0, 24.0)


def _line_masks(gray_small: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maschere delle linee orizzontali e verticali, tolleranti alle linee
    curve o piegate: unione delle aperture morfologiche con segmenti corti
    inclinati fino a ~30 gradi (nelle pieghe del foglio le linee scendono
    ripide per un tratto breve)."""
    bw = binarize(gray_small, c=12)
    h, w = bw.shape
    lh = max(11, w // 60)
    lv = max(11, h // 50)
    hm = np.zeros_like(bw)
    vm = np.zeros_like(bw)
    for a in _TRACE_ANGLES:
        hm |= cv2.morphologyEx(bw, cv2.MORPH_OPEN, _oriented_kernel(lh, a))
        if abs(a) <= 16:
            vm |= cv2.morphologyEx(bw, cv2.MORPH_OPEN, _oriented_kernel(lv, 90.0 + a))
    hm = cv2.dilate(hm, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    vm = cv2.dilate(vm, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    return hm, vm


def _strip_peaks(mask: np.ndarray, sw: int, max_thick: float, lo: int = 0,
                 hi: int | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Per ogni striscia verticale larga ``sw`` (colonne ``lo``..``hi``) le
    ordinate (sub-pixel) delle linee orizzontali che la attraversano."""
    h, w = mask.shape
    hi = w if hi is None else min(w, hi)
    lo = max(0, lo)
    k_n = (hi - lo) // sw
    if k_n <= 0:
        return np.zeros(0), []
    sub = (mask[:, lo:lo + k_n * sw] > 0).reshape(h, k_n, sw).mean(axis=2)
    on = sub >= 0.4
    pad = np.zeros((1, k_n), dtype=bool)
    edges = np.diff(np.vstack([pad, on, pad]).astype(np.int8), axis=0)
    ys = np.arange(h, dtype=np.float64)
    peaks: list[np.ndarray] = []
    for k in range(k_n):
        starts = np.nonzero(edges[:, k] == 1)[0]
        ends = np.nonzero(edges[:, k] == -1)[0]
        out = []
        for a, b in zip(starts, ends):
            segs = [(a, b)]
            if b - a > max_thick:
                # linea attraversata da un tratto di scrittura: resta la parte
                # coperta quasi per intero (la linea), se sottile
                strong = sub[a:b, k] >= 0.85
                e = np.diff(np.concatenate([[0], strong.astype(np.int8), [0]]))
                segs = [(a + s0, a + s1) for s0, s1 in zip(np.nonzero(e == 1)[0], np.nonzero(e == -1)[0])
                        if s1 - s0 <= 0.6 * max_thick]
            for s0, s1 in segs:
                wts = sub[s0:s1, k]
                out.append(float(np.dot(ys[s0:s1], wts) / max(float(wts.sum()), 1e-9)))
        peaks.append(np.asarray(out))
    centers = lo + sw * (np.arange(k_n) + 0.5)
    return centers, peaks


def _consensus_shift(pred: np.ndarray, q: np.ndarray, fixed_pos: np.ndarray, spacing: float) -> np.ndarray:
    """Spostamento comune delle linee vicine fra una striscia e la successiva.

    Per ogni linea si considerano le 4 vicine sopra e sotto e si sceglie lo
    spostamento (entro 0,7 righe) che porta il maggior numero di esse su un
    picco della striscia: una piega del foglio sposta tutte le linee insieme
    anche di mezza riga, cosa che una previsione linea per linea non segue.
    ``fixed_pos`` (NaN = libera) sono le linee gia' misurate nella striscia."""
    n = pred.size
    reach = max(2, int(math.ceil(0.7 * spacing)))
    nb = 2 * reach + 1
    tol = max(1, int(round(0.08 * spacing)))
    free = np.isnan(fixed_pos)
    pts = np.zeros((n, nb), dtype=bool)
    if q.size:
        sh = q[None, :] - pred[:, None]
        ok = (np.abs(sh) <= reach) & free[:, None]
        rows, cols = np.nonzero(ok)
        pts[rows, np.round(sh[rows, cols]).astype(np.intp) + reach] = True
    fx = ~free
    if fx.any():
        c = np.round(fixed_pos[fx] - pred[fx]).astype(np.intp) + reach
        inside = (c >= 0) & (c < nb)
        pts[np.nonzero(fx)[0][inside], c[inside]] = True
    hit = pts.copy()
    for t in range(1, tol + 1):
        hit[:, t:] |= pts[:, :-t]
        hit[:, :-t] |= pts[:, t:]
    cum = np.vstack([np.zeros((1, nb)), np.cumsum(hit, axis=0)])
    lo = np.clip(np.arange(n) - 4, 0, n)
    hi = np.clip(np.arange(n) + 5, 0, n)
    counts = cum[hi] - cum[lo]
    shifts = np.arange(nb, dtype=np.float64) - reach
    best = np.argmax(counts - 1e-3 * np.abs(shifts)[None, :], axis=1)
    out = shifts[best]
    out[counts[np.arange(n), best] < 1] = 0.0
    # raffinamento sub-pixel: mediana degli scarti delle linee vicine concordi
    if q.size:
        tgt = pred + out
        idx = np.clip(np.searchsorted(q, tgt), 1, max(1, q.size - 1)) if q.size > 1 else np.zeros(n, dtype=np.intp)
        left = q[np.clip(idx - 1, 0, q.size - 1)]
        right = q[np.clip(idx, 0, q.size - 1)]
        nearest = np.where(np.abs(left - tgt) <= np.abs(right - tgt), left, right)
        res = np.where(free, nearest - pred, fixed_pos - pred)
    else:
        res = np.where(free, np.nan, fixed_pos - pred)
    res_l = res.tolist()
    out_l = out.tolist()
    refined = list(out_l)
    for a in range(n):
        vals = sorted(v for v in res_l[max(0, a - 4):a + 5] if v == v and abs(v - out_l[a]) <= tol + 1)
        if vals:
            m = len(vals)
            refined[a] = vals[m // 2] if m % 2 else 0.5 * (vals[m // 2 - 1] + vals[m // 2])
    return np.array(refined)


def _track(peaks: list[np.ndarray], seed: int, spacing: float, max_miss: int) -> np.ndarray:
    """Collega i picchi delle strisce in linee (vedi la nota della sezione).

    Quattro passate (dal seme verso destra e verso sinistra, poi da un bordo
    all'altro nei due versi): ogni linea gia' misurata in una striscia resta
    fissa e fornisce lo spostamento di consenso alle vicine; le linee si
    allungano finche' trovano picchi compatibili (al massimo ``max_miss``
    strisce di seguito senza misura). Restituisce la matrice
    ``(linee, strisce)`` delle posizioni misurate (NaN = non misurata)."""
    k_n = len(peaks)
    rows: list[np.ndarray] = []

    def new_track(k: int, y: float) -> None:
        r = np.full(k_n, np.nan)
        r[k] = y
        rows.append(r)

    for y in peaks[seed]:
        new_track(seed, float(y))
    sweeps = ((seed + 1, 1), (seed - 1, -1), (1, 1), (k_n - 2, -1))
    for start, direction in sweeps:
        state: dict[int, list[float]] = {}          # linea -> [posizione, pendenza, mancanze]
        k = start
        while 0 <= k < k_n:
            prev = k - direction
            q = peaks[k]
            ids: list[int] = []
            pred: list[float] = []
            fixed: list[bool] = []
            for i, r in enumerate(rows):
                if not np.isnan(r[prev]):
                    p_prev = float(r[prev])
                    # nessuna pendenza propria: lo spostamento comune delle
                    # linee vicine (consenso) segue curve e pieghe, mentre la
                    # pendenza di una sola linea la porterebbe sui tratti di
                    # scrittura che la attraversano
                    slope = 0.0
                    if i in state and not np.isnan(r[k]):
                        del state[i]
                    elif np.isnan(r[k]):
                        state[i] = [p_prev, slope, 0.0]
                elif i in state:
                    p_prev, slope = state[i][0], state[i][1]
                else:
                    continue
                ids.append(i)
                pred.append(p_prev + slope)
                fixed.append(not np.isnan(r[k]))
            n = len(ids)
            if n == 0:
                for y in q:
                    new_track(k, float(y))
                k += direction
                continue
            order = np.argsort(pred)
            ids = [ids[o] for o in order]
            pr = np.array(pred)[order]
            fx = np.array(fixed)[order]
            gap = np.full(n, 1.5 * spacing)
            if n > 1:
                dpr = np.diff(pr)
                gap[:-1] = np.minimum(gap[:-1], dpr)
                gap[1:] = np.minimum(gap[1:], dpr)
            gap = np.maximum(gap, 0.5 * spacing)
            corr = _consensus_shift(pr, q, np.array([rows[i][k] if f else np.nan for i, f in zip(ids, fx)]),
                                    spacing)
            pr2 = pr + corr
            # due linee sulla stessa posizione: resta quella misurata
            drop = np.zeros(n, dtype=bool)
            for a in range(n - 1):
                if pr2[a + 1] - pr2[a] < 0.25 * spacing:
                    for x, y_ in ((a, a + 1), (a + 1, a)):
                        if not fx[x] and not drop[y_] and (fx[y_] or state[ids[x]][2] >= state[ids[y_]][2]):
                            drop[x] = True
                            break
            taken = np.zeros(q.size, dtype=bool)
            for a in range(n):
                if fx[a] and q.size:
                    b = int(np.argmin(np.abs(q - rows[ids[a]][k])))
                    if abs(q[b] - rows[ids[a]][k]) < 1e-6:
                        taken[b] = True
            matched = np.full(n, -1)
            if q.size:
                pairs = []
                for a in range(n):
                    if fx[a] or drop[a]:
                        continue
                    tol = max(2.0, 0.15 * spacing)
                    lo_i = int(np.searchsorted(q, pr2[a] - tol))
                    hi_i = int(np.searchsorted(q, pr2[a] + tol, side="right"))
                    for b in range(lo_i, hi_i):
                        if not taken[b]:
                            pairs.append((abs(q[b] - pr2[a]), a, b))
                pairs.sort()
                for _dist, a, b in pairs:
                    if matched[a] < 0 and not taken[b]:
                        matched[a] = b
                        taken[b] = True
            for a, i in enumerate(ids):
                if fx[a]:
                    continue
                if drop[a]:
                    del state[i]
                    continue
                st = state[i]
                if matched[a] >= 0:
                    rows[i][k] = float(q[matched[a]])
                    del state[i]                      # ripartira' dalla misura
                else:
                    st[1] *= 0.8
                    st[0] = float(pr2[a])
                    st[2] += 1
                    if st[2] > max_miss:
                        del state[i]
            # linee che iniziano qui (es. righe parziali dell'intestazione)
            for b in np.nonzero(~taken)[0]:
                y = float(q[b])
                if np.min(np.abs(pr2[~drop] - y)) < 0.4 * spacing:
                    continue
                new_track(k, y)
            k += direction
    return np.array(rows) if rows else np.zeros((0, k_n))


def _join_fragments(tracks: list[np.ndarray], spacing: float) -> list[np.ndarray]:
    """Unisce i pezzi della stessa linea interrotta (firme che la attraversano,
    tratti sbiaditi): il pezzo B prosegue il pezzo A se, nel buco fra i due, le
    linee vicine misurate su entrambi i lati si spostano quanto serve per
    passare dalla fine di A all'inizio di B."""
    tracks = [t for t in tracks if int((~np.isnan(t)).sum()) >= 2]
    while True:
        spans = []
        for t in tracks:
            idx = np.nonzero(~np.isnan(t))[0]
            spans.append((int(idx[0]), int(idx[-1])))
        best = None
        for a, ta in enumerate(tracks):
            ea = spans[a][1]
            ya = float(ta[ea])
            for b, tb in enumerate(tracks):
                sb = spans[b][0]
                if a == b or sb <= ea:
                    continue
                yb = float(tb[sb])
                if abs(yb - ya) > 1.2 * spacing:
                    continue
                deltas = [float(t[sb] - t[ea]) for t in tracks
                          if not np.isnan(t[ea]) and not np.isnan(t[sb]) and abs(float(t[ea]) - ya) <= 4 * spacing]
                if not deltas:
                    continue
                err = abs(yb - (ya + float(np.median(deltas))))
                if err <= 0.2 * spacing and (best is None or err < best[0]):
                    best = (err, a, b)
        if best is None:
            return tracks
        _, a, b = best
        merged = np.where(np.isnan(tracks[a]), tracks[b], tracks[a])
        tracks = [t for i, t in enumerate(tracks) if i not in (a, b)] + [merged]


def _clean_tracks(pos: np.ndarray, spacing: float, min_count: int) -> np.ndarray:
    """Scarta i punti anomali (tratti di scrittura sulla linea), unisce i pezzi
    della stessa linea, scarta le linee troppo corte e quelle doppie; ordina
    per posizione mediana."""
    cleaned = []
    for r in pos:
        idx = np.nonzero(~np.isnan(r))[0]
        if idx.size < 2:
            continue
        v = r[idx]
        med = np.array([np.median(v[max(0, i - 3):i + 4]) for i in range(v.size)])
        bad = np.abs(v - med) > max(1.5, 0.12 * spacing)
        r = r.copy()
        r[idx[bad]] = np.nan
        cleaned.append(r)
    keep = [r for r in _join_fragments(cleaned, spacing) if int((~np.isnan(r)).sum()) >= min_count]
    if not keep:
        return np.zeros((0, pos.shape[1]))
    arr = np.array(keep)
    order = np.argsort(np.nanmedian(arr, axis=1))
    arr = arr[order]
    out = [arr[0]]
    for r in arr[1:]:
        prev = out[-1]
        both = ~np.isnan(r) & ~np.isnan(prev)
        if both.sum() >= 3 and float(np.nanmedian(np.abs(r[both] - prev[both]))) < 0.3 * spacing:
            out[-1] = np.where(np.isnan(prev), r, prev)      # stessa linea seguita due volte
        elif both.sum() < 3 and abs(float(np.nanmedian(r)) - float(np.nanmedian(prev))) < 0.3 * spacing:
            out[-1] = np.where(np.isnan(prev), r, prev)
        else:
            out.append(r)
    return np.array(out)


def _dominant_spacing(peaks: list[np.ndarray], max_d: float) -> float | None:
    """Distanza piu' frequente fra linee consecutive (moda dell'istogramma delle
    distanze fra picchi vicini in tutte le strisce): l'altezza di riga."""
    d = np.concatenate([np.diff(p) for p in peaks if p.size > 1] or [np.zeros(0)])
    d = d[(d >= 4) & (d <= max_d)]
    if d.size < 10:
        return None
    hist = np.bincount(np.round(d).astype(np.intp)).astype(np.float64)
    hist = np.convolve(hist, np.ones(5), mode="same")
    mode = float(np.argmax(hist))
    close = d[np.abs(d - mode) <= 0.15 * mode]
    return float(np.median(close)) if close.size else mode


def _trace(mask: np.ndarray, spacing_hint: float | None, lo: int, hi: int, min_len: float) -> _Traced | None:
    """Segue le linee orizzontali di ``mask`` fra le colonne ``lo`` e ``hi``."""
    h, w = mask.shape
    sw = max(6, w // 100)
    max_thick = max(8.0, 0.012 * h)
    centers, peaks = _strip_peaks(mask, sw, max_thick, lo, hi)
    k_n = len(peaks)
    if k_n < 8:
        return None
    counts = np.array([p.size for p in peaks])
    mid = slice(int(0.2 * k_n), max(int(0.2 * k_n) + 1, int(0.8 * k_n)))
    seed = int(mid.start + np.argmax(counts[mid]))
    if counts[seed] < 3:
        return None
    if spacing_hint is None:
        sp = _dominant_spacing(peaks[mid], 0.1 * h)
        if sp is None:
            return None
        spacing = sp
    else:
        spacing = float(spacing_hint)
    pos = _track(peaks, seed, spacing, max_miss=max(4, int(0.3 * k_n)))
    pos = _clean_tracks(pos, spacing, max(4, int(math.ceil(min_len / sw))))
    if pos.shape[0] == 0:
        return None
    return _Traced(pos=pos, centers=centers, spacing=spacing)


def _drop_inconsistent(pos: np.ndarray, target: np.ndarray, tol: float) -> np.ndarray:
    """Indici delle linee coerenti con le vicine. Le linee vere del modulo si
    spostano come quelle accanto (curvatura e pieghe sono lisce o comuni a
    tutte); una "linea" seguita su tratti di scrittura (aste di firme, cifre
    in colonna) se ne discosta: il suo spostamento, previsto interpolando
    quello delle altre linee nella stessa striscia, differisce piu' di ``tol``.
    Si scarta una linea alla volta, la peggiore, finche' tutte sono coerenti."""
    keep = np.arange(pos.shape[0])
    disp = pos - target[:, None]
    while keep.size > 3:
        bad_score = np.zeros(keep.size)
        for a, i in enumerate(keep):
            others = keep[keep != i]
            res = []
            for k in np.nonzero(~np.isnan(disp[i]))[0]:
                col = disp[others, k]
                good = ~np.isnan(col)
                if good.sum() < 2:
                    continue
                t_o = target[others][good]
                # solo le vicine (entro 3 linee per parte) contano
                near = np.argsort(np.abs(t_o - target[i]))[:6]
                pred = float(np.interp(target[i], t_o[near][np.argsort(t_o[near])],
                                       col[good][near][np.argsort(t_o[near])]))
                res.append(abs(disp[i, k] - pred))
            if len(res) >= 3:
                bad_score[a] = float(np.median(res))
        worst = int(np.argmax(bad_score))
        if bad_score[worst] <= tol:
            break
        keep = np.delete(keep, worst)
    return keep


def _fill_field(tr: _Traced, lo_k: int, hi_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Spostamento trasversale di ogni linea in ogni striscia rispetto alla sua
    posizione ideale (mediana nelle strisce ``lo_k``..``hi_k``). I buchi sono
    riempiti interpolando fra le linee vicine (stessa striscia) e, fuori dalla
    zona misurata, ripetendo la striscia piu' vicina. Le linee che, cosi'
    completate, si incrocerebbero con le vicine (linee seguite male) sono
    scartate. Restituisce (posizioni ideali, spostamenti [linee, strisce])."""
    pos = tr.pos
    target = np.nanmedian(pos[:, lo_k:hi_k + 1], axis=1)
    ok = ~np.isnan(target)
    pos, target = pos[ok], target[ok]
    order = np.argsort(target)
    pos, target = pos[order], target[order]
    # fuori dalla tabella (oltre 2 strisce dai bordi) le misure non contano:
    # tratti dell'intestazione o del bordo del foglio
    pos = pos.copy()
    pos[:, :max(0, lo_k - 2)] = np.nan
    pos[:, hi_k + 3:] = np.nan
    keep = _drop_inconsistent(pos, target, max(2.0, 0.12 * tr.spacing))
    pos, target = pos[keep], target[keep]
    while True:
        filled = _fill(pos - target[:, None], target)
        if target.size < 3:
            break
        src = target[:, None] + filled
        gaps = np.diff(src, axis=0)
        need = 0.35 * np.diff(target)[:, None]
        bad = gaps < need
        if not bad.any():
            break
        # la linea con meno misure fra le due che si incrociano piu' spesso
        i = int(np.argmax(bad.sum(axis=1)))
        cov = (~np.isnan(pos)).sum(axis=1)
        j = i if cov[i] <= cov[i + 1] else i + 1
        pos = np.delete(pos, j, axis=0)
        target = np.delete(target, j)
    # leggera levigatura lungo le linee (rumore di misura di un pixel)
    k_n = filled.shape[1]
    if k_n >= 3:
        ker = np.array([1.0, 2.0, 1.0]) / 4.0
        padded = np.pad(filled, ((0, 0), (1, 1)), mode="edge")
        filled = np.stack([np.convolve(r, ker, mode="valid") for r in padded])
    return target, filled


def _fill(disp: np.ndarray, target: np.ndarray) -> np.ndarray:
    n, k_n = disp.shape
    filled = np.full_like(disp, np.nan)
    for k in range(k_n):
        col = disp[:, k]
        good = ~np.isnan(col)
        if good.sum() >= 2:
            filled[:, k] = np.interp(target, target[good], col[good])
        elif good.sum() == 1:
            filled[:, k] = col[good][0]
    have = ~np.isnan(filled[0]) if n else np.zeros(k_n, dtype=bool)
    if not have.any():
        return np.zeros_like(disp)
    idx = np.nonzero(have)[0]
    for k in range(k_n):
        if not have[k]:
            filled[:, k] = filled[:, idx[np.argmin(np.abs(idx - k))]]
    return filled


def _interp_field(centers: np.ndarray, targets: np.ndarray, field: np.ndarray,
                  along: np.ndarray, across: np.ndarray) -> np.ndarray:
    """Interpolazione bilineare del campo ``field[i, k]`` (linea ``i`` con
    posizione ideale ``targets[i]``, striscia ``k`` al centro ``centers[k]``)
    nei punti (``along``, ``across``); costante oltre gli estremi."""
    k = np.interp(along, centers, np.arange(centers.size, dtype=np.float64))
    i = np.interp(across, targets, np.arange(targets.size, dtype=np.float64))
    k0 = np.clip(np.floor(k).astype(np.intp), 0, max(0, centers.size - 2))
    i0 = np.clip(np.floor(i).astype(np.intp), 0, max(0, targets.size - 2))
    fk = np.clip(k - k0, 0.0, 1.0) if centers.size > 1 else np.zeros_like(k)
    fi = np.clip(i - i0, 0.0, 1.0) if targets.size > 1 else np.zeros_like(i)
    k1 = np.minimum(k0 + 1, centers.size - 1)
    i1 = np.minimum(i0 + 1, targets.size - 1)
    return ((1 - fi) * ((1 - fk) * field[i0, k0] + fk * field[i0, k1])
            + fi * ((1 - fk) * field[i1, k0] + fk * field[i1, k1]))


@dataclass
class PageMesh:
    """Maglia delle linee della tabella misurata su una pagina (coordinate di
    lavoro ``scale`` volte quelle della pagina).

    ``row_targets``/``row_field``: posizioni ideali delle linee orizzontali e
    loro spostamento verticale per striscia (``row_centers``);
    ``col_targets``/``col_field``/``col_centers``: idem per le verticali.
    ``deviation``: scostamento tipico delle linee dalla forma dritta, in
    frazioni dell'altezza di riga (0 = tabella perfettamente dritta).
    """

    scale: float
    row_centers: np.ndarray
    row_targets: np.ndarray
    row_field: np.ndarray
    col_centers: np.ndarray
    col_targets: np.ndarray
    col_field: np.ndarray
    period: float
    deviation: float
    table: tuple[float, float, float, float]      # x0, y0, x1, y1 della tabella (coordinate di lavoro)

    def maps(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Mappe ``cv2.remap`` (float32, dimensioni della pagina) che portano la
        pagina misurata su quella con le linee dritte."""
        s = self.scale
        ws, hs = width * s, height * s
        xs = np.arange(0.0, ws + _MAP_STEP, _MAP_STEP)
        ys = np.arange(0.0, hs + _MAP_STEP, _MAP_STEP)
        X, Y = np.meshgrid(xs, ys)
        sx, sy = X.copy(), Y.copy()
        for _ in range(3):
            sy = Y + _interp_field(self.row_centers, self.row_targets, self.row_field, sx.ravel(),
                                   Y.ravel()).reshape(Y.shape)
            sx = X + _interp_field(self.col_centers, self.col_targets, self.col_field, sy.ravel(),
                                   X.ravel()).reshape(X.shape)
        # ingrandimento alla pagina intera: il punto (i, j) della maglia e'
        # il pixel (j * step / s, i * step / s) della pagina
        mx = (sx / s).astype(np.float32)
        my = (sy / s).astype(np.float32)
        gx = (np.arange(width, dtype=np.float32) * s) / _MAP_STEP
        gy = (np.arange(height, dtype=np.float32) * s) / _MAP_STEP
        map_x = cv2.remap(mx, *np.meshgrid(gx, gy), interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
        map_y = cv2.remap(my, *np.meshgrid(gx, gy), interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
        return map_x, map_y


def _spans(tr: _Traced) -> tuple[np.ndarray, np.ndarray]:
    meas = ~np.isnan(tr.pos)
    first = np.array([tr.centers[np.nonzero(m)[0][0]] for m in meas])
    last = np.array([tr.centers[np.nonzero(m)[0][-1]] for m in meas])
    return first, last


def measure_mesh(img: np.ndarray) -> PageMesh | None:
    """Segue le linee orizzontali e verticali della tabella come curve e
    restituisce la maglia (``None`` se la tabella non si individua).

    Estensione orizzontale della tabella: estremi mediani delle linee
    orizzontali piu' lunghe; estensione verticale: estremi mediani delle linee
    verticali lunghe (le colonne esistono solo nella tabella, mentre righe
    lunghe si trovano anche nell'intestazione e sul bordo del foglio)."""
    gray = to_gray(img)
    small, s = resize_to_width(gray, _DEWARP_WIDTH)
    h, w = small.shape[:2]
    if h < 300 or w < 200:
        return None
    hmask, vmask = _line_masks(small)
    rows = _trace(hmask, None, 0, w, 0.3 * w)
    if rows is None or rows.n < 12:
        return None
    first, last = _spans(rows)
    span = last - first
    long_ = span >= 0.7 * float(span.max())
    if long_.sum() < 10:
        return None
    x0, x1 = float(np.median(first[long_])), float(np.median(last[long_]))
    tw = x1 - x0
    if tw < 0.4 * w:
        return None
    # verticali della tabella: maschera trasposta, solo fra i bordi della tabella
    vt = np.ascontiguousarray(vmask.T)
    vt[: max(0, int(x0 - 0.03 * w))] = 0
    vt[min(w, int(x1 + 0.03 * w)):] = 0
    cols = _trace(vt, 0.045 * tw, 0, h, 0.3 * h)
    if cols is None or cols.n < 4:
        return None
    vfirst, vlast = _spans(cols)
    vspan = vlast - vfirst
    vlong = vspan >= 0.7 * float(vspan.max())
    if vlong.sum() < 4:
        return None
    y0, y1 = float(np.median(vfirst[vlong])), float(np.median(vlast[vlong]))
    th = y1 - y0
    if th < 0.3 * h:
        return None
    ymed = np.nanmedian(rows.pos, axis=1)
    in_tab = (ymed >= y0 - 0.03 * th) & (ymed <= y1 + 0.06 * th) & (span >= 0.45 * tw)
    if in_tab.sum() < 10:
        return None
    d = np.diff(np.sort(ymed[in_tab]))
    d = d[d > 0.3 * rows.spacing]
    period = float(np.median(d)) if d.size else rows.spacing
    rows = _Traced(rows.pos[in_tab], rows.centers, rows.spacing)
    lo_k = int(np.searchsorted(rows.centers, x0))
    hi_k = max(lo_k, int(np.searchsorted(rows.centers, x1)) - 1)
    row_t, row_f = _fill_field(rows, lo_k, hi_k)
    xmed = np.nanmedian(cols.pos, axis=1)
    in_cols = (xmed >= x0 - 0.02 * tw) & (xmed <= x1 + 0.02 * tw) & (vspan >= 0.5 * th)
    if in_cols.sum() < 4:
        return None
    cols = _Traced(cols.pos[in_cols], cols.centers, cols.spacing)
    lo_c = int(np.searchsorted(cols.centers, y0))
    hi_c = max(lo_c, int(np.searchsorted(cols.centers, y1)) - 1)
    col_t, col_f = _fill_field(cols, lo_c, hi_c)
    # scostamento dalla forma dritta, solo dentro la tabella
    rk = slice(lo_k, hi_k + 1)
    ck = slice(lo_c, hi_c + 1)
    dev_r = np.abs(row_f[:, rk] - np.median(row_f[:, rk], axis=1, keepdims=True)).max(axis=1)
    dev_c = np.abs(col_f[:, ck] - np.median(col_f[:, ck], axis=1, keepdims=True)).max(axis=1)
    dev_r = float(np.median(dev_r)) / period
    dev_c = float(np.median(dev_c)) / period
    dev = max(dev_r, dev_c)
    log.debug("Maglia: %d righe, %d colonne, curvatura righe %.2f colonne %.2f", row_t.size, col_t.size,
              dev_r, dev_c)
    return PageMesh(scale=s, row_centers=rows.centers, row_targets=row_t, row_field=row_f,
                    col_centers=cols.centers, col_targets=col_t, col_field=col_f, period=period,
                    deviation=dev, table=(x0, y0, x1, y1))


def _grid_reliable(page: np.ndarray) -> bool:
    from sirio.vision.grid import detect_grid  # noqa: PLC0415

    return detect_grid(page).reliable


def _grid_quality(page: np.ndarray) -> tuple[bool, float]:
    """(griglia rilevata, allineamento) della pagina: usato per confermare il
    raddrizzamento a maglia (import pigro: grid.py usa questo modulo)."""
    from sirio.vision.grid import detect_grid  # noqa: PLC0415

    g = detect_grid(page)
    return bool(g.detected), float(g.alignment if g.alignment is not None else (1.0 if g.detected else 0.0))


def dewarp(img: np.ndarray, out: np.ndarray, transform: np.ndarray | None = None,
           min_dev: float = DEWARP_MIN_DEV, verify: bool = True,
           probe: np.ndarray | None = None):  # noqa: ANN201
    """Raddrizza a maglia la pagina ``out`` se le linee della tabella sono
    curve o spezzate (foglio incurvato o piegato).

    ``out`` e' la pagina gia' ruotata/raddrizzata ottenuta da ``img`` con la
    trasformazione proiettiva 3x3 ``transform`` (``img`` -> ``out``; ``None`` =
    identita'): il risultato e' ricampionato direttamente da ``img``, in
    un'unica interpolazione. Restituisce ``None`` se la tabella non si
    individua, se le linee sono gia' dritte (scostamento < ``min_dev`` righe)
    o se, con ``verify``, la griglia della pagina raddrizzata non risulta
    migliore di quella di partenza.

    Con ``probe`` (la versione di ``img`` usata per misurare, es. a contrasto
    aumentato, da cui deriva ``out``) restituisce la coppia (pagina
    raddrizzata, probe raddrizzato) oppure ``None``."""
    mesh = measure_mesh(out)
    if mesh is None:
        return None
    if mesh.deviation < min_dev:
        log.debug("Linee della tabella dritte (scostamento %.2f righe): nessun dewarp", mesh.deviation)
        return None
    h, w = out.shape[:2]
    map_x, map_y = mesh.maps(w, h)
    if transform is not None:
        pts = np.dstack([map_x, map_y]).reshape(-1, 1, 2)
        src = cv2.perspectiveTransform(pts, np.linalg.inv(transform)).reshape(h, w, 2)
        map_x = np.ascontiguousarray(src[:, :, 0], dtype=np.float32)
        map_y = np.ascontiguousarray(src[:, :, 1], dtype=np.float32)
    cand = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cand_probe = cand if probe is None else cv2.remap(probe, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                                      borderMode=cv2.BORDER_REPLICATE)
    if verify:
        det0, al0 = _grid_quality(out)
        det1, al1 = _grid_quality(cand_probe)
        log.info("Dewarp: scostamento %.2f righe; griglia prima %s/%.2f, dopo %s/%.2f",
                 mesh.deviation, det0, al0, det1, al1)
        if not det1 or (det0 and al1 < al0 - 0.05):
            return None
        if min_dev < DEWARP_MIN_DEV and not (al1 > al0 + 0.05 or (det1 and not det0)):
            return None                 # tentativo a soglia bassa: solo se migliora davvero
    return cand if probe is None else (cand, cand_probe)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Copia in scala di grigi con il contrasto locale aumentato (CLAHE): per
    trovare le linee di moduli stampati chiari o di foto in penombra. Serve solo
    a misurare la geometria; la pagina restituita e' sempre ricavata
    dall'originale."""
    gray = to_gray(img)
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)


def _normalize_with(img: np.ndarray, probe: np.ndarray,
                    info: dict | None = None, perspective: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Catena di ``normalize_page``: la geometria (verso, inclinazione,
    prospettiva, maglia) e' misurata su ``probe`` (``img`` stessa o la sua
    versione a contrasto aumentato, stesso formato) e applicata a ``img``.
    Restituisce (pagina normalizzata, probe normalizzato allo stesso modo);
    ``info`` (facoltativo) riceve le correzioni applicate (vedi ``normalize_page_info``);
    con ``perspective=False`` la prospettiva non viene corretta a parte (resta al
    raddrizzamento a maglia, che segue le linee reali)."""
    info = {} if info is None else info
    same = probe is img
    try:
        rot = detect_orientation(probe)
    except Exception:  # noqa: BLE001
        log.exception("Rilevamento dell'orientamento non riuscito")
        rot = 0
    info["rotazione"] = int(rot)
    if rot:
        log.info("Pagina ruotata di %d gradi", rot)
        img = rotate_right_angle(img, rot)
        probe = img if same else rotate_right_angle(probe, rot)
    try:
        angle = estimate_skew(probe)
    except Exception:  # noqa: BLE001 - il raddrizzamento non deve mai bloccare l'elaborazione
        log.exception("Stima dell'inclinazione non riuscita")
        angle = 0.0
    h, w = img.shape[:2]
    transform = None
    out, out_probe = img, probe
    info["inclinazione"] = round(float(angle), 2) if abs(angle) >= MIN_SKEW_DEG else 0.0
    if abs(angle) >= MIN_SKEW_DEG:
        log.debug("Inclinazione corretta: %.2f gradi", angle)
        transform = np.vstack([cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0), [0.0, 0.0, 1.0]])
        out = rotate_small_angle(img, -angle)
        out_probe = out if same else rotate_small_angle(probe, -angle)
    try:
        matrix = None
        quad = table_quad(out_probe) if perspective else None
        if quad is not None:
            matrix = _rectify(quad, PERSPECTIVE_MIN_DEV)
            if matrix is not None and transform is not None:
                # rotazione e prospettiva in un'unica interpolazione
                matrix = matrix @ transform
        elif transform is not None and perspective:
            # una prospettiva forte puo' ingannare la stima dell'inclinazione:
            # si riprova sulla pagina non ruotata
            quad = table_quad(probe)
            if quad is not None:
                matrix = _rectify(quad, PERSPECTIVE_MIN_DEV)
    except Exception:  # noqa: BLE001 - la correzione non deve mai bloccare l'elaborazione
        log.exception("Correzione della prospettiva non riuscita")
        matrix = None
    info["prospettiva"] = matrix is not None
    if matrix is not None:
        log.info("Prospettiva della pagina corretta")
        out = _warp(img, matrix)
        out_probe = out if same else _warp(probe, matrix)
        transform = matrix
    try:
        flat = dewarp(img, out_probe, transform, probe=None if same else probe)
        if flat is None and not _grid_reliable(out_probe):
            # curvatura modesta ma griglia comunque non affidabile: si prova
            # anche con una soglia bassa (il risultato e' tenuto solo se migliora)
            flat = dewarp(img, out_probe, transform, min_dev=DEWARP_RETRY_DEV, probe=None if same else probe)
    except Exception:  # noqa: BLE001 - il dewarp non deve mai bloccare l'elaborazione
        log.exception("Raddrizzamento a maglia non riuscito")
        flat = None
    info["maglia"] = flat is not None
    if flat is not None:
        log.info("Pagina incurvata raddrizzata a maglia")
        if same:
            out = out_probe = flat
        else:
            out, out_probe = flat
    info["capovolta_dalla_tabella"] = False
    try:
        if _confirm_upright(out_probe) is not out_probe:
            out = rotate_right_angle(out, 180)
            out_probe = out if same else rotate_right_angle(out_probe, 180)
            info["capovolta_dalla_tabella"] = True
            info["rotazione"] = (int(info.get("rotazione", 0)) + 180) % 360
    except Exception:  # noqa: BLE001
        log.exception("Verifica del verso della pagina non riuscita")
    return out, out_probe


def normalize_page(img: np.ndarray) -> np.ndarray:
    """Porta la pagina in verticale e nel verso giusto, la raddrizza, se la
    tabella e' deformata dalla prospettiva (foto scattata di sbieco) la riporta
    rettangolare e, se le linee della tabella sono curve o spezzate (foglio
    incurvato o piegato), la raddrizza a maglia (``dewarp``).

    Se alla fine la tabella non e' riconosciuta in modo affidabile (modulo
    stampato molto chiaro, foto in penombra) la geometria viene rimisurata su
    una copia a contrasto aumentato (``enhance_contrast``) e applicata alla
    pagina originale; si tiene il risultato solo se la griglia migliora.

    Restituisce sempre un'immagine BGR ``uint8``. Non ritaglia: il formato resta
    quello della pagina (scambiato se ruotata di 90 gradi). In caso di dubbio
    l'immagine non viene ruotata ne' deformata; una scansione gia' dritta esce
    identica. Solleva ``ValueError`` solo per immagini non valide.
    """
    return normalize_page_info(img)[0]


def normalize_page_info(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """Come ``normalize_page`` ma restituisce anche le correzioni applicate::

        {"rotazione": 0|90|180|270,     # gradi in senso orario (verso della pagina)
         "inclinazione": float,         # gradi corretti (0.0 = nessuna rotazione fine)
         "prospettiva": bool,           # foto di sbieco riportata rettangolare
         "maglia": bool,                # foglio incurvato/piegato raddrizzato a maglia
         "contrasto": bool,             # geometria misurata sulla copia a contrasto aumentato
         "capovolta_dalla_tabella": bool}

    Serve a spiegare all'utente che la pagina e' stata raddrizzata (``correzioni_testo``)."""
    img = to_bgr(img)
    info: dict = {}
    out, _ = _normalize_with(img, img, info)
    info["contrasto"] = False
    try:
        from sirio.vision.grid import detect_grid  # noqa: PLC0415

        g = detect_grid(out)
        if not g.reliable:
            enh = to_bgr(enhance_contrast(img))
            info2: dict = {}
            out2, _ = _normalize_with(img, enh, info2)
            g2 = detect_grid(out2)
            al = g.alignment if (g.detected and g.alignment is not None) else 0.0
            al2 = g2.alignment if (g2.detected and g2.alignment is not None) else 0.0
            if g2.detected and (al2 > al + 0.05 or not g.detected):
                log.info("Geometria misurata sulla copia a contrasto aumentato")
                out, g, info = out2, g2, {**info2, "contrasto": True}
        if not g.reliable and info.get("prospettiva"):
            # foto di sbieco di un foglio incurvato: il quadrilatero della tabella puo' essere
            # solo una parte della tabella (una piega sembra un bordo) e la correzione della
            # prospettiva impedisce poi il raddrizzamento a maglia. Si riprova senza: la
            # maglia segue le linee reali e corregge anche la prospettiva.
            info3: dict = {}
            out3, _ = _normalize_with(img, img, info3, perspective=False)
            g3 = detect_grid(out3)
            if _grid_better(g3, g):
                log.info("Pagina raddrizzata senza la correzione della prospettiva")
                out, g, info = out3, g3, {**info3, "contrasto": False}
    except Exception:  # noqa: BLE001
        log.exception("Seconda misura della geometria non riuscita")
    return np.ascontiguousarray(out), info


def _grid_better(new, old) -> bool:  # noqa: ANN001 - TableGrid (import circolare evitato)
    """``new`` e' una griglia migliore di ``old`` (rilevata, piu' allineata, piu' affidabile)."""
    if not new.detected:
        return False
    if not old.detected:
        return True
    if new.reliable != old.reliable:
        return bool(new.reliable)
    a_new = new.alignment if new.alignment is not None else new.score
    a_old = old.alignment if old.alignment is not None else old.score
    return a_new > a_old + 0.05


MIN_SKEW_NOTA = 1.0         # inclinazione (gradi) oltre la quale la correzione e' segnalata all'utente


def correzioni_testo(info: dict | None) -> list[str]:
    """Frasi brevi in italiano sulle correzioni di ``normalize_page_info`` degne di nota
    (le piccole inclinazioni di ogni scansione non sono segnalate)."""
    if not info:
        return []
    out: list[str] = []
    rot = int(info.get("rotazione") or 0) % 360
    if rot == 180:
        out.append("era capovolta")
    elif rot in (90, 270):
        out.append("era ruotata di 90 gradi")
    try:
        angle = abs(float(info.get("inclinazione") or 0.0))
    except (TypeError, ValueError):
        angle = 0.0
    if angle >= MIN_SKEW_NOTA:
        out.append(f"inclinazione di {angle:.1f} gradi corretta".replace(".", ","))
    if info.get("prospettiva"):
        out.append("foto di sbieco riportata in piano")
    if info.get("maglia"):
        out.append("foglio incurvato o piegato raddrizzato")
    return out


def _confirm_upright(page: np.ndarray) -> np.ndarray:
    """Seconda verifica del verso con la tabella gia' individuata: se la
    griglia della pagina non e' affidabile ma quella della pagina girata di
    180 gradi si' (numeri della colonna "Giorno" nell'ordine giusto,
    intestazione sopra e totali sotto), la pagina era capovolta."""
    from sirio.vision.grid import detect_grid  # noqa: PLC0415

    g = detect_grid(page)
    if g.reliable:
        return page
    flipped = rotate_right_angle(page, 180)
    g2 = detect_grid(flipped)
    al = g.alignment if (g.detected and g.alignment is not None) else 0.0
    if g2.reliable and g2.alignment is not None and g2.alignment >= al + 0.2:
        log.info("Pagina capovolta riconosciuta dalla tabella: girata di 180 gradi")
        return flipped
    return page
