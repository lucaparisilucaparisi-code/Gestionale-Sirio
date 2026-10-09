"""Generatore di fogli firma sintetici per i test (nessun dato personale).

API stabile, usata dai test di piu' moduli::

    from tests.synthetic import make_synthetic_sheet, default_data, sheet_to_pdf

    img, truth = make_synthetic_sheet()                      # febbraio 2026, nomi fittizi
    img, truth = make_synthetic_sheet(skew_deg=1.5, seed=3)  # pagina inclinata
    pdf_bytes = sheet_to_pdf([img])                          # PDF "scansionato" (una pagina per immagine)

``make_synthetic_sheet(data=None, seed=0, skew_deg=0.0, dpi=200, noise=True)``
restituisce ``(immagine BGR uint8, verita)``. ``verita`` ha la stessa forma del
JSON di verita' del foglio reale:

* ``"header"``: campi di ``sirio.models.Header`` (lotto, municipalita, ente,
  istituto, operatore, alunno, mese, anno, ore_pei, sostituzione,
  firma_coordinatore, timbro_referente, totale_mensile_dichiarato,
  anno_scolastico, data_compilazione);
* ``"rows"``: 31 dizionari con i campi di ``sirio.models.DayRow`` (giorno,
  prog_entrata, prog_uscita, eff_entrata, eff_uscita nel formato "HH:MM",
  ore_dichiarate, assenza_alunno, assenza_operatore, firma, note,
  trattino_effettivo);
* ``"griglia"`` (in piu'): geometria esatta della tabella disegnata, prima
  dell'eventuale inclinazione: ``col_x`` (11), ``row_y`` (32), ``header_top``,
  ``total_row``, ``width``, ``height``, ``skew_deg``.

``data`` ha la stessa forma (``header`` + ``rows``, anche parziali: i campi
mancanti prendono i valori predefiniti; le righe sono indicizzate da
``giorno``). Il disegno riproduce il modulo reale: titolo, intestazione con
valori scritti "a mano" (``cv2.FONT_HERSHEY_SCRIPT_SIMPLEX`` con variazioni
casuali), tabella con le proporzioni misurate sul modulo, intestazione su due
righe, 31 righe, riga dei totali, crocette, firme scarabocchiate che debordano
nella riga sopra, trattini, note, timbro circolare nel pie' di pagina.
"""

from __future__ import annotations

import copy
import datetime
import math
from typing import Any

import cv2
import numpy as np

# Proporzioni del modulo reale (frazioni della pagina A4 verticale).
TABLE_X = (0.0395, 0.8855)
COL_FRACTIONS = (0.0, 0.0548, 0.1306, 0.2059, 0.2807, 0.3561, 0.4280, 0.4914, 0.5553, 0.7555, 1.0)
HEADER_TOP = 0.1584
FIRST_ROW = 0.1818
LAST_ROW = 0.9305
TOTAL_BOTTOM = 0.9518
REFERENT_BOTTOM = 0.9735

A4_IN = (8.27, 11.69)

ROW_FIELDS = (
    "giorno", "prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita", "ore_dichiarate",
    "assenza_alunno", "assenza_operatore", "firma", "note", "trattino_effettivo",
)


# --------------------------------------------------------------------------
# Dati predefiniti (mese plausibile con nomi fittizi)
# --------------------------------------------------------------------------

def _row(giorno: int, **kw: Any) -> dict:
    r = {
        "giorno": giorno, "prog_entrata": None, "prog_uscita": None, "eff_entrata": None,
        "eff_uscita": None, "ore_dichiarate": None, "assenza_alunno": False,
        "assenza_operatore": False, "firma": False, "note": None, "trattino_effettivo": False,
    }
    r.update(kw)
    return r


def default_data() -> dict:
    """Febbraio 2026 (giorni feriali 2-6, 9-13, 16-20, 23-27) con nomi fittizi."""
    afternoon = {5, 12, 19, 26}
    rows = []
    for g in range(1, 32):
        if g > 28 or datetime.date(2026, 2, g).weekday() >= 5:
            rows.append(_row(g))
            continue
        a, b = ("11:00", "14:00") if g in afternoon else ("08:00", "11:00")
        if g == 4:      # assenza dell'alunno: ore parziali riconosciute
            rows.append(_row(g, prog_entrata=a, prog_uscita=b, ore_dichiarate=1.5,
                             assenza_alunno=True, firma=True, trattino_effettivo=True))
        elif g == 10:   # assenza dell'operatore (permesso L. 104)
            rows.append(_row(g, prog_entrata=a, prog_uscita=b, assenza_operatore=True,
                             firma=True, note="104", trattino_effettivo=True))
        elif g in (16, 17):
            rows.append(_row(g, prog_entrata=a, prog_uscita=b, note="PONTE DI CARNEVALE",
                             trattino_effettivo=True))
        else:
            rows.append(_row(g, prog_entrata=a, prog_uscita=b, eff_entrata=a, eff_uscita=b,
                             ore_dichiarate=3.0, firma=True))
    total = sum(r["ore_dichiarate"] or 0.0 for r in rows)
    header = {
        "anno_scolastico": "2025/2026",
        "lotto": "1",
        "municipalita": "3",
        "ente": "COOPERATIVA SOCIALE ESEMPIO",
        "istituto": "IC 1 VERDI",
        "operatore": "ROSSI MARIO",
        "alunno": "BIANCHI LUCA",
        "mese": 2,
        "anno": 2026,
        "ore_pei": 15.0,
        "sostituzione": "NO",
        "data_compilazione": None,
        "firma_coordinatore": True,
        "timbro_referente": True,
        "totale_mensile_dichiarato": total,
    }
    return {"header": header, "rows": rows}


def _merge_data(data: dict | None) -> dict:
    base = default_data()
    if not data:
        return base
    out = copy.deepcopy(base)
    out["header"].update(copy.deepcopy(data.get("header", {})))
    if "rows" in data:
        by_day = {r["giorno"]: _row(r["giorno"]) for r in out["rows"]}
        for r in data["rows"]:
            g = int(r["giorno"])
            if 1 <= g <= 31:
                by_day[g] = {**_row(g), **copy.deepcopy(r)}
        out["rows"] = [by_day[g] for g in range(1, 32)]
    return out


# --------------------------------------------------------------------------
# Primitive di disegno
# --------------------------------------------------------------------------

def _blend(canvas: np.ndarray, alpha: np.ndarray, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Compone ``alpha`` (0..255) sul canvas con il colore dato (l'inchiostro scurisce)."""
    h, w = alpha.shape
    H, W = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)[..., None] / 255.0
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    ink = np.array(color, dtype=np.float32)[None, None, :]
    mixed = region * (1 - a) + np.minimum(region, ink) * a
    canvas[y0:y1, x0:x1] = np.clip(mixed, 0, 255).astype(np.uint8)


_TXT_COMUNE = "COMUNE DI NAPOLI"


def _font(bold: bool) -> int:
    return cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX


def _text_width(text: str, height: float, bold: bool = False) -> float:
    th = max(1, int(round(height / (9 if bold else 14))))
    return float(cv2.getTextSize(text, _font(bold), height / 22.0, th)[0][0])


def _fit(text: str, height: float, max_width: float, bold: bool = False) -> float:
    """Altezza del testo ridotta, se serve, per stare nella larghezza indicata."""
    w = _text_width(text, height, bold)
    return height if w <= max_width else height * max_width / w


def _print(canvas: np.ndarray, text: str, x: float, y: float, height: float, bold: bool = False,
           center: bool = False, color: tuple[int, int, int] = (25, 25, 25)) -> None:
    """Testo stampato (baseline in y). ``center`` centra orizzontalmente su x."""
    font = _font(bold)
    scale = height / 22.0
    th = max(1, int(round(height / (9 if bold else 14))))
    (tw, _), _ = cv2.getTextSize(text, font, scale, th)
    xi = int(round(x - tw / 2 if center else x))
    cv2.putText(canvas, text, (xi, int(round(y))), font, scale, color, th, cv2.LINE_AA)


def _hand(canvas: np.ndarray, text: str, x: float, y: float, height: float, rng: np.random.Generator,
          color: tuple[int, int, int], max_width: float | None = None, center: bool = False) -> None:
    """Testo "scritto a mano": corsivo Hershey, carattere per carattere con
    piccole variazioni di posizione e scala, leggera rotazione e inclinazione.
    ``(x, y)``: inizio della linea di base."""
    if not text:
        return
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    scale = height / 22.0 * rng.uniform(0.92, 1.08)
    th = max(2, int(round(height / 11)))
    widths = [cv2.getTextSize(ch, font, scale, th)[0][0] for ch in text]
    total = sum(widths) + 0.08 * height * len(text)
    if max_width and total > max_width:
        f = max_width / total
        scale *= f
        widths = [w * f for w in widths]
        total = max_width
    pad = int(height * 1.2)
    pw = int(total + 2 * pad)
    ph = int(height * 2.2 + 2 * pad)
    patch = np.zeros((ph, pw), np.uint8)
    cx = float(pad)
    base = pad + height * 1.3
    for ch, w in zip(text, widths):
        dy = rng.normal(0, height * 0.05)
        s = scale * rng.uniform(0.9, 1.1)
        cv2.putText(patch, ch, (int(round(cx)), int(round(base + dy))), font, s, 255, th, cv2.LINE_AA)
        cx += w * rng.uniform(0.95, 1.08) + 0.08 * height
    angle = rng.uniform(-2.5, 2.5)
    shear = rng.uniform(-0.12, 0.12)
    m = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
    m[0, 1] += shear
    m[0, 2] -= shear * ph / 2
    patch = cv2.warpAffine(patch, m, (pw, ph), flags=cv2.INTER_LINEAR, borderValue=0)
    ox = x - pad - (total / 2 if center else 0)
    oy = y - base
    _blend(canvas, patch, int(round(ox)), int(round(oy)), color)


def _poly(canvas: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], th: int) -> None:
    cv2.polylines(canvas, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, color, th, cv2.LINE_AA)


def _signature(canvas: np.ndarray, box: tuple[float, float, float, float], rng: np.random.Generator,
               color: tuple[int, int, int], th: int) -> None:
    """Firma scarabocchiata: un tratto corsivo continuo (cicloide con occhielli
    di altezza variabile, inclinata) piu' uno svolazzo finale. Gli occhielli
    alti salgono oltre il bordo superiore della riga come nelle firme reali."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    base = y0 + h * rng.uniform(0.66, 0.76)
    xs = x0 + w * rng.uniform(0.05, 0.12)
    xe = x0 + w * rng.uniform(0.6, 0.85)
    n = int(rng.integers(5, 9))
    step = (xe - xs) / n
    radius = step / (2 * math.pi) * rng.uniform(1.2, 1.9)
    amps = [h * rng.uniform(0.8, 1.05)]          # iniziale maiuscola
    for _ in range(n - 1):
        amps.append(h * (rng.uniform(0.6, 0.95) if rng.random() < 0.35 else rng.uniform(0.18, 0.4)))
    slant = rng.uniform(0.15, 0.35)
    t = np.linspace(0.0, 2 * math.pi * n, 36 * n)
    k = np.minimum((t // (2 * math.pi)).astype(int), n - 1)
    amp = np.array(amps)[k]
    y = base - amp * (1 - np.cos(t)) / 2
    x = xs + step * t / (2 * math.pi) - radius * np.sin(t) + slant * (base - y)
    # svolazzo finale sotto la firma
    u = np.linspace(0, 1, 30)
    fx = x[-1] + (xs + 0.25 * (xe - xs) - x[-1]) * u
    fy = base + h * 0.14 * np.sin(math.pi * u) * rng.uniform(0.6, 1.0)
    pts = np.column_stack([np.concatenate([x, fx]), np.concatenate([y, fy])])
    wobble = np.cumsum(rng.normal(0, h * 0.004, pts.shape), axis=0)
    pts += wobble - wobble.mean(axis=0)
    _poly(canvas, pts, color, th)


def _cross(canvas: np.ndarray, box: tuple[float, float, float, float], rng: np.random.Generator,
           color: tuple[int, int, int], th: int) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx = x0 + w * rng.uniform(0.38, 0.62)
    cy = y0 + h * rng.uniform(0.42, 0.6)
    s = min(w, h) * rng.uniform(0.2, 0.3)
    a = rng.uniform(-0.25, 0.25)
    for sign in (1, -1):
        dx = s * math.cos(a + sign * math.pi / 4)
        dy = s * math.sin(a + sign * math.pi / 4)
        pts = np.array([(cx - dx, cy - dy), (cx + dx * rng.uniform(0.9, 1.15), cy + dy * rng.uniform(0.9, 1.15))])
        _poly(canvas, pts, color, th)


def _dash(canvas: np.ndarray, box: tuple[float, float, float, float], rng: np.random.Generator,
          color: tuple[int, int, int], th: int) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    length = w * rng.uniform(0.14, 0.22)
    cx = x0 + w * rng.uniform(0.35, 0.6)
    cy = y0 + h * rng.uniform(0.45, 0.62)
    tilt = rng.uniform(-0.06, 0.06) * length
    _poly(canvas, np.array([(cx - length / 2, cy + tilt / 2), (cx + length / 2, cy - tilt / 2)]), color, th)


def _stamp(canvas: np.ndarray, center: tuple[float, float], radius: float, rng: np.random.Generator) -> None:
    """Timbro circolare tenue (anello doppio con tratti interrotti e stella)."""
    color = (150, 90, 110)
    layer = np.zeros(canvas.shape[:2], np.uint8)
    cx, cy = center
    for r in (radius, radius * 0.8):
        for a0 in np.arange(0, 360, 24):
            a1 = a0 + rng.uniform(10, 20)
            cv2.ellipse(layer, (int(cx), int(cy)), (int(r), int(r)), 0, a0, a1, 255,
                        max(2, int(radius / 22)), cv2.LINE_AA)
    star = [(cx + radius * 0.45 * math.cos(math.pi / 2 + k * 4 * math.pi / 5),
             cy - radius * 0.45 * math.sin(math.pi / 2 + k * 4 * math.pi / 5)) for k in range(6)]
    cv2.polylines(layer, [np.round(star).astype(np.int32).reshape(-1, 1, 2)], False, 255,
                  max(2, int(radius / 25)), cv2.LINE_AA)
    _blend(canvas, layer, 0, 0, color)


def _fmt_time(t: str | None) -> str:
    if not t:
        return ""
    hh, mm = t.split(":")
    return f"{int(hh)}:{mm}"


def _fmt_hours(v: float | None) -> str:
    if v is None:
        return ""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}".replace(".", ",")


# --------------------------------------------------------------------------
# Generatore
# --------------------------------------------------------------------------

def make_synthetic_sheet(data: dict | None = None, seed: int = 0, skew_deg: float = 0.0, dpi: int = 200,
                         noise: bool = True) -> tuple[np.ndarray, dict]:
    """Disegna un foglio firma sintetico. Vedi la documentazione del modulo."""
    rng = np.random.default_rng(seed)
    d = _merge_data(data)
    hdr = d["header"]
    W = int(round(A4_IN[0] * dpi))
    H = int(round(A4_IN[1] * dpi))
    paper = int(rng.integers(238, 252)) if noise else 255
    img = np.full((H, W, 3), paper, np.uint8)
    pen = (120, 45, 25) if rng.random() < 0.6 else (40, 40, 40)   # biro blu o nera
    line_c = (30, 30, 30)
    lt = max(2, int(round(dpi / 75)))       # spessore delle linee della tabella
    pt = max(2, int(round(dpi / 90)))       # spessore della penna
    u = H / 100.0                           # unita' verticale (1% della pagina)

    # --- geometria della tabella
    tx0, tx1 = TABLE_X[0] * W, TABLE_X[1] * W
    col_x = [tx0 + f * (tx1 - tx0) for f in COL_FRACTIONS]
    r0, r31 = FIRST_ROW * H, LAST_ROW * H
    row_y = [r0 + i * (r31 - r0) / 31 for i in range(32)]
    rh = row_y[1] - row_y[0]
    htop = HEADER_TOP * H
    tbot = TOTAL_BOTTOM * H
    rbot = REFERENT_BOTTOM * H
    hmid = (htop + r0) / 2

    # fondini grigi (retino) come nel modulo stampato
    gray = (222, 222, 222)
    cv2.rectangle(img, (int(tx0), int(htop)), (int(tx1), int(r0)), gray, -1)
    cv2.rectangle(img, (int(col_x[0]), int(r0)), (int(col_x[1]), int(r31)), gray, -1)
    cv2.rectangle(img, (int(col_x[0]), int(r31)), (int(col_x[5]), int(tbot)), gray, -1)

    # --- intestazione della pagina
    # stemma stilizzato
    sx, sy = 0.07 * W, 0.045 * H
    cv2.rectangle(img, (int(sx), int(sy)), (int(sx + 0.045 * W), int(sy + 0.05 * H)), (60, 60, 60), 2)
    cv2.ellipse(img, (int(sx + 0.0225 * W), int(sy + 0.05 * H)), (int(0.0225 * W), int(0.018 * H)), 0, 0, 180,
                (60, 60, 60), 2)
    _print(img, "COMUNE DI NAPOLI", 0.035 * W, 0.147 * H, _fit(_TXT_COMUNE, 1.25 * u, 0.13 * W, True), bold=True)
    title_x = 0.44 * W
    t1 = "Assistenza Specialistica all'integrazione scolastica destinato agli alunni disabili"
    t2 = ("frequentanti le scuole del Comune di Napoli.  Anno Scolastico "
          + str(hdr.get("anno_scolastico") or "2025/2026"))
    th1 = _fit(t1, 0.75 * u, 0.42 * W, True)
    _print(img, t1, title_x, 0.046 * H, th1, bold=True, center=True)
    _print(img, t2, title_x, 0.060 * H, th1, bold=True, center=True)

    lab_h = 0.66 * u
    hand_h = 1.2 * u
    lx = 0.225 * W
    line_end = 0.665 * W

    def label_line(y: float, label: str, x: float, end: float) -> float:
        """Etichetta stampata e riga per la risposta; restituisce l'inizio della riga."""
        _print(img, label, x, y - 0.2 * u, lab_h, bold=True)
        start = x + _text_width(label, lab_h, True) + 0.004 * W
        cv2.line(img, (int(start), int(y)), (int(end), int(y)), line_c, 1, cv2.LINE_AA)
        return start

    y_lotto = 0.075 * H
    v_lotto = label_line(y_lotto, "LOTTO", lx, lx + _text_width("LOTTO", lab_h, True) + 0.035 * W)
    x_mun = lx + _text_width("LOTTO", lab_h, True) + 0.04 * W
    v_mun = label_line(y_lotto, "- Municipalita", x_mun, line_end)
    if hdr.get("lotto"):
        _hand(img, str(hdr["lotto"]), v_lotto + 0.008 * W, y_lotto - 0.25 * u, hand_h, rng, pen)
    if hdr.get("municipalita"):
        _hand(img, str(hdr["municipalita"]), v_mun + 0.03 * W, y_lotto - 0.25 * u, hand_h, rng, pen)
    for yf, label, key in (
        (0.093, "Ente:", "ente"),
        (0.112, "Istituto scolastico:", "istituto"),
        (0.130, "Nome e Cognome Operatore:", "operatore"),
        (0.149, "Nome e Cognome Alunno:", "alunno"),
    ):
        y = yf * H
        start = label_line(y, label, lx, line_end)
        if hdr.get(key):
            _hand(img, str(hdr[key]), start + 0.01 * W, y - 0.25 * u, hand_h, rng, pen,
                  max_width=line_end - start - 0.02 * W)
    # colonna destra
    _print(img, "MESE/ANNO DI RIF.:", 0.725 * W, 0.047 * H, lab_h, bold=True)
    cv2.line(img, (int(0.70 * W), int(0.093 * H)), (int(0.855 * W), int(0.093 * H)), line_c, 1, cv2.LINE_AA)
    if hdr.get("mese") and hdr.get("anno"):
        _hand(img, f"{int(hdr['mese']):02d}/{int(hdr['anno'])}", 0.715 * W, 0.093 * H - 0.3 * u, hand_h, rng, pen)
    v_pei = label_line(0.130 * H, "Ore da PEI:", 0.70 * W, 0.86 * W)
    if hdr.get("ore_pei") is not None:
        _hand(img, _fmt_hours(float(hdr["ore_pei"])), v_pei + 0.02 * W, 0.130 * H - 0.25 * u, hand_h, rng, pen)
    _print(img, "Sostituzione:", 0.68 * W, 0.147 * H, lab_h, bold=True)
    bx0, bx1, bx2 = 0.785 * W, 0.835 * W, 0.885 * W
    by0, by1 = 0.135 * H, 0.152 * H
    cv2.rectangle(img, (int(bx0), int(by0)), (int(bx2), int(by1)), line_c, max(1, lt - 1))
    cv2.line(img, (int(bx1), int(by0)), (int(bx1), int(by1)), line_c, max(1, lt - 1))
    _print(img, "SI", (bx0 + bx1) / 2, by1 - 0.5 * u, lab_h, bold=True, center=True)
    _print(img, "NO", (bx1 + bx2) / 2, by1 - 0.5 * u, lab_h, bold=True, center=True)
    if hdr.get("sostituzione") in ("SI", "NO"):
        cx = (bx0 + bx1) / 2 if hdr["sostituzione"] == "SI" else (bx1 + bx2) / 2
        _cross(img, (cx - 0.02 * W, by0, cx + 0.02 * W, by1), rng, pen, pt)

    # --- tabella: linee
    def hline(y: float, xa: float, xb: float) -> None:
        cv2.line(img, (int(round(xa)), int(round(y))), (int(round(xb)), int(round(y))), line_c, lt)

    def vline(x: float, ya: float, yb: float) -> None:
        cv2.line(img, (int(round(x)), int(round(ya))), (int(round(x)), int(round(yb))), line_c, lt)

    hline(htop, tx0, tx1)
    hline(hmid, col_x[1], col_x[5])
    for y in row_y:
        hline(y, tx0, tx1)
    hline(tbot, tx0, tx1)
    hline(rbot, col_x[6], tx1)
    for j, x in enumerate(col_x):
        if j in (2, 4):
            vline(x, hmid, r31)
        else:
            vline(x, htop, r31)
    for j in (0, 5, 6, 9, 10):
        vline(col_x[j], r31, tbot)
    for j in (6, 9, 10):
        vline(col_x[j], tbot, rbot)

    # --- intestazione della tabella
    th_h = 0.62 * u

    def center_text(lines: list[str], xa: float, xb: float, ya: float, yb: float) -> None:
        n = len(lines)
        for k, s in enumerate(lines):
            yy = ya + (yb - ya) * (k + 1) / (n + 1) + th_h * 0.45
            _print(img, s, (xa + xb) / 2, yy, th_h, bold=True, center=True)

    center_text(["Giorno"], col_x[0], col_x[1], htop, r0)
    center_text(["Orario programmato"], col_x[1], col_x[3], htop, hmid)
    center_text(["Orario effettivo"], col_x[3], col_x[5], htop, hmid)
    for j in (1, 3):
        center_text(["Entrata"], col_x[j], col_x[j + 1], hmid, r0)
        center_text(["Uscita"], col_x[j + 1], col_x[j + 2], hmid, r0)
    center_text(["Tot. Ore", "effettive"], col_x[5], col_x[6], htop, r0)
    center_text(["Assenza", "Alunno"], col_x[6], col_x[7], htop, r0)
    center_text(["Assenza", "Operat."], col_x[7], col_x[8], htop, r0)
    center_text(["Firma Operatore"], col_x[8], col_x[9], htop, r0)
    center_text(["Note"], col_x[9], col_x[10], htop, r0)
    for g in range(1, 32):
        _print(img, str(g), (col_x[0] + col_x[1]) / 2, (row_y[g - 1] + row_y[g]) / 2 + 0.3 * u, 0.6 * u,
               bold=True, center=True)
    _print(img, "Totale ore effettive mensili", col_x[0] + 0.004 * W, (r31 + tbot) / 2 + 0.3 * u, 0.62 * u,
           bold=True)
    _print(img, "Firma Coordinatore dell'Ente", col_x[6] + 0.004 * W, (r31 + tbot) / 2 + 0.3 * u, 0.62 * u,
           bold=True)
    _print(img, "Timbro e firma Referente Scolastico", col_x[6] + 0.004 * W, (tbot + rbot) / 2 + 0.3 * u,
           0.62 * u, bold=True)
    _print(img, "Napoli, ___/___/______", col_x[0], 0.967 * H, 0.62 * u, bold=True)

    # --- righe giornaliere
    text_h = rh * 0.42
    rows_truth = []
    for r in d["rows"]:
        g = int(r["giorno"])
        ya, yb = row_y[g - 1], row_y[g]

        def cell(j: int) -> tuple[float, float, float, float]:
            return col_x[j], ya, col_x[j + 1], yb

        for j, key in ((1, "prog_entrata"), (2, "prog_uscita"), (3, "eff_entrata"), (4, "eff_uscita")):
            s = _fmt_time(r.get(key))
            if s:
                x0, _, x1, _ = cell(j)
                _hand(img, s, x0 + (x1 - x0) * rng.uniform(0.06, 0.14), yb - rh * rng.uniform(0.2, 0.3),
                      text_h, rng, pen, max_width=(x1 - x0) * 0.8)
        if r.get("trattino_effettivo"):
            for j, key in ((3, "eff_entrata"), (4, "eff_uscita")):
                if not r.get(key):
                    _dash(img, cell(j), rng, pen, pt)
        if r.get("ore_dichiarate") is not None:
            x0, _, x1, _ = cell(5)
            _hand(img, _fmt_hours(float(r["ore_dichiarate"])), (x0 + x1) / 2, yb - rh * rng.uniform(0.2, 0.3),
                  text_h * 1.1, rng, pen, center=True, max_width=(x1 - x0) * 0.8)
        if r.get("assenza_alunno"):
            _cross(img, cell(6), rng, pen, pt)
        if r.get("assenza_operatore"):
            _cross(img, cell(7), rng, pen, pt)
        if r.get("firma"):
            _signature(img, cell(8), rng, pen, pt)
        if r.get("note"):
            x0, _, x1, _ = cell(9)
            _hand(img, str(r["note"]), x0 + (x1 - x0) * 0.06, yb - rh * 0.28, text_h * 0.9, rng, pen,
                  max_width=(x1 - x0) * 0.88)
        rows_truth.append({k: r.get(k) for k in ROW_FIELDS})

    # --- totali, firme e timbro
    if hdr.get("totale_mensile_dichiarato") is not None:
        _hand(img, _fmt_hours(float(hdr["totale_mensile_dichiarato"])), (col_x[5] + col_x[6]) / 2,
              tbot - rh * 0.22, text_h * 1.1, rng, pen, center=True)
    if hdr.get("firma_coordinatore"):
        _signature(img, (col_x[9], r31, col_x[10], tbot), rng, pen, pt)
    if hdr.get("timbro_referente"):
        rad = (rbot - tbot) * 1.15
        _stamp(img, (col_x[9] + (col_x[10] - col_x[9]) * 0.55, rbot + rad * 0.25), rad, rng)
        _signature(img, (col_x[9], tbot, col_x[10], rbot), rng, (40, 40, 40), pt)

    # --- rumore di scansione
    if noise:
        # interruzioni casuali delle linee della tabella
        for _ in range(int(rng.integers(6, 14))):
            y = row_y[int(rng.integers(0, 32))]
            x = rng.uniform(tx0, tx1)
            gl = rng.uniform(2, 0.006 * W)
            cv2.rectangle(img, (int(x), int(y - lt)), (int(x + gl), int(y + lt)), (paper,) * 3, -1)
        # puntini e polvere
        n = int(W * H / 6000)
        ys = rng.integers(0, H, n)
        xs = rng.integers(0, W, n)
        img[ys, xs] = rng.integers(60, 200, (n, 1)).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0.6)
        grain = rng.normal(0, 5.0, img.shape[:2]).astype(np.float32)[..., None]
        img = np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    if abs(skew_deg) > 1e-9:
        m = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), skew_deg, 1.0)
        img = cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(paper, paper, paper))

    truth = {
        "header": {
            k: hdr.get(k) for k in (
                "anno_scolastico", "lotto", "municipalita", "ente", "istituto", "operatore", "alunno", "mese",
                "anno", "ore_pei", "sostituzione", "data_compilazione", "firma_coordinatore", "timbro_referente",
                "totale_mensile_dichiarato",
            )
        },
        "rows": rows_truth,
        "griglia": {
            "width": W,
            "height": H,
            "col_x": [int(round(v)) for v in col_x],
            "row_y": [int(round(v)) for v in row_y],
            "header_top": int(round(htop)),
            "total_row": [int(round(r31)), int(round(tbot))],
            "skew_deg": float(skew_deg),
        },
    }
    return img, truth


def sheet_to_pdf(images: list[np.ndarray], dpi: int = 200, jpeg_quality: int = 85) -> bytes:
    """PDF "scansionato": una pagina A4 per immagine (JPEG incorporato)."""
    import pymupdf  # noqa: PLC0415

    doc = pymupdf.open()
    try:
        for img in images:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            if not ok:
                raise ValueError("Codifica JPEG non riuscita")
            h, w = img.shape[:2]
            page = doc.new_page(width=w * 72.0 / dpi, height=h * 72.0 / dpi)
            page.insert_image(page.rect, stream=buf.tobytes())
        return doc.tobytes()
    finally:
        doc.close()


__all__ = ["default_data", "make_synthetic_sheet", "sheet_to_pdf"]
