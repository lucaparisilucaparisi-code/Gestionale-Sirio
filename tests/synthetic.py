"""Generatore di fogli firma sintetici per i test (nessun dato personale).

API stabile, usata dai test di piu' moduli::

    from tests.synthetic import make_synthetic_sheet, default_data, sheet_to_pdf

    img, truth = make_synthetic_sheet()                      # febbraio 2026, nomi fittizi
    img, truth = make_synthetic_sheet(skew_deg=1.5, seed=3)  # pagina inclinata
    pdf_bytes = sheet_to_pdf([img])                          # PDF "scansionato" (una pagina per immagine)
    photo = photograph(img, keystone=0.06)                   # foto di sbieco su un tavolo
    curled = curl(img, sag=0.012, fold=0.012)                # foglio incurvato e piegato
    poor = degrade(img, contrast=0.45, shadow=0.4)           # stampa chiara e ombra

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

COLUMN_NAMES = ("giorno", "prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita", "ore_dichiarate",
                "assenza_alunno", "assenza_operatore", "firma", "note")

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
        if "totale_mensile_dichiarato" not in data.get("header", {}):
            hours = [r["ore_dichiarate"] for r in out["rows"] if r.get("ore_dichiarate") is not None]
            out["header"]["totale_mensile_dichiarato"] = float(sum(hours)) if hours else None
    return out


# --------------------------------------------------------------------------
# Primitive di disegno
# --------------------------------------------------------------------------

def _blend(canvas: np.ndarray, alpha: np.ndarray, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Compone ``alpha`` (0..255) sul canvas con il colore dato (l'inchiostro scurisce)."""
    bx, by, bw, bh = cv2.boundingRect(alpha)      # solo la zona effettivamente disegnata
    if bw == 0 or bh == 0:
        return
    alpha = alpha[by:by + bh, bx:bx + bw]
    x, y = x + bx, y + by
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


_FORM_COLOR: list[tuple[int, int, int]] = [(25, 25, 25)]     # colore della stampa del modulo


def _print(canvas: np.ndarray, text: str, x: float, y: float, height: float, bold: bool = False,
           center: bool = False, color: tuple[int, int, int] | None = None) -> None:
    """Testo stampato (baseline in y). ``center`` centra orizzontalmente su x."""
    color = _FORM_COLOR[0] if color is None else color
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


def _fmt_time(t: str | None, style: str = "colon") -> str:
    """Orario come lo scrive l'operatore: ``colon`` "8:00", ``comma`` "8,00",
    ``none`` "800" (senza separatore)."""
    if not t:
        return ""
    hh, mm = t.split(":")
    sep = {"colon": ":", "comma": ",", "none": ""}.get(style, ":")
    return f"{int(hh)}{sep}{mm}"


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
                         noise: bool = True, form_color: tuple[int, int, int] | None = None,
                         time_style: str = "colon", crossed_rows: tuple = (),
                         long_signatures: bool = False, footer_over_table: bool = False,
                         crease_column: str | None = None) -> tuple[np.ndarray, dict]:
    """Disegna un foglio firma sintetico. Vedi la documentazione del modulo.

    Varianti realistiche (tutte facoltative):

    * ``form_color``: colore BGR della stampa del modulo (linee e testi), es.
      ``(170, 170, 170)`` per una fotocopia chiara o ``(190, 140, 90)`` per un
      modulo stampato a colori; i fondini grigi si schiariscono di conseguenza;
    * ``time_style``: come sono scritti gli orari: ``"colon"`` (8:00),
      ``"comma"`` (8,00), ``"none"`` (800) o ``"mixed"`` (a caso riga per riga);
    * ``crossed_rows``: intervalli ``(dal_giorno, al_giorno)`` di righe non
      usate barrate con un lungo tratto diagonale che attraversa piu' celle e
      righe (la verita' delle righe resta vuota);
    * ``long_signatures``: firme lunghe che debordano nella colonna "Note"
      (nelle righe senza nota);
    * nelle righe dei dati la chiave ``"linea_effettivo": True`` disegna una
      linea orizzontale a penna attraverso le celle dell'orario effettivo
      (prestazione non svolta: nella verita' ``trattino_effettivo`` = True);
    * una nota con ``"\\n"`` e' scritta piccola su due righe nella cella (nella
      verita' le due righe sono unite da uno spazio);
    * ``footer_over_table``: il timbro rotondo della scuola e la firma del
      referente scolastico sono messi sopra l'angolo in basso a destra della
      tabella e sconfinano nelle celle "Firma operatore" e "Note" delle ultime
      righe (la verita' delle righe non cambia);
    * ``crease_column``: foglio piegato in verticale e fotografato: l'ombra della
      piega (``fold_crease``) attraversa tutta la pagina dentro la colonna indicata
      (es. ``"assenza_operatore"``); la verita' non cambia.
    """
    _FORM_COLOR[0] = tuple(int(v) for v in form_color) if form_color is not None else (25, 25, 25)
    try:
        return _make_sheet(data, seed, skew_deg, dpi, noise, form_color, time_style, crossed_rows,
                           long_signatures, footer_over_table, crease_column)
    finally:
        _FORM_COLOR[0] = (25, 25, 25)


def _make_sheet(data: dict | None, seed: int, skew_deg: float, dpi: int, noise: bool,
                form_color: tuple[int, int, int] | None, time_style: str, crossed_rows: tuple,
                long_signatures: bool, footer_over_table: bool = False,
                crease_column: str | None = None) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    d = _merge_data(data)
    hdr = d["header"]
    W = int(round(A4_IN[0] * dpi))
    H = int(round(A4_IN[1] * dpi))
    paper = int(rng.integers(238, 252)) if noise else 255
    img = np.full((H, W, 3), paper, np.uint8)
    pen = (120, 45, 25) if rng.random() < 0.6 else (40, 40, 40)   # biro blu o nera
    line_c = tuple(int(v) for v in form_color) if form_color is not None else (30, 30, 30)
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
    if form_color is not None:          # fondino proporzionato alla stampa (piu' chiara = piu' tenue)
        gray = tuple(int(255 - 0.25 * (255 - c)) for c in form_color)
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

    # interruzioni casuali delle linee (prima della scrittura, che non va cancellata)
    if noise:
        for _ in range(int(rng.integers(6, 14))):
            y = row_y[int(rng.integers(0, 32))]
            x = rng.uniform(tx0, tx1)
            gl = rng.uniform(2, 0.006 * W)
            cv2.rectangle(img, (int(x), int(y - lt)), (int(x + gl), int(y + lt)), (paper,) * 3, -1)

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
    if hdr.get("data_compilazione"):
        _hand(img, str(hdr["data_compilazione"]), col_x[0] + _text_width("Napoli, ", 0.62 * u, True),
              0.967 * H - 0.15 * u, 1.0 * u, rng, pen)

    # --- righe giornaliere
    text_h = rh * 0.42
    rows_truth = []
    for r in d["rows"]:
        g = int(r["giorno"])
        ya, yb = row_y[g - 1], row_y[g]

        def cell(j: int, ya: float = ya, yb: float = yb) -> tuple[float, float, float, float]:
            return col_x[j], ya, col_x[j + 1], yb

        style = time_style if time_style != "mixed" else str(rng.choice(["colon", "comma", "none"]))
        for j, key in ((1, "prog_entrata"), (2, "prog_uscita"), (3, "eff_entrata"), (4, "eff_uscita")):
            s = _fmt_time(r.get(key), style)
            if s:
                x0, _, x1, _ = cell(j)
                _hand(img, s, x0 + (x1 - x0) * rng.uniform(0.06, 0.14), yb - rh * rng.uniform(0.2, 0.3),
                      text_h, rng, pen, max_width=(x1 - x0) * 0.8)
        if r.get("linea_effettivo"):
            # linea a penna attraverso le celle dell'orario effettivo (e oltre)
            xa = col_x[3] + (col_x[4] - col_x[3]) * rng.uniform(0.05, 0.2)
            xb = col_x[rng.choice([5, 6, 8])] - (col_x[1] - col_x[0]) * rng.uniform(0.0, 0.3)
            yy = ya + rh * rng.uniform(0.4, 0.6)
            tilt = rh * rng.uniform(-0.12, 0.12)
            n = 24
            xs = np.linspace(xa, xb, n)
            ys = yy + np.linspace(-tilt, tilt, n) + rng.normal(0, rh * 0.02, n)
            _poly(img, np.column_stack([xs, ys]), pen, pt)
        elif r.get("trattino_effettivo"):
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
            if long_signatures and not r.get("note"):
                # firma lunga: parte nella colonna Firma e finisce in quella delle Note
                x0, _, x1, _ = cell(8)
                _signature(img, (x0, ya, x1 + (col_x[10] - col_x[9]) * rng.uniform(0.35, 0.6), yb), rng, pen, pt)
            else:
                _signature(img, cell(8), rng, pen, pt)
        if r.get("note"):
            x0, _, x1, _ = cell(9)
            lines = str(r["note"]).split("\n")
            if len(lines) == 1:
                _hand(img, lines[0], x0 + (x1 - x0) * 0.06, yb - rh * 0.28, text_h * 0.9, rng, pen,
                      max_width=(x1 - x0) * 0.88)
            else:
                # nota piccola su due righe dentro la cella
                for k, line in enumerate(lines[:2]):
                    _hand(img, line, x0 + (x1 - x0) * 0.06, ya + rh * (0.44 if k == 0 else 0.86), text_h * 0.62,
                          rng, pen, max_width=(x1 - x0) * 0.6)
        t = {k: r.get(k) for k in ROW_FIELDS}
        if t.get("note"):
            t["note"] = " ".join(str(t["note"]).split())
        if r.get("linea_effettivo"):
            t["trattino_effettivo"] = True
        rows_truth.append(t)

    # --- righe non usate barrate con lunghi tratti diagonali
    for g_from, g_to in crossed_rows:
        a, b = sorted((int(g_from), int(g_to)))
        y_a, y_b = row_y[a - 1] + 0.2 * rh, row_y[b] - 0.2 * rh
        xa = col_x[1] + rng.uniform(0.0, 0.5) * (col_x[2] - col_x[1])
        xb = col_x[9] - rng.uniform(0.0, 0.5) * (col_x[9] - col_x[8])
        n = 40
        xs = np.linspace(xa, xb, n)
        ys = np.linspace(y_b, y_a, n) + rng.normal(0, rh * 0.03, n)
        _poly(img, np.column_stack([xs, ys]), pen, pt)

    # --- totali, firme e timbro
    if hdr.get("totale_mensile_dichiarato") is not None:
        _hand(img, _fmt_hours(float(hdr["totale_mensile_dichiarato"])), (col_x[5] + col_x[6]) / 2,
              tbot - rh * 0.22, text_h * 1.1, rng, pen, center=True)
    if hdr.get("firma_coordinatore"):
        _signature(img, (col_x[9], r31, col_x[10], tbot), rng, pen, pt)
    if hdr.get("timbro_referente") and footer_over_table:
        # timbro e firma del referente appoggiati sull'angolo della tabella: sconfinano
        # nelle celle "Firma operatore" e "Note" delle ultime righe
        rad = 2.1 * rh
        _stamp(img, (col_x[9] - 0.1 * (col_x[9] - col_x[8]), r31 + 0.2 * rh), rad, rng)
        _signature(img, (col_x[8] + 0.45 * (col_x[9] - col_x[8]), r31 - 1.2 * rh,
                         col_x[10] - 0.15 * (col_x[10] - col_x[9]), rbot), rng, (40, 40, 40), pt)
    elif hdr.get("timbro_referente"):
        rad = (rbot - tbot) * 1.15
        _stamp(img, (col_x[9] + (col_x[10] - col_x[9]) * 0.55, rbot + rad * 0.25), rad, rng)
        _signature(img, (col_x[9], tbot, col_x[10], rbot), rng, (40, 40, 40), pt)

    # --- rumore di scansione
    if noise:
        # puntini e polvere
        n = int(W * H / 6000)
        ys = rng.integers(0, H, n)
        xs = rng.integers(0, W, n)
        img[ys, xs] = rng.integers(60, 200, (n, 1)).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0.6)
        grain = rng.normal(0, 5.0, img.shape[:2]).astype(np.float32)[..., None]
        img = np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    if crease_column is not None:
        j = COLUMN_NAMES.index(crease_column)
        xc = col_x[j] + 0.45 * (col_x[j + 1] - col_x[j])
        img = fold_crease(img, xc, darkness=110, width=5, seed=seed, bend=0.15 * (col_x[j + 1] - col_x[j]) / W)

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


def photograph(img: np.ndarray, keystone: float = 0.0, jitter: float = 0.0, seed: int = 0,
               background: tuple[int, int, int] = (70, 90, 110), shadow: bool = False) -> np.ndarray:
    """Foto del foglio scattata con il telefono: il foglio appoggiato su un
    piano (``background``, BGR) con margine attorno, deformato dalla prospettiva.

    ``keystone`` (frazione del formato) sposta i vertici come in una foto
    scattata di sbieco dall'alto a sinistra; ``jitter`` aggiunge a ogni vertice
    uno spostamento casuale fino a quella frazione (``seed``). ``shadow`` scurisce
    gradualmente il lato sinistro. Restituisce un'immagine BGR piu' grande del
    foglio (125% x 120%)."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    out_w, out_h = int(w * 1.25), int(h * 1.2)
    ox, oy = (out_w - w) / 2.0, (out_h - h) / 2.0
    d = float(keystone)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[ox + d * w, oy + d * 0.5 * h], [ox + w - d * 0.3 * w, oy],
                      [ox + w, oy + h], [ox + d * 0.2 * w, oy + h - d * 0.4 * h]])
    if jitter:
        dst += (rng.uniform(-jitter, jitter, (4, 2)) * np.float32([w, h])).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    canvas = np.empty((out_h, out_w, 3), np.uint8)
    canvas[:] = background
    out = cv2.warpPerspective(img, m, (out_w, out_h), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
    if shadow:
        light = 0.55 + 0.45 * np.linspace(0.0, 1.0, out_w, dtype=np.float32)
        out = out.astype(np.float32) * light[None, :, None]
    out = out.astype(np.float32) + rng.normal(0.0, 4.0, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def curl(img: np.ndarray, sag: float = 0.01, fold: float = 0.0, fold_x: float = 0.4, wave: float = 0.0,
         seed: int = 0) -> np.ndarray:
    """Foglio incurvato o piegato fotografato dall'alto (stesso formato).

    * ``sag``: incurvatura del foglio (frazione dell'altezza): le righe si
      inarcano, al centro scendono di ``sag * H`` rispetto ai bordi; le
      colonne si incurvano di un terzo di questo valore;
    * ``fold``: piega verticale alla posizione ``fold_x`` (frazione della
      larghezza): a destra della piega il foglio e' piu' basso di
      ``fold * H`` (salto ripido ma continuo, largo il 3% della pagina);
    * ``wave``: ondulazione casuale a bassa frequenza (frazione dell'altezza).
    Il riempimento ai bordi ripete la carta."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    u, v = X / w, Y / h
    dy = sag * h * np.sin(np.pi * u) * (0.6 + 0.4 * v)
    dx = (sag / 3.0) * w * np.sin(np.pi * v) * (u - 0.5) * 2.0
    if fold:
        t = np.clip((u - fold_x) / 0.03 + 0.5, 0.0, 1.0)
        dy = dy + fold * h * (t * t * (3 - 2 * t))
    if wave:
        ph = rng.uniform(0, 2 * np.pi, 3)
        dy = dy + wave * h * (np.sin(2 * np.pi * u * 1.3 + ph[0]) * np.cos(2 * np.pi * v * 0.7 + ph[1]))
        dx = dx + 0.5 * wave * w * np.sin(2 * np.pi * v * 1.1 + ph[2])
    map_x = (X - dx).astype(np.float32)
    map_y = (Y - dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def fold_crease(img: np.ndarray, x: float, darkness: float = 90.0, width: float = 3.0, bend: float = 0.004,
                slant: float = 0.0, shadow: float = 8.0, gaps: int = 0, seed: int = 0) -> np.ndarray:
    """Piega verticale di un foglio piegato in due e fotografato: una linea lunga,
    sottile e scura (l'ombra della piega) che attraversa la pagina dall'alto in basso.

    * ``x``: ascissa della piega a meta' pagina (pixel);
    * ``darkness``: quanto la piega scurisce la carta al centro (livelli di grigio);
    * ``width``: spessore del tratto scuro (pixel);
    * ``bend``: incurvatura (frazione della larghezza della pagina, al centro);
    * ``slant``: inclinazione (pixel di spostamento orizzontale per pixel verticale);
    * ``shadow``: larghezza (pixel) dell'ombra sfumata a destra della piega;
    * ``gaps``: tratti (lunghi l'1% della pagina) in cui la piega non si vede (luce radente).
    Non tocca la verita': la piega non e' scrittura."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    ys = np.arange(h, dtype=np.float32)
    wobble = np.cumsum(rng.normal(0, 0.15, h)).astype(np.float32)
    wobble = cv2.GaussianBlur((wobble - wobble.mean()).reshape(-1, 1), (1, 0), sigmaX=1, sigmaY=25).ravel()
    path = x + bend * w * np.sin(np.pi * ys / h) + slant * (ys - h / 2.0) + 0.5 * wobble
    reach = int(np.ceil(width + shadow + 3))
    x0 = max(0, int(np.floor(path.min())) - reach)
    x1 = min(w, int(np.ceil(path.max())) + reach + 1)
    xs = np.arange(x0, x1, dtype=np.float32)
    d = xs[None, :] - path[:, None]                       # distanza orizzontale dalla piega
    core = np.clip(width / 2.0 + 0.7 - np.abs(d), 0.0, 1.0)
    side = np.where(d > 0, np.clip(1.0 - d / max(1.0, width / 2.0 + shadow), 0.0, 1.0), 0.0) * 0.35
    dark = darkness * np.maximum(core, side) * rng.uniform(0.85, 1.0, (h, 1)).astype(np.float32)
    for y in rng.uniform(0.2 * h, 0.9 * h, int(gaps)):
        dark[int(y):int(y + 0.01 * h)] = 0.0
    out = img.copy()
    band = out[:, x0:x1].astype(np.float32) - dark[..., None]
    out[:, x0:x1] = np.clip(band, 0, 255).astype(np.uint8)
    return out


def degrade(img: np.ndarray, contrast: float = 1.0, shadow: float = 0.0, blur: float = 0.0,
            jpeg_quality: int | None = None, seed: int = 0) -> np.ndarray:
    """Scansione/foto di qualita' scadente: ``contrast`` < 1 schiarisce tutto
    l'inchiostro (toner scarico), ``shadow`` (0..1) ombra graduale da un
    angolo, ``blur`` sigma della sfocatura, ``jpeg_quality`` compressione."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32)
    if contrast != 1.0:
        out = 255.0 - (255.0 - out) * float(contrast)
    if shadow:
        h, w = out.shape[:2]
        gx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        gy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        light = 1.0 - shadow * np.clip(1.2 - (gx + gy), 0, 1)
        out = out * light[..., None]
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), float(blur))
    out = out + rng.normal(0, 2.0, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if jpeg_quality:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


__all__ = ["curl", "default_data", "degrade", "fold_crease", "make_synthetic_sheet", "photograph", "sheet_to_pdf"]
