"""Test dell'analisi d'immagine: raddrizzamento, orientamento, griglia, inchiostro.

I test sul foglio reale usano ``SIRIO_SAMPLE_PDF`` / ``SIRIO_SAMPLE_TRUTH`` e
vengono saltati se le variabili non sono impostate.
"""

from __future__ import annotations

import functools
import json
import os

import cv2
import numpy as np
import pytest

from sirio.vision import ink
from sirio.vision.grid import COLUMNS, TableGrid, detect_grid, draw_grid, template_grid
from sirio.vision.preprocess import (
    deskew,
    detect_orientation,
    estimate_skew,
    normalize_page,
    perspective_matrix,
    rotate_right_angle,
    table_quad,
)
from tests.synthetic import make_synthetic_sheet, photograph

TIME_FIELDS = ("prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita")


@functools.cache
def _sheet(seed: int = 0, skew: float = 0.0, dpi: int = 200) -> tuple[np.ndarray, dict]:
    return make_synthetic_sheet(seed=seed, skew_deg=skew, dpi=dpi)


@functools.cache
def _processed(seed: int = 0, skew: float = 0.0, dpi: int = 200) -> tuple[np.ndarray, TableGrid]:
    img, _ = _sheet(seed, skew, dpi)
    page = normalize_page(img)
    return page, detect_grid(page)


def classification_errors(img: np.ndarray, grid: TableGrid, truth: dict,
                          note_days: range | tuple = range(1, 32)) -> list[tuple]:
    """Confronta l'analisi dell'inchiostro con la verita' riga per riga.
    Restituisce l'elenco delle discrepanze (giorno, campo, controllo, valore)."""
    errors: list[tuple] = []
    for t in truth["rows"]:
        d = t["giorno"]
        for f in TIME_FIELDS:
            box = grid.cell(d, f)
            dash = t[f] is None and bool(t["trattino_effettivo"]) and f.startswith("eff")
            if ink.is_dash(img, box) != dash:
                errors.append((d, f, "trattino", not dash))
            if ink.is_blank(img, box) != (t[f] is None and not dash):
                errors.append((d, f, "vuota", t[f] is not None))
        box = grid.cell(d, "ore_dichiarate")
        if ink.is_blank(img, box) != (t["ore_dichiarate"] is None):
            errors.append((d, "ore_dichiarate", "vuota", t["ore_dichiarate"] is not None))
        for f in ("assenza_alunno", "assenza_operatore"):
            box = grid.cell(d, f)
            if ink.has_cross(img, box) != bool(t[f]):
                errors.append((d, f, "crocetta", not t[f]))
            if ink.is_blank(img, box) != (not t[f]):
                errors.append((d, f, "vuota", bool(t[f])))
        box = grid.cell(d, "firma")
        if ink.has_signature(img, box) != bool(t["firma"]):
            errors.append((d, "firma", "firma", not t["firma"]))
        if ink.is_blank(img, box) != (not t["firma"]):
            errors.append((d, "firma", "vuota", bool(t["firma"])))
        if d in note_days and ink.is_blank(img, grid.cell(d, "note")) != (not t["note"]):
            errors.append((d, "note", "vuota", bool(t["note"])))
    return errors


# --------------------------------------------------------------------------
# Raddrizzamento e orientamento
# --------------------------------------------------------------------------

@pytest.mark.parametrize("skew", [0.0, 1.5, -2.5])
def test_deskew_synthetic(skew):
    img, _ = _sheet(0, skew)
    assert estimate_skew(img) == pytest.approx(skew, abs=0.3)
    out, angle = deskew(img)
    assert out.shape == img.shape
    assert angle == pytest.approx(skew, abs=0.3)
    assert abs(estimate_skew(out)) < 0.3


@pytest.mark.parametrize("skew", [6.5, -6.8])
def test_deskew_large_angles(skew):
    img, _ = _sheet(2, skew)
    assert estimate_skew(img) == pytest.approx(skew, abs=0.3)


def test_correct_page_is_left_untouched():
    img, _ = _sheet(0, 0.0)
    assert detect_orientation(img) == 0
    out = normalize_page(img)
    assert out.shape == img.shape
    assert np.array_equal(out, img)


def test_blank_and_non_form_pages_are_not_rotated():
    blank = np.full((1400, 1000, 3), 255, np.uint8)
    assert detect_orientation(blank) == 0
    out, angle = deskew(blank)
    assert angle == 0.0 and out is blank
    noise = np.random.default_rng(0).integers(0, 256, (900, 700, 3), dtype=np.uint8)
    assert normalize_page(noise).shape == noise.shape


@pytest.mark.parametrize("rotation", [90, 180, 270])
@pytest.mark.parametrize("skew", [0.0, 1.5])
def test_landscape_and_upside_down_pages_are_normalized(rotation, skew):
    img, _ = _sheet(1, skew)
    turned = rotate_right_angle(img, rotation)
    assert detect_orientation(turned) == (360 - rotation) % 360
    page = normalize_page(turned)
    assert page.shape == img.shape                      # di nuovo verticale
    grid = detect_grid(page)
    _, ref = _processed(1, skew)
    assert grid.detected
    assert np.abs(np.array(grid.row_y) - np.array(ref.row_y)).max() <= 3
    assert np.abs(np.array(grid.col_x) - np.array(ref.col_x)).max() <= 3


def test_normalize_accepts_grayscale_and_rejects_invalid():
    img, _ = _sheet(0, 0.0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = normalize_page(gray)
    assert out.shape == img.shape and out.dtype == np.uint8
    with pytest.raises(ValueError):
        normalize_page(np.zeros((0, 0, 3), np.uint8))


# --------------------------------------------------------------------------
# Griglia
# --------------------------------------------------------------------------

def test_grid_matches_exact_geometry():
    img, truth = _sheet(0, 0.0)
    page, grid = _processed(0, 0.0)
    g = truth["griglia"]
    assert grid.detected and grid.score > 0.8
    assert (grid.width, grid.height) == (page.shape[1], page.shape[0])
    assert np.abs(np.array(grid.col_x) - g["col_x"]).max() <= 2
    assert np.abs(np.array(grid.row_y) - g["row_y"]).max() <= 2
    assert abs(grid.header_top - g["header_top"]) <= 3
    assert grid.total_row is not None
    assert abs(grid.total_row[0] - g["total_row"][0]) <= 2 and abs(grid.total_row[1] - g["total_row"][1]) <= 3


@pytest.mark.parametrize("skew", [0.0, 1.5, -2.5])
@pytest.mark.parametrize("dpi", [150, 300])
def test_grid_detected_on_skewed_sheets(skew, dpi):
    _, truth = _sheet(3, skew, dpi)
    _, grid = _processed(3, skew, dpi)
    assert grid.detected and grid.score > 0.8
    assert len(grid.col_x) == 11 and len(grid.row_y) == 32
    assert all(b > a for a, b in zip(grid.col_x, grid.col_x[1:]))
    diffs = np.diff(grid.row_y)
    expected = np.diff(truth["griglia"]["row_y"]).mean()
    assert np.all(np.abs(diffs - expected) <= max(2.0, 0.04 * expected))
    widths = np.diff(grid.col_x)
    assert np.allclose(widths, np.diff(truth["griglia"]["col_x"]), atol=3)


# --------------------------------------------------------------------------
# Foto scattate di sbieco (prospettiva)
# --------------------------------------------------------------------------

def test_table_quad_on_scans_and_no_warp_when_rectangular():
    img, truth = _sheet(0, 0.0)
    g = truth["griglia"]
    quad = table_quad(img)
    assert quad is not None
    expected = [(g["col_x"][0], g["header_top"]), (g["col_x"][-1], g["header_top"]),
                (g["col_x"][-1], g["total_row"][1]), (g["col_x"][0], g["total_row"][1])]
    assert np.abs(quad - np.array(expected, dtype=float)).max() <= 3
    assert perspective_matrix(img) is None                  # scansione: nessuna deformazione
    assert table_quad(np.full((1400, 1000, 3), 255, np.uint8)) is None


@functools.cache
def _photo(**kw) -> tuple[np.ndarray, np.ndarray, TableGrid]:
    img, _ = _sheet(3, 0.0)
    photo = photograph(img, **kw)
    page = normalize_page(photo)
    return photo, page, detect_grid(page)


@pytest.mark.parametrize("kw", [
    {"keystone": 0.03},
    {"keystone": 0.06},
    {"keystone": 0.08, "shadow": True},
    {"jitter": 0.03, "seed": 2},
    {"jitter": 0.04, "seed": 4, "background": (30, 30, 30)},
], ids=["keystone3", "keystone6", "keystone8-ombra", "casuale3", "casuale4-scuro"])
def test_photo_perspective_is_corrected(kw):
    """Foto del foglio scattata di sbieco: la tabella torna rettangolare e ogni
    cella e' letta nel giorno giusto (prima: griglia sfalsata di alcune righe
    accettata come rilevata, oppure foglio scartato)."""
    _, truth = _sheet(3, 0.0)
    photo, page, grid = _photo(**kw)
    assert page.shape == photo.shape
    assert grid.detected and grid.score > 0.9
    assert grid.details["intestazione_trovata"] and grid.details["totale_trovato"]
    assert perspective_matrix(page) is None                 # gia' rettangolare
    assert classification_errors(page, grid, truth) == []


@pytest.mark.parametrize("kw", [{"keystone": 0.06}, {"keystone": 0.06, "shadow": True}],
                         ids=["senza-ancoraggio", "righe-inclinate"])
def test_grid_rejects_rows_misaligned_by_perspective(kw):
    """Senza correzione della prospettiva le righe-giorno non seguono quelle del
    modulo: la griglia non deve essere data per rilevata (dati nei giorni sbagliati)."""
    img, _ = _sheet(3, 0.0)
    straight, _angle = deskew(photograph(img, **kw))        # solo rotazione, niente prospettiva
    grid = detect_grid(straight)
    assert not grid.detected and grid.score <= 0.3


def test_grid_never_raises_and_falls_back_to_template():
    for img in (
        np.full((2338, 1654, 3), 255, np.uint8),
        np.random.default_rng(1).integers(0, 256, (1200, 900), dtype=np.uint8),
        np.zeros((10, 10, 3), np.uint8),
        np.zeros((0, 0, 3), np.uint8),
        None,
    ):
        grid = detect_grid(img)  # type: ignore[arg-type]
        assert isinstance(grid, TableGrid)
        assert not grid.detected and grid.score <= 0.3
        assert len(grid.col_x) == 11 and len(grid.row_y) == 32
        assert all(b >= a for a, b in zip(grid.row_y, grid.row_y[1:]))
        if grid.height >= 500:
            assert all(b > a for a, b in zip(grid.row_y, grid.row_y[1:]))
            assert all(b > a for a, b in zip(grid.col_x, grid.col_x[1:]))
        x0, y0, x1, y1 = grid.cell(15, "firma")
        assert x1 > x0 and y1 > y0


def test_template_grid_proportions():
    grid = template_grid(2480, 3508)
    assert not grid.detected
    assert grid.header_top < grid.row_y[0] < grid.row_y[-1] < grid.total_row[1] < 3508
    assert 0 < grid.col_x[0] < grid.col_x[-1] < 2480
    assert np.all(np.diff(grid.col_x) > 0) and np.all(np.diff(grid.row_y) > 0)


def test_grid_api():
    grid = TableGrid(width=1000, height=2000, col_x=[50 + 80 * i for i in range(11)],
                     row_y=[300 + 50 * i for i in range(32)], header_top=240, total_row=(1850, 1900),
                     detected=True, score=0.9)
    assert grid.cell(1, "giorno") == (50, 300, 130, 350)
    assert grid.cell(31, "note") == (770, 1800, 850, 1850)
    assert grid.cell(2, 3) == grid.cell(2, "eff_entrata")
    assert grid.cell(1, "firma", pad=5) == (685, 295, 775, 355)
    assert grid.cell(1, "firma", pad=-5) == (695, 305, 765, 345)
    assert grid.row_box(3) == (50, 400, 850, 450)
    assert grid.rows_box(1, 16) == (50, 300, 850, 1100)
    assert grid.rows_box(16, 31, include_table_header=False) == (50, 1050, 850, 1850)
    assert grid.rows_box(1, 16, include_table_header=True) == (50, 240, 850, 1100)
    assert grid.table_box() == (50, 240, 850, 1900)
    assert grid.header_region() == (0, 0, 1000, 240)
    assert grid.footer_region() == (0, 1850, 1000, 2000)
    assert grid.total_value_box() == (450, 1850, 530, 1900)
    assert grid.coordinator_signature_box() == (770, 1850, 850, 1900)
    assert grid.row_height == pytest.approx(50.0)
    # i riquadri restano dentro l'immagine
    x0, y0, x1, y1 = grid.cell(31, "note", pad=500)
    assert x0 >= 0 and y0 >= 0 and x1 <= 1000 and y1 <= 2000
    for bad in ((0, "giorno"), (32, "giorno"), (1, "colonna_inesistente"), (1, 10)):
        with pytest.raises(ValueError):
            grid.cell(*bad)
    no_total = TableGrid(**{**grid.to_dict(), "total_row": None, "col_x": grid.col_x, "row_y": grid.row_y})
    assert no_total.total_value_box() is None
    assert no_total.table_box() == (50, 240, 850, 1850)


def test_grid_serialization_and_scaling():
    _, grid = _processed(0, 0.0)
    d = grid.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert set(d) == {"width", "height", "col_x", "row_y", "header_top", "total_row", "detected", "score",
                      "alignment", "row_quality"}
    back = TableGrid.from_dict(json.loads(json.dumps(d)))
    assert back == grid
    assert grid.alignment is not None and grid.alignment > 0.9 and grid.reliable
    assert len(grid.row_quality) == 31 and all(grid.row_reliable(g) for g in range(1, 32))
    # griglie salvate dalle versioni precedenti (senza autoverifica): restano valide
    old = {k: v for k, v in d.items() if k not in ("alignment", "row_quality")}
    legacy = TableGrid.from_dict(old)
    assert legacy.alignment is None and legacy.row_quality is None and legacy.reliable
    assert legacy.scaled(0.5, 0.5).alignment is None
    half = grid.scaled(0.5, 0.5)
    assert half.width == round(grid.width * 0.5) and half.col_x[3] == round(grid.col_x[3] * 0.5)
    assert half.row_y[10] == round(grid.row_y[10] * 0.5)
    assert half.total_row == (round(grid.total_row[0] * 0.5), round(grid.total_row[1] * 0.5))
    for bad in ({}, {**d, "col_x": d["col_x"][:5]}, {**d, "row_y": "x"}, {**d, "width": 0}):
        with pytest.raises(ValueError):
            TableGrid.from_dict(bad)


def test_draw_grid_returns_annotated_copy():
    page, grid = _processed(0, 0.0)
    before = page.copy()
    out = draw_grid(page, grid)
    assert out.shape == page.shape and out.dtype == np.uint8
    assert np.array_equal(page, before)
    assert not np.array_equal(out, page)
    small = cv2.resize(page, None, fx=0.5, fy=0.5)
    assert draw_grid(small, grid).shape == small.shape          # griglia riscalata


def test_columns_contract():
    assert COLUMNS == ("giorno", "prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita",
                       "ore_dichiarate", "assenza_alunno", "assenza_operatore", "firma", "note")


# --------------------------------------------------------------------------
# Inchiostro
# --------------------------------------------------------------------------

@pytest.mark.parametrize("skew", [0.0, 1.5, -2.5])
def test_ink_classification_synthetic(skew):
    _, truth = _sheet(0, skew)
    page, grid = _processed(0, skew)
    assert classification_errors(page, grid, truth) == []


@pytest.mark.parametrize("seed", [4, 5])
def test_ink_classification_other_sheets(seed):
    _, truth = _sheet(seed, 0.0, 300)
    page, grid = _processed(seed, 0.0, 300)
    assert classification_errors(page, grid, truth) == []


@pytest.mark.parametrize("present", [True, False])
def test_footer_boxes_synthetic(present):
    data = {"header": {"firma_coordinatore": present, "timbro_referente": present}}
    img, truth = make_synthetic_sheet(seed=3, data=data)
    page = normalize_page(img)
    grid = detect_grid(page)
    assert grid.detected
    coord, stamp, total = grid.coordinator_signature_box(), grid.referent_stamp_box(), grid.total_value_box()
    assert ink.has_signature(page, coord) is present
    assert ink.is_blank(page, stamp) is not present
    assert truth["header"]["totale_mensile_dichiarato"] == pytest.approx(49.5)
    assert not ink.is_blank(page, total)


def _cell_canvas() -> tuple[np.ndarray, TableGrid]:
    """Tre righe di una tabella disegnata a mano: celle 160x60 px."""
    img = np.full((400, 1000, 3), 250, np.uint8)
    cols = [40 + 90 * i for i in range(11)]
    rows = [40 + 60 * i for i in range(32)]
    for x in cols:
        cv2.line(img, (x, 40), (x, 220), (20, 20, 20), 3)
    for y in rows[:4]:
        cv2.line(img, (40, y), (940, y), (20, 20, 20), 3)
    grid = TableGrid(width=1000, height=400, col_x=cols, row_y=rows, header_top=10, total_row=None,
                     detected=True, score=1.0)
    return img, grid


def test_ink_primitives():
    img, grid = _cell_canvas()
    pen = (130, 50, 20)
    # riga 2: crocetta, trattino, "firma"
    x0, y0, x1, y1 = grid.cell(2, "assenza_alunno")
    cv2.line(img, (x0 + 25, y0 + 15), (x1 - 25, y1 - 15), pen, 3)
    cv2.line(img, (x0 + 25, y1 - 15), (x1 - 25, y0 + 15), pen, 3)
    x0, y0, x1, y1 = grid.cell(2, "eff_entrata")
    cv2.line(img, (x0 + 30, y0 + 32), (x0 + 55, y0 + 31), pen, 3)
    x0, y0, x1, y1 = grid.cell(2, "eff_uscita")
    cv2.putText(img, "8:00", (x0 + 8, y1 - 15), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, pen, 2, cv2.LINE_AA)
    x0, y0, x1, y1 = grid.cell(2, "firma")
    pts = np.array([(x0 + 5 + 4 * k, y0 + 35 + 18 * np.sin(k / 2.0)) for k in range(20)], np.int32)
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, pen, 2)
    # puntino isolato
    x0, y0, x1, y1 = grid.cell(3, "prog_entrata")
    cv2.circle(img, (x0 + 40, y0 + 30), 2, pen, -1)

    assert ink.has_cross(img, grid.cell(2, "assenza_alunno"))
    assert not ink.is_blank(img, grid.cell(2, "assenza_alunno"))
    assert not ink.is_dash(img, grid.cell(2, "assenza_alunno"))
    assert ink.is_dash(img, grid.cell(2, "eff_entrata"))
    assert not ink.has_cross(img, grid.cell(2, "eff_entrata"))
    assert not ink.is_blank(img, grid.cell(2, "eff_entrata"))
    assert not ink.is_dash(img, grid.cell(2, "eff_uscita"))
    assert not ink.is_blank(img, grid.cell(2, "eff_uscita"))
    assert ink.has_signature(img, grid.cell(2, "firma"))
    for col in COLUMNS[1:]:
        assert ink.is_blank(img, grid.cell(1, col)), col
    assert ink.is_blank(img, grid.cell(3, "prog_entrata"))          # il puntino non conta
    assert not ink.has_signature(img, grid.cell(1, "firma"))
    assert not ink.has_cross(img, grid.cell(1, "assenza_operatore"))


def test_signature_overflow_is_attributed_to_its_row():
    img, grid = _cell_canvas()
    x0, y0, x1, y1 = grid.cell(2, "firma")
    # firma della riga 2 con un'asta alta che sale fin quasi a meta' della riga 1
    pts = [(x0 + 10, y1 - 12), (x0 + 20, y0 - 25), (x0 + 28, y1 - 10), (x0 + 50, y0 + 20), (x0 + 80, y1 - 12)]
    cv2.polylines(img, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, (40, 40, 40), 3)
    assert ink.has_signature(img, grid.cell(2, "firma"))
    assert ink.is_blank(img, grid.cell(1, "firma"))
    assert not ink.has_signature(img, grid.cell(1, "firma"))
    assert ink.analyze_cell(img, grid.cell(1, "firma")).foreign > 0


def test_ink_ratio_and_crop():
    img, grid = _cell_canvas()
    box = grid.cell(2, "note")
    # riquadro che sborda di qualche pixel sulle linee della griglia
    wide = (box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4)
    assert ink.ink_ratio(img, wide, margin=0.0, remove_lines=True) == pytest.approx(0.0, abs=1e-3)
    assert ink.ink_ratio(img, wide, margin=0.0, remove_lines=False) > 0.03
    x0, y0, x1, y1 = box
    cv2.rectangle(img, (x0 + 20, y0 + 20), (x0 + 50, y0 + 40), (0, 0, 0), -1)
    r = ink.ink_ratio(img, box)
    assert 0.05 < r < 0.3
    c = ink.crop(img, box, pad=5)
    assert c.shape == (y1 - y0 + 10, x1 - x0 + 10, 3)
    c[:] = 0
    assert img[y0 + 2, x1 - 5].tolist() != [0, 0, 0]               # copia, non vista
    edge = ink.crop(img, (-50, -50, 30, 20))
    assert edge.shape == (20, 30, 3)
    assert ink.crop(img, (500, 500, 600, 600)).size == 0
    assert ink.ink_ratio(img, (10, 10, 10, 10)) == 0.0


# --------------------------------------------------------------------------
# Foglio reale
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _real() -> tuple[np.ndarray, TableGrid, dict]:
    pdf = os.environ.get("SIRIO_SAMPLE_PDF")
    truth_path = os.environ.get("SIRIO_SAMPLE_TRUTH")
    if not pdf or not truth_path or not os.path.exists(pdf) or not os.path.exists(truth_path):
        pytest.skip("SIRIO_SAMPLE_PDF / SIRIO_SAMPLE_TRUTH non impostate")
    from sirio.pdf_io import iter_pages

    with open(pdf, "rb") as fh:
        (_, img), = list(iter_pages(fh.read(), os.path.basename(pdf)))
    with open(truth_path, encoding="utf-8") as fh:
        truth = json.load(fh)
    page = normalize_page(img)
    return page, detect_grid(page), truth


def test_real_sample_grid():
    page, grid, _ = _real()
    assert grid.detected and grid.score > 0.9
    assert (grid.width, grid.height) == (page.shape[1], page.shape[0])
    assert len(grid.row_y) == 32 and all(b > a for a, b in zip(grid.row_y, grid.row_y[1:]))
    diffs = np.diff(grid.row_y)
    assert np.all(np.abs(diffs - np.median(diffs)) <= 0.06 * np.median(diffs))
    assert len(grid.col_x) == 11 and all(b > a for a, b in zip(grid.col_x, grid.col_x[1:]))
    # la colonna "Giorno" e' la piu' stretta, firma e note le piu' larghe
    widths = np.diff(grid.col_x)
    assert np.argmin(widths) == 0 and set(np.argsort(widths)[-2:]) == {8, 9}
    assert grid.header_top < grid.row_y[0] and grid.row_y[0] - grid.header_top < 1.5 * np.median(diffs)
    assert grid.total_row is not None and grid.total_row[0] == grid.row_y[-1]
    assert 0.6 * np.median(diffs) < grid.total_row[1] - grid.total_row[0] < 1.4 * np.median(diffs)


def test_real_sample_cells_with_content_have_ink():
    page, grid, truth = _real()
    for row in truth["rows"]:
        d = row["giorno"]
        for f in TIME_FIELDS:
            if row[f]:
                assert ink.ink_ratio(page, grid.cell(d, f)) > 0.03, (d, f)
        if row["ore_dichiarate"] is not None:
            assert ink.ink_ratio(page, grid.cell(d, "ore_dichiarate")) > 0.02, d
        if row["note"]:
            assert ink.ink_ratio(page, grid.cell(d, "note")) > 0.01, d
    value = grid.total_value_box()
    assert value is not None and ink.ink_ratio(page, value) > 0.02     # totale mensile scritto


def test_real_sample_ink_classification():
    page, grid, truth = _real()
    rows = truth["rows"]
    crosses_al = [d for d in range(1, 32) if ink.has_cross(page, grid.cell(d, "assenza_alunno"))]
    crosses_op = [d for d in range(1, 32) if ink.has_cross(page, grid.cell(d, "assenza_operatore"))]
    signatures = [d for d in range(1, 32) if ink.has_signature(page, grid.cell(d, "firma"))]
    dashes = [d for d in range(1, 32)
              if ink.is_dash(page, grid.cell(d, "eff_entrata")) and ink.is_dash(page, grid.cell(d, "eff_uscita"))]
    assert crosses_al == [r["giorno"] for r in rows if r["assenza_alunno"]]
    assert crosses_op == [r["giorno"] for r in rows if r["assenza_operatore"]]
    assert signatures == [r["giorno"] for r in rows if r["firma"]]
    assert dashes == [r["giorno"] for r in rows if r["trattino_effettivo"]]
    # Nelle note dei giorni 29-31 c'e' il timbro del referente che deborda
    # dal pie' di pagina: e' inchiostro reale, quindi quelle celle non sono vuote.
    assert classification_errors(page, grid, truth, note_days=range(1, 29)) == []


# --------------------------------------------------------------------------
# Fogli reali "difficili" simulati: orientamento, fogli incurvati, autoverifica
# --------------------------------------------------------------------------

from sirio.vision.grid import GRID_ALIGNMENT_MIN, check_alignment  # noqa: E402
from sirio.vision.preprocess import measure_mesh, rotate_small_angle  # noqa: E402
from tests.synthetic import curl, degrade  # noqa: E402

_DEGRADED = {
    "curvo": lambda i: curl(i, sag=0.02),
    "piegato": lambda i: curl(i, sag=0.012, fold=0.012, fold_x=0.35),
    "foto-curva": lambda i: photograph(curl(i, sag=0.015, fold=0.01), keystone=0.06),
    "foto-ombra": lambda i: photograph(curl(i, sag=0.02), keystone=0.05, shadow=True, background=(40, 40, 40)),
    "stampa-chiara": lambda i: degrade(i, contrast=0.3),
    "sfocata": lambda i: degrade(i, contrast=0.6, shadow=0.6, blur=1.5, jpeg_quality=50),
    "storta-9": lambda i: rotate_small_angle(i, 9.0),
}


@functools.cache
def _light_sheet() -> tuple[np.ndarray, dict]:
    """Modulo stampato a colori chiari, orari con la virgola o senza separatore."""
    return make_synthetic_sheet(seed=6, form_color=(175, 160, 150), time_style="mixed")


@pytest.mark.parametrize("case", sorted(_DEGRADED))
def test_orientation_on_degraded_photos_and_scans(case):
    """Il verso giusto anche su foto incurvate, stampe chiare, pagine storte
    oltre il raddrizzamento automatico; una pagina diritta non si gira mai."""
    for img in (_sheet(3)[0], _light_sheet()[0]):
        base = _DEGRADED[case](img)
        for rot in (0, 90, 180, 270):
            assert detect_orientation(rotate_right_angle(base, rot)) == (360 - rot) % 360, rot


@pytest.mark.parametrize("case", ["curvo", "piegato", "foto-curva"])
def test_curled_and_folded_sheets_are_flattened(case):
    """Foglio incurvato o piegato: dopo il raddrizzamento a maglia ogni cella e'
    letta nel giorno giusto (prima: griglia sfalsata di una riga data per buona)."""
    img, truth = _sheet(3)
    page = normalize_page(_DEGRADED[case](img))
    grid = detect_grid(page)
    assert grid.detected and grid.reliable and grid.alignment > 0.9
    mesh = measure_mesh(page)
    assert mesh is not None and mesh.deviation < 0.1          # linee ormai dritte
    assert classification_errors(page, grid, truth) == []


def test_upside_down_curled_photo_is_normalized():
    img, truth = _sheet(3)
    photo = rotate_right_angle(_DEGRADED["foto-ombra"](img), 180)
    page = normalize_page(photo)
    grid = detect_grid(page)
    assert grid.reliable
    assert classification_errors(page, grid, truth) == []


def test_light_colour_form_with_comma_times():
    img, truth = _light_sheet()
    page = normalize_page(img)
    grid = detect_grid(page)
    assert grid.reliable and grid.score > 0.9
    assert classification_errors(page, grid, truth) == []


def test_straight_scan_has_no_curvature():
    img, _ = _sheet(0)
    mesh = measure_mesh(img)
    assert mesh is not None and mesh.deviation < 0.05
    assert np.array_equal(normalize_page(img), img)           # nessun ricampionamento


def test_grid_self_check_catches_shifted_rows():
    """Griglia sfalsata di righe intere (le righe del modulo sono periodiche):
    i numeri stampati nella colonna "Giorno" la smascherano."""
    page, grid = _processed(0, 0.0)
    ok = check_alignment(page, grid)
    assert ok["alignment"] >= 0.95 and ok["sfalsamento"] == 0
    for shift in (-2, -1, 1, 2):
        d = grid.to_dict()
        dy = shift * grid.row_height
        d["row_y"] = [int(round(y + dy)) for y in grid.row_y]
        d["header_top"] = int(round(grid.header_top + dy))
        d["total_row"] = [int(round(v + dy)) for v in grid.total_row]
        bad = TableGrid.from_dict(d)
        chk = check_alignment(page, bad)
        assert chk["alignment"] < GRID_ALIGNMENT_MIN, shift
        if abs(shift) == 1:
            assert chk["sfalsamento"] == shift


def test_grid_on_curled_page_is_never_confidently_wrong():
    """Senza il raddrizzamento a maglia la griglia di un foglio piegato o e'
    giusta o e' dichiarata inaffidabile (mai valori nei giorni sbagliati come certi)."""
    for case in ("piegato", "curvo"):
        img, truth = _sheet(3)
        bent = _DEGRADED[case](img)
        grid = detect_grid(bent)
        if grid.reliable:
            assert classification_errors(bent, grid, truth) == [], case


def test_grid_self_check_on_blank_and_misaligned_grids():
    page, grid = _processed(0, 0.0)
    blank = np.full_like(page, 255)
    chk = check_alignment(blank, grid)
    assert chk["alignment"] < GRID_ALIGNMENT_MIN
    d = grid.to_dict()                                  # mezza riga di scarto: righe fuori dalle linee
    half = 0.5 * grid.row_height
    d["row_y"] = [int(round(y + half)) for y in grid.row_y]
    assert check_alignment(page, TableGrid.from_dict(d))["alignment"] < GRID_ALIGNMENT_MIN
    d = grid.to_dict()                                  # colonne spostate di mezza colonna
    d["col_x"] = [int(round(x + 0.5 * (grid.col_x[1] - grid.col_x[0]))) for x in grid.col_x]
    assert check_alignment(page, TableGrid.from_dict(d))["alignment"] < GRID_ALIGNMENT_MIN


def test_shift_correction_is_kept_only_if_the_shifted_grid_holds(monkeypatch):
    """La correzione dello sfalsamento (numeri della colonna "Giorno") si tiene solo se la
    griglia spostata poggia sulle linee almeno quanto l'originale e la sua prima e ultima
    riga sono righe vere: un foglio incurvato con le righe sbagliate solo in una parte non
    diventa "affidabile" spostando tutto di una riga."""
    from sirio.vision import grid as gm

    page, grid = _processed(0, 0.0)
    ok_rows = [1.0] * 31

    def fake(first: dict, second: dict):
        calls = iter([first, second])
        return lambda img, g: next(calls)

    base = {"alignment": 0.0, "row_quality": ok_rows, "linee": 0.92, "giorni": 0.0, "sfalsamento": 1,
            "totali_ok": None, "numeri_leggibili": 0.97, "capovolta": False}
    bad = {**base, "alignment": 0.9, "giorni": 1.0, "sfalsamento": 0, "linee": 0.90,
           "row_quality": [0.16] + ok_rows[1:]}
    monkeypatch.setattr(gm, "check_alignment", fake(base, bad))
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    assert not gm._apply_check(gray, grid).reliable
    good = {**bad, "linee": 0.92, "row_quality": ok_rows}
    monkeypatch.setattr(gm, "check_alignment", fake(base, good))
    fixed = gm._apply_check(gray, grid)
    assert fixed.reliable and fixed.details.get("sfalsamento_corretto") == 1


def test_day_numbers_readable_on_a_hatched_day_column():
    """Fotocopia con la colonna "Giorno" retinata scura: i numeri stampati restano
    riconoscibili (la verifica della griglia non declassa la pagina)."""
    page, grid = _processed(0, 0.0)
    img = page.copy()
    x0, x1 = grid.col_x[0] + 3, grid.col_x[1] - 3
    y0, y1 = grid.row_y[0] + 3, grid.row_y[-1] - 3
    zone = img[y0:y1, x0:x1].astype(np.int16)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    hatch = np.where(((xx + yy) % 4) < 2, 70, 0)             # retino diagonale fitto
    dark = np.clip(zone - 120 - hatch[..., None], 0, 255)
    numbers = zone < 110                                    # i numeri stampati restano piu' scuri
    dark[numbers] = np.clip(zone[numbers] - 150, 0, 255)
    img[y0:y1, x0:x1] = dark.astype(np.uint8)
    chk = check_alignment(img, grid)
    assert chk["numeri_leggibili"] >= 0.9 and chk["giorni"] == 1.0, chk


# --------------------------------------------------------------------------
# Inchiostro proprio: barrature, linee attraverso, debordamenti, "/"
# --------------------------------------------------------------------------

@functools.cache
def _struck_sheet() -> tuple[np.ndarray, TableGrid, dict]:
    from tests.synthetic import default_data

    data = default_data()
    for r in data["rows"]:
        if r["giorno"] in (16, 17):
            r.update(eff_entrata=None, eff_uscita=None, ore_dichiarate=None, trattino_effettivo=False,
                     linea_effettivo=True, note="SCUOLA CHIUSA")
    img, truth = make_synthetic_sheet(data=data, seed=2, crossed_rows=((28, 31),), long_signatures=True)
    page = normalize_page(img)
    return page, detect_grid(page), truth


def test_strike_through_and_lines_across_are_found():
    page, grid, truth = _struck_sheet()
    p = ink.PageInk(page, grid)
    assert {28, 29, 30, 31} <= p.struck_days
    assert p.line_days == {16, 17}
    for d in (16, 17):
        for col in ("eff_entrata", "eff_uscita"):
            res = p.own_ink(d, col)
            assert res.line_across and res.empty, (d, col)
    for d in (28, 29, 30, 31):
        assert p.own_ink(d, "prog_uscita").row_struck
    # le righe compilate non sono barrate e conservano la scrittura
    for t in truth["rows"]:
        d = t["giorno"]
        if d in (16, 17) or d >= 28:
            continue
        for col in TIME_FIELDS:
            res = p.own_ink(d, col)
            assert not res.struck and not res.line_across and not res.row_struck, (d, col)
            if t[col]:
                assert not res.empty and res.overflow_removed < 0.3, (d, col)
                assert res.crop.shape[:2] == (res.box[3] - res.box[1], res.box[2] - res.box[0])
        if t["note"]:
            assert not p.own_ink(d, "note").empty, d            # le note restano


def test_real_pages_have_no_false_strikes():
    for seed in (0, 4):
        page, grid = _processed(seed, 0.0)
        assert ink.PageInk(page, grid).strokes == []


def _note_canvas() -> tuple[np.ndarray, TableGrid]:
    img, grid = _cell_canvas()
    return img, grid


def test_signature_overflowing_into_note_is_removed():
    img, grid = _note_canvas()
    pen = (40, 40, 40)
    x0, y0, x1, y1 = grid.cell(2, "firma")
    nx0 = grid.cell(2, "note")[0]
    # firma corsiva che parte nella colonna Firma e finisce nelle Note
    xs = np.linspace(x0 + 5, nx0 + 30, 60)
    ys = (y0 + y1) / 2 + 12 * np.sin((xs - x0) / 6.0)
    cv2.polylines(img, [np.column_stack([xs, ys]).astype(np.int32).reshape(-1, 1, 2)], False, pen, 2)
    # riga 3: una nota vera, staccata dalla firma
    x0, y0, x1, y1 = grid.cell(3, "note")
    cv2.putText(img, "104", (x0 + 30, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, pen, 2, cv2.LINE_AA)
    xf0, yf0, xf1, yf1 = grid.cell(3, "firma")
    xs = np.linspace(xf0 + 5, xf1 - 20, 40)
    ys = (yf0 + yf1) / 2 + 12 * np.sin((xs - xf0) / 6.0)
    cv2.polylines(img, [np.column_stack([xs, ys]).astype(np.int32).reshape(-1, 1, 2)], False, pen, 2)
    p = ink.PageInk(img, grid)
    note2 = p.own_ink(2, "note")
    assert note2.empty and note2.overflow_removed > 0.5
    assert not p.own_ink(2, "firma").empty
    note3 = p.own_ink(3, "note")
    assert not note3.empty and note3.overflow_removed < 0.1


def test_descender_from_row_above_is_not_own_ink():
    img, grid = _cell_canvas()
    pen = (40, 40, 40)
    x0, y0, x1, y1 = grid.cell(1, "prog_entrata")
    cv2.putText(img, "10", (x0 + 10, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, pen, 2, cv2.LINE_AA)
    # virgola la cui coda scende nella riga sotto
    cx = x0 + 55
    cv2.line(img, (cx, y1 - 14), (cx - 3, y1 + 12), pen, 3)
    p = ink.PageInk(img, grid)
    assert p.own_ink(2, "prog_entrata").empty
    assert not p.own_ink(1, "prog_entrata").empty


def test_slash_is_recognized():
    img, grid = _cell_canvas()
    pen = (40, 40, 40)
    x0, y0, x1, y1 = grid.cell(1, "prog_entrata")
    cv2.line(img, (x0 + 20, y1 - 12), (x0 + 60, y0 + 14), pen, 3)          # "/"
    x0, y0, x1, y1 = grid.cell(2, "prog_entrata")
    cv2.putText(img, "8:00", (x0 + 8, y1 - 15), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, pen, 2, cv2.LINE_AA)
    x0, y0, x1, y1 = grid.cell(3, "ore_dichiarate")
    cv2.line(img, (x0 + 45, y1 - 10), (x0 + 49, y0 + 10), pen, 3)          # "1" quasi verticale
    p = ink.PageInk(img, grid)
    assert p.own_ink(1, "prog_entrata").slash
    assert not p.own_ink(2, "prog_entrata").slash
    one = p.own_ink(3, "ore_dichiarate")
    assert not one.slash and one.stroke_angle > 75
    assert p.own_ink(1, "eff_entrata").stroke_angle is None


def test_faint_dark_photo_is_never_confidently_wrong():
    """Modulo stampato chiaro, foto in penombra, prospettiva e foglio
    ondulato: se la griglia non e' sicura deve dirlo (``reliable`` False);
    se si dichiara affidabile, ogni cella e' nel giorno giusto."""
    img, truth = make_synthetic_sheet(seed=93, dpi=150, form_color=(195, 177, 167), time_style="mixed")
    img = curl(img, sag=0.012, wave=0.005, fold_x=0.66, seed=93)
    img = photograph(img, keystone=0.04, jitter=0.02, seed=93, shadow=True)
    img = degrade(img, contrast=0.4, shadow=0.33, blur=0.6, jpeg_quality=68, seed=93)
    page = normalize_page(img)
    grid = detect_grid(page)
    if grid.reliable:
        assert classification_errors(page, grid, truth) == []
    else:
        assert grid.alignment is None or grid.alignment < GRID_ALIGNMENT_MIN


def test_normalize_page_info_reports_the_corrections():
    from sirio.vision.preprocess import correzioni_testo, normalize_page_info

    img, _ = _sheet(0)
    page, info = normalize_page_info(img)
    assert np.array_equal(page, normalize_page(img))
    assert info["rotazione"] == 0 and not info["maglia"] and not info["prospettiva"]
    assert correzioni_testo(info) == []
    page, info = normalize_page_info(rotate_right_angle(img, 180))
    assert info["rotazione"] == 180 and detect_grid(page).reliable
    assert correzioni_testo(info) == ["era capovolta"]
    assert correzioni_testo({"rotazione": 90, "inclinazione": -2.26, "prospettiva": True, "maglia": True}) == [
        "era ruotata di 90 gradi", "inclinazione di 2,3 gradi corretta", "foto di sbieco riportata in piano",
        "foglio incurvato o piegato raddrizzato"]
    assert correzioni_testo({"inclinazione": 0.4}) == [] and correzioni_testo(None) == []


def test_curled_photo_with_strong_keystone_and_background_is_found():
    """Foto di un foglio incurvato e piegato, di sbieco, con lo sfondo attorno al foglio:
    la tabella deve essere trovata e affidabile (raddrizzamento a maglia)."""
    from sirio.vision.preprocess import normalize_page_info
    from tests.synthetic import curl

    img, truth = make_synthetic_sheet(seed=42)
    photo = photograph(curl(img, sag=0.02, fold=0.012, seed=42), keystone=0.06, seed=42, shadow=True)
    page, info = normalize_page_info(photo)
    grid = detect_grid(page)
    assert grid.detected and grid.reliable
    assert info["maglia"]


@pytest.mark.parametrize("column", ["assenza_operatore", "firma", "eff_entrata"])
def test_fold_crease_is_found_and_removed_from_cell_ink(column):
    """Foglio piegato in verticale e fotografato: l'ombra della piega attraversa tutta
    la tabella dentro una colonna. E' riconosciuta come piega (non scrittura) e tolta
    dall'inchiostro proprio delle celle: una cella con la sola piega resta vuota."""
    from tests.synthetic import fold_crease

    img, truth = _sheet(1)
    gx = truth["griglia"]["col_x"]
    j = COLUMNS.index(column)
    xc = gx[j] + 0.45 * (gx[j + 1] - gx[j])
    page = normalize_page(fold_crease(img, xc, darkness=110, width=5, seed=1))
    grid = detect_grid(page)
    creases, mask = ink.find_creases(page, grid)
    assert len(creases) == 1 and mask is not None and mask.any()
    c = creases[0]
    assert len(c.days) >= 25
    assert grid.col_x[j] < float(np.median(c.path)) < grid.col_x[j + 1]
    pi = ink.PageInk(page, grid)
    empty_rows = [t["giorno"] for t in truth["rows"] if not t[column] and not t["note"]
                  and t["giorno"] in c.days and not t["trattino_effettivo"]]
    assert empty_rows
    for g in empty_rows:
        res = pi.own_ink(g, column)
        assert res.crease and res.empty, (g, res.ink_ratio)


def test_fold_crease_seen_only_in_part_is_still_one_crease():
    """Piega poco visibile in alcuni punti (luce radente): i pezzi restano una piega."""
    from tests.synthetic import curl, fold_crease

    img, truth = _sheet(1)
    gx = truth["griglia"]["col_x"]
    xc = gx[7] + 0.5 * (gx[8] - gx[7])
    photo = photograph(curl(fold_crease(img, xc, darkness=70, width=4, gaps=4, seed=2), sag=0.006,
                            fold=0.004, fold_x=xc / img.shape[1]), keystone=0.03, seed=2)
    page = normalize_page(photo)
    grid = detect_grid(page)
    creases, _mask = ink.find_creases(page, grid)
    assert len(creases) == 1 and len(creases[0].days) >= 25
    assert grid.col_x[7] < float(np.median(creases[0].path)) < grid.col_x[8]


def test_synthetic_sheet_with_crease_option():
    img, truth = make_synthetic_sheet(seed=2, crease_column="assenza_operatore")
    page = normalize_page(img)
    grid = detect_grid(page)
    creases, _mask = ink.find_creases(page, grid)
    assert len(creases) == 1
    assert grid.col_x[7] < float(np.median(creases[0].path)) < grid.col_x[8]


def test_no_creases_on_regular_sheets():
    """Scrittura di righe consecutive allineata in colonna (firme, ore) non e' una piega."""
    for seed in (0, 3, 5):
        page, grid = _processed(seed)
        assert ink.find_creases(page, grid) == ([], None)


def test_pagina_a_righe_fitte_non_blocca_il_raddrizzamento():
    """Una pagina piena di righe verticali fitte (retino, foglio a righe, rumore)
    produce centinaia di "linee": il raddrizzamento deve comunque terminare in
    pochi secondi invece di bloccare la coda di lettura (prima: oltre 11 minuti)."""
    import time

    img, _ = make_synthetic_sheet(seed=0, dpi=150)
    img[:, ::7] = 0
    t = time.monotonic()
    out = normalize_page(img)
    assert out.ndim == 3
    assert time.monotonic() - t < 60
