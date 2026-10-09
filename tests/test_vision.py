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
    rotate_right_angle,
)
from tests.synthetic import make_synthetic_sheet

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
    assert set(d) == {"width", "height", "col_x", "row_y", "header_top", "total_row", "detected", "score"}
    back = TableGrid.from_dict(json.loads(json.dumps(d)))
    assert back == grid
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
