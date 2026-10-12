"""Test delle barrature e delle barrette ("/") riconosciute dal motore locale."""

from __future__ import annotations

import functools

import cv2
import numpy as np

from sirio.engines import strokes as st
from sirio.vision.grid import detect_grid
from sirio.vision.preprocess import normalize_page
from tests.synthetic import make_synthetic_sheet


@functools.cache
def _sheet(seed: int = 0):
    img, truth = make_synthetic_sheet(seed=seed)
    page = normalize_page(img)
    return page, detect_grid(page), truth


def _wobbly_line(img: np.ndarray, p0, p1, th: int, color=(60, 60, 60), gaps: bool = False, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = 60
    xs = np.linspace(p0[0], p1[0], n)
    ys = np.linspace(p0[1], p1[1], n) + np.cumsum(rng.normal(0, 0.6, n))
    for i in range(n - 1):
        if gaps and i % 7 == 3:
            continue          # tratto leggero e spezzato
        cv2.line(img, (int(xs[i]), int(ys[i])), (int(xs[i + 1]), int(ys[i + 1])), color, th, cv2.LINE_AA)


def test_synthetic_sheet_has_no_strikes():
    """Scrittura, firme (anche con svolazzi e sconfinamenti), crocette e trattini non sono barrature."""
    for seed in (0, 1, 2):
        page, grid, _ = _sheet(seed)
        info = st.find_strikes(cv2.cvtColor(page, cv2.COLOR_BGR2GRAY), grid)
        assert not info.strokes, (seed, info.strokes)
        assert not info.cells


def test_diagonal_strike_over_unused_rows_is_found_and_erased():
    page, grid, _ = _sheet(0)
    img = page.copy()
    # righe 29-31 annullate con una lunga linea diagonale leggera e spezzata
    p0 = (grid.col_x[1] + 5, grid.row_y[31] - 5)
    p1 = (grid.col_x[4] - 5, grid.row_y[28] + 5)
    _wobbly_line(img, p0, p1, 2, color=(110, 110, 110), gaps=True)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    info = st.find_strikes(gray, grid)
    assert len(info.strokes) == 1 and not info.strokes[0].horizontal
    days = {g for g, _c in info.cells}
    assert days <= {29, 30, 31} and len(days) >= 2
    erased = st.erase(gray, info.mask)
    drawn = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY).astype(int) - gray.astype(int) > 40
    assert drawn.sum() > 100
    assert (erased[drawn] < 200).sum() < 0.1 * drawn.sum()


def test_horizontal_line_across_effective_times():
    page, grid, truth = _sheet(0)
    img = page.copy()
    g = 7           # riga vuota (sabato): linea dagli orari effettivi alla firma
    ym = (grid.row_y[g - 1] + grid.row_y[g]) // 2
    _wobbly_line(img, (grid.col_x[3] + 10, ym), (grid.col_x[8] - 10, ym + 3), 2)
    info = st.find_strikes(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), grid)
    assert info.strokes and all(s.horizontal for s in info.strokes)
    assert info.horizontal(g, 3) and info.horizontal(g, 4)
    assert all(day == g for day, _c in info.cells)


def test_line_through_written_cells_is_not_taken_for_writing():
    """Una linea che attraversa celle scritte non fonde le cifre in una barratura:
    restano trovati (e cancellati) solo i pezzi nelle celle vuote."""
    page, grid, _truth = _sheet(0)
    img = page.copy()
    g = 18
    ym = (grid.row_y[g - 1] + grid.row_y[g]) // 2
    _wobbly_line(img, (grid.col_x[1] + 10, ym), (grid.col_x[8] - 10, ym + 3), 2)
    info = st.find_strikes(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), grid)
    assert all(day == g for day, _c in info.cells)
    assert info.horizontal(g, 6) and info.horizontal(g, 7)


def test_slash_versus_digits():
    unit = 40.0

    def canvas():
        return np.full((60, 90), 255, np.uint8)

    slash = canvas()
    cv2.line(slash, (25, 48), (60, 14), 0, 3, cv2.LINE_AA)
    assert st.is_slash(slash, unit)
    back = canvas()
    cv2.line(back, (25, 14), (60, 48), 0, 3, cv2.LINE_AA)
    assert st.is_slash(back, unit)
    one = canvas()
    cv2.line(one, (44, 50), (50, 10), 0, 3, cv2.LINE_AA)          # "1" quasi verticale
    assert not st.is_slash(one, unit)
    dash = canvas()
    cv2.line(dash, (25, 30), (65, 31), 0, 3, cv2.LINE_AA)         # trattino
    assert not st.is_slash(dash, unit)
    digits = canvas()
    cv2.putText(digits, "8", (10, 45), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)
    cv2.line(digits, (50, 48), (80, 14), 0, 3, cv2.LINE_AA)
    assert not st.is_slash(digits, unit)
    assert not st.is_slash(canvas(), unit)


def test_stroke_shape_of_a_straight_line():
    xs = np.arange(0, 200, dtype=float)
    ys = 0.5 * xs + 3
    length, thick, dev, angle = st.stroke_shape(xs, ys)
    assert 220 <= length <= 225 and dev < 1e-6 and 26 <= angle <= 27.5
