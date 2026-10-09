"""Test di sirio.pdf_io: lettura PDF/immagini, codifica e messaggi d'errore."""

from __future__ import annotations

import io
import os

import cv2
import numpy as np
import pytest
from PIL import Image

from sirio import pdf_io
from tests.synthetic import make_synthetic_sheet, sheet_to_pdf


@pytest.fixture(scope="module")
def sheet() -> np.ndarray:
    img, _ = make_synthetic_sheet(seed=11, dpi=150)
    return img


def _pattern(h: int = 120, w: int = 80) -> np.ndarray:
    """Immagine BGR asimmetrica (per verificare rotazioni e canali)."""
    img = np.full((h, w, 3), 255, np.uint8)
    img[: h // 4, : w // 2] = (255, 0, 0)        # blu in alto a sinistra
    img[-h // 4:, w // 2:] = (0, 0, 255)         # rosso in basso a destra
    return img


def _pil_bytes(im: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format=fmt, **kw)
    return buf.getvalue()


def _rgb(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# --------------------------------------------------------------------------
# Riconoscimento dei file
# --------------------------------------------------------------------------

def test_supported_extensions_and_names():
    assert pdf_io.SUPPORTED_EXTENSIONS == {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    assert pdf_io.is_supported("Foglio firma – città è ñ 日本.PDF")
    assert pdf_io.is_supported(r"C:\Scansioni\Febbraio\pagina.JpEg")
    assert pdf_io.is_supported("cartella/sotto/scan.tiff")
    assert not pdf_io.is_supported("documento.docx")
    assert not pdf_io.is_supported("senza_estensione")
    assert pdf_io.file_extension("archivio.tar.PDF") == ".pdf"


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

def test_pdf_roundtrip_multipage(sheet):
    second = cv2.rotate(sheet, cv2.ROTATE_180)
    data = sheet_to_pdf([sheet, second], dpi=150)
    name = "Fogli firma – Febbraio è.pdf"
    assert pdf_io.count_pages(data, name) == 2
    pages = list(pdf_io.iter_pages(data, name))
    assert [i for i, _ in pages] == [0, 1]
    for _, img in pages:
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
        # A4 a 300 dpi
        assert abs(img.shape[1] - 2481) <= 3 and abs(img.shape[0] - 3507) <= 3
    # il contenuto corrisponde all'originale (a meno della compressione JPEG)
    back = cv2.resize(pages[0][1], (sheet.shape[1], sheet.shape[0]), interpolation=cv2.INTER_AREA)
    diff = np.abs(back.astype(np.int16) - sheet.astype(np.int16)).mean()
    assert diff < 6.0
    back2 = cv2.resize(pages[1][1], (sheet.shape[1], sheet.shape[0]), interpolation=cv2.INTER_AREA)
    assert np.abs(back2.astype(np.int16) - second.astype(np.int16)).mean() < 6.0


def test_pdf_custom_dpi(sheet):
    data = sheet_to_pdf([sheet], dpi=150)
    (_, img), = list(pdf_io.iter_pages(data, "x.pdf", dpi=100))
    assert abs(img.shape[1] - 827) <= 2 and abs(img.shape[0] - 1169) <= 2


def test_pdf_page_rotation_is_applied(sheet):
    import pymupdf

    doc = pymupdf.open(stream=sheet_to_pdf([sheet], dpi=150), filetype="pdf")
    doc[0].set_rotation(90)
    data = doc.tobytes()
    doc.close()
    (_, img), = list(pdf_io.iter_pages(data, "ruotato.pdf", dpi=100))
    assert img.shape[1] > img.shape[0]          # pagina orizzontale come la vede il lettore PDF


def test_pdf_detected_by_content_even_with_wrong_extension(sheet):
    data = sheet_to_pdf([sheet], dpi=150)
    assert pdf_io.count_pages(data, "scansione.jpg") == 1
    assert pdf_io.count_pages(data, "senza_estensione") == 1


def test_encrypted_pdf_gives_clear_error(sheet):
    import pymupdf

    doc = pymupdf.open(stream=sheet_to_pdf([sheet], dpi=150), filetype="pdf")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="proprietario", user_pw="segreto")
    doc.close()
    with pytest.raises(ValueError, match="password"):
        pdf_io.count_pages(data, "protetto.pdf")
    with pytest.raises(ValueError, match="password"):
        list(pdf_io.iter_pages(data, "protetto.pdf"))


def test_pdf_with_owner_password_only_is_readable(sheet):
    import pymupdf

    doc = pymupdf.open(stream=sheet_to_pdf([sheet], dpi=150), filetype="pdf")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="proprietario", user_pw="")
    doc.close()
    assert pdf_io.count_pages(data, "solo_permessi.pdf") == 1
    assert len(list(pdf_io.iter_pages(data, "solo_permessi.pdf", dpi=72))) == 1


@pytest.mark.parametrize(
    "data, name, pattern",
    [
        (b"", "vuoto.pdf", "vuoto"),
        (b"%PDF-1.7\nquesto non e' un pdf", "rotto.pdf", "danneggiato"),
        (b"testo qualunque", "rotto.pdf", "danneggiato"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, "rotta.png", "danneggiata"),
        (b"testo qualunque", "rotta.jpg", "danneggiata"),
        (b"testo qualunque", "documento.docx", "non supportato"),
        (b"testo qualunque", "senza_estensione", "riconoscere"),
    ],
)
def test_invalid_files_raise_italian_value_error(data, name, pattern):
    with pytest.raises(ValueError, match=pattern):
        pdf_io.count_pages(data, name)
    with pytest.raises(ValueError, match=pattern):
        list(pdf_io.iter_pages(data, name))


def test_truncated_jpeg_is_reported(sheet):
    data = pdf_io.encode_jpeg(sheet, quality=85)
    with pytest.raises(ValueError, match="danneggiata"):
        list(pdf_io.iter_pages(data[: len(data) // 3], "troncata.jpg"))


def test_corrupted_pdf_pages_are_reported(sheet):
    data = bytearray(sheet_to_pdf([sheet], dpi=150))
    # danneggia il flusso dell'immagine incorporata
    start = len(data) // 3
    data[start:start + 4000] = b"\x00" * 4000
    try:
        pages = list(pdf_io.iter_pages(bytes(data), "danneggiato.pdf", dpi=72))
    except ValueError as exc:
        assert "PDF" in str(exc)
    else:  # MuPDF puo' riparare il file: in tal caso deve restituire pagine valide
        assert all(p.ndim == 3 and p.dtype == np.uint8 for _, p in pages)


# --------------------------------------------------------------------------
# Immagini
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fmt, ext", [("PNG", ".png"), ("BMP", ".bmp"), ("WEBP", ".webp"), ("TIFF", ".tif")])
def test_lossless_image_roundtrip(fmt, ext):
    img = _pattern()
    kw = {"lossless": True} if fmt == "WEBP" else {}
    data = _pil_bytes(_rgb(img), fmt, **kw)
    assert pdf_io.count_pages(data, "pagina" + ext) == 1
    pages = list(pdf_io.iter_pages(data, "pagina" + ext))
    assert len(pages) == 1 and pages[0][0] == 0
    assert np.array_equal(pages[0][1], img)
    assert np.array_equal(pdf_io.decode_image(data), img)


def test_jpeg_roundtrip():
    img = _pattern(300, 200)
    data = _pil_bytes(_rgb(img), "JPEG", quality=95)
    out = pdf_io.decode_image(data)
    assert out.shape == img.shape
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 4


@pytest.mark.parametrize(
    "orientation, rotate",
    [(3, cv2.ROTATE_180), (6, cv2.ROTATE_90_CLOCKWISE), (8, cv2.ROTATE_90_COUNTERCLOCKWISE)],
)
def test_exif_orientation_is_applied(orientation, rotate):
    img = _pattern(160, 100)
    exif = Image.Exif()
    exif[0x0112] = orientation
    data = _pil_bytes(_rgb(img), "JPEG", quality=95, exif=exif.tobytes())
    out = pdf_io.decode_image(data)
    expected = cv2.rotate(img, rotate)
    assert out.shape == expected.shape
    assert np.abs(out.astype(int) - expected.astype(int)).mean() < 5
    (_, page), = list(pdf_io.iter_pages(data, "foto.jpg"))
    assert page.shape == expected.shape


def test_multipage_tiff():
    frames = [np.full((90, 60, 3), v, np.uint8) for v in (30, 120, 210)]
    pil = [_rgb(f) for f in frames]
    buf = io.BytesIO()
    pil[0].save(buf, format="TIFF", save_all=True, append_images=pil[1:], compression="tiff_lzw")
    data = buf.getvalue()
    assert pdf_io.count_pages(data, "scansioni.TIFF") == 3
    pages = list(pdf_io.iter_pages(data, "scansioni.TIFF"))
    assert [i for i, _ in pages] == [0, 1, 2]
    for (_, page), frame in zip(pages, frames):
        assert np.array_equal(page, frame)


def test_grayscale_bilevel_alpha_and_16bit_images():
    gray = np.tile(np.arange(0, 256, 2, dtype=np.uint8), (20, 1))
    out = pdf_io.decode_image(_pil_bytes(Image.fromarray(gray, "L"), "PNG"))
    assert out.shape == (20, 128, 3) and np.array_equal(out[:, :, 0], gray)

    bilevel = Image.fromarray((gray > 127).astype(np.uint8) * 255, "L").convert("1")
    out = pdf_io.decode_image(_pil_bytes(bilevel, "PNG"))
    assert out.shape == (20, 128, 3) and set(np.unique(out)) <= {0, 255}

    rgba = np.zeros((10, 10, 4), np.uint8)
    rgba[:5, :, :] = (0, 0, 0, 255)          # meta' nera opaca, meta' trasparente
    out = pdf_io.decode_image(_pil_bytes(Image.fromarray(rgba, "RGBA"), "PNG"))
    assert (out[:5] == 0).all() and (out[5:] == 255).all()

    deep = (np.tile(np.linspace(0, 65535, 64), (8, 1))).astype(np.uint16)
    out = pdf_io.decode_image(_pil_bytes(Image.fromarray(deep), "PNG"))
    assert out.dtype == np.uint8 and out[0, 0, 0] == 0 and out[0, -1, 0] == 255


def test_high_resolution_images_are_reduced(sheet):
    # scansione a 600 dpi (dichiarata nel file): riportata a ~300 dpi
    hi = np.full((2400, 1800, 3), 255, np.uint8)
    cv2.rectangle(hi, (100, 100), (1700, 2300), (0, 0, 0), 8)
    data = _pil_bytes(_rgb(hi), "PNG", dpi=(600, 600), compress_level=1)
    (_, page), = list(pdf_io.iter_pages(data, "alta_risoluzione.png", dpi=300))
    assert page.shape[:2] == (1200, 900)
    # senza informazione DPI vale il limite in pixel (A4 a 300 dpi con margine)
    huge = np.full((5200, 3600, 3), 250, np.uint8)
    data = _pil_bytes(_rgb(huge), "PNG", compress_level=1)
    (_, page), = list(pdf_io.iter_pages(data, "senza_dpi.png", dpi=300))
    assert page.shape[0] * page.shape[1] <= 2481 * 3508 * 1.6 + 10
    assert page.shape[0] / page.shape[1] == pytest.approx(5200 / 3600, rel=0.01)
    # le immagini piccole non vengono ingrandite
    (_, page), = list(pdf_io.iter_pages(_pil_bytes(_rgb(sheet), "PNG"), "piccola.png", dpi=300))
    assert page.shape == sheet.shape


# --------------------------------------------------------------------------
# Codifica e ridimensionamento
# --------------------------------------------------------------------------

def test_resize_max():
    img = np.zeros((400, 300, 3), np.uint8)
    assert pdf_io.resize_max(img) is img
    assert pdf_io.resize_max(img, max_side=1000) is img
    assert pdf_io.resize_max(img, max_side=200).shape[:2] == (200, 150)
    out = pdf_io.resize_max(img, max_pixels=30_000)
    assert out.shape[0] * out.shape[1] <= 30_000 and out.shape[0] / out.shape[1] == pytest.approx(4 / 3, rel=0.02)
    out = pdf_io.resize_max(img, max_side=300, max_pixels=10_000)
    assert max(out.shape[:2]) <= 300 and out.shape[0] * out.shape[1] <= 10_000
    gray = np.zeros((50, 1000), np.uint8)
    assert pdf_io.resize_max(gray, max_side=100).shape == (5, 100)


def test_encode_jpeg_and_png(sheet):
    jpg = pdf_io.encode_jpeg(sheet, quality=80, max_side=600)
    assert jpg[:3] == b"\xff\xd8\xff"
    dec = pdf_io.decode_image(jpg)
    assert max(dec.shape[:2]) == 600
    small = pdf_io.encode_jpeg(sheet, quality=30)
    big = pdf_io.encode_jpeg(sheet, quality=95)
    assert len(small) < len(big)

    png = pdf_io.encode_png(sheet)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert np.array_equal(pdf_io.decode_image(png), sheet)
    assert max(pdf_io.decode_image(pdf_io.encode_png(sheet, max_side=500)).shape[:2]) == 500

    gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
    assert pdf_io.decode_image(pdf_io.encode_jpeg(gray)).shape == sheet.shape
    bgra = cv2.cvtColor(sheet, cv2.COLOR_BGR2BGRA)
    assert np.array_equal(pdf_io.decode_image(pdf_io.encode_png(bgra)), sheet)
    with pytest.raises(ValueError):
        pdf_io.encode_jpeg(np.zeros((0, 0, 3), np.uint8))


# --------------------------------------------------------------------------
# Foglio reale (solo se disponibile in locale)
# --------------------------------------------------------------------------

def test_real_sample_pdf():
    path = os.environ.get("SIRIO_SAMPLE_PDF")
    if not path or not os.path.exists(path):
        pytest.skip("SIRIO_SAMPLE_PDF non impostata")
    with open(path, "rb") as fh:
        data = fh.read()
    assert pdf_io.count_pages(data, os.path.basename(path)) == 1
    (idx, img), = list(pdf_io.iter_pages(data, os.path.basename(path)))
    assert idx == 0 and img.shape == (3509, 2480, 3) and img.dtype == np.uint8
    # la scansione non e' vuota: ci sono testo e linee scure
    assert (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 100).mean() > 0.02
