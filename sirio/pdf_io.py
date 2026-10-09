"""Lettura di PDF e immagini come pagine (numpy BGR) e codifica JPEG/PNG.

* PDF (anche multipagina) tramite PyMuPDF: ogni pagina viene renderizzata a
  ``dpi`` (300 per le scansioni), rispettando la rotazione della pagina.
* Immagini JPG/PNG/TIFF (anche multipagina)/BMP/WEBP tramite Pillow, con
  applicazione dell'orientamento EXIF.
* Si lavora sempre sui byte del file: i nomi con caratteri accentati o non
  latini non sono un problema (il nome serve solo a riconoscere l'estensione).
* Ogni errore di formato produce un ``ValueError`` con un messaggio in
  italiano comprensibile per l'utente.
"""

from __future__ import annotations

import io
import logging
import math
import os
from collections.abc import Iterator

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
IMAGE_EXTENSIONS = SUPPORTED_EXTENSIONS - {".pdf"}

DEFAULT_DPI = 300
# Limite di sicurezza per il rendering (A3 a 300 dpi ~ 17 Mpx): oltre si riduce la risoluzione.
MAX_RENDER_PIXELS = 40_000_000
# Le immagini molto grandi (es. scansioni a 600 dpi) vengono ridotte a circa
# ``dpi`` per un A4, con un margine generoso.
_A4_INCHES = (8.27, 11.69)
_IMAGE_PIXEL_FACTOR = 1.6

_MAGIC_IMAGE = (
    b"\x89PNG\r\n\x1a\n",     # PNG
    b"\xff\xd8\xff",          # JPEG
    b"II*\x00",               # TIFF little endian
    b"MM\x00*",               # TIFF big endian
    b"II+\x00",               # BigTIFF
    b"MM\x00+",
    b"BM",                    # BMP
)


# --------------------------------------------------------------------------
# Riconoscimento del formato
# --------------------------------------------------------------------------

def file_extension(filename: str) -> str:
    """Estensione in minuscolo (con il punto) del nome file, anche con percorso."""
    name = os.path.basename(str(filename or "").replace("\\", "/"))
    return os.path.splitext(name)[1].lower()


def is_supported(filename: str) -> bool:
    """True se l'estensione del file e' tra quelle gestite."""
    return file_extension(filename) in SUPPORTED_EXTENSIONS


def _display_name(filename: str) -> str:
    name = os.path.basename(str(filename or "").replace("\\", "/"))
    return name or "file"


def _looks_like_pdf(data: bytes) -> bool:
    return b"%PDF" in data[:1024]


def _looks_like_image(data: bytes) -> bool:
    if data.startswith(_MAGIC_IMAGE):
        return True
    return data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def _detect_kind(data: bytes, filename: str) -> str:
    """'pdf' o 'image' in base al contenuto (e, in subordine, all'estensione)."""
    if not data:
        raise ValueError(f"Il file «{_display_name(filename)}» è vuoto.")
    if _looks_like_pdf(data):
        return "pdf"
    if _looks_like_image(data):
        return "image"
    ext = file_extension(filename)
    if ext == ".pdf":
        return "pdf"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext:
        raise ValueError(
            f"Formato «{ext}» non supportato per «{_display_name(filename)}»: "
            "sono accettati PDF e immagini JPG, PNG, TIFF, BMP o WEBP."
        )
    raise ValueError(
        f"Impossibile riconoscere il formato di «{_display_name(filename)}»: "
        "sono accettati PDF e immagini JPG, PNG, TIFF, BMP o WEBP."
    )


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

_pymupdf_ready = False


def _pymupdf():
    """Importa PyMuPDF (import pesante, solo quando serve) e silenzia i messaggi
    di MuPDF su stderr: gli errori arrivano comunque come eccezioni."""
    global _pymupdf_ready
    import pymupdf  # noqa: PLC0415

    if not _pymupdf_ready:
        try:
            pymupdf.TOOLS.mupdf_display_errors(False)
            pymupdf.TOOLS.mupdf_display_warnings(False)
        except Exception:  # noqa: BLE001 - versioni senza queste opzioni
            pass
        _pymupdf_ready = True
    return pymupdf


def _open_pdf(data: bytes, filename: str):
    pymupdf = _pymupdf()
    name = _display_name(filename)
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - PyMuPDF solleva tipi diversi per i file corrotti
        raise ValueError(f"Il PDF «{name}» è danneggiato o non è un PDF valido.") from exc
    try:
        if doc.needs_pass and not doc.authenticate(""):
            raise ValueError(
                f"Il PDF «{name}» è protetto da password: rimuovere la protezione e riprovare."
            )
        if doc.page_count <= 0:
            raise ValueError(f"Il PDF «{name}» non contiene pagine.")
    except ValueError:
        doc.close()
        raise
    except Exception as exc:  # noqa: BLE001
        doc.close()
        raise ValueError(f"Il PDF «{name}» è danneggiato o non leggibile.") from exc
    return doc


def _pixmap_to_bgr(pix) -> np.ndarray:
    n = pix.n
    buf = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = buf.reshape(pix.height, pix.stride)[:, : pix.width * n].reshape(pix.height, pix.width, n)
    if n == 1:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if n == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    raise ValueError(f"Formato colore della pagina non gestito ({n} canali).")


def _render_page(page, dpi: int):
    pymupdf = _pymupdf()
    rect = page.rect
    zoom = max(dpi, 10) / 72.0
    pixels = rect.width * zoom * rect.height * zoom
    if pixels > MAX_RENDER_PIXELS:
        zoom *= math.sqrt(MAX_RENDER_PIXELS / pixels)
        log.info("Pagina molto grande: rendering ridotto a %.0f dpi", zoom * 72)
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False,
                           annots=True)


def _iter_pdf(data: bytes, filename: str, dpi: int) -> Iterator[tuple[int, np.ndarray]]:
    doc = _open_pdf(data, filename)
    name = _display_name(filename)
    try:
        for i in range(doc.page_count):
            try:
                page = doc.load_page(i)
                pix = _render_page(page, dpi)
                img = _pixmap_to_bgr(pix)
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"Impossibile leggere la pagina {i + 1} del PDF «{name}»: il file è danneggiato."
                ) from exc
            yield i, img
    finally:
        doc.close()


# --------------------------------------------------------------------------
# Immagini
# --------------------------------------------------------------------------

def _open_image(data: bytes, filename: str = "") -> Image.Image:
    subject = f"L'immagine «{_display_name(filename)}»" if filename else "L'immagine"
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Image.DecompressionBombError as exc:
        raise ValueError(f"{subject} è troppo grande per essere elaborata.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"{subject} è danneggiata o in un formato non riconosciuto.") from exc
    return im


def _pil_to_bgr(im: Image.Image) -> np.ndarray:
    """Converte un fotogramma Pillow (qualsiasi modo) in BGR uint8, sfondo bianco
    per la trasparenza, orientamento EXIF applicato."""
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:  # noqa: BLE001 - EXIF malformati: si usa l'immagine com'e'
        log.debug("Orientamento EXIF non applicabile", exc_info=True)
    mode = im.mode
    if mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        arr = np.asarray(im, dtype=np.float64)
        if mode.startswith("I;16"):
            arr = arr / 257.0
        else:
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
            arr = (arr - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(arr)
        gray = np.clip(arr, 0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, rgba).convert("RGB")
    elif mode in ("1", "L"):
        gray = np.asarray(im.convert("L"), dtype=np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode != "RGB":
        im = im.convert("RGB")
    rgb = np.asarray(im, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def decode_image(data: bytes) -> np.ndarray:
    """Decodifica un'immagine (primo fotogramma) in BGR uint8, con orientamento EXIF."""
    if not data:
        raise ValueError("L'immagine è vuota.")
    im = _open_image(data)
    try:
        return _pil_to_bgr(im)
    finally:
        im.close()


def _image_frames(im: Image.Image) -> Iterator[Image.Image]:
    n = int(getattr(im, "n_frames", 1) or 1)
    if n <= 1:
        yield im
        return
    for frame in ImageSequence.Iterator(im):
        yield frame


def _limit_image(img: np.ndarray, dpi: int, source_dpi: float | None) -> np.ndarray:
    """Riduce le immagini molto piu' risolute del necessario (mai ingrandisce)."""
    if source_dpi and source_dpi > dpi * 1.2:
        scale = dpi / float(source_dpi)
        h, w = img.shape[:2]
        if min(h, w) * scale >= 600:
            return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                              interpolation=cv2.INTER_AREA)
    max_pixels = int(_A4_INCHES[0] * dpi * _A4_INCHES[1] * dpi * _IMAGE_PIXEL_FACTOR)
    return resize_max(img, max_pixels=max_pixels)


def _source_dpi(im: Image.Image) -> float | None:
    info = im.info.get("dpi")
    try:
        if info:
            val = float(max(info[0], info[1]))
            if 50 <= val <= 2400:
                return val
    except (TypeError, ValueError, IndexError):
        return None
    return None


def _iter_image(data: bytes, filename: str, dpi: int) -> Iterator[tuple[int, np.ndarray]]:
    im = _open_image(data, filename)
    name = _display_name(filename)
    try:
        for i, frame in enumerate(_image_frames(im)):
            try:
                img = _pil_to_bgr(frame.copy())
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"Impossibile leggere la pagina {i + 1} dell'immagine «{name}»: il file è danneggiato."
                ) from exc
            yield i, _limit_image(img, dpi, _source_dpi(frame))
    finally:
        im.close()


# --------------------------------------------------------------------------
# API pubblica
# --------------------------------------------------------------------------

def count_pages(data: bytes, filename: str) -> int:
    """Numero di pagine del file (PDF o immagine, TIFF multipagina incluso)."""
    kind = _detect_kind(data, filename)
    if kind == "pdf":
        doc = _open_pdf(data, filename)
        try:
            return int(doc.page_count)
        finally:
            doc.close()
    im = _open_image(data, filename)
    try:
        return max(1, int(getattr(im, "n_frames", 1) or 1))
    finally:
        im.close()


def iter_pages(data: bytes, filename: str, dpi: int = DEFAULT_DPI) -> Iterator[tuple[int, np.ndarray]]:
    """Pagine del file come ``(indice 0-based, immagine BGR uint8)``.

    I PDF sono renderizzati a ``dpi``; le immagini sono restituite alla loro
    risoluzione (ridotte solo se molto piu' grandi di un A4 a ``dpi``).
    Solleva ``ValueError`` (messaggio in italiano) per file vuoti, danneggiati,
    protetti da password o di formato non supportato.
    """
    kind = _detect_kind(data, filename)
    if kind == "pdf":
        yield from _iter_pdf(data, filename, dpi)
    else:
        yield from _iter_image(data, filename, dpi)


def resize_max(img: np.ndarray, max_side: int | None = None, max_pixels: int | None = None) -> np.ndarray:
    """Riduce l'immagine (mai la ingrandisce) perche' il lato lungo sia <= ``max_side``
    e l'area <= ``max_pixels``. Se non serve, restituisce l'immagine invariata."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    scale = 1.0
    if max_side and max(h, w) > max_side:
        scale = min(scale, max_side / float(max(h, w)))
    if max_pixels and h * w > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / float(h * w)))
    if scale >= 1.0:
        return img
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _prepare_for_encoding(img: np.ndarray) -> np.ndarray:
    if not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Immagine vuota: impossibile codificarla.")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    return img


def encode_jpeg(img: np.ndarray, quality: int = 90, max_side: int | None = None) -> bytes:
    """Codifica in JPEG (BGR o grigio), eventualmente ridotta a ``max_side``."""
    img = _prepare_for_encoding(resize_max(img, max_side=max_side))
    q = int(max(1, min(100, quality)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    if not ok:
        raise ValueError("Codifica JPEG non riuscita.")
    return buf.tobytes()


def encode_png(img: np.ndarray, max_side: int | None = None) -> bytes:
    """Codifica in PNG (senza perdita), eventualmente ridotta a ``max_side``."""
    img = _prepare_for_encoding(resize_max(img, max_side=max_side))
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError("Codifica PNG non riuscita.")
    return buf.tobytes()


__all__ = [
    "SUPPORTED_EXTENSIONS",
    "count_pages",
    "decode_image",
    "encode_jpeg",
    "encode_png",
    "file_extension",
    "is_supported",
    "iter_pages",
    "resize_max",
]
