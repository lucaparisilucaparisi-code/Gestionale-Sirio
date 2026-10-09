"""Genera l'icona dell'applicazione (sirio.ico / sirio.png) con Pillow.

Uso (solo per chi sviluppa; i file generati sono già nel repository):
    python launcher/crea_icona.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
SIZE = 1024  # disegno ad alta risoluzione, poi ridimensionato


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def _star(cx: float, cy: float, r_out: float, r_in: float, points: int = 4, rot: float = -90.0):
    pts = []
    for i in range(points * 2):
        r = r_out if i % 2 == 0 else r_in
        a = math.radians(rot + i * 180.0 / points)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def draw_icon() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))

    # Sfondo: quadrato arrotondato blu notte con sfumatura diagonale.
    bg = Image.new("RGBA", (SIZE, SIZE))
    top, bottom = (20, 33, 61), (8, 14, 30)
    px = bg.load()
    for y in range(SIZE):
        for x in range(SIZE):
            t = (x + y) / (2 * SIZE)
            px[x, y] = (*_lerp(top, bottom, t), 255)
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, SIZE - 1, SIZE - 1), radius=int(SIZE * 0.22), fill=255)
    img.paste(bg, (0, 0), mask)

    c = SIZE / 2
    # Alone luminoso azzurro dietro la stella.
    glow = Image.new("RGBA", (SIZE, SIZE), (59, 130, 246, 0))  # colore uniforme: la sfocatura non scurisce i bordi
    ImageDraw.Draw(glow).ellipse((c - 300, c - 300, c + 300, c + 300), fill=(59, 130, 246, 150))
    glow = glow.filter(ImageFilter.GaussianBlur(110))
    img = Image.alpha_composite(img, Image.composite(glow, Image.new("RGBA", glow.size), mask))

    d = ImageDraw.Draw(img)
    # Stella secondaria (diagonale), poi stella principale a 4 punte: Sirio.
    d.polygon(_star(c, c, 250, 62, rot=-45), fill=(96, 165, 250, 210))
    d.polygon(_star(c, c, 390, 78), fill=(255, 255, 255, 255))
    # Riflesso azzurro sulla metà inferiore destra della stella principale.
    shade = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(shade).polygon(
        [(c, c - 390), (c + 78 * 0.7, c - 78 * 0.7), (c + 390, c), (c, c)], fill=(191, 219, 254, 255)
    )
    ImageDraw.Draw(shade).polygon(
        [(c, c), (c - 78 * 0.7, c + 78 * 0.7), (c, c + 390)], fill=(191, 219, 254, 255)
    )
    img = Image.alpha_composite(img, shade)
    # Nucleo luminoso.
    core = Image.new("RGBA", (SIZE, SIZE), (255, 255, 255, 0))
    ImageDraw.Draw(core).ellipse((c - 46, c - 46, c + 46, c + 46), fill=(255, 255, 255, 255))
    core = core.filter(ImageFilter.GaussianBlur(10))
    return Image.alpha_composite(img, core)


def main() -> None:
    icon = draw_icon()
    icon.resize((512, 512), Image.LANCZOS).save(HERE / "sirio.png")
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    icon.resize((256, 256), Image.LANCZOS).save(HERE / "sirio.ico", sizes=sizes)
    print("Creati:", HERE / "sirio.png", HERE / "sirio.ico")


if __name__ == "__main__":
    main()
