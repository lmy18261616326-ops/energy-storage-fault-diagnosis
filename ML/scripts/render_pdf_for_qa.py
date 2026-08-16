#!/usr/bin/env python3
"""Render a PDF to page PNGs and compact contact sheets for visual QA."""

from __future__ import annotations

import sys
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: render_pdf_for_qa.py INPUT.pdf OUTPUT_DIR")
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(source))
    pages: list[Path] = []
    thumbs: list[Image.Image] = []
    for index in range(len(pdf)):
        page = pdf[index]
        bitmap = page.render(scale=1.75)
        image = bitmap.to_pil().convert("RGB")
        target = output / f"page_{index + 1:02d}.png"
        image.save(target, optimize=True)
        pages.append(target)
        thumb = image.copy()
        thumb.thumbnail((820, 1060), Image.Resampling.LANCZOS)
        thumbs.append(thumb)

    for start in range(0, len(thumbs), 4):
        group = thumbs[start:start + 4]
        sheet = Image.new("RGB", (1720, 2220), "#D8DEE5")
        draw = ImageDraw.Draw(sheet)
        for offset, thumb in enumerate(group):
            x = 25 + (offset % 2) * 850
            y = 55 + (offset // 2) * 1090
            sheet.paste(thumb, (x, y))
            draw.text((x, 22 + (offset // 2) * 1090), f"Page {start + offset + 1}", fill="#17324D")
        sheet.save(output / f"contact_{start + 1:02d}_{start + len(group):02d}.png", optimize=True)
    print(f"rendered_pages={len(pages)}")


if __name__ == "__main__":
    main()
