#!/usr/bin/env python3
"""
CLI utility to extract headings/text plus captioned tables/images from DOCX.

Tables are exported as CSV, figures as their original binary files.

Example:
    poetry run python scripts/docx_etl.py input.docx output.txt
"""

from __future__ import annotations

import argparse
import csv
import mimetypes
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from docx import Document
from docx.oxml.ns import nsmap
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

PICTURE_NS = {
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": nsmap["r"],
}
CAPTION_STYLE_NAMES = {"Caption"}
FIGURE_PREFIXES = ("рис", "figure", "рисунок")
TABLE_PREFIXES = ("таблица", "table")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract headings and text from a DOCX file.",
    )
    parser.add_argument("input_docx", type=Path, help="Path to the DOCX file to parse.")
    parser.add_argument(
        "output_txt",
        type=Path,
        help="Path to write the formatted heading: text output.",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        help="Directory to store extracted tables/images (default: next to output).",
    )
    parser.add_argument(
        "--heading-prefix",
        default="Heading",
        help='Prefix that identifies heading styles (default: "Heading").',
    )
    return parser.parse_args()


def iter_sections(document: Document, heading_prefix: str) -> Iterable[tuple[str, list[str]]]:
    current_title: str | None = None
    current_chunks: list[str] = []

    def flush():
        nonlocal current_title, current_chunks
        if current_title:
            paragraphs = [chunk for chunk in current_chunks if chunk]
            if paragraphs:
                yield (current_title, paragraphs)
        current_title = None
        current_chunks = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        style_name = paragraph.style.name if paragraph.style else ""
        is_heading = style_name.startswith(heading_prefix)

        if is_heading:
            if current_title is not None:
                yield from flush()
            current_title = text
            current_chunks = []
            continue

        if current_title is not None:
            current_chunks.append(text)

    if current_title is not None:
        yield from flush()


def iter_block_items(document: Document) -> Iterator[Paragraph | Table]:
    body = document.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def looks_like_caption(text: str, style_name: str | None) -> bool:
    if not text:
        return False
    if style_name and style_name in CAPTION_STYLE_NAMES:
        return True
    lowered = text.lower()
    return any(lowered.startswith(prefix) for prefix in FIGURE_PREFIXES + TABLE_PREFIXES)


def caption_kind(text: str) -> str:
    lowered = text.lower()
    if any(lowered.startswith(prefix) for prefix in TABLE_PREFIXES):
        return "table"
    return "figure"


def sanitize_caption(text: str) -> str:
    base = re.sub(r"\s+", "_", text.strip())
    base = re.sub(r"[^\w.\-]", "_", base, flags=re.UNICODE)
    return base.strip("._")[:128] or "asset"


def unique_name(base: str, extension: str, used: set[str]) -> str:
    candidate = f"{base}{extension}"
    counter = 1
    while candidate in used:
        candidate = f"{base}_{counter}{extension}"
        counter += 1
    used.add(candidate)
    return candidate


def save_image(paragraph: Paragraph, document: Document, caption: str, dest_dir: Path, used: set[str]) -> int:
    drawings = paragraph._element.xpath(".//*[local-name()='pic']")
    if not drawings:
        return 0

    saved = 0
    for drawing in drawings:
        blip = drawing.xpath(".//*[local-name()='blip']")
        if not blip:
            continue
        embed_id = blip[0].get(f"{{{PICTURE_NS['r']}}}embed")
        if not embed_id or embed_id not in document.part.related_parts:
            continue
        image_part = document.part.related_parts[embed_id]
        ext = Path(image_part.filename).suffix or mimetypes.guess_extension(image_part.content_type) or ".bin"
        filename = unique_name(sanitize_caption(caption), ext, used)
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / filename).write_bytes(image_part.blob)
        saved += 1
    return saved


def save_table(table: Table, caption: str, dest_dir: Path, used: set[str]) -> bool:
    filename = unique_name(sanitize_caption(caption), ".csv", used)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with (dest_dir / filename).open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        wrote = False
        for row in table.rows:
            cells = [" ".join(cell.text.split()) for cell in row.cells]
            writer.writerow(cells)
            wrote = True
    return wrote


def extract_assets(document: Document, assets_dir: Path) -> tuple[int, int]:
    pending_caption: str | None = None
    pending_kind: str | None = None
    used_names: set[str] = set()
    saved_images = 0
    saved_tables = 0

    for block in iter_block_items(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            style_name = block.style.name if block.style else ""

            if looks_like_caption(text, style_name):
                pending_caption = text
                pending_kind = caption_kind(text)
                continue

            if pending_caption and (pending_kind in (None, "figure")):
                saved = save_image(block, document, pending_caption, assets_dir, used_names)
                if saved:
                    saved_images += saved
                    pending_caption = None
                    pending_kind = None
            continue

        if isinstance(block, Table) and pending_caption and pending_kind == "table":
            if save_table(block, pending_caption, assets_dir, used_names):
                saved_tables += 1
                pending_caption = None
                pending_kind = None

    return saved_images, saved_tables


def write_output(sections: Iterable[tuple[str, list[str]]], output_path: Path) -> None:
    lines = []
    for title, paragraphs in sections:
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            lines.append(f"{title}: {paragraph}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.input_docx.exists():
        print(f"Input file not found: {args.input_docx}", file=sys.stderr)
        return 1

    document = Document(args.input_docx)
    sections = list(iter_sections(document, heading_prefix=args.heading_prefix))
    if not sections:
        print("No headings found. Nothing to write.", file=sys.stderr)
        return 2

    write_output(sections, args.output_txt)
    assets_dir = args.assets_dir or args.output_txt.parent / f"{args.output_txt.stem}_assets"
    images_saved, tables_saved = extract_assets(document, assets_dir)
    print(
        f"Wrote {len(sections)} sections to {args.output_txt} | "
        f"assets saved: {images_saved} images, {tables_saved} tables -> {assets_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

