#!/usr/bin/env python3
"""Convert Eric Lengyel's official C++ Transvoxel tables into Rust constants."""

from __future__ import annotations

import pathlib
import re
import sys


def table(source: str, name: str) -> str:
    start = source.index(name)
    start = source.index("{", start)
    depth = 0
    for cursor in range(start, len(source)):
        if source[cursor] == "{":
            depth += 1
        elif source[cursor] == "}":
            depth -= 1
            if depth == 0:
                return source[start : cursor + 1]
    raise ValueError(f"unterminated {name}")


def numbers(body: str) -> list[int]:
    return [int(value, 0) for value in re.findall(r"0x[0-9A-Fa-f]+|\d+", body)]


def rows(body: str) -> list[list[int]]:
    result: list[list[int]] = []
    depth = 0
    start = None
    for cursor, character in enumerate(body[1:-1], 1):
        if character == "{":
            if depth == 0:
                start = cursor
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start is not None:
                result.append(numbers(body[start : cursor + 1]))
                start = None
    return result


def padded(values: list[int], length: int, width: int) -> str:
    values = values + [0] * (length - len(values))
    return "[" + ", ".join(f"0x{value:0{width}x}" for value in values) + "]"


def main() -> None:
    source = pathlib.Path(sys.argv[1]).read_text()
    destination = pathlib.Path(sys.argv[2])
    regular_class = numbers(table(source, "regularCellClass"))
    regular_cells = rows(table(source, "regularCellData"))
    regular_vertices = rows(table(source, "regularVertexData"))
    transition_class = numbers(table(source, "transitionCellClass"))
    transition_cells = rows(table(source, "transitionCellData"))
    transition_corners = numbers(table(source, "transitionCornerData"))
    transition_vertices = rows(table(source, "transitionVertexData"))

    assert len(regular_class) == 256
    assert len(regular_cells) == 16
    assert len(regular_vertices) == 256
    assert len(transition_class) == 512
    assert len(transition_cells) == 56
    assert len(transition_corners) == 13
    assert len(transition_vertices) == 512

    lines = [
        "// Generated from https://github.com/EricLengyel/Transvoxel (main).",
        "// Copyright 2009 Eric Lengyel. MIT license: transvoxel/LICENSE.",
        "// Do not edit by hand; regenerate with scripts/generate_transvoxel_tables.py.",
        "",
        "#[derive(Clone, Copy)]",
        "pub(crate) struct CellData<const N: usize> {",
        "    pub(crate) geometry_counts: u8,",
        "    pub(crate) vertex_index: [u8; N],",
        "}",
        "",
        f"pub(crate) const REGULAR_CELL_CLASS: [u8; 256] = {padded(regular_class, 256, 2)};",
        "pub(crate) const REGULAR_CELL_DATA: [CellData<15>; 16] = [",
    ]
    for row in regular_cells:
        lines.append(
            f"    CellData {{ geometry_counts: 0x{row[0]:02x}, vertex_index: {padded(row[1:], 15, 2)} }},"
        )
    lines.extend([
        "];",
        "pub(crate) const REGULAR_VERTEX_DATA: [[u16; 12]; 256] = [",
    ])
    lines.extend(f"    {padded(row, 12, 4)}," for row in regular_vertices)
    lines.extend([
        "];",
        f"pub(crate) const TRANSITION_CELL_CLASS: [u8; 512] = {padded(transition_class, 512, 2)};",
        "pub(crate) const TRANSITION_CELL_DATA: [CellData<36>; 56] = [",
    ])
    for row in transition_cells:
        lines.append(
            f"    CellData {{ geometry_counts: 0x{row[0]:02x}, vertex_index: {padded(row[1:], 36, 2)} }},"
        )
    lines.extend([
        "];",
        f"pub(crate) const TRANSITION_CORNER_DATA: [u8; 13] = {padded(transition_corners, 13, 2)};",
        "pub(crate) const TRANSITION_VERTEX_DATA: [[u16; 12]; 512] = [",
    ])
    lines.extend(f"    {padded(row, 12, 4)}," for row in transition_vertices)
    lines.append("];\n")
    destination.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
