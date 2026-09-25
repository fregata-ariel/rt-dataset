"""Mitsuba scene XML file references, parsed with defusedxml."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import defusedxml
import defusedxml.ElementTree

DEFAULT_MAX_SCENE_XML_BYTES = 256 * 1024 * 1024
FILE_SHAPE_TYPES = ("ply", "obj", "serialized")

_DRIVE_RE = re.compile(r"[A-Za-z]:")


class SceneFileError(ValueError):
    """A scene XML could not be read (malformed, unsafe, too large, not <scene>)."""


@dataclass(frozen=True)
class ShapeRef:
    """One <shape> element's file reference and material link."""

    shape_id: str | None
    shape_type: str | None
    filename: str | None
    bsdf_id: str | None
    path: Path | None
    unsafe: bool


def _is_unsafe_filename(filename: str) -> bool:
    """Return True when a scene XML filename value must not be opened."""
    if filename == "" or "\x00" in filename:
        return True
    if filename.startswith("/") or filename.startswith("\\"):
        return True
    if _DRIVE_RE.match(filename) is not None:
        return True
    return any(segment == ".." for segment in filename.replace("\\", "/").split("/"))


def scene_file_references(
    xml_path: Path | str, *, max_bytes: int = DEFAULT_MAX_SCENE_XML_BYTES
) -> list[ShapeRef]:
    """Return one ShapeRef per <shape> element in document order."""
    path = Path(xml_path)
    size = os.path.getsize(path)
    if size > max_bytes:
        raise SceneFileError(f"{path}: {size} bytes exceeds {max_bytes}")
    try:
        tree = defusedxml.ElementTree.parse(
            os.fspath(path), forbid_dtd=True, forbid_entities=True, forbid_external=True
        )
    except defusedxml.DefusedXmlException as exc:
        raise SceneFileError(f"{path}: unsafe XML: {exc}") from exc
    except ET.ParseError as exc:
        raise SceneFileError(f"{path}: malformed XML: {exc}") from exc
    root = tree.getroot()
    if root is None:
        raise SceneFileError(f"{path}: malformed XML: empty document")
    if root.tag != "scene":
        raise SceneFileError(f"{path}: root element is <{root.tag}>, not <scene>")
    refs: list[ShapeRef] = []
    for shape in root.iter("shape"):
        shape_id = shape.get("id")
        shape_type = shape.get("type")
        filename: str | None = None
        for child in shape:
            if child.tag == "string" and child.get("name") == "filename":
                filename = child.get("value")
                break
        bsdf_id: str | None = None
        for child in shape:
            if child.tag == "ref" and child.get("name") in ("bsdf", None):
                bsdf_id = child.get("id")
                break
        if bsdf_id is None:
            for child in shape:
                if child.tag == "bsdf":
                    bsdf_id = child.get("id")
                    break
        if filename is not None and _is_unsafe_filename(filename):
            refs.append(
                ShapeRef(
                    shape_id=shape_id,
                    shape_type=shape_type,
                    filename=filename,
                    bsdf_id=bsdf_id,
                    path=None,
                    unsafe=True,
                )
            )
        else:
            refs.append(
                ShapeRef(
                    shape_id=shape_id,
                    shape_type=shape_type,
                    filename=filename,
                    bsdf_id=bsdf_id,
                    path=path.parent / filename if filename is not None else None,
                    unsafe=False,
                )
            )
    return refs
