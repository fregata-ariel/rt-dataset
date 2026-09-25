"""Doc-consistency tests for docs/viewer_bundle.md (issue #31)."""

import json
import re
from pathlib import Path

DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "viewer_bundle.md"

REQUIRED_SECTIONS = [
    "## Options considered",
    "## Layout",
    "## Member kinds",
    "## Bundle root",
    "## bundle.json keys",
    "## Detection without bundle.json",
    "## Validation errors",
    "## Paths",
    "### Member paths",
    "### Paths inside the data",
    "## Link resolution",
    "### Scene to dataset",
    "### Partial to source dataset",
    "### Tomography run to source dataset",
    "## Invariants",
    "## Downstream issues",
]

EXPECTED_KEYS = {
    "bundle_format_version",
    "members",
    "members[].id",
    "members[].kind",
    "members[].path",
    "members[].for",
    "members[].source",
    "created_by",
    "created_by.tool",
    "created_by.tool_version",
}

EXPECTED_REQUIRED = {
    "bundle_format_version": "required",
    "members": "required",
    "members[].id": "required",
    "members[].kind": "required",
    "members[].path": "required",
    "members[].for": "optional",
    "members[].source": "optional",
    "created_by": "optional",
    "created_by.tool": "optional",
    "created_by.tool_version": "optional",
}

ALLOWED_KINDS = {"rf_dataset", "rf_partial", "tomo_run", "scene"}
ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
KEY_TABLE_HEADER = "| Key | Type | Required | Meaning |"


def _doc_text() -> str:
    """Return the raw text of the viewer bundle document."""
    return DOC_PATH.read_text(encoding="utf-8")


def _json_blocks(text: str) -> list[str]:
    """Extract the bodies of ```json fenced blocks."""
    return [m.group(1).strip() for m in re.finditer(r"```json[^\n]*\n(.*?)```", text, re.DOTALL)]


def _key_table(text: str) -> dict[str, tuple[str, str]]:
    """Map key -> (type, required) from the key table under ## bundle.json keys."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line == "## bundle.json keys")
    table: dict[str, tuple[str, str]] = {}
    in_table = False
    for line in lines[start + 1 :]:
        if line == KEY_TABLE_HEADER:
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            if table:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4:
            continue
        if set(cells) == {"---"}:
            continue
        key = cells[0].strip("`")
        table[key] = (cells[1], cells[2])
    assert table, "key table not found under ## bundle.json keys"
    return table


def _section_text(text: str, heading: str) -> str:
    """Return the text between heading and the next ## heading."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line == heading)
    end = next(
        (i for i, line in enumerate(lines[start + 1 :], start + 1) if line.startswith("## ")),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_doc_has_required_sections() -> None:
    """All outline headings are present as exact lines."""
    lines = _doc_text().splitlines()
    for heading in REQUIRED_SECTIONS:
        assert heading in lines, f"missing heading: {heading}"


def test_json_examples_parse() -> None:
    """Every json fenced block parses with json.loads."""
    blocks = _json_blocks(_doc_text())
    assert len(blocks) >= 1, "no json blocks found"
    for block in blocks:
        json.loads(block)


def test_key_table_is_complete() -> None:
    """The key table has exactly the documented keys with correct Required values."""
    table = _key_table(_doc_text())
    assert set(table) == EXPECTED_KEYS
    assert {k: v[1] for k, v in table.items()} == EXPECTED_REQUIRED


def test_examples_use_only_documented_keys() -> None:
    """Bundle examples use documented keys, version 1, valid kinds/ids/links."""
    text = _doc_text()
    table = _key_table(text)
    checked = 0
    for block in _json_blocks(text):
        example = json.loads(block)
        if not isinstance(example, dict) or "bundle_format_version" not in example:
            continue
        checked += 1
        flat: set[str] = set(example)
        for member in example.get("members", []):
            flat.update(f"members[].{k}" for k in member)
        created_by = example.get("created_by", {})
        if isinstance(created_by, dict):
            flat.update(f"created_by.{k}" for k in created_by)
        for key in flat:
            assert key in table, f"undocumented key in example: {key}"
        assert example["bundle_format_version"] == 1
        members = example["members"]
        assert all(m["kind"] in ALLOWED_KINDS for m in members)
        ids = [m["id"] for m in members]
        assert len(set(ids)) == len(ids), "duplicate member ids"
        assert all(ID_RE.fullmatch(i) for i in ids), "invalid member id"
        id_to_kind = {m["id"]: m["kind"] for m in members}
        for member in members:
            for link in ("for", "source"):
                if link in member:
                    target = member[link]
                    assert target in id_to_kind, f"unknown link target: {target}"
                    assert id_to_kind[target] == "rf_dataset", f"link to non-dataset: {target}"
    assert checked >= 1, "no bundle example found"


def test_kind_names_documented() -> None:
    """Member kinds section names the four kinds and no others as kinds."""
    section = _section_text(_doc_text(), "## Member kinds")
    for kind in ("rf_dataset", "rf_partial", "tomo_run", "scene"):
        assert kind in section, f"kind not documented: {kind}"
    for line in section.splitlines():
        if line.startswith("|"):
            first = line.strip().strip("|").split("|")[0].strip()
            assert first not in ("`placement`", "`tomography_gt`"), f"not a kind: {first}"
