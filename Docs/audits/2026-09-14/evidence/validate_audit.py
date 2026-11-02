#!/usr/bin/env python3
"""Check this audit's coverage, source references, links and retained hashes."""
from pathlib import Path
import argparse
import hashlib
import json
import re
from urllib.parse import unquote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-source", action="store_true",
                        help="Also require current tracked files to match the audited revision snapshot.")
    args = parser.parse_args()
    evidence = Path(__file__).resolve().parent
    audit = evidence.parent
    root = audit.parents[2]
    data = json.loads((audit / "claims.json").read_text())
    manifest = json.loads((evidence / "manifest.json").read_text())
    records = data["records"]
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "Duplicate record IDs"
    required = {*(f"A{i}" for i in range(1, 11)), *(f"C{i}" for i in range(1, 21)),
                *(f"PERF{i}" for i in range(1, 10)), *(f"B{i}" for i in range(1, 8))}
    mapped = {c for r in records for c in r["canonical_ids"]}
    assert mapped == required, ("Canonical coverage mismatch", required - mapped, mapped - required)
    categories = {"architecture", "code quality", "performance"}
    verdicts = {"confirmed-reproduced", "confirmed-source", "qualified", "rejected", "unverified"}
    fields = {"id", "original_refs", "claim", "verdict", "priority", "category", "exposure",
              "sources", "evidence", "correction", "improvement", "acceptance", "dependencies"}
    for record in records:
        assert fields <= record.keys(), record["id"]
        assert record["category"] in categories, record["id"]
        assert record["verdict"] in verdicts, record["id"]
        assert record["priority"] in {"P1", "P2", "P3", "Context"}, record["id"]
        assert record["evidence"] and record["acceptance"] and record["sources"], record["id"]
        assert all(d in ids for d in record["dependencies"]), record["id"]
        for location in record["sources"]:
            name, line = location.rsplit(":", 1)
            source = root / name
            assert source.is_file(), (record["id"], location)
            assert 1 <= int(line) <= len(source.read_text().splitlines()), (record["id"], location)
    for document in audit.rglob("*.md"):
        text = document.read_text()
        for match in re.finditer(r"\[[^\]\n]+\]\(([^\n]+?)\)", text):
            target = unquote(match[1].strip().strip("<>"))
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue
            path, _, fragment = target.partition("#")
            # Historical absolute /private/tmp paths in notes are provenance only.
            if path.startswith("/private/tmp/"):
                continue
            source_match = re.search(r":([0-9]+)$", path)
            if source_match:
                path = path[:source_match.start()]
            resolved = (document.parent / path) if path else document
            assert resolved.exists(), (str(document.relative_to(audit)), target)
            if fragment and resolved.name == "CLAIMS.md":
                assert fragment.upper() in ids, target
    for name, expected in manifest["artifact_sha256"].items():
        actual = hashlib.sha256((audit / name).read_bytes()).hexdigest()
        assert actual == expected, ("Changed retained artifact", name)
    if args.verify_source:
        for name, expected in manifest["tracked_file_sha256"].items():
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            assert actual == expected, ("Source differs from audited snapshot", name)
    print(f"Validated {len(records)} records, {len(required)} canonical labels, "
          f"{len(manifest['artifact_sha256'])} retained files, source references and local links.")
    if args.verify_source:
        print(f"Verified {len(manifest['tracked_file_sha256'])} tracked-file hashes.")


if __name__ == "__main__":
    main()
