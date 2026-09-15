#!/usr/bin/env python3
"""Compile and execute complete examples from the refreshed documentation.

All Swift blocks in REFERENCE_DOCS must be standalone programs. Only the
README Quick Start is included: installation snippets and deferred guides are
not programs. Use --list-examples to inspect selection without running Swift.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

REFERENCE_DOCS = (
    "Docs/API_Overview_Map.md",
    "Docs/Memory_Alignment.md",
    "Docs/SoA_Layout_Contract.md",
    "Docs/Linear_Algebra_and_Projection.md",
)


def swift_blocks(text, document):
    starts = re.findall(r"^```swift[ \t]*$", text, re.M)
    blocks = re.findall(r"^```swift[ \t]*\n(.*?)^```[ \t]*$", text, re.M | re.S)
    if len(starts) != len(blocks):
        raise SystemExit(f"Unclosed Swift code fence in {document}")
    if not blocks:
        raise SystemExit(f"No complete Swift examples found in {document}")
    return [{"document": document, "block": index, "source": source}
            for index, source in enumerate(blocks, start=1)]


def collect_examples(root):
    readme = (root / "README.md").read_text()
    section = re.search(r"^## Quick Start\n(.*?)(?=^## |\Z)", readme, re.M | re.S)
    if section is None:
        raise SystemExit("README.md must contain a ## Quick Start section")
    examples = swift_blocks(section.group(1), "README.md")
    if len(examples) != 1:
        raise SystemExit("README.md Quick Start must contain exactly one complete Swift program")
    for document in REFERENCE_DOCS:
        examples.extend(swift_blocks((root / document).read_text(), document))
    return examples


def run_examples(root, examples):
    with tempfile.TemporaryDirectory(prefix="vectorcore-consumer-") as directory:
        fixture = Path(directory)
        targets = []
        for index, example in enumerate(examples):
            name = f"Consumer{index}"
            source = fixture / "Sources" / name
            source.mkdir(parents=True)
            (source / "main.swift").write_text(example["source"])
            targets.append(
                f'.executableTarget(name: "{name}", dependencies: ['
                '.product(name: "VectorCore", package: "VectorCore")])')
            print(f'{name}: {example["document"]}, Swift block {example["block"]}', flush=True)

        manifest = """// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "DocumentationConsumers",
    platforms: [.macOS(.v14)],
    dependencies: [.package(name: "VectorCore", path: ROOT)],
    targets: [
        TARGETS
    ]
)
""".replace("ROOT", json.dumps(str(root))).replace("TARGETS", ",\n        ".join(targets))
        (fixture / "Package.swift").write_text(manifest)
        # Build the dependency once and each example as its own executable.
        build = ["swift", "build", "--package-path", str(fixture), "-c", "release"]
        subprocess.run(build, check=True)
        binary_directory = Path(subprocess.check_output(
            build + ["--show-bin-path"], text=True).strip())
        for index, example in enumerate(examples):
            print(f'Running {example["document"]}, Swift block {example["block"]}', flush=True)
            subprocess.run([str(binary_directory / f"Consumer{index}")], check=True, timeout=60)
        print(f"Passed {len(examples)} documentation consumer programs.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-examples", action="store_true",
                        help="print selected programs as JSON without compiling")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    examples = collect_examples(root)
    if args.list_examples:
        print(json.dumps(examples, indent=2))
    else:
        run_examples(root, examples)


if __name__ == "__main__":
    main()
