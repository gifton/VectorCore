#!/usr/bin/env python3
"""Compile and execute complete examples from the refreshed documentation, and compile the must-not-compile contracts.

All Swift blocks in REFERENCE_DOCS must be standalone programs. Only the
README Quick Start is included: installation snippets and deferred guides are
not programs. Use --list-examples to inspect selection without running Swift.

CONTRACTS (PR #56 review SELF1) are compiled from the same scratch package, each
as its own executable target with `define` as a compilation condition when given.
A row with expect "fail" must not build: with errorLines "code", exactly one
`error:` on every non-blank, non-comment, non-import line of the fixture and none
elsewhere (the D02 rule); with a list, one error per listed line (a line listed
twice expects exactly two errors there), and `optionalErrorLines` may error or not
(compiler-dependent diagnostics); with neither, at least one error in the fixture. `diagnostics` are substrings the error
text must contain and `diagnosticsAny` substrings of which one must appear. A
"compile" row must build; a "run" row must build, run and print `marker`. Use
--list-contracts to check the table without running Swift.
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

_R01 = "Scripts/ci/fixtures/compile-contracts/R01MetricScalarContract.swift"
_R02 = "Scripts/ci/fixtures/compile-contracts/R02CollectionUpdateConsumer.swift"
_B01 = "Scripts/ci/fixtures/B01BareConformanceNegative.swift"
CONTRACTS = (
    # R01: a DistanceMetric whose Scalar is Double is rejected by every public search entry; Float compiles.
    {"id": "R01-reject-double", "path": _R01, "define": "REJECT_DOUBLE", "expect": "fail", "errorLines": [5, 8, 11]},
    {"id": "R01-float-control", "path": _R01, "expect": "compile"},
    # R02: a MutableVectorCollection without `update(id:vector:)` does not conform; with it, the public
    # transactional witness runs.
    {"id": "R02-missing-update", "path": _R02, "define": "MISSING_UPDATE", "expect": "fail",
     "diagnostics": ["does not conform to protocol 'MutableVectorCollection'"]},
    {"id": "R02-transactional-witness", "path": _R02, "expect": "run", "marker": "R02_PUBLIC_TRANSACTIONAL_WITNESS_OK"},
    # D02 negatives: every code line names a removed member and is one compile error; the Logger pair is the
    # scoped-import contract.
    {"id": "D02-storage-negative", "path": "Scripts/ci/tests/D02/StorageNegative.swift", "expect": "fail", "errorLines": "code"},
    {"id": "D02-routed-negative", "path": "Scripts/ci/tests/D02/RoutedNegative.swift", "expect": "fail", "errorLines": "code"},
    {"id": "D02-hooks-negative", "path": "Scripts/ci/tests/D02/HooksNegative.swift", "expect": "fail", "errorLines": "code"},
    {"id": "D02-device-negative", "path": "Scripts/ci/tests/D02/DeviceNegative.swift", "expect": "fail", "errorLines": "code"},
    # LoggerNegative line 10 (`VectorCore.Logger`) errors on every supported compiler; line 9 (a bare `Logger` with both
    # modules imported) is ambiguous under Swift 6.4 and accepted by Swift 6.0 (Xcode 16.2, the consumer job). The line
    # numbers are absolute: editing the fixture's header comment moves them.
    {"id": "D02-logger-negative", "path": "Scripts/ci/tests/D02/LoggerNegative.swift", "expect": "fail", "errorLines": [10],
     "optionalErrorLines": [9], "diagnostics": ["type 'VectorCore' has no member 'Logger'"]},
    {"id": "D02-logger-qualification", "path": "Scripts/ci/tests/D02/LoggerQualification.swift", "expect": "run",
     "marker": "D02_LOGGER_QUALIFICATION_COMPLETE"},
    # B01: no vector type conforms to BinaryCodable bare. `BinaryCodable` is BinaryEncodable & BinaryDecodable, so the
    # compiler reports one missing conformance per protocol: exactly two errors on the one line. With no condition
    # the body is empty and compiles.
    {"id": "B01-bare-conformance-fixed", "path": _B01, "define": "B01_FIXED", "expect": "fail", "errorLines": [5, 5],
     "diagnostics": ["requires that", "BinaryEncodable", "BinaryDecodable"]},
    {"id": "B01-bare-conformance-dynamic", "path": _B01, "define": "B01_DYNAMIC", "expect": "fail", "errorLines": [7, 7],
     "diagnostics": ["requires that", "BinaryEncodable", "BinaryDecodable"]},
    {"id": "B01-bare-conformance-optimized", "path": _B01, "define": "B01_OPTIMIZED", "expect": "fail", "errorLines": [9, 9],
     "diagnostics": ["requires that", "BinaryEncodable", "BinaryDecodable"]},
    {"id": "B01-bare-conformance-int8", "path": _B01, "define": "B01_INT8", "expect": "fail", "errorLines": [11, 11],
     "diagnostics": ["requires that", "BinaryEncodable", "BinaryDecodable"]},
    {"id": "B01-bare-conformance-control", "path": _B01, "expect": "compile"},
    # R05: the metric-free `computedScore` property is gone; only the method remains.
    {"id": "R05-old-property", "path": "Scripts/ci/fixtures/R05OldPropertyNegative.swift", "expect": "fail", "errorLines": [5],
     "diagnostics": ["cannot convert value of type", "(any DistanceMetric<Float>) -> Float?"]},
)


def code_lines(text):
    """Line numbers the D02 negative rule expects to error: non-blank, not a `//` comment, not an import."""
    return [number for number, line in enumerate(text.split("\n"), 1)
            if line.strip() and not line.strip().startswith("//") and not line.startswith("import ")]


def judge_contract(row, status, output, source, file_name=None):
    """(passed, detail) for one contract given the build (or run) exit status and its combined output."""
    name = file_name or Path(row["path"]).name
    if row["expect"] == "compile":
        return status == 0, f"exit {status}"
    if row["expect"] == "run":
        present = row["marker"] in output
        return status == 0 and present, f"exit {status}; marker {'present' if present else 'absent'}"
    plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output)   # swift build colours its diagnostics
    errors = re.findall(rf"{re.escape(name)}:(\d+):\d+: error: ([^\n]*)", plain)
    lines = sorted(int(number) for number, _ in errors)
    detail = f"exit {status}; error lines {lines}"
    if status == 0:
        return False, detail + "; the fixture compiled"
    expected = row.get("errorLines")
    if expected == "code":
        expected = code_lines(source)
    optional = sorted(row.get("optionalErrorLines", ()))
    if expected is not None:
        required = sorted(expected)
        present = [line for line in lines if line not in optional]
        if present != required or any(lines.count(line) > 1 for line in optional):
            return False, detail + (f"; expected exactly one error on each of {required}"
                                    + (f" and at most one on each of {optional}" if optional else ""))
    elif not lines:
        return False, detail + "; no error in the fixture"
    text = "\n".join(message for _, message in errors)
    for needed in row.get("diagnostics", ()):
        if needed not in text:
            return False, detail + f"; missing diagnostic {needed!r}"
    alternatives = row.get("diagnosticsAny", ())
    if alternatives and not any(option in text for option in alternatives):
        return False, detail + f"; none of {list(alternatives)} in the diagnostics"
    return True, detail


def collect_contracts(root):
    for row in CONTRACTS:
        if not (root / row["path"]).is_file():
            raise SystemExit(f"Missing compile-contract fixture: {row['path']}")
    return [dict(row) for row in CONTRACTS]


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


def run_examples(root, examples, contracts=()):
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
        # Each contract is its own target; a `@main` fixture keeps its file name, any other is the target's main.swift.
        for index, row in enumerate(contracts):
            name = f"Contract{index}"
            text = (root / row["path"]).read_text()
            row["file"] = Path(row["path"]).name if "@main" in text else "main.swift"
            row["target"] = name
            source = fixture / "Sources" / name
            source.mkdir(parents=True)
            (source / row["file"]).write_text(text)
            settings = f', swiftSettings: [.define("{row["define"]}")]' if row.get("define") else ""
            targets.append(
                f'.executableTarget(name: "{name}", dependencies: ['
                f'.product(name: "VectorCore", package: "VectorCore")]{settings})')
            print(f'{name}: {row["id"]} ({row["path"]}, expect {row["expect"]})', flush=True)

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
        # One product per build: the first builds the dependency, the rest are incremental. A negative contract must
        # not be part of a whole-package build, so every product is built on its own.
        build = ["swift", "build", "--package-path", str(fixture), "-c", "release"]
        for index in range(len(examples)):
            subprocess.run(build + ["--product", f"Consumer{index}"], check=True, timeout=1800)
        binary_directory = Path(subprocess.check_output(
            build + ["--show-bin-path"], text=True, timeout=600).strip())
        for index, example in enumerate(examples):
            print(f'Running {example["document"]}, Swift block {example["block"]}', flush=True)
            subprocess.run([str(binary_directory / f"Consumer{index}")], check=True, timeout=60)
        print(f"Passed {len(examples)} documentation consumer programs.", flush=True)
        failures = []
        for row in contracts:
            result = subprocess.run(build + ["--product", row["target"]], capture_output=True, text=True, timeout=1800)
            output = result.stdout + result.stderr
            status = result.returncode
            if row["expect"] == "run" and status == 0:
                run = subprocess.run([str(binary_directory / row["target"])], capture_output=True, text=True, timeout=60)
                status, output = run.returncode, run.stdout + run.stderr
            passed, detail = judge_contract(row, status, output, (root / row["path"]).read_text(), row["file"])
            print(f'{row["id"]}: {"passed" if passed else "FAILED"} ({detail})', flush=True)
            if not passed:
                plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output)
                diagnostics = [line for line in plain.splitlines() if re.search(r"\.swift:\d+:\d+: ", line)]
                failures.append(f'{row["id"]}: {detail}\n' + "\n".join(diagnostics[:40] or [plain[-2000:]]))
        if failures:
            raise SystemExit("Compile contracts failed:\n" + "\n".join(failures))
        print(f"Passed {len(contracts)} compile contracts.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-examples", action="store_true",
                        help="print selected programs as JSON without compiling")
    parser.add_argument("--list-contracts", action="store_true",
                        help="print the compile-contract table as JSON without compiling")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.list_contracts:
        print(json.dumps(collect_contracts(root), indent=2))
        return
    examples = collect_examples(root)
    if args.list_examples:
        print(json.dumps(examples, indent=2))
    else:
        run_examples(root, examples, collect_contracts(root))


if __name__ == "__main__":
    main()
