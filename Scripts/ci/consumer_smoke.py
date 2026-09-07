#!/usr/bin/env python3
"""Compile and execute the README Quick Start as a downstream package."""
import json
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
readme = (root / "README.md").read_text()
section = readme.split("## Quick Start\n", 1)[1].split("\n## ", 1)[0]
blocks = re.findall(r"```swift\n(.*?)```", section, re.S)
if len(blocks) != 1:
    raise SystemExit("Quick Start must contain exactly one complete Swift program")
with tempfile.TemporaryDirectory(prefix="vectorcore-consumer-") as directory:
    fixture = Path(directory)
    source = fixture / "Sources" / "Consumer"
    source.mkdir(parents=True)
    (source / "main.swift").write_text(blocks[0])
    manifest = """// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "Consumer",
    platforms: [.macOS(.v14)],
    dependencies: [.package(name: "VectorCore", path: ROOT)],
    targets: [.executableTarget(name: "Consumer", dependencies: [
        .product(name: "VectorCore", package: "VectorCore")
    ])]
)
""".replace("ROOT", json.dumps(str(root)))
    (fixture / "Package.swift").write_text(manifest)
    subprocess.run(["swift", "run", "--package-path", str(fixture),
                    "-c", "release", "Consumer"], check=True)
