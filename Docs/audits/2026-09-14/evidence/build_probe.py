#!/usr/bin/env python3
"""Compile an audit observation client against an already-built local VectorCore.

This does not build the package or execute the client. Use the same toolchain and
source revision as the chosen Debug/Release objects. Some clients deliberately
trigger faults when given special arguments; consult README.md before running.
"""
from pathlib import Path
import argparse
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--configuration", choices=("debug", "release"), default="release")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    build = root / ".build" / args.configuration
    objects = sorted((build / "VectorCore.build").glob("*.o"))
    objects += sorted((build / "VectorCoreC.build").rglob("*.o"))
    if not objects or not (build / "Modules" / "VectorCore.swiftmodule").exists():
        parser.error("Build the matching package configuration first (swift test [-c release]).")
    output_dir = Path(tempfile.gettempdir()) / "vectorcore-validated-audit-replay"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / (args.source.stem + "-" + args.configuration)
    command = ["swiftc", "-parse-as-library", "-g"]
    if args.configuration == "release":
        command.append("-O")
    command += ["-I", str(build / "Modules"),
                "-Xcc", "-fmodule-map-file=" + str(build / "VectorCoreC.build/module.modulemap"),
                "-Xcc", "-I" + str(root / "Sources/VectorCoreC/include"),
                str(args.source.resolve()), *map(str, objects), "-o", str(output)]
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
