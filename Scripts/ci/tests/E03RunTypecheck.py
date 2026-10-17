#!/usr/bin/env python3
"""Compile and run E03TypecheckDriver.swift against a SwiftPM build, native or swiftbuild layout.

usage: E03RunTypecheck.py <bin-path> <evidence-dir>
<bin-path> is `swift build --show-bin-path` output. Every child has a finite timeout.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import swiftpm_products as products  # noqa: E402

BUILD_TIMEOUT_S = 300
RUN_TIMEOUT_S = 120


def main(argv):
    bin_path, evidence = Path(argv[1]), Path(argv[2])
    evidence.mkdir(parents=True, exist_ok=False)
    modules = products.module_search_path(bin_path)
    modulemap = products.module_map(bin_path, 'VectorCoreC')
    driver = evidence / 'E03TypecheckDriver'
    subprocess.run(['swiftc', str(HERE / 'E03TypecheckDriver.swift'), '-o', str(driver)],
                   check=True, timeout=BUILD_TIMEOUT_S)
    completed = subprocess.run([str(driver), str(HERE / 'E03ConfigurationConsumer.swift'), str(modules),
                                str(modulemap), str(evidence)], timeout=RUN_TIMEOUT_S)
    return completed.returncode


if __name__ == '__main__':
    sys.exit(main(sys.argv))
