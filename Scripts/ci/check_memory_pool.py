#!/usr/bin/env python3
"""Bound MemoryPool regressions outside Swift's cooperative executor.

Compile the current production MemoryPool.swift directly, with a test-only
tracked allocator boundary. No copied pool implementation can drift from it.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import tempfile


def bounded_run(command, *, timeout, env=None):
    """A compiler can spawn children; terminate the entire private process group."""
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=["debug", "release"], default="debug")
    parser.add_argument("--timeout", type=int, default=60, help="seconds allowed per probe process")
    parser.add_argument("--mode", action="append", choices=["lifetime", "retention", "cleanup", "saturation"],
                        help="select modes for diagnosis; the default runs every mode")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    root = Path(__file__).resolve().parents[2]
    fixtures = root / "Tests" / "MemoryPoolRegression"
    failures = []
    with tempfile.TemporaryDirectory(prefix="vectorcore-memorypool-regression-") as directory:
        executable = Path(directory) / "probe"
        command = ["xcrun", "swiftc", "-swift-version", "6", "-parse-as-library",
                   "-O" if args.configuration == "release" else "-Onone",
                   "-module-cache-path", str(Path(directory) / "cache"),
                   str(root / "Sources/VectorCore/Utilities/MemoryPool.swift"),
                   str(fixtures / "AllocationSupport.swift"), str(fixtures / "Probe.swift"),
                   "-o", str(executable)]
        print(f"Compiling MemoryPool regression ({args.configuration})", flush=True)
        bounded_run(command, timeout=180)
        modes = args.mode or ["lifetime", "retention", "cleanup", "saturation"]
        runs = [(strict, mode) for mode in modes
                for strict in ([False, True] if mode == "saturation" else [False])]
        for strict, mode in runs:
            environment = dict(os.environ)
            environment.pop("LIBDISPATCH_COOPERATIVE_POOL_STRICT", None)
            if strict:
                # Apple documents this process-local diagnostic in WWDC21 session 10254.
                environment["LIBDISPATCH_COOPERATIVE_POOL_STRICT"] = "1"
            label = f"{mode}, strict cooperative pool={strict}"
            print(f"Running {label} (timeout {args.timeout}s)", flush=True)
            try:
                bounded_run([str(executable), mode], env=environment, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                failures.append(f"{label}: exceeded {args.timeout}s")
            except subprocess.CalledProcessError as error:
                failures.append(f"{label}: exit {error.returncode}")
    if failures:
        raise SystemExit("MemoryPool regression failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
