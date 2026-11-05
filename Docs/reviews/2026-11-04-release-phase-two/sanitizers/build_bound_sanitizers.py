#!/usr/bin/env python3
"""Produce a fresh source→mandatory-build→bundle association; no driver changes."""
import argparse
from pathlib import Path
from source_binding import produce


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sanitizer', required=True, choices=['address', 'thread'])
    parser.add_argument('--scratch-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    association = produce(Path(__file__).resolve().parents[4], args.output, args.scratch_root, args.sanitizer)
    print(association)


if __name__ == '__main__': main()
