#!/usr/bin/env python3
"""Validate GitHub Actions required job results."""
import json
import os
import re
import sys


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main():
    expected = sys.argv[1:]
    if (not expected or len(set(expected)) != len(expected)
            or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", job) for job in expected)):
        raise ValueError("supply a nonempty, unique list of expected job IDs")
    needs = json.loads(os.environ.get("NEEDS_JSON", ""), object_pairs_hook=unique_object)
    if not isinstance(needs, dict):
        raise ValueError("NEEDS_JSON must be an object")
    if set(needs) != set(expected):
        raise ValueError(f"job IDs differ: missing={sorted(set(expected) - set(needs))}, "
                         f"unexpected={sorted(set(needs) - set(expected))}")
    unsuccessful = [job for job in expected
                    if not isinstance(needs[job], dict) or needs[job].get("result") != "success"]
    if unsuccessful:
        raise ValueError("required jobs did not succeed: " + ", ".join(unsuccessful))
    print("All required jobs succeeded: " + ", ".join(expected))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, TypeError) as error:
        print(f"CI Required: {error}", file=sys.stderr)
        sys.exit(1)
