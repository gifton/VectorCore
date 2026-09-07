#!/bin/bash
# Optional installation; never replace a different hook without --force.
set -euo pipefail

force=false
case "${1:-}" in
    "") ;;
    --force) force=true; shift ;;
    *) echo "Usage: $0 [--force]" >&2; exit 2 ;;
esac
if [ "$#" -ne 0 ]; then
    echo "Usage: $0 [--force]" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
cd -- "$repo_root"
# Git resolves linked-worktree common directories and core.hooksPath here.
hooks_dir="$(git rev-parse --git-path hooks)"
destination="$hooks_dir/pre-commit"
if [ -e "$destination" ] || [ -L "$destination" ]; then
    if cmp -s "$script_dir/pre-commit" "$destination"; then
        chmod +x "$destination"
        echo "The VectorCore pre-commit hook is already installed."
        exit 0
    fi
    if [ "$force" != true ]; then
        echo "Existing hook preserved at $destination; use --force to replace it." >&2
        exit 1
    fi
fi
mkdir -p -- "$hooks_dir"
# Remove the destination first so --force replaces a symlink, not its target.
if [ -e "$destination" ] || [ -L "$destination" ]; then
    rm -- "$destination"
fi
cp -- "$script_dir/pre-commit" "$destination"
chmod +x "$destination"
echo "Installed optional pre-commit hook at $destination."
if ! command -v swiftlint >/dev/null 2>&1; then
    echo "SwiftLint is unavailable; install it before using the optional lint hook."
fi
