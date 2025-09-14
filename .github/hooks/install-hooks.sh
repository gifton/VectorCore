#!/bin/bash
#
# Install Git hooks for the VectorCore project

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
GIT_HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "Installing Git hooks for VectorCore..."

# Install pre-commit hook
if [ -f "$SCRIPT_DIR/pre-commit" ]; then
    cp "$SCRIPT_DIR/pre-commit" "$GIT_HOOKS_DIR/pre-commit"
    chmod +x "$GIT_HOOKS_DIR/pre-commit"
    echo "✅ Installed pre-commit hook"
else
    echo "⚠️  pre-commit hook not found"
fi

# Check if SwiftLint is installed
if command -v swiftlint &> /dev/null; then
    echo "✅ SwiftLint is installed ($(swiftlint --version))"
else
    echo "⚠️  SwiftLint is not installed"
    echo "   Install with: brew install swiftlint"
fi

echo ""
echo "Git hooks installation complete!"
echo "The pre-commit hook will run SwiftLint on staged Swift files before each commit."
echo ""
echo "To bypass the hook (not recommended), use: git commit --no-verify"