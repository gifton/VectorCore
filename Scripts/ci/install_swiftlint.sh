#!/usr/bin/env bash
set -euo pipefail
# SHA-256 from realm/SwiftLint's 0.65.1 portable_swiftlint.zip release asset.
version=0.65.1
checksum=c1e429b0599cf1b516f369a2d9ec04eaf0e436f3c12b637df8851fa52ff694d0
install_dir="${1:?usage: install_swiftlint.sh destination-directory}"
mkdir -p "$install_dir"
archive="$install_dir/swiftlint.zip"
curl --fail --silent --show-error --location --retry 3 \
  "https://github.com/realm/SwiftLint/releases/download/$version/portable_swiftlint.zip" \
  --output "$archive"
printf '%s  %s\n' "$checksum" "$archive" | shasum -a 256 --check
unzip -oq "$archive" -d "$install_dir"
"$install_dir/swiftlint" version
