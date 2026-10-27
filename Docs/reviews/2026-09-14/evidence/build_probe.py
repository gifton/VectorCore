"""Link one trusted review client against existing Apple arm64 Release objects."""

from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[4]
source = Path(sys.argv[1]).resolve()
build = root / ".build/arm64-apple-macosx/release"
objects = sorted((build / "VectorCore.build").glob("*.o"))
objects += sorted((build / "VectorCoreC.build").rglob("*.o"))
if not objects or not (build / "Modules/VectorCore.swiftmodule").exists():
    raise SystemExit("Run swift test -c release on the reviewed revision first.")

destination = Path("/private/tmp/vectorcore-review-replay")
destination.mkdir(exist_ok=True)
output = destination / source.stem
subprocess.run(
    [
        "swiftc", "-O", "-parse-as-library", "-g",
        "-I", str(build / "Modules"),
        "-Xcc", "-fmodule-map-file=" + str(build / "VectorCoreC.build/module.modulemap"),
        "-Xcc", "-I" + str(root / "Sources/VectorCoreC/include"),
        str(source), *map(str, objects), "-o", str(output),
    ],
    check=True,
)
print(output)
