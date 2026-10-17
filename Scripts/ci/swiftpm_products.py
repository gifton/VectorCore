"""Build-system-agnostic SwiftPM product discovery for the V03 gate drivers.

Products are located from the bin path SwiftPM itself reports for the exact
flags of the build, never from directory-name heuristics. Exactly two test
bundle layouts are accepted (tests: test_swiftpm_products.py):

* native:    <bin>/VectorCorePackageTests.xctest/Contents/MacOS/VectorCorePackageTests
             (one aggregate bundle holding every test target);
* swiftbuild: <bin>/<Target>.xctest/Contents/MacOS/<Target> (one bundle per target).

Library objects, module search paths and module maps are likewise located per
layout (`detect_layout`), for external-consumer and link drivers:

* native:    objects `<bin>/<Target>.build/**/*.o` (one per source), module
             search path `<bin>/Modules`, module map `<bin>/<Target>.build/module.modulemap`;
* swiftbuild: one merged `<bin>/<Target>.o` per target (no module.modulemap beside
             it), per-source objects under
             `<bin>/../../Intermediates.noindex/*/<Config>/<Target>-t.build/Objects-normal/<arch>/*.o`,
             module search path `<bin>` (holding `<Target>.swiftmodule/`), module map
             `<bin>/../../Intermediates.noindex/GeneratedModuleMaps/<Target>.modulemap`.

A missing or ambiguous product is an error; one of several candidates is never
chosen silently. Stdlib only.
"""
from pathlib import Path
import re

from v03_process import run

NATIVE_AGGREGATE = 'VectorCorePackageTests'
BUILD_SYSTEMS = ('native', 'swiftbuild')


class ProductError(ValueError):
    """A required SwiftPM product is missing or ambiguous."""


def build_system_flags(build_system):
    """`--build-system X` when requested; no flag (toolchain default) when None."""
    if build_system is None:
        return []
    if build_system not in BUILD_SYSTEMS:
        raise ProductError('Unknown build system: ' + repr(build_system))
    return ['--build-system', build_system]


def parse_bin_path(text):
    """The reported bin path is the last non-empty output line and must be absolute."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not Path(lines[-1]).is_absolute():
        raise ProductError('swift build --show-bin-path did not report an absolute path: ' + repr(text))
    return Path(lines[-1])


def find_swift(output, *, cwd, environment=None, timeout=60):
    """Resolve the toolchain swift once (`xcrun --find swift`); every later swift invocation uses it."""
    _, text = run(['xcrun', '--find', 'swift'], output, 'find-swift', cwd=cwd, environment=environment, timeout=timeout)
    path = Path(text.strip())
    if not path.is_absolute() or not path.is_file():
        raise ProductError('xcrun --find swift did not report a swift executable: ' + repr(text))
    return str(path)


def check_no_rebuild(log, name):
    """`swift test --skip-build` must reuse the built products (flags identical to the build's)."""
    if re.search(r'^\[\d+/\d+\] (Compiling|Linking)|^Build complete!|^Compiling ', log, re.M):
        raise ProductError('Test invocation rebuilt products: ' + name)


def resolve_bin_path(swift, flags, output, name, *, cwd, environment=None, timeout=120):
    """Run `<swift> build <flags> --show-bin-path` (flags identical to the build's)."""
    record, _ = run([swift, 'build', *flags, '--show-bin-path'], output, name,
                    cwd=cwd, environment=environment, timeout=timeout)
    return parse_bin_path(Path(record['stdout']['path']).read_text())


def _bundle_executables(bin_path, name):
    return sorted(bundle / 'Contents' / 'MacOS' / name
                  for bundle in Path(bin_path).rglob(name + '.xctest')
                  if bundle.is_dir() and (bundle / 'Contents' / 'MacOS' / name).is_file())


def find_test_bundle(bin_path, target):
    """The single loadable bundle executable that contains `target`."""
    candidates = _bundle_executables(bin_path, target)
    if target != NATIVE_AGGREGATE:
        candidates += _bundle_executables(bin_path, NATIVE_AGGREGATE)
    if not candidates:
        raise ProductError(f'No test bundle for {target} under {bin_path}')
    if len(candidates) > 1:
        raise ProductError(f'Ambiguous test bundles for {target} under {bin_path}: '
                           + ', '.join(map(str, candidates)))
    return candidates[0]


def find_test_bundles(bin_path, targets):
    """Map each distinct target to its bundle executable."""
    return {target: find_test_bundle(bin_path, target) for target in sorted(set(targets))}


def find_executable(bin_path, name):
    """A built executable product is `<bin>/<name>`."""
    path = Path(bin_path) / name
    if not path.is_file() or path.is_symlink():
        raise ProductError(f'Executable product {name} not found at {path}')
    return path


def target_of_filter(selector):
    """Test target of an anchored filter such as `^ComprehensiveTests\\.Suite/test`."""
    match = re.match(r'\^(\w+)\\?\.', selector)
    if match is None:
        raise ProductError('Filter does not begin with an anchored target: ' + selector)
    return match.group(1)


def in_log(path, log):
    """True when the path (as reported, or its resolved spelling) appears in the log."""
    path = Path(path)
    return str(path) in log or str(path.resolve()) in log


# Swift Testing's run summary differs by toolchain (both pinned by test_test_output_counts):
#   Swift 6.0 (Xcode 16.2): `Test run with 1 test passed after 0.1 seconds.`
#   Swift 6.2+ (Xcode 26.3, 27): `Test run with 1 test in 1 suite passed after 0.1 seconds.`
# A substring check written against one rendering misread the other on PR #56 (T20 round 3).
_SWIFT_TESTING_RUN = re.compile(r'Test run with (\d+) tests?(?: in (\d+) suites?)? (passed|failed)\b')
_XCTEST_EXECUTED = re.compile(r'Executed (\d+) tests?, with ')


def swift_testing_runs(log):
    """(tests, outcome) for every Swift Testing run summary in the log, in either rendering."""
    return [(int(m[1]), m[3]) for m in _SWIFT_TESTING_RUN.finditer(log)]


def xctest_executed(log):
    """Largest XCTest `Executed N test(s)` count in the log; 0 when XCTest ran nothing or did not report."""
    return max((int(n) for n in _XCTEST_EXECUTED.findall(log)), default=0)


def detect_layout(bin_path):
    """'native' or 'swiftbuild'; anything else (or both at once) is an error.

    swiftbuild: `<scratch>/out/Products/<Config>` with a sibling
    `<scratch>/out/Intermediates.noindex`. native: `<bin>` holding `<Target>.build`
    directories and no Intermediates.noindex beside Products.
    """
    bin_path = Path(bin_path)
    intermediates = bin_path.parent.parent / 'Intermediates.noindex'
    swiftbuild = bin_path.parent.name == 'Products' and intermediates.is_dir()
    native = bin_path.is_dir() and any(p.is_dir() and p.name.endswith('.build') for p in bin_path.iterdir())
    if swiftbuild and native:
        raise ProductError(f'Ambiguous SwiftPM layout (native and swiftbuild) at {bin_path}')
    if swiftbuild:
        return 'swiftbuild'
    if native:
        return 'native'
    raise ProductError(f'Unrecognized SwiftPM layout (neither native nor swiftbuild) at {bin_path}')


def intermediates_dir(bin_path):
    return Path(bin_path).parent.parent / 'Intermediates.noindex'


def module_search_path(bin_path):
    """Directory to pass as `-I` so `import <Target>` finds its swiftmodule."""
    layout = detect_layout(bin_path)
    path = Path(bin_path) / 'Modules' if layout == 'native' else Path(bin_path)
    if not path.is_dir():
        raise ProductError(f'Module directory missing: {path}')
    return path


def module_map(bin_path, target):
    """The clang module map of a C target, for `-fmodule-map-file=`."""
    layout = detect_layout(bin_path)
    if layout == 'native':
        path = Path(bin_path) / f'{target}.build' / 'module.modulemap'
    else:
        path = intermediates_dir(bin_path) / 'GeneratedModuleMaps' / f'{target}.modulemap'
    if not path.is_file():
        raise ProductError(f'Module map for {target} missing ({layout}): {path}')
    return path


def source_objects(bin_path, target):
    """Per-source object files of `target`; never empty."""
    layout = detect_layout(bin_path)
    if layout == 'native':
        found = sorted(Path(bin_path, f'{target}.build').rglob('*.o'))
    else:
        config = Path(bin_path).name
        roots = sorted(intermediates_dir(bin_path).glob(f'*/{config}/{target}-t.build/Objects-normal'))
        if len(roots) > 1:
            raise ProductError(f'Ambiguous object directories for {target}: ' + ', '.join(map(str, roots)))
        if not roots:
            raise ProductError(f'No object directory for {target} (swiftbuild) under {bin_path}')
        architectures = [p for p in roots[0].iterdir() if p.is_dir()]
        if len(architectures) != 1:
            raise ProductError(f'Expected exactly one architecture under {roots[0]}, found {len(architectures)}')
        found = sorted(architectures[0].glob('*.o'))
    if not found:
        raise ProductError(f'No objects for {target} ({layout}) under {bin_path}')
    return found


def link_objects(bin_path, targets):
    """Objects to link for `targets`: native per-source objects, swiftbuild's single merged `<Target>.o`."""
    layout = detect_layout(bin_path)
    result = []
    for target in targets:
        if layout == 'native':
            result += source_objects(bin_path, target)
        else:
            merged = Path(bin_path) / f'{target}.o'
            if not merged.is_file():
                raise ProductError(f'Merged object for {target} missing (swiftbuild): {merged}')
            result.append(merged)
    return result


def snapshot_consumer_inputs(bin_path, output, targets=('VectorCore', 'VectorCoreC')):
    """Copy the link objects, `VectorCore.*` module files and C module map of a build into `output`.

    Layout-agnostic (native or swiftbuild). Returns a dict with `objects` (copied link
    objects), `modules` (copied module search directory), `modulemap` (copied map; its
    umbrella header path is absolute in both layouts) and `records` (relative path to
    the original file, for provenance). Nothing in the build is modified.
    """
    import shutil
    bin_path, output = Path(bin_path), Path(output)
    records, objects = [], []
    for source in link_objects(bin_path, targets):
        destination = output / 'objects' / source.relative_to(bin_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        objects.append(destination)
        records.append((source.relative_to(bin_path), destination))
    modules = output / 'Modules'
    modules.mkdir(parents=True, exist_ok=True)
    found = False
    for entry in sorted(module_search_path(bin_path).glob('VectorCore.*')):
        if entry.suffix == '.o':
            continue
        found = True
        destination = modules / entry.name
        if entry.is_dir():
            shutil.copytree(entry, destination)
            records += [(f.relative_to(bin_path), destination / f.relative_to(entry)) for f in sorted(entry.rglob('*')) if f.is_file()]
        else:
            shutil.copy2(entry, destination)
            records.append((entry.relative_to(bin_path), destination))
    if not found:
        raise ProductError(f'No VectorCore module files under {module_search_path(bin_path)}')
    modulemap = output / 'VectorCoreC.modulemap'
    shutil.copy2(module_map(bin_path, 'VectorCoreC'), modulemap)
    return {'objects': objects, 'modules': modules, 'modulemap': modulemap, 'records': records}
