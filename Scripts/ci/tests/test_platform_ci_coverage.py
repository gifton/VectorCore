"""CI must build EVERY Package.swift library product on EVERY declared platform.

Reads Package.swift (products, platforms) and the `platforms` job of ci.yml as text; stdlib only,
no git history, no toolchain. Negative controls mutate both inputs and must be rejected; positive
controls (a consistent product/platform addition) must be accepted.
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Scripts/ci'))
import package_products as pkg

CATALYST = 'macOS,variant=Mac Catalyst'
SIMULATOR = {'iOS': 'iOS Simulator', 'tvOS': 'tvOS Simulator', 'watchOS': 'watchOS Simulator',
             'visionOS': 'visionOS Simulator'}


def expected_destinations(platforms):
    unknown = [p for p in platforms if p != 'macOS' and p not in SIMULATOR]
    if unknown:
        raise AssertionError(f'platform(s) {unknown} have no CI destination mapping in this test')
    dests = {SIMULATOR[p] for p in platforms if p in SIMULATOR}
    if 'iOS' in platforms:
        dests.add(CATALYST)  # Catalyst builds the iOS-declared sources
    return dests


def platforms_job(ci_text):
    lines = ci_text.splitlines()
    start = next((i for i, l in enumerate(lines) if l == '  platforms:'), None)
    if start is None:
        raise AssertionError('ci.yml has no platforms job')
    end = next((i for i in range(start + 1, len(lines)) if re.match(r'^  \S', lines[i])), len(lines))
    return '\n'.join(lines[start:end])


def build_coverage(job_text, products):
    """Return {(product, destination)} that the job's xcodebuild commands cover (Release only)."""
    text = re.sub(r'\\\n\s*', ' ', job_text)
    covered, platform_list, product_list = set(), None, None
    pending, guarded = None, False
    for raw in text.splitlines():
        line = re.sub(r'\s+', ' ', raw.strip())
        loop = re.match(r"for platform in ((?:'[^']*' ?)+); do$", line)
        if loop:
            platform_list = re.findall(r"'([^']*)'", loop.group(1))
            continue
        # Fail-closed enumeration: assignment (no `|| true`), then `test -n`, then the loop over the variable.
        # A bare `for product in $(...)` is rejected because `bash -e` ignores a failed substitution there.
        gen = re.match(r'products="\$\(python3 Scripts/ci/package_products\.py((?: --exclude \w+)*)\)"$', line)
        if gen:
            excluded = re.findall(r'--exclude (\w+)', gen.group(1))
            pending, guarded, product_list = [p for p in products if p not in excluded], False, None
            continue
        if line == 'test -n "$products"' and pending is not None:
            guarded = True
            continue
        if line == 'for product in $products; do':
            product_list = pending if guarded else None
            continue
        if not line.startswith('xcodebuild '):
            continue
        scheme = re.search(r'-scheme (?:"\$product"|(\w+))', line)
        if not scheme or '-configuration Release' not in line or not line.endswith('build -quiet') \
                or 'CODE_SIGNING_ALLOWED=NO' not in line:
            continue
        if scheme.group(1):
            schemes = [scheme.group(1)]
        elif '-scheme "$product"' in line and product_list is not None:
            schemes = product_list
        else:
            continue
        if '-destination "generic/platform=$platform"' in line and platform_list is not None:
            dests = platform_list
        else:
            literal = re.search(r'-destination "generic/platform=([^$"]+)"', line)
            dests = [literal.group(1)] if literal else []
        covered.update((s, d) for s in schemes for d in dests)
    return covered


def divergences(package_text, ci_text):
    products = pkg.library_products(package_text)
    wanted = {(p, d) for p in products for d in expected_destinations(pkg.declared_platforms(package_text))}
    covered = build_coverage(platforms_job(ci_text), products)
    problems = [f'not built in CI: {p} on {d}' for p, d in sorted(wanted - covered)]
    problems += [f'CI builds a destination Package.swift does not declare: {p} on {d}'
                 for p, d in sorted(covered - wanted)
                 if d not in {x for _, x in wanted}]
    return problems


class PlatformCICoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = (ROOT / 'Package.swift').read_text()
        cls.ci = (ROOT / '.github/workflows/ci.yml').read_text()

    def test_actual_repository_has_no_divergence(self):
        self.assertEqual(divergences(self.package, self.ci), [])
        # Guard against a vacuous pass: the real inputs must really enumerate the expected sets.
        self.assertIn('VectorCoreBenchmarking', pkg.library_products(self.package))
        self.assertEqual(len(expected_destinations(pkg.declared_platforms(self.package))), 5)

    def test_executable_products_are_not_library_products(self):
        self.assertNotIn('vectorcore-bench', pkg.library_products(self.package))

    def test_hand_listed_products_are_not_accepted_as_enumeration(self):
        handlisted = self.ci.replace('for product in $products; do', 'for product in VectorCoreBenchmarking; do')
        self.assertNotEqual(handlisted, self.ci)
        problems = divergences(self.package, handlisted)
        self.assertTrue(any('VectorCoreBenchmarking' in p for p in problems), problems)

    def test_new_library_product_is_picked_up_by_the_generated_loop(self):
        mutated = self.package.replace(
            '        .executable(', '        .library(name: "VectorCoreExtra", targets: ["VectorCore"]),\n        .executable(', 1)
        self.assertNotEqual(mutated, self.package)
        self.assertEqual(divergences(mutated, self.ci), [])  # positive control
        # A CI that stops using the generator (literal step per product) misses the new product.
        literal = self.ci.replace('for product in $products; do', 'for product in VectorCoreBenchmarking; do')
        self.assertTrue(any('VectorCoreExtra' in p for p in divergences(mutated, literal)))

    def test_new_declared_platform_is_checked_against_ci(self):
        # Already-covered platform added again: positive control.
        mutated = self.package.replace('.visionOS(.v1)', '.visionOS(.v1),\n        .tvOS(.v17)', 1)
        self.assertEqual(divergences(mutated, self.ci), [])
        # A declared platform with no CI destination mapping must fail loudly, never pass silently.
        unmapped = self.package.replace('.visionOS(.v1)', '.visionOS(.v1),\n        .macCatalyst(.v17)', 1)
        with self.assertRaises((AssertionError, ValueError)):
            divergences(unmapped, self.ci)

    def test_dropping_each_ci_platform_is_rejected(self):
        for name in list(SIMULATOR.values()) + [CATALYST]:
            mutated = self.ci.replace(f"'{name}' ", '').replace(f" '{name}';", ';')
            self.assertNotEqual(mutated, self.ci, name)
            problems = divergences(self.package, mutated)
            with self.subTest(platform=name):
                self.assertTrue(problems and all(name in p for p in problems), problems)
                self.assertGreaterEqual(len(problems), 2)  # both products lose it

    def test_removing_a_platform_from_package_while_ci_keeps_it_is_rejected(self):
        mutated = self.package.replace('        .visionOS(.v1)\n', '').replace('.watchOS(.v10),', '.watchOS(.v10)')
        self.assertNotEqual(mutated, self.package)
        problems = divergences(mutated, self.ci)
        self.assertTrue(any('visionOS Simulator' in p and 'does not declare' in p for p in problems), problems)

    def test_consistent_platform_removal_and_product_addition_is_accepted(self):
        pkg_mut = self.package.replace('        .executable(', '        .library(name: "VectorCoreExtra", targets: ["VectorCore"]),\n        .executable(', 1)
        pkg_mut = pkg_mut.replace('        .watchOS(.v10),\n', '')
        ci_mut = self.ci.replace("'watchOS Simulator' ", '')
        self.assertNotEqual(pkg_mut, self.package)
        self.assertNotEqual(ci_mut, self.ci)
        self.assertEqual(divergences(pkg_mut, ci_mut), [])  # positive control
        self.assertTrue(divergences(pkg_mut, self.ci))  # CI still building the removed platform diverges

    def test_loop_must_fail_closed(self):
        old_form = self.ci.replace(
            'products="$(python3 Scripts/ci/package_products.py --exclude VectorCore)"\n          test -n "$products"\n          for product in $products; do',
            'for product in $(python3 Scripts/ci/package_products.py --exclude VectorCore); do')
        self.assertNotEqual(old_form, self.ci)
        no_guard = self.ci.replace('          test -n "$products"\n', '')
        or_true = self.ci.replace('--exclude VectorCore)"', '--exclude VectorCore) || true"')
        or_true2 = self.ci.replace('--exclude VectorCore)"\n', '--exclude VectorCore)" || true\n')
        guard_before = self.ci.replace('products="$(python3 Scripts/ci/package_products.py --exclude VectorCore)"\n          test -n "$products"',
                                       'test -n "$products"\n          products="$(python3 Scripts/ci/package_products.py --exclude VectorCore)"')
        for name, mutated in [('old for-in-substitution form', old_form), ('missing test -n', no_guard),
                              ('|| true inside', or_true), ('|| true after', or_true2), ('guard before assignment', guard_before)]:
            self.assertNotEqual(mutated, self.ci, name)
            with self.subTest(name):
                self.assertTrue(any('VectorCoreBenchmarking' in p for p in divergences(self.package, mutated)))

    def test_shell_body_fails_when_lister_fails(self):
        import subprocess, tempfile
        job = platforms_job(self.ci)
        body = job[job.index('products="$('):job.index('          for product in $products; do')]
        script = 'set -e\n' + body.replace('python3 Scripts/ci/package_products.py', 'false') + 'echo REACHED_LOOP\n'
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(['bash', '-e', '-c', script], cwd=tmp, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('REACHED_LOOP', result.stdout)

    def test_missing_release_configuration_or_quiet_build_is_rejected(self):
        job = platforms_job(self.ci)
        products = pkg.library_products(self.package)
        self.assertTrue(build_coverage(job, products))
        self.assertEqual(build_coverage(job.replace('-configuration Release', '-configuration Debug'), products), set())

    def test_excluding_a_product_in_the_generator_without_a_literal_step_is_rejected(self):
        mutated = self.ci.replace('package_products.py --exclude VectorCore', 'package_products.py --exclude VectorCore --exclude VectorCoreBenchmarking')
        self.assertTrue(any('VectorCoreBenchmarking' in p for p in divergences(self.package, mutated)))

    def test_original_vectorcore_only_ci_is_rejected(self):
        # The pre-T07 CI (VectorCore step only) is exactly the gap this policy closes.
        start = self.ci.index('      - name: Build every other library product')
        end = self.ci.index('      - name: Compile library and tests for Intel macOS')
        problems = divergences(self.package, self.ci[:start] + self.ci[end:])
        self.assertEqual(len(problems), 5)
        self.assertTrue(all('VectorCoreBenchmarking' in p for p in problems))

    def test_parser_fails_closed(self):
        base = self.package
        nonliteral = base.replace('name: "VectorCoreBenchmarking",', 'name: benchName,', 1)
        self.assertNotEqual(nonliteral, base)
        with self.assertRaisesRegex(ValueError, 'literal names'):
            pkg.library_products(nonliteral)
        for bad in ('products: [.library(name: "A", targets: ["A"])] + more,',
                    'products: [.library(name: "A", targets: ["A"])].filter { true },'):
            with self.subTest(bad), self.assertRaises(ValueError):
                pkg.library_products('let p = Package(\n' + bad + '\n)')
        with self.assertRaises(ValueError):
            pkg.library_products('let p = Package(products: shared)')
        with self.assertRaises(ValueError):
            pkg.declared_platforms('Package(\nplatforms: [.iOS(v17)]\n)')
        self.assertEqual(pkg.library_products('Package(\nproducts: [.library(name: "A", targets: [])]\n)'), ['A'])

    def test_missing_platforms_job_is_rejected(self):
        with self.assertRaises(AssertionError):
            divergences(self.package, self.ci.replace('  platforms:', '  other:'))


if __name__ == '__main__':
    unittest.main()
